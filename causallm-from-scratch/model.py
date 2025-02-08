import math
import inspect
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F

from config import ModelConfig


def precompute_freqs(dim: int, max_position_embeddings: int = 4096, base: int = 10000):
    # \theta_i=10000^{-2i/d},i \in [0,2,...d/2-1]
    # shape: [d/2]
    theta = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))[: (dim // 2)]

    # shape: [max_position_embeddings]
    t = torch.arange(max_position_embeddings)

    # shape: [max_position_embeddings, d/2]
    freqs = torch.outer(t, theta).float()
    freqs_cos = torch.cos(freqs)
    freqs_sin = torch.sin(freqs)
    return freqs_cos, freqs_sin


def apply_rope(x: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor):
    seq_len = x.shape[-2]
    # 偶数奇数位置的元素分解
    # eg. [1,2,3,4] -> [1,3],[2,4]
    # shape: [bs, n_heads, len, d_q/2]
    x_r, x_i = x.float().reshape(x.shape[:-1] + (-1, 2)).unbind(-1)
    cos = freqs_cos[:seq_len].reshape(1, 1, seq_len, -1)
    sin = freqs_sin[:seq_len].reshape(1, 1, seq_len, -1)
    # 偶数位置元素的旋转 = 偶数位置元素的余弦 - 奇数位置元素的正弦
    # 奇数位置元素的旋转 = 奇数位置元素的余弦 + 偶数位置元素的正弦
    # shape: [bs, n_heads, len, d_q/2]
    x_out_r = x_r * cos - x_i * sin
    x_out_i = x_r * sin + x_i * cos

    # flatten start from: dimensions
    # eg.tensor([[[[[0],[2],[4]]]]]),tensor([[[[[1],[3],[5]]]]])
    # -> tensor([[[[[[0, 1]], [[2, 3]],[[4, 5]]]]]])
    # -> tensor([[[[0, 1, 2, 3, 4, 5]]]])
    # shape: [bs, n_heads, len, d_q/2, 2] -> [bs, n_heads, len, d_q]
    x_embed = torch.stack([x_out_r, x_out_i], dim=-1).flatten(3)

    return x_embed.type_as(x)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class FFN(nn.Module):
    def __init__(self, d_model: int, d_ffn: int, dropout: float):
        super().__init__()
        if d_ffn is None:
            d_ffn = int(8 * d_model / 3)

        self.w1 = nn.Linear(d_model, d_ffn)
        self.w2 = nn.Linear(d_model, d_ffn)
        self.w3 = nn.Linear(d_ffn, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x -> w1 -> silu
                           * -> w3 -> dropout
        x -> w2
        """
        return self.dropout(self.w3(F.silu(self.w1(x)) * self.w2(x)))


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    bs, n_kv_heads, slen, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, None, :, :]
        .expand(bs, n_kv_heads, n_rep, slen, head_dim)
        .reshape(bs, n_kv_heads * n_rep, slen, head_dim)
    )


class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask):
        """
        Args:
          q: query, shape [batch_size,n_heads,q_len,d_q]
          k: key, shape [batch_size,n_heads,k_len,d_k]
          v: value, shape [batch_size,n_heads,len_v,d_v]
          mask: mask, shape [batch_size,n_heads,q_len,k_len]

        d_q = d_k

        Return:
          weighted value, shape [batch_size,n_heads,q_len,d_v]
        """
        d_k = q.shape[-1]
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d_k)
        scores = scores + mask
        attention_weights = self.dropout(
            nn.functional.softmax(scores, dim=-1).type_as(q)
        )
        return torch.matmul(attention_weights, v)


class Attention(nn.Module):
    def __init__(
        self,
        n_heads: int,
        d_model: int,
        max_seq_len: int,
        dropout: float,
        n_kv_heads: Optional[int] = None,
    ):
        super().__init__()
        assert n_kv_heads is None or n_heads % n_kv_heads == 0
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads or n_heads
        self.n_rep = self.n_heads // self.n_kv_heads

        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, self.n_kv_heads * self.d_k)
        self.wv = nn.Linear(d_model, self.n_kv_heads * self.d_k)
        self.wo = nn.Linear(d_model, d_model)
        self.dropout = dropout
        self.attention = ScaledDotProductAttention(dropout=self.dropout)
        self.output_dropout = nn.Dropout(dropout)
        """
        tensor([[[[0., -inf, -inf, -inf],
          [0., 0., -inf, -inf],
          [0., 0., 0., -inf],
          [0., 0., 0., 0.]]]])
        """
        mask = torch.full((1, 1, max_seq_len, max_seq_len), float("-inf"))
        mask = torch.triu(mask, diagonal=1)
        self.register_buffer("mask", mask)

    def forward(
        self, x: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor
    ):
        """
        Args:
            x: [batch_size, seq_len, d_model]

        Returns:
            [batch_size, seq_len, d_model]
        """
        bs, seq_len, _ = x.shape
        q = self.wq(x).view(bs, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = self.wk(x).view(bs, -1, self.n_kv_heads, self.d_k).transpose(1, 2)
        v = self.wv(x).view(bs, -1, self.n_kv_heads, self.d_k).transpose(1, 2)

        q = apply_rope(q, freqs_cos, freqs_sin)
        k = apply_rope(k, freqs_cos, freqs_sin)

        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)

        mask = self.mask[:, :, :seq_len, :seq_len]
        output = self.attention(q, k, v, mask)

        output = output.transpose(1, 2).contiguous().view(bs, seq_len, -1)
        return self.output_dropout(self.wo(output))


class DecoderLayers(nn.Module):
    def __init__(
        self,
        n_heads: int,
        d_model: int,
        d_ffn: int,
        max_seq_len: int,
        dropout: float,
        norm_eps: float,
        n_kv_heads: Optional[int] = None,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_ffn = d_ffn
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.n_kv_heads = n_kv_heads or n_heads

        self.attention = Attention(
            n_heads=n_heads,
            d_model=d_model,
            max_seq_len=max_seq_len,
            dropout=dropout,
            n_kv_heads=n_kv_heads,
        )
        self.ffn = FFN(d_model=d_model, d_ffn=d_ffn, dropout=dropout)
        self.norm1 = RMSNorm(d_model, norm_eps)
        self.norm2 = RMSNorm(d_model, norm_eps)

    def forward(
        self, x: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor
    ):
        """
        Args:
            x: [batch_size, seq_len, d_model]
        Returns:
            [batch_size, seq_len, d_model]

        x -> norm1 -> attention ->
                                    -> norm2 -> ffn ->
                                  +                     +
                                    ----------------->
        x ----------------------->
        """
        x = x + self.attention(self.norm1(x), freqs_cos, freqs_sin)
        x = x + self.ffn(self.norm2(x))
        return x


class CausalLM(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
    ):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.n_layers = config.n_layers
        self.dropout = config.dropout
        self.max_seq_len = config.max_seq_len

        # 这里要注意，cos和sin根据的维度是d_model//n_heads，而不是d_model
        freqs_cos, freqs_sin = precompute_freqs(
            config.d_model // config.n_heads, config.max_seq_len
        )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)
        self.embed_dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList(
            [
                DecoderLayers(
                    n_heads=config.n_heads,
                    d_model=config.d_model,
                    d_ffn=config.d_ffn,
                    max_seq_len=config.max_seq_len,
                    dropout=config.dropout,
                    norm_eps=config.norm_eps,
                    n_kv_heads=config.n_kv_heads,
                )
                for _ in range(config.n_layers)
            ]
        )
        self.postnorm = RMSNorm(dim=config.d_model, eps=config.norm_eps)
        self.output = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.embed_tokens.weight = self.output.weight  # 权重共享

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            # gpt2里对attention/ffn的最后一个线性层
            if pn.endswith("w3.weight") or pn.endswith("wo.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layers)
                )

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            # torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, tokens: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        _, seq_len = tokens.shape
        freqs_cos = self.freqs_cos[:seq_len]
        freqs_sin = self.freqs_sin[:seq_len]

        x = self.embed_tokens(tokens)
        x = self.embed_dropout(x)
        for layer in self.layers:
            x = layer(x, freqs_cos, freqs_sin)
        x = self.postnorm(x)

        if targets is not None:
            logits = self.output(x)
        else:
            logits = self.output(x[:, [-1], :])
        return logits

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        ##################### Notice ######################################################
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        ##################### Notice ######################################################
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(
            f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
        )
        print(
            f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
        )
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas, **extra_args
        )
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    @torch.inference_mode()
    def generate(
        self,
        idx,
        max_new_tokens,
        eos_token_id: int,
        temperature=1.0,
        repetition_penalty=1.0,
        top_k=None,
    ):
        end_by_eos = False
        for _ in range(max_new_tokens):
            idx_cond = (
                idx if idx.size(1) <= self.max_seq_len else idx[:, -self.max_seq_len :]
            )
            logits = self(idx_cond)
            logits = logits[:, -1, :]

            # 重复惩罚:https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py#L292
            # 从logits中提取已经出现过的token_id
            score = torch.gather(logits, 1, idx_cond)
            # 惩罚
            # if score < 0 then repetition penalty has to be multiplied to reduce the token probabilities
            score = torch.where(
                score < 0, score * repetition_penalty, score / repetition_penalty
            )
            # 把惩罚过的logits值再放回去
            logits = logits.scatter(1, idx_cond, score)

            # 采样
            if temperature == 0.0:
                idx_next = torch.argmax(logits, dim=-1)
            else:
                # logits ->> /t ->> [topk ->>] softmax ->> multinomial
                logits = logits / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float("-inf")
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
            if idx_next == eos_token_id:
                end_by_eos = True
                break
            idx = torch.cat((idx, idx_next), dim=1)
        if end_by_eos:
            print("End by EOS.")
        else:
            print("End by max length.")
        return idx
