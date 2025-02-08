import torch
import torch.nn as nn
import math
from config import DataConfig

PAD_IDX = DataConfig.pad_idx
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_pad_mask(q, k):
    """
    Attention mask for padded token.

    Args:
        q: torch.Tensor, shape [batch_size, seq_len_q]
        k: torch.Tensor, shape [batch_size, seq_len_k]
    Returns:
        mask: torch.Tensor, shape [batch_size, seq_len_q, seq_len_k]
    """
    bs, len_q = q.size()
    bs, len_k = k.size()
    mask = (k != PAD_IDX).unsqueeze(1).repeat(1, len_q, 1)
    return mask


def get_subsequent_mask(q):
    """
    Attention mask for subsequent tokens.

    Args:
        q: torch.Tensor, shape [batch_size, seq_len_q]

    Returns:
        subsequent_mask: torch.Tensor, shape [batch_size, seq_len_q, seq_len_q]

    Example: seq_len_q = 3, one token.
    1 1 1       1 0 0
    1 1 1  ->   1 1 0
    1 1 1       1 1 1
    """
    shape = [q.size(0), q.size(1), q.size(1)]
    subsequent_mask = torch.tril(torch.ones(shape)).type(torch.uint8)
    return subsequent_mask


class PositionEmbedding(nn.Module):
    def __init__(self, d_model, dropout, max_len=1000):
        super(PositionEmbedding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.PE = torch.zeros(1, max_len, d_model)

        pos = torch.arange(max_len).unsqueeze(1)  # [max_len,1]
        # 2024/8/7：10000^(-2i/d) -> exp{-2i/d*log(10000)}
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )  # [d_model/2,]

        self.PE[:, :, 0::2] = torch.sin(pos * div_term)
        self.PE[:, :, 1::2] = torch.cos(pos * div_term)

    def forward(self, X):
        """
        Args:
            X: token embedding, shape [batch_size,seq_len,d_model]
        Returns:
            PE + token embedding, shape [batch_size,seq_len,d_model]
        """
        X = X + self.PE[:, : X.size(1), :].to(X.device)
        # 2024/8/7：在PE+E之后dropout
        return self.dropout(X)


class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, mask):
        """
        Args:
          Q: query, shape [batch_size,num_heads,len_q,d_q]
          K: key, shape [batch_size,num_heads,len_k,d_k]
          V: value, shape [batch_size,num_heads,len_v,d_v]
          mask: mask, shape [batch_size,num_heads,len_q,len_k]

        d_q = d_k

        Return:
          weighted value, shape [batch_size,num_heads,len_q,d_v]
        """
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(
            d_k
        )  # [bs,h,len_q,len_k]
        scores.masked_fill_(mask == 0, float("-inf"))
        # 2024/8/7：softmax分母是行和
        self.attention_weights = self.dropout(nn.functional.softmax(scores, dim=-1))
        return torch.matmul(self.attention_weights, V)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, h, dropout):
        super(MultiHeadAttention, self).__init__()

        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention(dropout)
        self.linear = nn.Linear(d_model, d_model)  # W_o

    def forward(self, Q, K, V, mask):
        """
        Args:
          Q: query, shape [batch_size,len_q,d_model]
          K: key, shape [batch_size,len_k,d_model]
          V: value, shape [batch_size,len_v,d_model]
          mask: mask, shape [batch_size,len_q,len_k]

        d_q = d_k = d_v = d_model//h

        Return:
          context: shape [batch_size,len_q,d_model]
        """
        bs = Q.size(0)
        # W_Q(Q): [bs,len_q,d_q*h] -> [bs,len_q,h,d_q] -> [bs,h,len_q,d_q]
        qs = self.W_Q(Q).view(bs, -1, self.h, self.d_k).transpose(1, 2)
        ks = self.W_K(K).view(bs, -1, self.h, self.d_k).transpose(1, 2)
        vs = self.W_V(V).view(bs, -1, self.h, self.d_k).transpose(1, 2)

        # mask: [bs,len_q,len_k] -> [bs,1,len_q,len_k] -> [bs,h,len_q,len_k]
        mask = mask.unsqueeze(1).repeat(1, self.h, 1, 1)

        # context: [bs,h,len_q,d_k] -> [bs,len_q,h,d_k] -> [bs,len_q,h * d_k]
        context = self.attention(qs, ks, vs, mask)
        context = context.transpose(1, 2).contiguous().view(bs, -1, self.h * self.d_k)
        return self.linear(context)


class AddNorm(nn.Module):
    def __init__(self, d_model, dropout):
        super(AddNorm, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)


class PositionWiseFFN(nn.Module):
    def __init__(self, d_model, d_ffn, dropout):
        super(PositionWiseFFN, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X):
        """
        Args:
          X: input, shape [batch_size,len_q,d_model]
        Return:
          output, shape [batch_size,len_q,d_model]
        """
        return self.linear2(self.dropout(self.relu(self.linear1(X))))


class EncoderBlock(nn.Module):
    def __init__(self, d_model, d_ffn, num_heads, dropout):
        super(EncoderBlock, self).__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.addnorm1 = AddNorm(d_model, dropout)
        self.ffn = PositionWiseFFN(d_model, d_ffn, dropout)
        self.addnorm2 = AddNorm(d_model, dropout)

    def forward(self, enc_inputs, src_mask):
        enc_outputs = self.addnorm1(
            enc_inputs, self.attention(enc_inputs, enc_inputs, enc_inputs, src_mask)
        )
        return self.addnorm2(enc_outputs, self.ffn(enc_outputs))


class DecoderBlock(nn.Module):
    def __init__(self, d_model, d_ffn, num_heads, dropout):
        super(DecoderBlock, self).__init__()
        # self.attention1:casual-dec
        self.attention1 = MultiHeadAttention(d_model, num_heads, dropout)
        self.addnorm1 = AddNorm(d_model, dropout)

        # self.attention2:dec-enc
        self.attention2 = MultiHeadAttention(d_model, num_heads, dropout)
        self.addnorm2 = AddNorm(d_model, dropout)

        self.ffn = PositionWiseFFN(d_model, d_ffn, dropout)
        self.addnorm3 = AddNorm(d_model, dropout)

    def forward(self, dec_inputs, enc_outputs, tgt_mask, dec_enc_mask):
        dec_outputs = self.addnorm1(
            dec_inputs, self.attention1(dec_inputs, dec_inputs, dec_inputs, tgt_mask)
        )
        dec_outputs = self.addnorm2(
            dec_outputs,
            self.attention2(dec_outputs, enc_outputs, enc_outputs, dec_enc_mask),
        )
        return self.addnorm3(dec_outputs, self.ffn(dec_outputs))


class Encoder(nn.Module):
    def __init__(self, src_vocab_size, num_enc_blk, d_model, d_ffn, num_heads, dropout):
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.src_emb = nn.Embedding(src_vocab_size, d_model)
        self.pos_emb = PositionEmbedding(d_model, dropout)
        self.blks = nn.ModuleList(
            [
                EncoderBlock(d_model, d_ffn, num_heads, dropout)
                for _ in range(num_enc_blk)
            ]
        )

    def forward(self, enc_inputs):
        """
        Args:
            enc_inputs: [bs,x_len]
        """
        # mask: pad mask
        src_pad_mask = get_pad_mask(enc_inputs, enc_inputs).to(DEVICE)

        # embedding: [bs,x_len] -> [bs,x_len,d_model]
        enc_outputs = self.src_emb(enc_inputs) * math.sqrt(self.d_model)
        enc_outputs = self.pos_emb(enc_outputs)

        self.attention_weights = [None] * len(self.blks)
        for i, blk in enumerate(self.blks):
            enc_outputs = blk(enc_outputs, src_pad_mask)
            self.attention_weights[i] = blk.attention.attention.attention_weights
        return enc_outputs


class Decoder(nn.Module):
    def __init__(self, tgt_vocab_size, num_dec_blk, d_model, d_ffn, num_heads, dropout):
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_emb = PositionEmbedding(d_model, dropout)
        self.blks = nn.ModuleList(
            [
                DecoderBlock(d_model, d_ffn, num_heads, dropout)
                for _ in range(num_dec_blk)
            ]
        )

    def forward(self, dec_inputs, enc_inputs, enc_outputs):
        """
        Args:
            dec_inputs: [bs,y_len]
            enc_inputs: [bs,x_len]
            enc_outputs: [bs,x_len,d_model]
        """
        # casual mask: pad + subsequent
        tgt_pad_mask = get_pad_mask(dec_inputs, dec_inputs).to(DEVICE)
        tgt_subsequent_mask = get_subsequent_mask(dec_inputs).to(DEVICE)
        tgt_mask = tgt_pad_mask & tgt_subsequent_mask

        # dec_enc_mask: pad mask
        dec_enc_mask = get_pad_mask(dec_inputs, enc_inputs).to(DEVICE)

        # embedding
        dec_outputs = self.tgt_emb(dec_inputs) * math.sqrt(self.d_model)
        dec_outputs = self.pos_emb(dec_outputs)

        # self.attention_weights[0]:dec attention
        # self.attention_weights[1]:dec_enc_attention
        self.attention_weights = [[None] * len(self.blks) for _ in range(2)]

        for i, blk in enumerate(self.blks):
            dec_outputs = blk(dec_outputs, enc_outputs, tgt_mask, dec_enc_mask)
            self.attention_weights[0][i] = blk.attention1.attention.attention_weights
            self.attention_weights[1][i] = blk.attention1.attention.attention_weights
        return dec_outputs


class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        num_enc_blk,
        num_dec_blk,
        d_model,
        d_ffn,
        num_heads,
        dropout,
    ):
        super(Transformer, self).__init__()
        self.encoder = Encoder(
            src_vocab_size, num_enc_blk, d_model, d_ffn, num_heads, dropout
        )
        self.decoder = Decoder(
            tgt_vocab_size, num_dec_blk, d_model, d_ffn, num_heads, dropout
        )
        self.projection = nn.Linear(d_model, tgt_vocab_size, bias=False)

    def forward(self, enc_inputs, dec_inputs):
        enc_outputs = self.encoder(enc_inputs)
        dec_outputs = self.decoder(dec_inputs, enc_inputs, enc_outputs)
        dec_logits = self.projection(dec_outputs)  # [bs,?,tgt_vocab_size]
        return dec_logits.view(-1, dec_logits.size(-1))
