import torch
import torch.nn as nn
import math

# H: image height
# W: image width
# C: number of channels
# P: patch size
# n_patches: number of patches inside our image, N = HW/P^2


class PatchEmbedding(nn.Module):
    """
    2D Image to Patch Embedding
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.img_height = config.img_height
        self.img_width = config.img_width
        self.channels = config.channels

        self.patch_size = config.patch_size
        # self.n_patches = self.img_height * self.img_width // (self.patch_size**2)

        self.d_model = config.d_model

        # H' = (H - kernel_size)/stride + 1 -> H//P
        # W' = (W - kernel_size)/stride + 1 -> W//P
        self.projection = nn.Conv2d(
            in_channels=self.channels,
            out_channels=self.d_model,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

    def forward(self, x):
        """
        x: (batch_size, C, H, W)

        return: (batch_size, n_patches, d_model)

        """
        # (bz, C, H, W) -> (bz, d_model, H/P, W/P)
        x = self.projection(x)
        # -> (bz, d_model, n_patches) -> (bz, n_patches, d_model)
        x = x.flatten(2).transpose(1, 2)
        return x


class Embedding(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_patches = (
            self.config.img_height
            * self.config.img_width
            // (self.config.patch_size**2)
        )
        self.d_model = self.config.d_model

        # patch embedding
        self.patch_embedding = PatchEmbedding(config)
        # cls token
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.d_model))
        # # position embedding
        # self.PE = torch.randn(1, self.n_patches + 1, self.d_model)
        # pos = torch.arange(self.n_patches + 1).unsqueeze(1)
        # div_term = torch.exp(
        #     torch.arange(0, self.d_model, 2)
        #     * (-torch.log(torch.tensor(10000.0)) / self.d_model)
        # )
        # self.PE[:, :, 0::2] = torch.sin(pos * div_term)
        # self.PE[:, :, 1::2] = torch.cos(pos * div_term)
        self.PE = nn.Parameter(torch.randn(1, self.n_patches + 1, self.d_model))
        # dropout
        self.dropout = nn.Dropout(self.config.dropout)

    def forward(self, x):
        """
        x: (bz, C, H, W)
        return: (bz, n_patches + 1, d_model)
        """
        bz, _, _, _ = x.shape
        # (bz, n_patches, d_model)
        x = self.patch_embedding(x)
        # (1, 1, d_model) -> (bz, 1, d_model)
        cls_tokens = self.cls_token.expand(bz, -1, -1)
        # (bz, n_patches + 1, d_model)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.PE.to(x.device)
        x = self.dropout(x)
        return x


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.d_model = self.config.d_model
        self.n_heads = self.config.n_heads
        assert self.d_model % self.n_heads == 0

        # qkv projection
        self.wq = nn.Linear(self.d_model, self.d_model)
        self.wk = nn.Linear(self.d_model, self.d_model)
        self.wv = nn.Linear(self.d_model, self.d_model)
        # output projection
        self.wo = nn.Linear(self.d_model, self.d_model)
        # dropout
        self.scores_dropout = nn.Dropout(self.config.dropout)

    def forward(self, x):
        """
        x: (bz, n_patches + 1, d_model)
        return: (bz, n_patches + 1, d_model)
        """
        # seq_len = n_patches + 1
        bz, seq_len, d_model = x.shape
        d_k = d_model // self.n_heads
        # (bz, seq_len, d_model) -> (bz, n_heads, seq_len, d_k)
        q = self.wq(x).view(bz, seq_len, self.n_heads, d_k).transpose(1, 2)
        k = self.wk(x).view(bz, seq_len, self.n_heads, d_k).transpose(1, 2)
        v = self.wv(x).view(bz, seq_len, self.n_heads, d_k).transpose(1, 2)

        # (bz, n_heads, seq_len, seq_len)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        scores = torch.softmax(scores, dim=-1)
        # dropout
        scores = self.scores_dropout(scores)
        # (bz, n_heads, seq_len, d_k) -> (bz, seq_len, d_model)
        output = (
            torch.matmul(scores, v)
            .transpose(1, 2)
            .contiguous()
            .view(bz, seq_len, d_model)
        )
        output = self.wo(output)
        return output


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.d_model = self.config.d_model
        self.d_ffn = self.config.d_ffn

        # mlp
        self.net = nn.Sequential(
            nn.Linear(self.d_model, self.d_ffn),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.d_ffn, self.d_model),
            nn.Dropout(self.config.dropout),
        )

    def forward(self, x):
        """
        x: (bz, n_patches + 1, d_model)
        return: (bz, n_patches + 1, d_model)
        """
        x = self.net(x)
        return x


class EncoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.attn = Attention(config)
        self.mlp = MLP(config)
        self.norm1 = nn.LayerNorm(self.config.d_model)
        self.norm2 = nn.LayerNorm(self.config.d_model)

    def forward(self, x):
        """
        x: (bz, n_patches + 1, d_model)
        return: (bz, n_patches + 1, d_model)
        """
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder_blocks = nn.ModuleList(
            [EncoderBlock(config) for _ in range(self.config.n_layers)]
        )

    def forward(self, x):
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x)
        return x


class ViT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = Embedding(config)
        self.encoder = Encoder(config)
        self.post_norm = nn.LayerNorm(self.config.d_model)
        self.linear_head = nn.Linear(self.config.d_model, self.config.n_classes)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, Embedding):
            nn.init.normal_(m.cls_token, mean=0.0, std=0.02)
            nn.init.normal_(m.PE, mean=0.0, std=0.02)

    def forward(self, x):
        """
        x: (bz, C, H, W)
        return: (bz, n_classes)
        """
        # (bz, n_patches + 1, d_model)
        x = self.embedding(x)
        x = self.encoder(x)
        # cls token
        logits = self.linear_head(self.post_norm(x[:, 0, :]))
        return logits
