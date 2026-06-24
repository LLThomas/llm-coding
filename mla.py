import math
import torch
import torch.nn as nn


class MLA(nn.Module):
    def __init__(self, d_model, num_heads, latent_dim):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.w_q   = nn.Linear(d_model, d_model, bias=False)     # Q(不压缩)
        self.w_dkv = nn.Linear(d_model, latent_dim, bias=False)  # 下采:压成 latent c  ← 推理只 cache 它
        self.w_uk  = nn.Linear(latent_dim, d_model, bias=False)  # 上采:latent → per-head K
        self.w_uv  = nn.Linear(latent_dim, d_model, bias=False)  # 上采:latent → per-head V
        self.w_o   = nn.Linear(d_model, d_model, bias=False)     # 输出投影:合并头 + 跨头混合

    def forward(self, x, causal=True):
        B, S, _ = x.shape

        # Q, K, V
        # Q: BSD -> BSHD -> BHSD
        q = self.w_q(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        # latent c: BSD -> B S latent_dim
        c = self.w_dkv(x)
        # K/V: B S latent_dim -> BSD -> BSHD -> BHSD
        k = self.w_uk(c).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.w_uv(c).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        # score
        # BHSD @ BHDS -> BHSS
        score = q @ k.transpose(-1, -2) / math.sqrt(self.head_dim)
        if causal:
            mask = torch.triu(torch.ones(S, S, dtype=torch.bool), diagonal=1)
            score = score.masked_fill(mask, float("-inf"))
        attn = torch.softmax(score, dim=-1) @ v

        # merge heads: BHSD -> [B, S, num_q_heads * head_dim] = [B, S, d_model]
        attn = attn.transpose(1, 2).reshape(B, S, -1)
        # output projection
        return self.w_o(attn)


mla = MLA(d_model=512, num_heads=8, latent_dim=128)
x = torch.randn(2, 16, 512)
print("[MLA] x.shape: ", mla.forward(x, True).shape)