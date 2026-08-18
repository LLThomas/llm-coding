import math
import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, d_model, q_head, kv_head):
        super().__init__()

        self.head_dim = d_model // q_head
        self.q_head = q_head
        self.kv_head = kv_head
        self.q_per_kv = q_head // kv_head

        # BSD
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, self.head_dim * kv_head, bias=False)
        self.v_proj = nn.Linear(d_model, self.head_dim * kv_head, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, causal=True):
        B, S, _ = x.shape

        # qkv proj
        # BSD -> BHSD
        q = self.q_proj(x).view(B, S, self.q_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.kv_head, self.head_dim).transpose(1, 2)

        # repeat kv head
        k = torch.repeat_interleave(k, self.q_per_kv, dim=1)
        v = torch.repeat_interleave(v, self.q_per_kv, dim=1)

        # score (mask)
        # BHSD @ BHDS -> BHSS
        score = q @ k.transpose(-1, -2) / math.sqrt(self.head_dim)
        if causal:
            mask = torch.triu(torch.ones(S, S, dtype=bool), 1)
            score = torch.masked_fill(score, mask, float("-inf"))
        score = torch.softmax(score, dim=-1)

        # attn
        # BHSS @ BHSD -> BHSD
        attn = score @ v

        # out proj
        # BHSD -> BSHD -> BSD
        return self.out_proj(attn.transpose(1, 2).reshape(B, S, -1))


# BSD -> (2, 2, 128)
x = torch.randn(2, 2, 128)
GQA = Attention(128, 4, 2)
print("GQA: ", GQA(x).shape)
