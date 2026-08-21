import torch
import torch.nn as nn
import torch.nn.functional as F


class FFN(nn.Module):
    def __init__(self, d_model, ffn_dim):
        super().__init__()

        # gate
        self.gate_proj = nn.Linear(d_model, ffn_dim, bias=False)
        # up
        self.up_proj = nn.Linear(d_model, ffn_dim, bias=False)
        # down
        self.down_proj = nn.Linear(ffn_dim, d_model, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MoE(nn.Module):
    def __init__(self, d_model, ffn_dim, num_experts, topk):
        super().__init__()

        self.topk = topk

        # gate / experts
        self.gate = nn.Linear(d_model, num_experts, bias=False)
        self.experts = nn.ModuleList([FFN(d_model, ffn_dim) for _ in range(num_experts)])

    def forward(self, x):
        B, S, D = x.shape
        x = x.reshape(-1, D)

        # gate
        # ND @ DE -> NE
        logits = self.gate(x)

        # topk
        # NE -> NK
        topk_logits, topk_idx = torch.topk(logits, self.topk)
        weights = torch.softmax(topk_logits, dim=-1)

        # moe
        out = torch.zeros_like(x)
        for e, expert in enumerate(self.experts):
            for k in range(self.topk):
                mask = topk_idx[:, k] == e
                if mask.any():
                    out[mask] += weights[mask, k:k+1] * expert(x[mask])

        return out.reshape(B, S, D)


moe = MoE(d_model=8, ffn_dim=32, num_experts=4, topk=2)
x = torch.randn(2, 5, 8)
print("x:", x.shape, ", out:", moe(x).shape)