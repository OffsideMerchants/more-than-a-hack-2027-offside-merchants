"""Masked trajectory model: a transformer that alternates attention over time and attention across agents.

Each (frame, agent) pair is one token. Every block lets each agent attend over its own
trajectory (time attention), then lets all agents in a frame attend to each other
(agent attention). Agents carry no identity embedding, only home/away/ball, so the
model is permutation-equivariant within each team.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import AGENT_TYPES


class Attention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        n, L, d = x.shape
        q, k, v = self.qkv(x).view(n, L, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        out = F.scaled_dot_product_attention(q, k, v)
        return self.proj(out.transpose(1, 2).reshape(n, L, d))


class AxialBlock(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4, dropout=0.0):
        super().__init__()
        self.norm_t, self.attn_t = nn.LayerNorm(dim), Attention(dim, heads)
        self.norm_a, self.attn_a = nn.LayerNorm(dim), Attention(dim, heads)
        self.norm_m = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(mlp_ratio * dim, dim)
        )

    def forward(self, x):
        B, T, A, D = x.shape
        h = x.transpose(1, 2).reshape(B * A, T, D)
        x = x + self.attn_t(self.norm_t(h)).view(B, A, T, D).transpose(1, 2)
        h = x.reshape(B * T, A, D)
        x = x + self.attn_a(self.norm_a(h)).view(B, T, A, D)
        return x + self.mlp(self.norm_m(x))


class TrajectoryMAE(nn.Module):
    def __init__(self, dim=128, depth=6, heads=4, max_len=128, dropout=0.0):
        super().__init__()
        self.in_proj = nn.Sequential(nn.Linear(2, dim), nn.GELU(), nn.Linear(dim, dim))
        self.mask_token = nn.Parameter(torch.zeros(dim))     # hidden by the masking objective
        self.missing_token = nn.Parameter(torch.zeros(dim))  # genuinely absent in the data
        self.type_emb = nn.Embedding(3, dim)
        self.time_emb = nn.Parameter(torch.zeros(max_len, dim))
        self.register_buffer("agent_types", torch.as_tensor(AGENT_TYPES), persistent=False)
        self.blocks = nn.ModuleList([AxialBlock(dim, heads, dropout=dropout) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, 2)
        for p in (self.mask_token, self.missing_token, self.time_emb):
            nn.init.normal_(p, std=0.02)

    def encode(self, pos, hidden=None):
        """pos: [B, T, A, 2], NaN where missing. hidden: [B, T, A] bool, True for tokens to hide.

        Returns token features [B, T, A, D].
        """
        B, T, A, _ = pos.shape
        observed = ~pos.isnan().any(-1)
        visible = observed if hidden is None else observed & ~hidden
        tok = self.in_proj(torch.nan_to_num(pos))
        tok = torch.where(visible[..., None], tok, self.mask_token)
        tok = torch.where(observed[..., None], tok, self.missing_token)
        tok = tok + self.type_emb(self.agent_types)[None, None] + self.time_emb[:T][None, :, None]
        for blk in self.blocks:
            tok = blk(tok)
        return self.norm(tok)

    def forward(self, pos, hidden):
        return self.head(self.encode(pos, hidden))

    @torch.no_grad()
    def embed_clip(self, pos):
        """Clip-level embedding [B, D]: mean of features over observed tokens, with nothing hidden."""
        feats = self.encode(pos)
        w = (~pos.isnan().any(-1)).float()[..., None]
        return (feats * w).sum((1, 2)) / w.sum((1, 2)).clamp(min=1)
