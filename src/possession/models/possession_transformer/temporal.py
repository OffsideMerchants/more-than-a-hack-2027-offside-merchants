import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .encoder import maybe_checkpoint


def rope(x: Tensor, pos: Tensor) -> Tensor:
    """x [B,h,L,hd], pos [L] time index. Relative-time position by rotary embedding."""
    hd = x.shape[-1]
    freq = 1.0 / (10000 ** (torch.arange(0, hd, 2, device=x.device).float() / hd))
    ang = pos[:, None].float() * freq[None]
    cos, sin = ang.cos().to(x.dtype), ang.sin().to(x.dtype)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1).flatten(-2)


class TemporalBlock(nn.Module):
    def __init__(self, d: int, heads: int, dropout: float, drop_path: float):
        super().__init__()
        self.h, self.dp = heads, drop_path
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.out = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(4 * d, d))
        self.drop = nn.Dropout(dropout)

    def _path(self, x: Tensor) -> Tensor:
        if not self.training or self.dp == 0:
            return x
        keep = (torch.rand(x.shape[0], 1, 1, device=x.device) > self.dp).to(x.dtype)
        return x * keep / (1 - self.dp)

    def forward(self, x: Tensor, pos: Tensor, key_ok: Tensor | None) -> Tensor:
        B, L, d = x.shape
        q, k, v = self.qkv(self.n1(x)).reshape(B, L, 3, self.h, d // self.h).unbind(2)
        q, k, v = (z.transpose(1, 2) for z in (q, k, v))
        q, k = rope(q, pos), rope(k, pos)
        mask = None if key_ok is None else key_ok[:, None, None, :]
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        x = x + self._path(self.drop(self.out(a.transpose(1, 2).reshape(B, L, d))))
        return x + self._path(self.drop(self.mlp(self.n2(x))))


class TemporalTransformer(nn.Module):
    """Attention over (time x tokens). Tokens only carry a type (team / ball / control), never a
    team identity, so the stack is equivariant to a home/away swap."""

    def __init__(self, d: int, heads: int, blocks: int, dropout: float, drop_path: float, checkpoint: bool):
        super().__init__()
        self.types = nn.Parameter(torch.zeros(3, d))
        self.register_buffer("type_ids", torch.tensor([0, 0, 1, 2]), persistent=False)
        rates = torch.linspace(0, drop_path, blocks).tolist()
        self.blocks = nn.ModuleList(TemporalBlock(d, heads, dropout, r) for r in rates)
        self.norm = nn.LayerNorm(d)
        self.ckpt = checkpoint

    def forward(self, tokens: Tensor, valid: Tensor | None = None) -> Tensor:
        """tokens [B,T,4,d] in order home, away, ball, control; valid [B,T] bool."""
        B, T, K, d = tokens.shape
        x = (tokens + self.types[self.type_ids]).reshape(B, T * K, d)
        pos = torch.arange(T, device=x.device).repeat_interleave(K)
        key_ok = None if valid is None else valid.repeat_interleave(K, dim=1)
        for blk in self.blocks:
            x = maybe_checkpoint(blk, self.ckpt, x, pos, key_ok)
        return self.norm(x).reshape(B, T, K, d)
