import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint

MASK_NEG = -1e4


class TrackEncoder(nn.Module):
    """Shared 1D conv per track over about +-1 s."""

    def __init__(self, in_dim: int, d: int, kernel: int):
        super().__init__()
        p = kernel // 2
        self.c1 = nn.Conv1d(in_dim, d, kernel, padding=p)
        self.c2 = nn.Conv1d(d, d, kernel, padding=p)

    def forward(self, x: Tensor, pres: Tensor) -> Tensor:
        B, T, N, F_ = x.shape
        h = x.permute(0, 2, 3, 1).reshape(B * N, F_, T)
        h = self.c2(F.gelu(self.c1(h)))
        h = h.reshape(B, N, -1, T).permute(0, 3, 1, 2)
        return h * pres[..., None]


class SocialBlock(nn.Module):
    """Permutation-equivariant self-attention over the tracks of one time step.

    There is no positional encoding and no team identity: the only team signal is a learned
    same-team bias on the attention logits, which is invariant under a team swap.
    """

    def __init__(self, d: int, heads: int, dropout: float):
        super().__init__()
        self.h = heads
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)
        self.same = nn.Parameter(torch.zeros(heads))
        self.mlp = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor, pres: Tensor, same_team: Tensor) -> Tensor:
        B, T, N, d = x.shape
        q, k, v = self.qkv(self.n1(x)).reshape(B, T, N, 3, self.h, d // self.h).unbind(3)
        q, k, v = (z.transpose(2, 3) for z in (q, k, v))  # [B,T,h,N,hd]
        bias = same_team[:, None, None].float() * self.same[None, None, :, None, None]
        bias = bias + (~pres)[:, :, None, None, :].float() * MASK_NEG
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=bias.to(q.dtype))
        a = a.transpose(2, 3).reshape(B, T, N, d)
        x = x + self.drop(self.out(a))
        x = x + self.drop(self.mlp(self.n2(x)))
        return x * pres[..., None]


class PoolToken(nn.Module):
    """Learned-query cross-attention pooling of the selected tracks into one token."""

    def __init__(self, d: int, heads: int, conditioned: bool = False):
        super().__init__()
        self.h = heads
        self.q = nn.Parameter(torch.randn(heads, d // heads) * 0.02)
        self.cond = nn.Linear(d, d) if conditioned else None
        self.k, self.v = nn.Linear(d, d), nn.Linear(d, d)
        self.out = nn.Linear(d, d)
        self.empty = nn.Parameter(torch.zeros(d))

    def forward(self, x: Tensor, select: Tensor, cond: Tensor | None = None) -> Tensor:
        B, T, N, d = x.shape
        hd = d // self.h
        k = self.k(x).reshape(B, T, N, self.h, hd)
        v = self.v(x).reshape(B, T, N, self.h, hd)
        q = self.q.expand(B, T, self.h, hd)
        if self.cond is not None:
            q = q + self.cond(cond).reshape(B, T, self.h, hd)
        s = (k * q[:, :, None]).sum(-1) / hd ** 0.5  # [B,T,N,h]
        s = s + (~select)[..., None].float() * MASK_NEG
        w = s.softmax(2) * select.any(2)[:, :, None, None]
        o = (w[..., None] * v).sum(2).reshape(B, T, d)
        return self.out(o) + (~select.any(2))[..., None].float() * self.empty


def maybe_checkpoint(fn, use: bool, *args):
    if use and torch.is_grad_enabled():
        return checkpoint(fn, *args, use_reentrant=False)
    return fn(*args)
