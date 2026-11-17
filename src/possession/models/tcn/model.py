"""Permutation-invariant TCN for possession (alive/dead + owner); see arsh-research/03_model.md.

Swap symmetry is exact: every window is encoded twice, once from each team's point of view
(own team attacks +x), and the home logit is own_logit(home view) - own_logit(away view).
Inputs deliberately exclude Z, roles and IDs (see arsh-research/03_model.md).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from ..possession_transformer.temporal import TemporalBlock

GAP_CAP_S = 30.0
X_SCALE, Y_SCALE, V_SCALE, REL_SCALE = 52.5, 34.0, 10.0, 30.0
TRACK_F = 9  # x y vx vy speed relx rely dist present
BALL_F = 8  # x y vx vy present since next gaplen


@dataclass
class TCNConfig:
    hidden: int = 64
    track_blocks: int = 3
    ball_blocks: int = 3
    joint_blocks: int = 6
    kernel: int = 3
    dropout: float = 0.1
    use_rel_ball: bool = True
    attention_pool: bool = True
    rate_hz: float = 12.5
    attn_blocks: int = 0  # >0: RoPE self-attention blocks over time after the conv stack (hybrid transformer)
    heads: int = 4
    drop_path: float = 0.0

    def receptive_field_s(self) -> float:
        steps = 1
        for n in (self.track_blocks, self.joint_blocks):
            steps += (self.kernel - 1) * sum(2**i for i in range(n))
        return steps / self.rate_hz


class ChannelLN(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.ln = nn.LayerNorm(c)

    def forward(self, x):  # [B, C, T]
        return self.ln(x.transpose(1, 2)).transpose(1, 2)


class Block(nn.Module):
    """Residual dilated non-causal conv block; LayerNorm so no batch statistics need adapting."""

    def __init__(self, c: int, dilation: int, k: int, p: float):
        super().__init__()
        self.net = nn.Sequential(
            ChannelLN(c), nn.GELU(),
            nn.Conv1d(c, c, k, dilation=dilation, padding=dilation * (k - 1) // 2),
            ChannelLN(c), nn.GELU(), nn.Dropout(p), nn.Conv1d(c, c, 1),
        )

    def forward(self, x):
        return x + self.net(x)


def tcn(c_in: int, c: int, n: int, k: int, p: float) -> nn.Sequential:
    return nn.Sequential(nn.Conv1d(c_in, c, 1), *[Block(c, 2**i, k, p) for i in range(n)])


def ball_gap_features(present: torch.Tensor, dt: float) -> torch.Tensor:
    """present [B,T] bool -> [B,T,3] (since seen, to next seen, gap length), seconds/10, capped at 30 s."""
    B, T = present.shape
    idx = torch.arange(T, device=present.device).expand(B, T)
    cap = GAP_CAP_S / dt
    last = torch.cummax(torch.where(present, idx, torch.full_like(idx, -1)), 1).values
    since = torch.where(last >= 0, idx - last, torch.full_like(idx, int(cap))).float().clamp(max=cap)
    nxt_f = torch.cummax(torch.where(present, T - 1 - idx, torch.full_like(idx, -1)).flip(1), 1).values.flip(1)
    to_next = torch.where(nxt_f >= 0, (T - 1 - nxt_f) - idx, torch.full_like(idx, int(cap))).float().clamp(max=cap)
    gap = torch.where(present, torch.zeros_like(since), (since + to_next).clamp(max=cap))
    return torch.stack([since, to_next, gap], -1) * dt / 10.0


class MaskedPool(nn.Module):
    """Mean + max (+ attention) over tracks at each step -> [B,T,H]."""

    def __init__(self, h: int, attention: bool):
        super().__init__()
        self.att = nn.Linear(h, 1) if attention else None
        self.out = nn.Linear(h * (3 if attention else 2), h)

    def forward(self, z, m):  # z [B,N,T,H], m [B,N,T]
        z = z.float()
        mf = m.unsqueeze(-1).float()
        cnt = mf.sum(1)
        mean = (z * mf).sum(1) / cnt.clamp(min=1)
        mx = z.masked_fill(~m.unsqueeze(-1), -1e4).max(1).values * (cnt > 0)
        parts = [mean, mx]
        if self.att is not None:
            s = self.att(z).squeeze(-1).masked_fill(~m, -1e4)
            w = torch.softmax(s, 1) * m
            parts.append((w.unsqueeze(-1) * z).sum(1))
        return self.out(torch.cat(parts, -1).to(self.out.weight.dtype))


class PossessionTCN(nn.Module):
    def __init__(self, cfg: TCNConfig | None = None):
        super().__init__()
        self.cfg = c = cfg or TCNConfig()
        h = c.hidden
        self.track_enc = tcn(TRACK_F, h, c.track_blocks, c.kernel, c.dropout)
        self.ball_enc = tcn(BALL_F, h, c.ball_blocks, c.kernel, c.dropout)
        self.pool_own = MaskedPool(h, c.attention_pool)
        self.pool_opp = MaskedPool(h, c.attention_pool)
        self.fuse = nn.Conv1d(3 * h, h, 1)
        self.joint = nn.Sequential(*[Block(h, 2**i, c.kernel, c.dropout) for i in range(c.joint_blocks)])
        self.attn = nn.ModuleList(TemporalBlock(h, c.heads, c.dropout, c.drop_path) for _ in range(c.attn_blocks))
        self.attn_norm = nn.LayerNorm(h) if c.attn_blocks else None
        self.head_alive = nn.Conv1d(h, 1, 1)
        self.head_owner = nn.Conv1d(h, 1, 1)  # g(z_own, z_opp), scalar "own team has ball"
        self.head_turn = nn.Conv1d(h, 1, 1)

    def _view(self, players, pmask, team, ball, bmask, own: int):
        """Encode from `own`'s point of view: own team attacks +x. Returns dict of [B,T] logits."""
        c = self.cfg
        dt = 1.0 / c.rate_hz
        sgn = 1.0 if own == 0 else -1.0
        fl = torch.tensor([sgn, 1, sgn, 1, 1], device=players.device, dtype=players.dtype)
        p = players * fl
        b = ball * fl[:4]
        B, T, N, _ = p.shape
        xy = torch.stack([p[..., 0] / X_SCALE, p[..., 1] / Y_SCALE], -1)
        vel = p[..., 2:4] / V_SCALE
        spd = p[..., 4:5] / V_SCALE
        rel = (p[..., :2] - b[:, :, None, :2]) / REL_SCALE
        rm = pmask & bmask[:, :, None]
        if not c.use_rel_ball:
            rm = torch.zeros_like(rm)
        dist = rel.norm(dim=-1, keepdim=True)
        rel = rel * rm.unsqueeze(-1)
        dist = dist * rm.unsqueeze(-1)
        f = torch.cat([xy, vel, spd, rel, dist, pmask.unsqueeze(-1).to(p.dtype)], -1) * pmask.unsqueeze(-1)
        f = f.permute(0, 2, 3, 1).reshape(B * N, TRACK_F, T)
        z = self.track_enc(f).reshape(B, N, -1, T).permute(0, 1, 3, 2)  # [B,N,T,H]
        is_own = (team == own).unsqueeze(-1)
        is_opp = ((team == 1 - own)).unsqueeze(-1)
        zo = self.pool_own(z, pmask.permute(0, 2, 1) & is_own)
        zp = self.pool_opp(z, pmask.permute(0, 2, 1) & is_opp)
        gaps = ball_gap_features(bmask, dt)
        bf = torch.cat([b[..., :2] / torch.tensor([X_SCALE, Y_SCALE], device=b.device),
                        b[..., 2:4] / V_SCALE, bmask.unsqueeze(-1).to(b.dtype), gaps], -1)
        bf = bf * torch.cat([bmask.unsqueeze(-1).expand(-1, -1, 5).to(b.dtype), torch.ones_like(gaps)], -1)
        zb = self.ball_enc(bf.permute(0, 2, 1)).permute(0, 2, 1)
        h = self.fuse(torch.cat([zo, zp, zb], -1).permute(0, 2, 1))
        h = self.joint(h)
        if self.attn:
            x = h.permute(0, 2, 1)
            pos = torch.arange(x.shape[1], device=x.device)
            for blk in self.attn:
                x = blk(x, pos, None)
            h = self.attn_norm(x).permute(0, 2, 1)
        return {k: getattr(self, f"head_{k}")(h).squeeze(1) for k in ("alive", "owner", "turn")}

    def forward(self, players, pmask, team, ball, bmask):
        """players [B,T,N,5] (x y vx vy speed, NaN-free), pmask [B,T,N], team [B,N] in {0,1,-1 pad},
        ball [B,T,4], bmask [B,T]. Returns logits {alive, home, turn} each [B,T]."""
        vh = self._view(players, pmask, team, ball, bmask, 0)
        va = self._view(players, pmask, team, ball, bmask, 1)
        return {
            "alive": 0.5 * (vh["alive"] + va["alive"]),
            "home": vh["owner"] - va["owner"],  # exactly antisymmetric under team swap
            "turn": 0.5 * (vh["turn"] + va["turn"]),
        }
