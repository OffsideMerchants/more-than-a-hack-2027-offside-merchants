from dataclasses import dataclass, fields

import torch
import torch.nn as nn
from torch import Tensor

from . import features as ft
from .encoder import PoolToken, SocialBlock, TrackEncoder, maybe_checkpoint
from .temporal import TemporalTransformer


@dataclass
class S4Config:
    rate_hz: float = 6.25
    window_s: float = 60.0
    d: int = 96
    heads: int = 4
    social_blocks: int = 2
    temporal_blocks: int = 4
    max_tracks: int = 30
    track_kernel: int = 7
    dropout: float = 0.1
    drop_path: float = 0.1
    checkpoint: bool = True
    max_offset: float = 0.5  # steps

    @classmethod
    def from_dict(cls, cfg: dict) -> "S4Config":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in cfg.items() if k in names})

    @property
    def window_steps(self) -> int:
        return int(round(self.window_s * self.rate_hz))


def _mlp(i: int, d: int, o: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(i, d), nn.GELU(), nn.Linear(d, o))


class PossessionTransformer(nn.Module):
    """Set-encoded game states -> long temporal context -> CRF potentials over {H, A, D}.

    Inputs: players [B,T,N,6], team [B,N] (0 home / 1 away), ball [B,T,5], lengths [B] (optional).
    Every head is either invariant to a home/away swap or built as f(own, opp), so a team swap
    swaps the outputs exactly.
    """

    def __init__(self, cfg: S4Config | None = None):
        super().__init__()
        self.cfg = c = cfg or S4Config()
        d = c.d
        self.track = TrackEncoder(ft.PLAYER_DIM, d, c.track_kernel)
        self.social = nn.ModuleList(SocialBlock(d, c.heads, c.dropout) for _ in range(c.social_blocks))
        self.pool_team = PoolToken(d, c.heads)
        self.pool_ctrl = PoolToken(d, c.heads)
        self.pool_ball = PoolToken(d, c.heads, conditioned=True)
        self.team_in = nn.Linear(d + ft.TEAM_GAP_DIM, d)
        self.ball_in = _mlp(5 + ft.CTX_DIM, d, d)
        self.temporal = TemporalTransformer(d, c.heads, c.temporal_blocks, c.dropout, c.drop_path, c.checkpoint)

        self.team_head = _mlp(4 * d, d, 3)  # gain, restart, stay
        self.sym_head = _mlp(4 * d, d, 4)   # alive, stop, stay_dead, offset
        self.bias = nn.Parameter(torch.tensor([-3.0, -3.0, -3.0]))  # gain, stop, restart
        self.aux_phase = _mlp(2 * d, d, 3)
        self.aux_possessor = _mlp(2 * d, d, 1)
        self.aux_ball = _mlp(2 * d, d, 2)

    def _encode(self, pf, bf, pres, bok, ball_xy, team, same):
        c = self.cfg
        xy = torch.stack([pf[..., 0] * ft.PITCH_X, pf[..., 1] * ft.PITCH_Y], -1)
        cent = ft.masked_centroids(xy, pres, team)
        shared, per_team = ft.gap_context(ball_xy, bok, cent, c.rate_hz)

        h = self.track(pf, pres)
        for blk in self.social:
            h = maybe_checkpoint(blk, c.checkpoint, h, pres, same)

        sel = [pres & (team == k)[:, None] for k in (0, 1)]
        team_tok = [self.team_in(torch.cat([self.pool_team(h, s), per_team[:, :, k]], -1))
                    for k, s in enumerate(sel)]
        ball_vec = self.ball_in(torch.cat([bf, shared], -1))
        ball_tok = ball_vec + self.pool_ball(h, pres, ball_vec)
        ctrl_tok = self.pool_ctrl(h, pres)
        return h, torch.stack([team_tok[0], team_tok[1], ball_tok, ctrl_tok], 2)

    def forward(self, players: Tensor, team: Tensor, ball: Tensor, lengths: Tensor | None = None) -> dict:
        B, T = ball.shape[:2]
        pf, bf, pres, bok, ball_xy = ft.player_inputs(players, ball)
        team = team.long()
        same = team[:, :, None] == team[:, None, :]
        valid = None if lengths is None else torch.arange(T, device=ball.device)[None] < lengths[:, None]

        h, tokens = self._encode(pf, bf, pres, bok, ball_xy, team, same)
        z = self.temporal(tokens, valid)
        home, away, ball_t, ctrl = z.unbind(2)

        th = self.team_head(torch.cat([home, away, ball_t, ctrl], -1))  # (own, opp) = (home, away)
        ta = self.team_head(torch.cat([away, home, ball_t, ctrl], -1))
        sym = self.sym_head(torch.cat([home + away, home * away, ball_t, ctrl], -1))
        b_gain, b_stop, b_restart = self.bias
        gain_home, gain_away = th[..., 0] + b_gain, ta[..., 0] + b_gain
        restart_home, restart_away = th[..., 1] + b_restart, ta[..., 1] + b_restart
        stay_h, stay_a = th[..., 2], ta[..., 2]
        alive, stop, stay_d = sym[..., 0], sym[..., 1] + b_stop, sym[..., 2]
        offset = torch.tanh(sym[..., 3]) * self.cfg.max_offset

        trans = torch.stack([
            torch.stack([stay_h, gain_away, stop], -1),
            torch.stack([gain_home, stay_a, stop], -1),
            torch.stack([restart_home, restart_away, stay_d], -1),
        ], -2)
        emit = torch.stack([alive / 2, alive / 2, -alive / 2], -1)

        cb = torch.cat([ctrl, ball_t], -1)
        poss = self.aux_possessor(torch.cat([h, ctrl[:, :, None].expand_as(h)], -1)).squeeze(-1)
        return dict(
            emit=emit, trans=trans, offset=offset, alive_logit=alive,
            aux_phase=self.aux_phase(cb),
            aux_possessor=poss.masked_fill(~pres, -1e4),
            aux_ball=self.aux_ball(cb) * ft.PITCH_X,
            valid=valid,
        )
