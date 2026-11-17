import torch
from torch import Tensor

PITCH_X, PITCH_Y, SPEED = 52.5, 34.0, 10.0
GAP_CAP_S = 60.0
CTX_DIM = 5
TEAM_GAP_DIM = 8
PLAYER_DIM = 9


def masked_centroids(xy: Tensor, present: Tensor, team: Tensor) -> Tensor:
    """xy [B,T,N,2], present [B,T,N] bool, team [B,N] -> centroids [B,T,2(team),2]."""
    out = []
    for k in (0, 1):
        m = (present & (team == k)[:, None]).unsqueeze(-1).to(xy.dtype)
        out.append((xy * m).sum(2) / m.sum(2).clamp(min=1))
    return torch.stack(out, 2)


def gap_context(ball_xy: Tensor, ball_ok: Tensor, cent: Tensor, rate_hz: float):
    """Gap edges around every step.

    Returns shared [B,T,5] (team-invariant) and per_team [B,T,2,8]: the last-seen and next-seen
    ball position relative to the own and the opposing centroid, in (own, opp) order per team.
    """
    B, T = ball_ok.shape
    idx = torch.arange(T, device=ball_ok.device).expand(B, T)
    last = torch.where(ball_ok, idx, torch.full_like(idx, -1)).cummax(1).values
    nxt = torch.where(ball_ok, idx, torch.full_like(idx, 2 * T)).flip(1).cummin(1).values.flip(1)
    has_last, has_next = last >= 0, nxt < T
    cap = torch.full(idx.shape, GAP_CAP_S, device=idx.device)
    since = torch.where(has_last, (idx - last).float() / rate_hz, cap).clamp(max=GAP_CAP_S)
    until = torch.where(has_next, (nxt - idx).float() / rate_hz, cap).clamp(max=GAP_CAP_S)
    gap_len = torch.where(ball_ok, torch.zeros_like(since), (since + until).clamp(max=GAP_CAP_S))
    shared = torch.stack([torch.log1p(since), torch.log1p(until), torch.log1p(gap_len),
                          has_last.float(), has_next.float()], -1)

    def at(i):
        return ball_xy.gather(1, i.clamp(0, T - 1)[..., None].expand(-1, -1, 2))

    lxy, nxy = at(last), at(nxt)
    edge = torch.cat([has_last[..., None].expand(-1, -1, 4), has_next[..., None].expand(-1, -1, 4)], -1)
    per_team = []
    for k in (0, 1):
        own, opp = cent[:, :, k], cent[:, :, 1 - k]
        f = torch.cat([lxy - own, lxy - opp, nxy - own, nxy - opp], -1) / PITCH_X
        per_team.append(f * edge)
    return shared, torch.stack(per_team, 2)


def player_inputs(players: Tensor, ball: Tensor):
    """players [B,T,N,6] (x,y,vx,vy,speed,present), ball [B,T,5] (x,y,vx,vy,present).

    Missing values may be NaN. Returns player features [B,T,N,9], ball features [B,T,5],
    player presence [B,T,N] and ball presence [B,T] as bool, and ball xy with zeros where absent.
    """
    pres = (players[..., 5] > 0.5) & ~torch.isnan(players[..., :2]).any(-1)
    bok = (ball[..., 4] > 0.5) & ~torch.isnan(ball[..., :2]).any(-1)
    p = torch.nan_to_num(players[..., :5]) * pres[..., None]
    b = torch.nan_to_num(ball[..., :4]) * bok[..., None]
    scale = torch.tensor([PITCH_X, PITCH_Y, SPEED, SPEED, SPEED], dtype=p.dtype, device=p.device)
    rel = (p[..., :2] - b[:, :, None, :2]) / PITCH_X * (pres & bok[..., None])[..., None]
    bflag = bok[:, :, None, None].expand(-1, -1, p.shape[2], 1).to(p.dtype)
    pf = torch.cat([p / scale, pres[..., None].to(p.dtype), rel, bflag], -1)
    bscale = torch.tensor([PITCH_X, PITCH_Y, SPEED, SPEED], dtype=b.dtype, device=b.device)
    bf = torch.cat([b / bscale, bok[..., None].to(b.dtype)], -1)
    return pf, bf, pres, bok, b[..., :2]
