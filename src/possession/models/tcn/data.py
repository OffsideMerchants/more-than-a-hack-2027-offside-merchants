"""Match container, event-aware window sampling, augmentation and collation for S2."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

# Window sampling mixture
MIX = {"turnover": 0.40, "restart": 0.20, "gap": 0.15, "uniform": 0.25}


@dataclass
class Match:
    """Canonical 25 Hz arrays. NaN = unobserved. Labels may be None at inference."""
    players: np.ndarray  # [T,N,5] x y vx vy speed, centred metres, home attacks +x
    team: np.ndarray  # [N] 0 home, 1 away
    ball: np.ndarray  # [T,4] x y vx vy
    alive: np.ndarray | None = None  # [T] bool
    home: np.ndarray | None = None  # [T] bool, meaningful where alive

    @property
    def T(self):
        return len(self.players)


def ball_present(ball):
    return np.isfinite(ball).all(-1)


def turnover_targets(home, alive, width=0):
    """1 at the first alive frame whose owner differs from the last alive owner; dilated +-width frames."""
    t = np.zeros(len(home), np.float32)
    last = None
    for i in np.flatnonzero(alive):
        if last is not None and home[i] != last:
            t[i] = 1
        last = home[i]
    if width:
        t = np.minimum(np.convolve(t, np.ones(2 * width + 1, np.float32), "same"), 1)
    return t


def event_frames(m: Match):
    """Frames of turnovers, restarts (dead->alive) and ball-gap starts."""
    a = m.alive.astype(np.int8)
    bp = ball_present(m.ball).astype(np.int8)
    return {
        "turnover": np.flatnonzero(turnover_targets(m.home, m.alive)),
        "restart": np.flatnonzero(np.diff(a, prepend=a[0]) == 1),
        "gap": np.flatnonzero(np.diff(bp, prepend=1) == -1),
    }


class WindowSampler:
    """Draws (match_idx, start) with event-aware mixture; every window has `win` frames at 25 Hz."""

    def __init__(self, matches, win, mix=None):
        self.matches, self.win = matches, win
        mix = mix or MIX
        self.events = [event_frames(m) for m in matches]
        self.kinds = list(mix)
        self.p = np.array([mix[k] for k in self.kinds], float)
        self.p /= self.p.sum()

    def sample(self, rng):
        mi = int(rng.integers(len(self.matches)))
        m, ev = self.matches[mi], self.events[mi]
        kind = self.kinds[rng.choice(len(self.kinds), p=self.p)]
        hi = m.T - self.win
        if kind != "uniform" and len(ev[kind]):
            f = int(rng.choice(ev[kind]))
            s = f - int(rng.integers(self.win // 8, self.win - self.win // 8))
        else:
            s = int(rng.integers(0, hi + 1))
        return mi, int(np.clip(s, 0, hi))


def augment(s: dict, rng, p_ball_mask=0.15, p_track_drop=0.1, swap=None, yflip=None):
    """In-place on one sample. Team swap is coupled with an x mirror so the new home still attacks +x."""
    swap = rng.random() < 0.5 if swap is None else swap
    yflip = rng.random() < 0.5 if yflip is None else yflip
    pl, ball = s["players"], s["ball"]
    if swap:
        s["team"] = np.where(s["team"] < 0, -1, 1 - s["team"])
        for a in (pl, ball):
            a[..., 0] *= -1
            a[..., 2] *= -1
        if "home" in s:
            s["home"] = 1 - s["home"]
    if yflip:
        for a in (pl, ball):
            a[..., 1] *= -1
            a[..., 3] *= -1
    perm = rng.permutation(pl.shape[1])  # slot permutation
    s["players"], s["team"] = pl[:, perm], s["team"][perm]
    s["players"][:, rng.random(pl.shape[1]) < p_track_drop] = np.nan  # track dropout
    if rng.random() < p_ball_mask:  # whole-ball masking forces use of player movement
        s["ball"][:] = np.nan
    return s


def make_sample(m: Match, start, win, stride, offset=0):
    """Slice [start, start+win) at `stride`; `offset` shifts the stride phase (random time crop)."""
    sl = slice(start + offset, start + win, stride)
    s = {"players": m.players[sl].copy(), "team": m.team.copy(), "ball": m.ball[sl].copy()}
    if m.alive is not None:
        s["alive"], s["home"] = m.alive[sl].astype(np.float32), m.home[sl].astype(np.float32)
        s["turn"] = turnover_targets(m.home, m.alive, 5)[sl]
    return s


def collate(samples, device="cpu"):
    """Pad slots to the max N in the batch; NaN -> 0 with masks."""
    B, T = len(samples), len(samples[0]["players"])
    N = max(s["players"].shape[1] for s in samples)
    pl = np.full((B, T, N, 5), np.nan, np.float32)
    team = np.full((B, N), -1, np.int64)
    for i, s in enumerate(samples):
        n = s["players"].shape[1]
        pl[i, :, :n], team[i, :n] = s["players"], s["team"]
    ball = np.stack([s["ball"] for s in samples]).astype(np.float32)
    pm = np.isfinite(pl).all(-1) & (team[:, None, :] >= 0)
    bm = np.isfinite(ball).all(-1)
    out = {"players": np.nan_to_num(pl), "pmask": pm, "team": team, "ball": np.nan_to_num(ball), "bmask": bm}
    if "alive" in samples[0]:
        for k in ("alive", "home", "turn"):
            out[k] = np.stack([s[k] for s in samples]).astype(np.float32)
    return {k: torch.from_numpy(v).to(device) for k, v in out.items()}
