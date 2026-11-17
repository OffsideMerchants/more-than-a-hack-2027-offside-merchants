"""Canonical per-match representation shared by IDSSE and BAR-ATH.

players[T,N,6] x y vx vy speed present | team[N] 0 home 1 away | ball[T,5] x y vx vy present
half[T] | frame[T]. Centred metres, home attacks +x in both halves (algorithmic, see `attack_signs`).
NaN plus mask where unobserved, never zero-filled. Speeds are recomputed from positions with one shared
filter for both sources, because DFL `S` and ATD speed units are not assumed equal.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

FPS = 25


@dataclass
class Canon:
    match_id: str
    source: str  # "idsse" | "atd"
    players: np.ndarray  # [T,N,6]
    team: np.ndarray  # [N] int8
    ball: np.ndarray  # [T,5]
    half: np.ndarray  # [T] int8
    frame: np.ndarray  # [T] int64 (IDSSE: DFL N; ATD: video frame = frame_id - 1)
    aux: dict = field(default_factory=dict)  # IDSSE only: ball_z, role, person_id. Never model inputs.

    @property
    def T(self):
        return len(self.frame)

    @property
    def N(self):
        return len(self.team)

    def pos_mask(self):
        return self.players[..., 5] > 0

    def ball_mask(self):
        return self.ball[:, 4] > 0


def central_velocity(xy: np.ndarray, fps: int = FPS) -> np.ndarray:
    """xy [T,...,2] -> velocity m/s with NaN-aware fallback: 5-point central, 3-point central, one-sided, else NaN."""
    T = xy.shape[0]
    v = np.full(xy.shape, np.nan, np.float32)
    def sh(k):  # xy shifted so out[t] = xy[t+k], NaN padded
        out = np.full_like(xy, np.nan)
        if k > 0:
            out[:-k] = xy[k:]
        elif k < 0:
            out[-k:] = xy[:k]
        else:
            out[:] = xy
        return out
    cands = [(sh(2) - sh(-2)) / (4 / fps), (sh(1) - sh(-1)) / (2 / fps), (sh(1) - xy) * fps, (xy - sh(-1)) * fps]
    for c in cands:
        v = np.where(np.isnan(v), c, v)
    return v.astype(np.float32)


def attack_signs(xy_home_slots, xy_away_slots, half: np.ndarray, min_presence=0.1):
    """Per half sign s with s=+1 if home already attacks +x. Home's goalkeeper is the home slot whose mean x is
    most extreme; it sits in the box its team defends. Returns {half: +1/-1} and asserts the two teams disagree."""
    out = {}
    for h in np.unique(half):
        sel = half == h
        side = []
        for slots in (xy_home_slots, xy_away_slots):
            x = slots[sel, :, 0]
            pres = np.isfinite(x).mean(0)
            with np.errstate(all='ignore'):
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    mean = np.where(pres >= min_presence, np.nanmean(x, 0), 0.0)
            j = int(np.argmax(np.abs(mean)))
            side.append(float(mean[j]))
        if side[0] * side[1] >= 0:
            raise AssertionError(f"half {h}: teams do not defend opposite ends (means {side})")
        out[int(h)] = 1 if side[0] < 0 else -1  # home defends -x -> attacks +x
    return out


def build(match_id, source, xy, team, ball_xy, half, frame, aux=None) -> Canon:
    """xy [T,N,2] metres (NaN missing), team [N], ball_xy [T,2], oriented here so home attacks +x."""
    xy = xy.astype(np.float32).copy()
    ball_xy = ball_xy.astype(np.float32).copy()
    signs = attack_signs(xy[:, team == 0], xy[:, team == 1], half)
    for h, s in signs.items():
        if s < 0:
            sel = half == h
            xy[sel] *= -1
            ball_xy[sel] *= -1
    aux = dict(aux or {})
    aux["attack_signs"] = signs
    return assemble(match_id, source, xy, team, ball_xy, half, frame, aux)


def assemble(match_id, source, xy, team, ball_xy, half, frame, aux=None) -> Canon:
    """Already-oriented positions -> Canon with velocities/speed recomputed from positions (one shared filter)."""
    v = np.full(xy.shape, np.nan, np.float32)
    bv = np.full(ball_xy.shape, np.nan, np.float32)
    for h in np.unique(half):
        sel = np.flatnonzero(half == h)
        v[sel] = central_velocity(xy[sel])
        bv[sel] = central_velocity(ball_xy[sel])
    present = np.isfinite(xy).all(-1)
    spd = np.linalg.norm(v, axis=-1)
    players = np.concatenate([xy, v, spd[..., None], present[..., None].astype(np.float32)], -1)
    bpres = np.isfinite(ball_xy).all(-1)
    ball = np.concatenate([ball_xy, bv, bpres[:, None].astype(np.float32)], -1)
    return Canon(match_id, source, players, np.asarray(team, np.int8), ball, half.astype(np.int8), np.asarray(frame, np.int64), dict(aux or {}))


def from_idsse(raw) -> Canon:
    team = np.array([0 if t == raw.home_team else 1 for t in raw.team_id], np.int8)
    c = build(raw.match_id, "idsse", raw.xy, team, raw.ball, raw.half, raw.frame_n,
              aux={"ball_z": raw.ball_z, "role": np.array(raw.role), "person_id": np.array(raw.person_id)})
    return c


def to_model_match(c: Canon, alive=None, home=None):
    """Adapter to the S2 TCN `Match` (NaN where unobserved, 5 features)."""
    from .models.tcn.data import Match
    p = c.players[..., :5].copy()
    p[c.players[..., 5] == 0] = np.nan
    b = c.ball[:, :4].copy()
    b[c.ball[:, 4] == 0] = np.nan
    return Match(p, c.team.astype(np.int64), b, alive, home)


def save(c: Canon, root: str):
    d = os.path.join(root, f"{c.source}_{c.match_id}")
    os.makedirs(d, exist_ok=True)
    for k in ("players", "team", "ball", "half", "frame"):
        np.save(os.path.join(d, k + ".npy"), getattr(c, k))
    np.savez(os.path.join(d, "aux.npz"), **{k: v for k, v in c.aux.items() if isinstance(v, np.ndarray)},
             attack_signs=np.array(sorted(c.aux["attack_signs"].items())))
    return d


def load(match_id, source, root, mmap=True) -> Canon:
    d = os.path.join(root, f"{source}_{match_id}")
    mm = "r" if mmap else None
    g = lambda k: np.load(os.path.join(d, k + ".npy"), mmap_mode=mm)
    aux = dict(np.load(os.path.join(d, "aux.npz"), allow_pickle=False))
    aux["attack_signs"] = {int(h): int(s) for h, s in aux["attack_signs"]}
    return Canon(match_id, source, g("players"), g("team"), g("ball"), g("half"), g("frame"), aux)
