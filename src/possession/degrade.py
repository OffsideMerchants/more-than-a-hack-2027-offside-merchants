"""Degradation simulator: turns clean IDSSE tracking into BAR-ATH-like automatic tracking.

Processes: semi-Markov ball dropout with hazard from ball height, speed and crowding; ball noise and false
detections; per-player gaps with a spatially correlated far-side component; track fragmentation into slots; ID swaps
between nearby teammates; duplicate tracks; coloured position noise. `Z`, identity and the label are used only
inside the hazard (simulating the sensor), never as model inputs. Parameters are global, never frame-indexed.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields

import numpy as np
from numba import njit
from scipy.ndimage import uniform_filter1d
from scipy.signal import lfilter

from . import canonical

FPS = 25
SLOTS_PER_TEAM = 23


@dataclass
class SimParams:
    ball_gap_rate: float = 0.08  # gap starts per second of visible ball at zero covariates
    ball_z_gain: float = 1.0
    ball_speed_gain: float = 0.5
    ball_crowd_gain: float = 0.5
    ball_gap_mu: float = 0.7  # lognormal over seconds
    ball_gap_sigma: float = 1.2
    ball_sigma: float = 0.35  # metres
    ball_false_rate: float = 0.01  # false detections per second
    player_gap_rate: float = 0.03  # per player per second
    player_gap_mu: float = -0.3
    player_gap_sigma: float = 0.9
    far_side_boost: float = 1.5  # log-hazard boost on the far touchline strip
    far_side_y: float = 22.0
    frag_mean_s: float = 45.0  # mean fragment length before a forced break
    swap_rate: float = 0.02  # ID swaps per team per second
    dup_rate: float = 0.004  # duplicate-track starts per player per second
    pos_sigma: float = 0.2
    pos_corr_s: float = 2.0
    ball_still_gain: float = 0.0  # log-hazard boost for dropout while the ball is near-stationary (dead ball)
    ball_smooth_s: float = 0.0  # moving-average window on the observed ball track (seconds)
    pl_smooth_s: float = 0.0  # moving-average window on player positions (seconds)

    def vector(self):
        return np.array([getattr(self, f.name) for f in fields(self)], float)

    @classmethod
    def from_vector(cls, v):
        return cls(**{f.name: float(x) for f, x in zip(fields(cls), v)})


# (low, high) for the optimiser
BOUNDS = {
    "ball_gap_rate": (0.005, 0.5), "ball_z_gain": (0.0, 4.0), "ball_speed_gain": (0.0, 3.0), "ball_crowd_gain": (0.0, 3.0),
    "ball_gap_mu": (-1.0, 3.5), "ball_gap_sigma": (0.2, 2.0), "ball_sigma": (0.0, 1.5), "ball_false_rate": (0.0, 0.2),
    "player_gap_rate": (0.0, 0.3), "player_gap_mu": (-1.5, 2.0), "player_gap_sigma": (0.2, 1.8), "far_side_boost": (0.0, 4.0),
    "far_side_y": (10.0, 32.0), "frag_mean_s": (5.0, 300.0), "swap_rate": (0.0, 0.3), "dup_rate": (0.0, 0.05),
    "pos_sigma": (0.0, 0.8), "pos_corr_s": (0.2, 10.0),
    "ball_still_gain": (0.0, 4.0), "ball_smooth_s": (0.0, 1.0), "pl_smooth_s": (0.0, 1.0),
}


@njit(cache=True)
def _semi_markov(hazard, mu, sigma, seed):
    """Gap mask: start a gap at frame t with prob hazard[t]; gap length lognormal(mu, sigma) frames (>=1)."""
    np.random.seed(seed)
    n = len(hazard)
    out = np.zeros(n, np.bool_)
    t = 0
    while t < n:
        if np.random.random() < hazard[t]:
            L = max(int(np.exp(mu + sigma * np.random.randn())), 1)
            out[t:min(t + L, n)] = True
            t += L
        else:
            t += 1
    return out


def _runs(mask):
    """Start/stop of True runs."""
    d = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


def degrade(canon: canonical.Canon, p: SimParams, seed: int, a: int = 0, b: int | None = None, tag: str = "deg") -> canonical.Canon:
    """Degrade frames [a, b) of a clean IDSSE Canon. Returns a Canon with 46 anonymous slots (23 per team)."""
    b = canon.T if b is None else b
    rng = np.random.default_rng(seed)
    T = b - a
    half = np.asarray(canon.half[a:b])
    xy = np.asarray(canon.players[a:b, :, :2]).astype(np.float64)
    pres = np.asarray(canon.players[a:b, :, 5]) > 0
    team = np.asarray(canon.team)
    ball = np.asarray(canon.ball[a:b, :2]).astype(np.float64)
    bspeed = np.linalg.norm(np.asarray(canon.ball[a:b, 2:4]), axis=1)
    z = np.asarray(canon.aux["ball_z"][a:b], float) if "ball_z" in canon.aux else np.zeros(T)

    # --- ball dropout, hazard from height, speed, crowding
    dist = np.linalg.norm(xy - ball[:, None], axis=-1)
    crowd = np.nansum(dist < 6.0, 1)
    cov = p.ball_z_gain * np.minimum(z / 5.0, 1) + p.ball_speed_gain * np.minimum(bspeed / 30.0, 1) + p.ball_crowd_gain * np.minimum(crowd / 8.0, 1) + p.ball_still_gain * (bspeed < 1.0)
    hz = np.clip(p.ball_gap_rate / FPS * np.exp(cov), 0, 0.5)
    mu = p.ball_gap_mu + np.log(FPS)
    ball_gap = _semi_markov(hz, mu, p.ball_gap_sigma, int(rng.integers(2**31)))
    vis = ~ball_gap
    # --- ball noise and false detections (inside visible stretches)
    ball_obs = ball + rng.normal(0, p.ball_sigma, ball.shape)
    n_false = rng.poisson(p.ball_false_rate * T / FPS)
    for _ in range(n_false):
        t0 = int(rng.integers(T))
        L = int(rng.integers(1, 9))
        j = int(rng.integers(xy.shape[1]))
        ball_obs[t0:t0 + L] = np.where(np.isfinite(xy[t0:t0 + L, j]), xy[t0:t0 + L, j], ball_obs[t0:t0 + L]) + rng.normal(0, 1.0, (min(L, T - t0), 2))
    if p.ball_smooth_s * FPS >= 2:
        ball_obs = uniform_filter1d(ball_obs, int(p.ball_smooth_s * FPS), axis=0, mode="nearest")
    ball_obs[~vis] = np.nan

    # --- player gaps (individual + far-touchline correlated)
    N0 = xy.shape[1]
    keep = pres.copy()
    # boost applies to whichever touchline the data puts at y>0; orientation is arbitrary, so the calibration finds the magnitude
    for j in range(N0):
        if not pres[:, j].any():
            continue
        far = np.clip((np.nan_to_num(xy[:, j, 1]) - p.far_side_y) / 8.0, 0, 1)
        hzp = np.clip(p.player_gap_rate / FPS * np.exp(p.far_side_boost * far), 0, 0.5)
        keep[:, j] &= ~_semi_markov(hzp, p.player_gap_mu + np.log(FPS), p.player_gap_sigma, int(rng.integers(2**31)))
    # --- coloured position noise
    if p.pos_sigma > 0:
        a_ = np.exp(-1 / (p.pos_corr_s * FPS))
        noise = lfilter([np.sqrt(1 - a_**2)], [1, -a_], rng.normal(0, 1, (T, N0, 2)), axis=0) * p.pos_sigma
        xy = xy + noise
    if p.pl_smooth_s * FPS >= 2:
        ok = np.isfinite(xy)
        xy = np.where(ok, uniform_filter1d(np.nan_to_num(xy), int(p.pl_smooth_s * FPS), axis=0, mode="nearest"), np.nan)

    # --- fragments: contiguous kept runs, split at random breaks; duplicates
    frags = []  # (team, t0, t1, player, dx, dy)
    for j in range(N0):
        s, e = _runs(keep[:, j])
        for s0, e0 in zip(s, e):
            cuts = [s0]
            t = s0
            while True:
                t += max(int(rng.exponential(p.frag_mean_s * FPS)), 5)
                if t >= e0 - 5:
                    break
                cuts.append(t)
            cuts.append(e0)
            for c0, c1 in zip(cuts[:-1], cuts[1:]):
                frags.append((team[j], c0, c1, j, 0.0, 0.0))
                if rng.random() < p.dup_rate * (c1 - c0) / FPS:
                    d0 = c0 + int(rng.integers(0, c1 - c0))
                    L = int(rng.exponential(3 * FPS)) + 5
                    frags.append((team[j], d0, min(d0 + L, c1), j, *rng.normal(0, 0.8, 2)))
    frags.sort(key=lambda f: f[1])
    slot_xy = np.full((T, 2 * SLOTS_PER_TEAM, 2), np.nan)
    free = {0: list(range(SLOTS_PER_TEAM)), 1: list(range(SLOTS_PER_TEAM, 2 * SLOTS_PER_TEAM))}
    active = []  # (t1, team, slot)
    for tm, t0, t1, j, dx, dy in frags:
        for item in [x for x in active if x[0] <= t0]:
            active.remove(item)
            free[item[1]].append(item[2])
        if not free[tm]:
            continue
        s = free[tm].pop(int(rng.integers(len(free[tm]))))
        active.append((t1, tm, s))
        slot_xy[t0:t1, s] = xy[t0:t1, j] + (dx, dy)
    # --- ID swaps between nearby teammates
    for tm in (0, 1):
        sl = np.arange(tm * SLOTS_PER_TEAM, (tm + 1) * SLOTS_PER_TEAM)
        for t in np.sort(rng.integers(0, T, rng.poisson(p.swap_rate * T / FPS))):
            act = [s for s in sl if np.isfinite(slot_xy[t, s, 0])]
            if len(act) < 2:
                continue
            P = slot_xy[t, act]
            d = np.linalg.norm(P[:, None] - P[None], axis=-1) + np.eye(len(act)) * 1e9
            i, k = np.unravel_index(np.argmin(d), d.shape)
            if d[i, k] < 5.0:
                sa, sb = act[i], act[k]
                tmp = slot_xy[t:, sa].copy()
                slot_xy[t:, sa], slot_xy[t:, sb] = slot_xy[t:, sb], tmp
    team_slots = np.array([0] * SLOTS_PER_TEAM + [1] * SLOTS_PER_TEAM, np.int8)
    out = canonical.assemble(canon.match_id, tag, slot_xy.astype(np.float32), team_slots, ball_obs.astype(np.float32), half,
                             np.asarray(canon.frame[a:b]), aux={"true_ball_visible": vis, "attack_signs": canon.aux.get("attack_signs", {}),
                                                                "ball_z": z, "src_slice": (a, b)})
    return out


def degrade_arrays_for_training(canon, p: SimParams, rng: np.random.Generator, start: int, win: int):
    """Fresh degradation for one training window (with 5 s lead-in for fragment/gap state)."""
    lead = 5 * FPS
    a = max(start - lead, 0)
    b = min(start + win, canon.T)
    d = degrade(canon, p, int(rng.integers(2**31)), a, b)
    return d, start - a
