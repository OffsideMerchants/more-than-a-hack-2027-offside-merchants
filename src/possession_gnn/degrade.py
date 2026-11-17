"""Minimal 5 Hz degradation (S3 stand-in for S0 degrade.py). Parameters are global, never frame-indexed.

Targets from DATA.md: ball missing ~41% of graded frames, ~20.5 tracked objects. Everything the model
sees is derived AFTER degradation (positions, velocities, graph edges).
"""
import numpy as np

HZ = 5


def _semi_markov(T, rng, p_missing, mean_gap_s, sigma=1.0):
    """Alternating visible/missing blocks; missing lengths lognormal, visible set to hit p_missing."""
    mu = np.log(mean_gap_s * HZ) - sigma ** 2 / 2
    mean_vis = mean_gap_s * HZ * (1 - p_missing) / p_missing
    out = np.zeros(T, bool)
    i = 0
    miss = rng.random() < p_missing
    while i < T:
        n = max(1, int(rng.lognormal(mu, sigma))) if miss else max(1, int(rng.exponential(mean_vis)))
        out[i:i + n] = miss
        i += n
        miss = not miss
    return out


def degrade(d, rng, ball_missing=0.41, ball_gap_s=3.5, player_missing=0.07, player_gap_s=1.5, noise=0.3):
    xy = d["xy"].copy()
    T, P, _ = xy.shape
    for j in range(P):
        xy[_semi_markov(T, rng, player_missing, player_gap_s), j] = np.nan
    xy += rng.normal(0, noise, xy.shape).astype(np.float32)
    ball = d["ball"].copy()
    ball[_semi_markov(T, rng, ball_missing, ball_gap_s)] = np.nan
    ball += rng.normal(0, noise, ball.shape).astype(np.float32)
    return xy, ball


PMAX = 32


def _pad(xy, team):
    k = PMAX - xy.shape[1]
    if k > 0:
        xy = np.concatenate([xy, np.full((xy.shape[0], k, 2), np.nan, np.float32)], 1)
        team = np.concatenate([team, np.full(k, -1, team.dtype)])
    return xy, team


def clean(d):
    return d["xy"].copy(), d["ball"].copy()


def node_features(xy, ball, team, ball_blind=False):
    """-> nodes [T,P+1,8] float32 (x,y,vx,vy,speed,is_home,is_away,present) and present mask [T,P+1].

    Last node is the ball (is_home=is_away=0). ball_blind drops it on every frame.
    Velocities are finite differences of the degraded positions only (zero when a neighbour is missing).
    """
    xy, team = _pad(xy, team)
    T, P, _ = xy.shape
    allp = np.concatenate([xy, ball[:, None, :]], 1)
    pres = ~np.isnan(allp[..., 0])
    if ball_blind:
        pres[:, -1] = False
    v = np.zeros_like(allp)
    ok = pres[1:] & pres[:-1]
    v[1:] = np.where(ok[..., None], (allp[1:] - allp[:-1]) * HZ, 0.0)
    v[:-1] = np.where(ok[..., None], v[1:], v[:-1])
    v = np.clip(np.nan_to_num(v), -12, 12)
    sp = np.linalg.norm(v, axis=-1, keepdims=True)
    tm = np.concatenate([team, [-1]])
    f = np.concatenate([np.nan_to_num(allp) / np.array([52.5, 34.0], np.float32), v / 8, sp / 8,
                        np.broadcast_to((tm == 0)[None, :, None], (T, P + 1, 1)),
                        np.broadcast_to((tm == 1)[None, :, None], (T, P + 1, 1)),
                        pres[..., None]], -1).astype(np.float32)
    f[~pres] = 0
    return f, pres
