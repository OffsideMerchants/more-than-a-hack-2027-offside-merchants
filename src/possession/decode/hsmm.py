"""Explicit-duration HSMM decoding (H, A, D) on a decimated grid, durations learned from IDSSE labels.

Optional evidence conditions transitions: `p_turnover` rewards an H<->A switch at that step, `p_restart` rewards D->H/A.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit

from .viterbi import emissions

NEG = -1e18


@dataclass
class DurationModel:
    logdur: np.ndarray  # [3, Dmax+1] log P(duration = d steps | state), index 0 unused
    logtrans: np.ndarray  # [3,3] log P(next state | state), diagonal NEG
    stride: int  # frames per step (25 Hz / stride = decode rate)
    start: np.ndarray  # [3] log initial distribution


def fit_durations(label_sets, stride=5, dmax_s=60.0, smooth=1.5):
    """label_sets: iterable of (alive bool[T], home bool[T], half[T]) from training folds only.
    Spells never cross half boundaries. Durations are histogrammed in decode steps and smoothed."""
    dmax = int(dmax_s * 25 / stride)
    cnt = np.zeros((3, dmax + 1))
    tr = np.ones((3, 3))
    for alive, home, half in label_sets:
        code = np.where(alive, np.where(home, 0, 1), 2)
        for h in np.unique(half):
            c = code[half == h]
            edges = np.flatnonzero(np.diff(c)) + 1
            starts = np.r_[0, edges]
            stops = np.r_[edges, len(c)]
            for k, (a, b) in enumerate(zip(starts, stops)):
                d = max(int(round((b - a) / stride)), 1)
                cnt[c[a], min(d, dmax)] += 1
                if k + 1 < len(starts):
                    tr[c[a], c[starts[k + 1]]] += 1
    x = np.arange(dmax + 1)
    logdur = np.full_like(cnt, NEG)
    for s in range(3):
        sm = np.zeros(dmax + 1)
        for d in range(1, dmax + 1):
            if cnt[s, d] == 0:
                continue
            w = np.exp(-0.5 * ((x - d) / (smooth * (1 + d * 0.1))) ** 2)
            sm += cnt[s, d] * w / w[1:].sum()
        sm += 1e-6
        logdur[s, 1:] = np.log(sm[1:] / sm[1:].sum())
    np.fill_diagonal(tr, 0)
    logtrans = np.where(tr > 0, np.log(np.maximum(tr, 1e-12) / tr.sum(1, keepdims=True)), NEG)
    return DurationModel(logdur, logtrans, stride, np.log(np.ones(3) / 3))


@njit(cache=True)
def _hsmm(cum, logdur, logtrans, start, bonus, dmax):
    """cum [T+1,3] cumulative emissions; bonus [T+1,3,3] added to transition r->s entering at step t."""
    T = cum.shape[0] - 1
    delta = np.full((T + 1, 3), -1e18)
    bprev = np.full((T + 1, 3), -1e18)
    arg_r = np.zeros((T + 1, 3), np.int8)
    arg_d = np.zeros((T + 1, 3), np.int32)
    for s in range(3):
        bprev[0, s] = start[s]
        arg_r[0, s] = -1
    for t in range(1, T + 1):
        for s in range(3):
            best, bd = -1e18, 1
            for d in range(1, min(dmax, t) + 1):
                v = bprev[t - d, s] + logdur[s, d] + cum[t, s] - cum[t - d, s]
                if v > best:
                    best, bd = v, d
            delta[t, s] = best
            arg_d[t, s] = bd
        for s in range(3):
            b, r_arg = -1e18, 0
            for r in range(3):
                if r == s:
                    continue
                v = delta[t, r] + logtrans[r, s] + bonus[t, r, s]
                if v > b:
                    b, r_arg = v, r
            bprev[t, s] = b
            arg_r[t, s] = r_arg
    out = np.zeros(T, np.int8)
    s = np.argmax(delta[T])
    t = T
    while t > 0:
        d = arg_d[t, s]
        out[t - d:t] = s
        r = arg_r[t - d, s]
        t -= d
        if r >= 0:
            s = r
    return out


def hsmm_decode(p_alive, p_home, dm: DurationModel, p_turnover=None, p_restart=None, temp=0.3,
                turn_w=1.0, restart_w=1.0):
    """Per-frame state codes at 25 Hz, upsampled from the decode grid (refine() sharpens boundaries)."""
    st = dm.stride
    em = emissions(p_alive, p_home) * temp
    T = len(em)
    n = -(-T // st)
    pad = n * st - T
    if pad:
        em = np.vstack([em, np.repeat(em[-1:], pad, 0)])
    cum = np.vstack([np.zeros((1, 3)), np.cumsum(em.reshape(n, st, 3).sum(1), 0)])
    bonus = np.zeros((n + 1, 3, 3))

    def per_step(p):
        return np.clip(np.pad(np.asarray(p, float), (0, pad), mode="edge").reshape(n, st).max(1), 1e-4, 1 - 1e-4)

    if p_turnover is not None:
        pt = per_step(p_turnover)
        bonus[:n, 0, 1] = bonus[:n, 1, 0] = turn_w * np.log(pt / (1 - pt))
    if p_restart is not None:
        pr = per_step(p_restart)
        bonus[:n, 2, 0] = bonus[:n, 2, 1] = restart_w * np.log(pr / (1 - pr))
    states = _hsmm(cum, dm.logdur, dm.logtrans, dm.start, bonus, dm.logdur.shape[1] - 1)
    return np.repeat(states, st)[:T]
