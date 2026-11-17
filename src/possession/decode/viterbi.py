"""3-state decoders over states H=0, A=1, D=2 (same codes as evaluation/possession_metrics.py)."""
from __future__ import annotations

import numpy as np
from numba import njit

EPS = 1e-4


def emissions(p_alive, p_home, eps=EPS):
    """Per-frame log-likelihoods [T,3] for H, A, D from the model contract."""
    pa = np.clip(np.asarray(p_alive, np.float64), eps, 1 - eps)
    ph = np.clip(np.asarray(p_home, np.float64), eps, 1 - eps)
    return np.stack([np.log(pa) + np.log(ph), np.log(pa) + np.log(1 - ph), np.log(1 - pa)], 1)


def argmax_decode(p_alive, p_home):
    """Reference only: no sequence model."""
    return np.where(np.asarray(p_alive) >= 0.5, np.where(np.asarray(p_home) >= 0.5, 0, 1), 2).astype(np.int8)


@njit(cache=True)
def _viterbi(em, cost):
    T = em.shape[0]
    d = np.zeros((T, 3))
    bp = np.zeros((T, 3), np.int8)
    d[0] = em[0]
    for t in range(1, T):
        for s in range(3):
            best, arg = -1e300, 0
            for r in range(3):
                v = d[t - 1, r] - cost[r, s]
                if v > best:
                    best, arg = v, r
            d[t, s] = best + em[t, s]
            bp[t, s] = arg
    out = np.zeros(T, np.int8)
    out[T - 1] = np.argmax(d[T - 1])
    for t in range(T - 1, 0, -1):
        out[t - 1] = bp[t, out[t]]
    return out


def switch_costs(c_team=8.0, c_dead=6.0):
    """Symmetric fixed costs: H<->A pays c_team, entering or leaving D pays c_dead."""
    return np.array([[0, c_team, c_dead], [c_team, 0, c_dead], [c_dead, c_dead, 0]], np.float64)


def viterbi(p_alive, p_home, c_team=8.0, c_dead=6.0, temp=0.3):
    """`temp` tempers frame emissions: frames are strongly correlated, so raw products are over-confident."""
    return _viterbi(emissions(p_alive, p_home) * temp, switch_costs(c_team, c_dead))
