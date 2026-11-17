"""Reference baseline on canonical arrays: nearest player to the ball, then a centred majority vote (starter/baseline.py)."""
from __future__ import annotations

import numpy as np

NOBODY = -1


def nearest_team(c, chunk=20000):
    """Per-frame team (0 home, 1 away) of the player nearest the ball, NOBODY where the ball is missing or no player is seen."""
    out = np.full(c.T, NOBODY, np.int8)
    for a in range(0, c.T, chunk):
        b = min(a + chunk, c.T)
        P = np.asarray(c.players[a:b, :, :2])
        B = np.asarray(c.ball[a:b, :2])
        d = np.linalg.norm(P - B[:, None], axis=-1)
        d = np.where(np.isnan(d), np.inf, d)
        j = d.argmin(1)
        ok = np.isfinite(d.min(1)) & np.isfinite(B[:, 0])
        out[a:b][ok] = c.team[j[ok]]
    return out


def smooth(codes, window, half):
    """Majority vote over +-window frames within each half; D (=2) where nobody has the ball. Codes H=0, A=1, D=2."""
    res = np.full(len(codes), 2, np.int8)
    for h in np.unique(half):
        sel = np.flatnonzero(half == h)
        cd = codes[sel]
        if window <= 0:
            res[sel] = np.where(cd == 0, 0, np.where(cd == 1, 1, 2))
            continue
        hc = np.r_[0, np.cumsum(cd == 0)]
        ac = np.r_[0, np.cumsum(cd == 1)]
        i = np.arange(len(cd))
        lo, hi = np.maximum(0, i - window), np.minimum(len(cd), i + window + 1)
        hm, am = hc[hi] - hc[lo], ac[hi] - ac[lo]
        lab = np.full(len(cd), 2, np.int8)
        seen = (hm + am) > 0
        lab[seen] = np.where(hm[seen] >= am[seen], 0, 1)
        res[sel] = lab
    return res
