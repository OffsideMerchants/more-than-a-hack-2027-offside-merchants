"""25 Hz boundary refinement: move each boundary to the best local position under the frame emissions."""
from __future__ import annotations

import numpy as np

from .viterbi import emissions


def refine(codes, p_alive, p_home, radius=50, temp=1.0, min_len=3):
    c = np.asarray(codes, np.int8).copy()
    em = emissions(p_alive, p_home) * temp
    cum = np.vstack([np.zeros((1, 3)), np.cumsum(em, 0)])
    b = np.flatnonzero(np.diff(c)) + 1
    for k, p in enumerate(b):
        left, right = c[p - 1], c[p]
        lo = (b[k - 1] + min_len) if k else min_len
        hi = (b[k + 1] - min_len) if k + 1 < len(b) else len(c) - min_len
        lo, hi = max(lo, p - radius), min(hi, p + radius)
        if lo >= hi:
            continue
        cand = np.arange(lo, hi + 1)
        score = (cum[cand, left] - cum[lo, left]) + (cum[hi + 1, right] - cum[cand, right])
        q = int(cand[np.argmax(score)])
        c[lo:q], c[q:hi + 1] = left, right
        b[k] = q
    return c
