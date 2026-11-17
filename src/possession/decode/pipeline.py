from __future__ import annotations

import numpy as np

from .hsmm import hsmm_decode
from .refine import refine
from .viterbi import argmax_decode, viterbi


def decode(pred: dict, method="hsmm+refine", dm=None, params=None, half=None):
    """pred: contract dict (p_alive, p_home, optional p_turnover/p_restart). Each half is decoded independently.
    Methods: argmax, viterbi, hsmm, hsmm+turn, hsmm+refine, hsmm+turn+refine."""
    params = params or {}
    T = len(pred["p_alive"])
    half = np.ones(T, int) if half is None else half
    out = np.zeros(T, np.int8)
    for h in np.unique(half):
        sel = half == h
        pa, ph = pred["p_alive"][sel], pred["p_home"][sel]
        pt = pred["p_turnover"][sel] if pred.get("p_turnover") is not None else None
        pr = pred["p_restart"][sel] if pred.get("p_restart") is not None else None
        if method == "argmax":
            c = argmax_decode(pa, ph)
        elif method == "viterbi":
            c = viterbi(pa, ph, **{k: v for k, v in params.items() if k in ("c_team", "c_dead", "temp")})
        else:
            hp = {k: v for k, v in params.items() if k in ("temp", "turn_w", "restart_w")}
            use = "turn" in method
            c = hsmm_decode(pa, ph, dm, pt if use else None, pr if use else None, **hp)
            if "refine" in method:
                c = refine(c, pa, ph, radius=params.get("radius", 50), temp=params.get("refine_temp", 1.0))
        out[sel] = c
    return out
