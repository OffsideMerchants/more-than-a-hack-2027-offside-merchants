"""Decode 5 Hz probabilities, upsample to 25 Hz, score with evaluation/possession_metrics (not reimplemented)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "evaluation"))
import possession_metrics as pm  # noqa: E402

UP = 5


def viterbi(p_alive, p_home, switch=4.0, dead_switch=6.0):
    """3-state (H,A,D) Viterbi with fixed switching costs; emission = -log p."""
    eps = 1e-4
    pa, ph = np.clip(p_alive, eps, 1 - eps), np.clip(p_home, eps, 1 - eps)
    em = -np.log(np.stack([pa * ph, pa * (1 - ph), 1 - pa], 1))
    cost = np.array([[0, switch, dead_switch], [switch, 0, dead_switch], [dead_switch, dead_switch, 0]])
    T = len(em)
    dp, bp = em[0].copy(), np.zeros((T, 3), np.int8)
    for t in range(1, T):
        m = dp[:, None] + cost
        bp[t] = m.argmin(0)
        dp = m.min(0) + em[t]
    path = np.zeros(T, np.int8)
    path[-1] = dp.argmin()
    for t in range(T - 1, 0, -1):
        path[t - 1] = bp[t, path[t]]
    return path


def decode_25hz(d, p_alive, p_home, decode="viterbi"):
    """-> (gt25, pred25) code arrays with 25 X frames between halves (so halves never merge)."""
    h5, h25, y25 = d["half"], d["half25"], d["y25"]
    gt, pr = [], []
    for h in (0, 1):
        m5 = h5 == h
        pa, ph = p_alive[m5], p_home[m5]
        path = viterbi(pa, ph) if decode == "viterbi" else np.where(pa < .5, 2, np.where(ph > .5, 0, 1))
        up = np.repeat(path, UP)
        yy = y25[h25 == h]
        up = np.concatenate([up, np.full(max(0, len(yy) - len(up)), up[-1])])[:len(yy)]
        gt += [yy, np.full(25, 3)]
        pr += [up, np.full(25, 3)]
    return np.concatenate(gt).astype(np.int8), np.concatenate(pr).astype(np.int8)


def score(d, p_alive, p_home, decode="viterbi", bootstrap_n=0):
    gt, pr = decode_25hz(d, p_alive, p_home, decode)
    return pm.evaluate(gt, pr, bootstrap_n=bootstrap_n)


def gap_len(missing):
    """Length (steps) of the ball-gap run containing each step; 0 where the ball is visible."""
    out = np.zeros(len(missing), int)
    i, T = 0, len(missing)
    while i < T:
        if missing[i]:
            j = i
            while j < T and missing[j]:
                j += 1
            out[i:j] = j - i
            i = j
        else:
            i += 1
    return out


BUCKETS = [("visible", 0, 0), ("<1s", 1, 5), ("1-3s", 6, 15), ("3-10s", 16, 50), (">10s", 51, 10 ** 9)]


def gap_report(d, ball_missing, p_home):
    """Owner accuracy on alive steps and turnover recall (+-2 s) by ball-gap bucket, at 5 Hz argmax."""
    y = d["y"]
    gl = gap_len(ball_missing)
    alive = y < 2
    ph = (p_home > .5).astype(int)
    ok = (ph == 1) == (y == 0)
    tt = np.where((y[1:] < 2) & (y[:-1] < 2) & (y[1:] != y[:-1]))[0] + 1
    flips = set(np.where(ph[1:] != ph[:-1])[0] + 1)
    rep = {}
    for name, lo, hi in BUCKETS:
        m = alive & (gl >= lo) & (gl <= hi)
        sel = [t for t in tt if lo <= gl[t] <= hi]
        hit = sum(1 for t in sel if any(s in flips and ph[s] == int(y[t] == 0) for s in range(t - 10, t + 11)))
        rep[name] = {"n": int(m.sum()), "owner_acc": float(ok[m].mean()) if m.any() else None,
                     "turnovers": len(sel), "turnover_recall_2s": hit / len(sel) if sel else None}
    return rep
