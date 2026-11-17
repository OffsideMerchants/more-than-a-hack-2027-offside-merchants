"""Confusion matrix, per-class scores and validation-tuned decoding (IDSSE only)."""
import itertools

import numpy as np

from . import evalbridge as eb

NAMES = ["H", "A", "D"]


def confusion(gt, pr):
    """gt/pr: 25 Hz code arrays (0 H, 1 A, 2 D, 3 X). Graded frames are gt != 3. -> dict."""
    m = gt != 3
    cm = np.zeros((3, 3), np.int64)
    np.add.at(cm, (gt[m].astype(int), np.clip(pr[m], 0, 2).astype(int)), 1)
    tp = np.diag(cm).astype(float)
    prec = tp / np.maximum(cm.sum(0), 1)
    rec = tp / np.maximum(cm.sum(1), 1)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
    return {"labels": NAMES, "matrix": cm.tolist(), "accuracy": float(tp.sum() / cm.sum()),
            "precision": prec.tolist(), "recall": rec.tolist(), "f1": f1.tolist(),
            "macro_f1": float(f1.mean())}


def decode_with(d, pa, ph, switch, dead_switch):
    orig = eb.viterbi
    eb.viterbi = lambda a, h, s=switch, ds=dead_switch: orig(a, h, s, ds)
    try:
        return eb.decode_25hz(d, pa, ph)
    finally:
        eb.viterbi = orig


GRID = list(itertools.product([2.0, 4.0, 6.0, 9.0], [3.0, 6.0, 9.0, 13.0]))


def tune_decoder(d, pa, ph):
    """Pick (switch, dead_switch) maximising spellF1 on a *validation* match."""
    best, arg = -1, GRID[0]
    for s, ds in GRID:
        gt, pr = decode_with(d, pa, ph, s, ds)
        f = eb.pm.evaluate(gt, pr)["spell_f1_mean"]
        if f > best:
            best, arg = f, (s, ds)
    return arg, best


def score_full(d, pa, ph, params=(4.0, 6.0), bootstrap_n=0):
    gt, pr = decode_with(d, pa, ph, *params)
    m = eb.pm.evaluate(gt, pr, bootstrap_n=bootstrap_n)
    return m, confusion(gt, pr)
