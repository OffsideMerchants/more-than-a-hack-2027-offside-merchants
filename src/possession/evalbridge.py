"""Bridge between model labels/predictions and `evaluation/possession_metrics.py` (never reimplement the metric).

Frame labels are (alive bool, home bool). Codes: H=0, A=1, D=2, X=3. Halves are separated by an X block so
`valid_runs` treats them separately.
"""
from __future__ import annotations

import os
import sys

import numpy as np

from .paths import ROOT

sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402

X_GAP = 100


def to_codes(alive, home):
    alive, home = np.asarray(alive, bool), np.asarray(home, bool)
    return np.where(alive, np.where(home, pm.H, pm.A), pm.D).astype(np.int8)


def with_halves(codes, half, gap=X_GAP):
    """Insert X blocks between halves. Returns codes with len = T + gap*(n_halves-1)."""
    parts = []
    for h in np.unique(half):
        if parts:
            parts.append(np.full(gap, pm.X, np.int8))
        parts.append(codes[half == h])
    return np.concatenate(parts)


def score(gt_codes, pred_codes, half, bootstrap_n=0, mask=None):
    """gt/pred: per-frame codes (no X), same length as `half`. `mask` marks scored frames (others become X)."""
    gt = np.asarray(gt_codes, np.int8).copy()
    if mask is not None:
        gt[~mask] = pm.X
    return pm.evaluate(with_halves(gt, half), with_halves(np.asarray(pred_codes, np.int8), half), bootstrap_n=bootstrap_n)


def headline(res):
    t = res["transitions"]
    out = {k: res.get(k) for k in ("spell_f1_mean", "macro_f1", "alive_dead_f1", "team_acc_both_alive",
                                   "n_gt_spells", "n_pred_spells", "gt_mean_spell_duration_s", "mean_spell_duration_s")}
    out["trans_f1_12"], out["trans_f1_25"] = t[12]["f1"], t[25]["f1"]
    out["median_latency"] = t[25]["median_latency"]
    return out
