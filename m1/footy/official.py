"""Score per-period predictions with the official MTAH scorer (evaluation/possession_metrics.py).

Our label codes match the scorer's: 0 = H, 1 = A, 2 = D; our -1 (ignore / no answer) becomes X (3).
Periods are joined with a block of X frames in between, so the scorer's valid_runs() treats each
period as its own graded run and never matches across a half boundary, as it does on BAR-ATH.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "evaluation"))
import possession_metrics as pm  # noqa: E402

GAP = 100  # X frames between periods


def to_match_array(per_period):
    """List of per-period label arrays (-1 allowed) -> one scorer-coded array with X gaps."""
    parts = []
    for i, lab in enumerate(per_period):
        if i:
            parts.append(np.full(GAP, pm.X, dtype=np.int8))
        lab = np.asarray(lab)
        parts.append(np.where(lab < 0, pm.X, lab).astype(np.int8))
    return np.concatenate(parts)


def evaluate(gt_periods, pred_periods, **kw):
    """Official metrics for one match. gt/pred: lists of per-period arrays at 25 Hz."""
    return pm.evaluate(to_match_array(gt_periods), to_match_array(pred_periods), **kw)


def summary(res):
    """The headline numbers as a flat dict."""
    return {k: res[k] for k in ("spell_f1_mean", "macro_f1", "n_pred_spells", "n_gt_spells",
                                "mean_spell_duration_s", "gt_mean_spell_duration_s", "alive_dead_f1",
                                "team_acc_both_alive")}
