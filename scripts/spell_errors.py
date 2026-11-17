"""Spell-level error taxonomy for saved SC test predictions (python scripts/spell_errors.py TAG)."""
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from possession_gnn import data_sc, evalbridge as eb  # noqa: E402
pm = eb.pm
tag = sys.argv[1]
cat = {"matched>=.5": 0, "boundary off (.3-.5)": 0, "weak overlap (<.3)": 0, "wrong team only": 0, "predicted dead/none": 0}
errs_s, errs_e, dur_missed, dur_all = [], [], [], []
n_pred_total = n_fp = 0
for m in data_sc.matches()[::5]:
    d = data_sc.load(m)
    f = next((ROOT / "artifacts/preds").glob(f"m_{tag}/{tag}_*_s0/sc/{m}.npz"))
    p = np.load(f)
    gt, pr = eb.decode_25hz(d, p["p_alive"], p["p_home"])
    runs = pm.valid_runs(gt)
    gs, ps = pm.possession_spells(gt, runs), pm.possession_spells(pr, runs)
    n_pred_total += len(ps)
    ov = pm.spell_overlaps(gs, ps)
    best = {}
    for iou, g, q in ov:
        best[g] = max(best.get(g, (0, -1)), (iou, q))
    used_pred = set()
    for gi, s in enumerate(gs):
        L = (s[1] - s[0]) / 25
        dur_all.append(L)
        iou, q = best.get(gi, (0, -1))
        if iou >= .5:
            cat["matched>=.5"] += 1; used_pred.add(q)
            errs_s.append((ps[q][0] - s[0]) / 25); errs_e.append((ps[q][1] - s[1]) / 25)
        elif iou >= .3:
            cat["boundary off (.3-.5)"] += 1; used_pred.add(q)
        elif iou > 0:
            cat["weak overlap (<.3)"] += 1; dur_missed.append(L)
        else:
            prs = pr[s[0]:s[1]]
            cat["predicted dead/none" if (prs == 2).mean() > .5 else "wrong team only"] += 1; dur_missed.append(L)
    n_fp += len(ps) - len(used_pred)
tot = sum(cat.values())
print({k: f"{v} ({v/tot:.0%})" for k, v in cat.items()}, "true spells", tot, "pred spells", n_pred_total, "unmatched preds", n_fp)
print("matched boundary error (s): start median %.2f p90 %.2f | end median %.2f p90 %.2f" % (
    np.median(np.abs(errs_s)), np.percentile(np.abs(errs_s), 90), np.median(np.abs(errs_e)), np.percentile(np.abs(errs_e), 90)))
print("true spell duration median %.1fs; unmatched-spell median %.1fs" % (np.median(dur_all), np.median(dur_missed) if dur_missed else 0))
