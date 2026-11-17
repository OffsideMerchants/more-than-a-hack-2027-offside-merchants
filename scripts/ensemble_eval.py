"""Average saved 5 Hz probabilities of several runs, tune decoder costs on the SC validation match, score SC test.

  python scripts/ensemble_eval.py --tags a_tcn96_f85 c_tcn128_f85 [--out arsh-research/results/ensemble.json]
Test matches are never used for tuning (costs come from the validation match only).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from possession_gnn import data_sc, diagnostics as dg, evalbridge as eb  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--tags", nargs="+", required=True)
ap.add_argument("--seed", default="s0")
ap.add_argument("--out", default=None)
ap.add_argument("--no_tune", action="store_true", help="use default decoder costs (needed when runs have different validation matches)")
a = ap.parse_args()
ALL = data_sc.matches()
TEST, VAL = ALL[::5], ALL[1]  # fold-0 test; validation match differs for runs made after the fold refactor


def avg(sub, m):
    ps = []
    for t in a.tags:
        f = next((ROOT / "artifacts" / "preds").glob(f"m_{t}/{t}_*_s*/{sub}/{m}.npz"), None)
        if f is None:
            raise SystemExit(f"missing {t}/{sub}/{m}")
        ps.append(np.load(f))
    la = np.mean([np.log(np.clip(p["p_alive"], 1e-4, 1 - 1e-4) / (1 - np.clip(p["p_alive"], 1e-4, 1 - 1e-4))) for p in ps], 0)
    lh = np.mean([np.log(np.clip(p["p_home"], 1e-4, 1 - 1e-4) / (1 - np.clip(p["p_home"], 1e-4, 1 - 1e-4))) for p in ps], 0)
    return 1 / (1 + np.exp(-la)), 1 / (1 + np.exp(-lh))


if a.no_tune:
    cost, vf = (4.0, 6.0), float("nan")
else:
    dv = data_sc.load(VAL)
    (cost, vf) = dg.tune_decoder(dv, *avg("val", VAL))
print("decoder costs (from validation match):", cost, "val spellF1 %.3f" % vf)
rows, cm = {}, np.zeros((3, 3), np.int64)
for m in TEST:
    d = data_sc.load(m)
    pa, ph = avg("sc", m)
    met, conf = dg.score_full(d, pa, ph, cost)
    base = eb.score(d, pa, ph)["spell_f1_mean"]
    rows[m] = {"spell_f1": met["spell_f1_mean"], "spell_f1_default_decoder": base, "macro_f1": met["macro_f1"],
               "n_gt": met["n_gt_spells"], "n_pred": met["n_pred_spells"]}
    cm += np.array(conf["matrix"])
    print(m, {k: round(v, 3) for k, v in rows[m].items()})
tp = np.diag(cm)
P, R = tp / cm.sum(0), tp / cm.sum(1)
F = 2 * P * R / (P + R)
res = {"tags": a.tags, "costs": cost, "mean_spell_f1": float(np.mean([r["spell_f1"] for r in rows.values()])),
       "mean_macro_f1": float(np.mean([r["macro_f1"] for r in rows.values()])), "accuracy": float(tp.sum() / cm.sum()),
       "confusion": cm.tolist(), "precision": P.tolist(), "recall": R.tolist(), "f1": F.tolist(), "per_match": rows}
print("MEAN spellF1 %.3f macroF1 %.3f acc %.3f" % (res["mean_spell_f1"], res["mean_macro_f1"], res["accuracy"]))
if a.out:
    Path(a.out).write_text(json.dumps(res, indent=1))
