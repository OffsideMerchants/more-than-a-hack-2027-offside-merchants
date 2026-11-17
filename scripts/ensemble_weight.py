"""Choose the M1 weight of the M1 + TCN ensemble on the validation matches, then score it once on the test matches.

  python scripts/ensemble_weight.py [--out arsh-research/results/ensemble_weight.json]

Same path as scripts/predict_ensemble.py, on the open matches of splits.json (val: weight choice, test: one score):
  - input: the halves of m1/data/finetune_cache.pkl (IDSSE as simulated ATD, SkillCorner as broadcast tracking), so
    both models see the same degraded positions;
  - M1: four test-time passes (m1/scripts/predict.py: m1_probs);
  - TCN: the five models of models/possession_ensemble.pt averaged in logit space, each half oriented so home attacks
    +x (possession.canonical.attack_signs, as for BAR-ATH);
  - log-linear pool (predict_ensemble.pool), M1's decoder (m1/runs/bar_ath/decoder.npz), official scorer on the
    controlled-possession labels (2 s rule).
The weight is the best mean spell-F1 over the validation matches; the test matches are scored after the choice, for
the chosen weight and for each model alone (w = 1: M1, w = 0: TCN) as reference points.
"""
import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "m1"), str(REPO / "m1" / "scripts"), str(REPO / "scripts"), str(REPO / "src")]
import predict_ensemble as ens  # noqa: E402  (imports m1/scripts/predict.py as ens.m1, scripts/predict_barath.py as ens.tcn)
from footy import official  # noqa: E402
from footy.data import BALL, MAX_PER_TEAM  # noqa: E402
from footy.pipeline import EVAL, VAL  # noqa: E402
from possession.canonical import attack_signs  # noqa: E402
from possession_gnn import run_s2s4 as R  # noqa: E402

WEIGHTS = np.round(np.arange(0, 1.0001, 0.1), 1)
TEAM = np.r_[np.zeros(MAX_PER_TEAM), np.ones(MAX_PER_TEAM), np.full(R.NSLOT - 2 * MAX_PER_TEAM, -1)].astype(np.int64)


def tcn_probs(models, pos5):
    """TCN ensemble on one half (M1 slot layout, 5 Hz) -> H/A/D probabilities [n5, 3]."""
    xy, ball = pos5[:, :BALL].astype(np.float32), pos5[:, BALL].astype(np.float32)
    s = attack_signs(xy[:, :MAX_PER_TEAM], xy[:, MAX_PER_TEAM:BALL], np.zeros(len(xy), int))[0]
    pl, bl, _ = R.raw({"xy": xy * s, "ball": ball * s, "native": np.ones(1)}, 0, clean=True)
    team = torch.from_numpy(TEAM)
    la, lh = [], []
    for m in models:
        pa, ph = R.infer("tcn", m, pl, bl, team, R.WIN["tcn"])
        la.append(ens.tcn.logit(pa)); lh.append(ens.tcn.logit(ph))
    pa, ph = 1 / (1 + np.exp(-np.mean(la, 0))), 1 / (1 + np.exp(-np.mean(lh, 0)))
    pa, ph = np.clip(pa, 1e-4, 1 - 1e-4), np.clip(ph, 1e-4, 1 - 1e-4)
    return np.stack([pa * ph, pa * (1 - ph), 1 - pa], 1)


def score(halves, probs, w, dec):
    pred = [ens.m1.decode(ens.pool(pm, pt, w), h["rules"], h["T"], *dec) for h, (pm, pt) in zip(halves, probs)]
    r = official.evaluate([h["gt"] for h in halves], pred)
    return {k: float(r[k]) for k in ("spell_f1_mean", "macro_f1")} | {k: int(r[k]) for k in ("n_pred_spells", "n_gt_spells")}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cache", default=str(REPO / "m1" / "data" / "finetune_cache.pkl"))
    ap.add_argument("--tags", nargs="+", default=[f"cs{k}" for k in range(5)])
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=str(REPO / "arsh-research" / "results" / "ensemble_weight.json"))
    a = ap.parse_args()
    t0 = time.time()
    device = ens.m1.setup_device(a.device)
    R.dev = device
    data = pickle.loads(Path(a.cache).read_bytes())
    os.chdir(REPO / "m1")                                   # M1's model and decoder paths are relative to m1/
    net, window = ens.m1.load_net(ens.m1.MODEL, device)
    dec = ens.m1.load_decoder()
    models = [ens.tcn.load_model(t) for t in a.tags]
    probs = {}
    for k in VAL + EVAL:
        probs[k] = [(ens.m1.m1_probs(net, window, h["pos5"], device), tcn_probs(models, h["pos5"])) for h in data[k]]
        assert all(pm.shape == pt.shape for pm, pt in probs[k]), k
        print(f"{k[0]} {k[1]}: {len(data[k])} halves ({time.time() - t0:.0f}s)", flush=True)

    name = lambda k: f"{k[0]}_{k[1]}"
    val = {float(w): {name(k): score(data[k], probs[k], w, dec) for k in VAL} for w in WEIGHTS}
    mean = {w: float(np.mean([r["spell_f1_mean"] for r in v.values()])) for w, v in val.items()}
    print("validation spell-F1 per M1 weight:")
    for w in val:
        print(f"  w={w:.1f}  mean {mean[w]:.4f}  " + "  ".join(f"{m} {r['spell_f1_mean']:.4f}" for m, r in val[w].items()))
    best = max(mean, key=mean.get)
    print(f"chosen weight (validation only): {best:.1f}", flush=True)

    test = {}
    for w in dict.fromkeys([best, 1.0, 0.0]):                # chosen weight first; the choice is already fixed
        test[w] = {name(k): score(data[k], probs[k], w, dec) for k in EVAL}
        print(f"test w={w:.1f}{' (chosen)' if w == best else ''}: mean {np.mean([r['spell_f1_mean'] for r in test[w].values()]):.4f}  "
              + "  ".join(f"{m} {r['spell_f1_mean']:.4f}" for m, r in test[w].items()), flush=True)

    out = {"weight": best, "val_mean_spellF1": mean, "val": val,
           "test": {str(w): v for w, v in test.items()},
           "test_mean_spellF1": {str(w): float(np.mean([r["spell_f1_mean"] for r in v.values()])) for w, v in test.items()},
           "m1_model": ens.m1.MODEL, "tcn_tags": a.tags, "val_matches": [name(k) for k in VAL],
           "test_matches": [name(k) for k in EVAL], "minutes": (time.time() - t0) / 60}
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(f"-> {a.out}", flush=True)


if __name__ == "__main__":
    main()
