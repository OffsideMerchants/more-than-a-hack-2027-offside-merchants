"""Ensemble of M1 (m1/) and the five TCN seeds (models/possession_ensemble.pt) -> BAR-ATH submission CSV.

  python scripts/predict_ensemble.py --data-dir "/path/to/dataset" --out evaluation/results/<team_name>.csv

Per graded half (games.half_video_blocks):
  1. M1: H/A/D probabilities at 5 Hz, test-time augmented (m1/scripts/predict.py: half_probs).
  2. TCN: five seeds averaged in logit space -> p_alive, p_home at 5 Hz over each detected half
     (scripts/predict_barath.py: half_probs), read off at M1's 5 Hz frames and turned into
     H = p_alive * p_home, A = p_alive * (1 - p_home), D = 1 - p_alive.
  3. Log-linear pooling: p ∝ p_M1^w * p_TCN^(1 - w).
  4. M1's decoder: HMM (m1/runs/bar_ath/decoder.npz) -> 25 Hz -> boundaries snapped to rule-based transitions.
Frames outside the graded halves are X. --weight 1 reproduces m1/scripts/predict.py exactly.

Runs on the GPU (CUDA) with PyTorch's deterministic settings, so repeated runs give the same file; falls back to the
CPU when no GPU is available (--device). Prints only counts and PASS/FAIL.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "m1" / "scripts"), str(REPO / "scripts"), str(REPO / "src")]
import predict as m1  # noqa: E402  (m1/scripts/predict.py)
import predict_barath as tcn  # noqa: E402
from possession.guard import quiet, verdict  # noqa: E402

WEIGHT = 0.3   # weight of M1 in the log-linear pool; chosen on the validation matches (scripts/ensemble_weight.py), never on BAR-ATH


def tcn_on_grid(probs, first, n5, step):
    """TCN half probabilities -> H/A/D probabilities [n5, 3] at video frames first + step * k."""
    frames = first + step * np.arange(n5)
    for idx, pa, ph in probs.values():
        k = np.searchsorted(idx, frames)
        if k.max() < len(idx) and (idx[np.minimum(k, len(idx) - 1)] == frames).all():
            j = np.minimum(k // step, len(pa) - 1)          # 5 Hz sample covering each frame (as in predict_barath)
            pa, ph = np.clip(pa[j], 1e-4, 1 - 1e-4), np.clip(ph[j], 1e-4, 1 - 1e-4)
            return np.stack([pa * ph, pa * (1 - ph), 1 - pa], 1)
    raise ValueError("graded half not inside one detected half")


def pool(p_m1, p_tcn, w):
    """Log-linear pooling, renormalised per frame (w = 1 / 0: one model's probabilities unchanged)."""
    if w in (0, 1):
        return p_m1 if w == 1 else p_tcn
    lp =w * np.log(np.clip(p_m1, 1e-6, 1)) + (1 - w) * np.log(np.clip(p_tcn, 1e-6, 1))
    p = np.exp(lp - lp.max(1, keepdims=True))
    return p / p.sum(1, keepdims=True)


@quiet
def main():
    from games import GAME, GAMES, half_video_blocks, require, source_frames

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", help="data bundle root holding tactical/ (default: $MTAH_DATA_DIR or <repository root>/data)")
    ap.add_argument("--out", default=str(REPO / "evaluation" / "results" / "ensemble.csv"))
    ap.add_argument("--weight", type=float, default=WEIGHT, help="M1 weight in [0, 1] (1 = M1 only, 0 = TCN only)")
    ap.add_argument("--tags", nargs="+", default=[f"cs{k}" for k in range(5)], help="TCN models (seeds) in models/possession_ensemble.pt")
    ap.add_argument("--device", default=None, help="cuda or cpu (default: cuda if available)")
    args = ap.parse_args()
    t0 = time.time()
    device = m1.setup_device(args.device)

    meta, raw = require(["atd_meta", "atd_raw"], GAME, args.data_dir)
    n_frames = source_frames("tactical", GAME)
    halves = m1.half_probs(meta, raw, half_video_blocks(GAME), n_frames, GAMES[GAME]["pitch"], device=device)
    c = tcn.load_canon(args.data_dir, use_cache=False)              # always from the raw ATD
    verdict("same frame count", c.T == n_frames, f"n={c.T}")
    probs = tcn.half_probs(c, args.tags, device)
    print(f"TCN: {len(args.tags)} models done ({time.time() - t0:.0f}s)", flush=True)

    prior, log_trans, scale = m1.load_decoder()
    labels = np.full(n_frames, "X", dtype="<U1")
    for period, first, last, p_m1, rules, T in halves:
        p = pool(p_m1, tcn_on_grid(probs, first, len(p_m1), m1.STEP), args.weight)
        labels[first:last + 1] = m1.LETTERS[m1.decode(p, rules, T, prior, log_trans, scale)]
    m1.write_csv(labels, args.out)
    verdict("rows written", len(labels) == n_frames, f"n={len(labels)} weight={args.weight:g} device={device} ({time.time() - t0:.0f}s)")
    r = subprocess.run([sys.executable, str(REPO / "evaluation" / "validate_submission.py"), args.out], capture_output=True, text=True)
    verdict("validate_submission", r.returncode == 0)


if __name__ == "__main__":
    main()
