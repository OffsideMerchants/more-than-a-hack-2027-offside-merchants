"""RED tier: ensemble of the five TCN seeds -> BAR-ATH possession CSV in artifacts/barath/ (never opened).

  python scripts/predict_barath.py --out artifacts/barath/pred_ensemble.csv   (uses models/possession_ensemble.pt)
Prints only counts and PASS/FAIL. Frames outside the published graded blocks are written as X.
half_probs() is also used by scripts/predict_ensemble.py.
"""
import argparse
import os
import subprocess
import sys

import numpy as np
import torch

from possession import canonical, io_atd
from possession.guard import quiet, verdict
from possession.paths import BARATH_ART, ROOT
from possession_gnn import evalbridge as eb
from possession_gnn import run_s2s4 as R

STEP = 5  # 25 Hz -> 5 Hz


def load_model(tag):
    ck = torch.load(os.path.join(ROOT, "models", "possession_ensemble.pt"), map_location=R.dev)[tag]
    a = ck["args"]
    m = R.make_model("tcn", a["hidden"], a["blocks"], a["attn"]).to(R.dev)
    m.load_state_dict(ck["state"])
    return m.eval()


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def load_canon(data_dir=None, use_cache=True):
    """BAR-ATH in the canonical layout: the cache in artifacts/barath/ if present (and use_cache), else the raw ATD."""
    if use_cache:
        try:
            return canonical.load("BAR-ATH", "atd", BARATH_ART)
        except Exception:  # noqa: BLE001
            pass
    return io_atd.load_canon(data_dir)[0]


def half_probs(c, tags, device=None):
    """Fold models averaged in logit space -> {half: (frame indices of the half, p_alive [n5], p_home [n5])}.

    5 Hz sample k of a half is frame idx[STEP * k]. device: torch device for the models (default: R.dev).
    """
    if device is not None:
        R.dev = torch.device(device)
    models = [load_model(t) for t in tags]
    T = R.WIN["tcn"]
    keep = np.sort(np.argsort(-(c.players[..., 5] > 0).sum(0))[:R.NSLOT])  # models have NSLOT slots: most-present first
    verdict("slots", True, f"raw={c.N} used={len(keep)}")
    tm = c.team[keep]
    team = torch.from_numpy(np.concatenate([tm, np.full(R.NSLOT - len(tm), -1)]).astype(np.int64))
    xy = c.players[:, keep, :2].copy()
    xy[c.players[:, keep, 5] == 0] = np.nan
    ball = c.ball[:, :2].copy()
    ball[c.ball[:, 4] == 0] = np.nan
    out = {}
    for h in np.unique(c.half):
        idx = np.flatnonzero(c.half == h)
        sel = idx[::STEP]
        d = {"xy": xy[sel], "ball": ball[sel], "native": np.ones(1)}
        pl, bl, _ = R.raw(d, 0, clean=True)
        la, lh = [], []
        for m in models:
            pa, ph = R.infer("tcn", m, pl, bl, team, T)
            la.append(logit(pa)); lh.append(logit(ph))
        out[h] = (idx, 1 / (1 + np.exp(-np.mean(la, 0))), 1 / (1 + np.exp(-np.mean(lh, 0))))
    return out


@quiet
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", default=[f"cs{k}" for k in range(5)])
    ap.add_argument("--out", required=True)
    ap.add_argument("--data-dir", default=None, help="data bundle root (default: $MTAH_DATA_DIR or data/)")
    ap.add_argument("--device", default=None, help="cpu or cuda (default: cuda if available)")
    a = ap.parse_args()
    c = load_canon(a.data_dir)
    codes = np.full(c.T, 3, np.int8)
    probs = half_probs(c, a.tags, a.device)
    for h, (idx, pa, ph) in probs.items():
        path = np.repeat(eb.viterbi(pa, ph), STEP)
        path = np.concatenate([path, np.full(max(0, len(idx) - len(path)), path[-1])])[:len(idx)]
        codes[idx] = path
    graded = io_atd.graded_mask()
    codes[~graded] = 3
    lab = np.array(list("HADX"))[codes]
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fh:
        fh.write("frame,label\n")
        fh.write("".join(f"{i},{l}\n" for i, l in enumerate(lab)))
    verdict("rows written", len(lab) == io_atd.N_FRAMES, f"n={len(lab)} models={len(a.tags)}")
    r = subprocess.run([sys.executable, os.path.join(ROOT, "evaluation", "validate_submission.py"), a.out], capture_output=True, text=True)
    verdict("validate_submission", r.returncode == 0)


if __name__ == "__main__":
    main()
