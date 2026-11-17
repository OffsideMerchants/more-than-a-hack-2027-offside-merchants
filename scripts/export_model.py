"""Bundle the five TCN seeds (cs0..cs4) into models/possession_ensemble.pt (weights + training args)."""
import glob
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
tags = sys.argv[1:] or [f"cs{k}" for k in range(5)]
bundle = {}
for t in tags:
    f = sorted(glob.glob(str(ROOT / "artifacts" / "models" / f"{t}_J*_s*.pt")))[0]
    ck = torch.load(f, map_location="cpu")
    bundle[t] = {"state": ck["state"], "args": {k: ck["args"].get(k, "folds") for k in ("hidden", "blocks", "attn", "win", "lam_turn", "lam_spell", "split", "sc_fold", "steps", "seed")}}
out = ROOT / "models"
out.mkdir(exist_ok=True)
torch.save(bundle, out / "possession_ensemble.pt")
print("saved", list(bundle), round((out / "possession_ensemble.pt").stat().st_size / 1e6, 1), "MB")
