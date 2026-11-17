"""Prediction-file contract: artifacts/preds/<model>/<run_id>/<source>/<match>.npz."""
from __future__ import annotations

import os

import numpy as np

from .paths import ARTIFACTS

REQUIRED = ("frame", "p_alive", "p_home")
OPTIONAL = ("p_turnover", "p_restart")


def pred_path(model, run_id, source, match):
    return os.path.join(ARTIFACTS, "preds", model, run_id, source, f"{match}.npz")


def write_pred(path, frame, p_alive, p_home, **optional):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    arrs = {"frame": np.asarray(frame, np.int64), "p_alive": np.round(p_alive, 4).astype(np.float32),
            "p_home": np.round(p_home, 4).astype(np.float32)}
    for k, v in optional.items():
        assert k in OPTIONAL, k
        arrs[k] = np.round(v, 4).astype(np.float32)
    np.savez(path, **arrs)


def validate(path_or_dict, n_frames=None):
    """Returns (ok, [problems]). Prints nothing: safe to call on BAR-ATH files."""
    d = dict(np.load(path_or_dict)) if isinstance(path_or_dict, str) else path_or_dict
    bad = []
    for k in REQUIRED:
        if k not in d:
            bad.append(f"missing {k}")
    if bad:
        return False, bad
    n = len(d["frame"])
    if n_frames is not None and n != n_frames:
        bad.append(f"length {n} != {n_frames}")
    if not np.all(np.diff(d["frame"]) > 0):
        bad.append("frame not strictly increasing")
    for k in REQUIRED[1:] + tuple(k for k in OPTIONAL if k in d):
        v = np.asarray(d[k], float)
        if v.shape != (n,):
            bad.append(f"{k} shape")
        elif not np.isfinite(v).all():
            bad.append(f"{k} non-finite")
        elif v.min() < 0 or v.max() > 1:
            bad.append(f"{k} outside [0,1]")
        elif not np.allclose(v, np.round(v, 4), atol=1e-6):
            bad.append(f"{k} not rounded to 1e-4")
    return not bad, bad
