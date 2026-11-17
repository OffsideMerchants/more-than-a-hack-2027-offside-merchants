"""Evaluation conditions on IDSSE: clean, deg_A, deg_B (fixed seeded degraded copies, built from the calibrated params)."""
from __future__ import annotations

import json
import os

import numpy as np

from . import canonical, degrade, labels, io_idsse
from .paths import ARTIFACTS

CANON = os.path.join(ARTIFACTS, "canon")
SEEDS = {"deg_A": 1_000_003, "deg_B": 2_000_029}


def sim_params():
    pj = os.path.join(ARTIFACTS, "sim", "params.json")
    return degrade.SimParams(**json.load(open(pj))) if os.path.exists(pj) else degrade.SimParams()


def get(match_id, cond="clean") -> canonical.Canon:
    """Canon for a match under a condition. Degraded copies are cached on disk; they are deterministic given the params."""
    if cond == "clean":
        return canonical.load(match_id, "idsse", CANON)
    src = f"idsse_{cond}"
    try:
        return canonical.load(match_id, src, CANON)
    except FileNotFoundError:
        clean = canonical.load(match_id, "idsse", CANON)
        seed = SEEDS[cond] + io_idsse.MATCHES.index(match_id)
        d = degrade.degrade(clean, sim_params(), seed, tag=src)
        d.aux["attack_signs"] = clean.aux["attack_signs"]
        d.source = src
        canonical.save(d, CANON)
        return d


_META = {}


def get_labels(match_id, version="v2_control"):
    """(alive bool, home bool) per frame. v2_control is cached because it scans events and tracking."""
    path = os.path.join(ARTIFACTS, "labels", f"{match_id}_{version}_{labels.RULE_VERSION.replace('/', '')}.npz")
    if os.path.exists(path):
        z = np.load(path)
        return z["alive"], z["home"]
    meta = _META.get(match_id)
    if meta is None:
        meta = _META[match_id] = labels.prepare_meta(labels.load_meta(match_id))
    alive, home = labels.v1_raw(meta) if version == "v1_raw" else labels.v2_control(canonical.load(match_id, "idsse", CANON), meta)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, alive=alive, home=home)
    return alive, home
