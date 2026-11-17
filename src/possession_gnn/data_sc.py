"""SkillCorner open data (MIT, github.com/SkillCorner/opendata) -> the same 5 Hz dict as data.load.

Labels come from the provider's phases of play: the team in possession during a phase is H/A, frames
between phases are D. The ball is NaN where it was not detected; players keep the provider's
(partly extrapolated) positions. Home attacks +x in each period. Z, ids and roles are never used.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import ROOT

SC = ROOT / "artifacts" / "external" / "skillcorner" / "data" / "matches"
CACHE = ROOT / "artifacts" / "skillcorner"
STEP = 2  # 10 fps -> 5 Hz
PMAX = 32


def matches():
    """SkillCorner match ids; empty when the open data is not downloaded (prediction does not need it)."""
    return sorted(p.name for p in SC.iterdir() if p.is_dir()) if SC.is_dir() else []


def _parse(mid):
    m = json.loads((SC / mid / f"{mid}_match.json").read_text(encoding="utf-8"))
    home = m["home_team"]["id"]
    team_of = {p["id"]: (0 if p["team_id"] == home else 1) for p in m["players"]}
    periods = {p["period"]: (p["start_frame"], p["end_frame"]) for p in m["match_periods"]}
    side = m["home_team_side"]
    n_frames = max(e for _, e in periods.values()) + 1
    ids = sorted(team_of, key=lambda i: (team_of[i], i))
    col = {i: j for j, i in enumerate(ids)}
    xy = np.full((n_frames, len(ids), 2), np.nan, np.float32)
    ball = np.full((n_frames, 2), np.nan, np.float32)
    with open(SC / mid / f"{mid}_tracking_extrapolated.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            fr = r["frame"]
            if r["period"] is None or fr >= n_frames:
                continue
            b = r["ball_data"]
            if b["is_detected"]:
                ball[fr] = (b["x"], b["y"])
            for p in r["player_data"]:
                j = col.get(p["player_id"])
                if j is not None and p["x"] is not None:
                    xy[fr, j] = (p["x"], p["y"])
    ph = pd.read_csv(SC / mid / f"{mid}_phases_of_play.csv")
    y = np.full(n_frames, 2, np.int8)
    for r in ph.itertuples():
        y[r.frame_start:r.frame_end] = 0 if r.team_in_possession_id == home else 1
    team = np.array([team_of[i] for i in ids], np.int8)
    fr_sel, half = [], []
    for per in (1, 2):
        s, e = periods[per]
        idx = np.arange(s, e + 1, STEP)
        if side[per - 1] != "left_to_right":  # home attacks -x -> mirror
            xy[idx] *= -1
            ball[idx] *= -1
        fr_sel.append(idx)
        half.append(np.full(len(idx), per - 1, np.int8))
    sel = np.concatenate(fr_sel)
    xy, ball, y = xy[sel], ball[sel], y[sel]
    seen = np.argsort(-np.isfinite(xy[..., 0]).sum(0), kind="stable")[:PMAX]
    keep = np.sort(seen)
    half = np.concatenate(half)
    return dict(xy=xy[:, keep], team=team[keep], ball=ball, y=y, half=half, frame=sel,
                y25=np.repeat(y, 5), half25=np.repeat(half, 5), native=np.array(1, np.int8))


def load(mid):
    CACHE.mkdir(parents=True, exist_ok=True)
    p = CACHE / f"{mid}_5hz.npz"
    if p.exists():
        return dict(np.load(p))
    d = _parse(mid)
    np.savez_compressed(p, **d)
    return d
