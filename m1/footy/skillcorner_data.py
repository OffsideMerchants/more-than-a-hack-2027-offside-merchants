"""SkillCorner open data (20 A-League 2024/25 matches) -> 29-slot positions at 25 Hz + H/A/D labels.

Source: github.com/SkillCorner/opendata (MIT licence; SkillCorner asks to be credited).
Tracking is broadcast-based at 10 fps. Players outside the camera view are *extrapolated*
(`is_detected` = False), which is about 45% of player positions, and the ball has x/y/z and its
own `is_detected` flag. Coordinates are metres with the pitch centred at (0, 0).

Labels come from the phases of play (`{id}_phases_of_play.csv`): phases only exist while the
ball is in play and carry the team in possession, so frames inside a phase are that team's,
and frames outside every phase are DeadBall. Convert them with footy.semantics.controlled_possession
for the challenge definition, as for IDSSE.

Cached per match as data/cache/skillcorner_<id>.npz (25 Hz): extrapolated positions, a
player-detected mask, the ball-detected mask and labels per period.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import BALL, MAX_PER_TEAM, NUM_AGENTS
from .kloppy_data import CACHE, _assign_slots

ROOT = Path("data/skillcorner")
SRC_HZ, FPS = 10, 25


def match_ids():
    return [int(x["id"]) for x in json.load(open(ROOT / "matches.json"))]


def _upsample(arr, n_out):
    """Linear interpolation along axis 0 from 10 Hz to 25 Hz; NaN where either neighbour is NaN."""
    t = np.arange(n_out) * SRC_HZ / FPS
    i0 = np.clip(np.floor(t).astype(int), 0, len(arr) - 1)
    i1 = np.clip(i0 + 1, 0, len(arr) - 1)
    w = (t - i0).reshape((-1,) + (1,) * (arr.ndim - 1))
    return arr[i0] * (1 - w) + arr[i1] * w


def _parse(mid):
    meta = json.load(open(ROOT / str(mid) / f"{mid}_match.json"))
    home_id, away_id = meta["home_team"]["id"], meta["away_team"]["id"]
    team_of = {p["id"]: (0 if p["team_id"] == home_id else 1) for p in meta["players"]}
    frames, periods, ball, ball_det, players = [], [], [], [], []
    with open(ROOT / str(mid) / f"{mid}_tracking_extrapolated.jsonl") as fh:
        for line in fh:
            fr = json.loads(line)
            if fr.get("period") is None:
                continue
            b = fr.get("ball_data") or {}
            frames.append(fr["frame"])
            periods.append(fr["period"])
            ball.append((b.get("x") if b.get("x") is not None else np.nan, b.get("y") if b.get("y") is not None else np.nan))
            ball_det.append(bool(b.get("is_detected")) and b.get("x") is not None)
            players.append({p["player_id"]: (p["x"], p["y"], p.get("is_detected", False)) for p in fr.get("player_data") or []})
    return meta, home_id, away_id, team_of, np.array(frames), np.array(periods), np.array(ball, float), np.array(ball_det), players


def load_skillcorner_match(mid, use_cache=True):
    """List of dicts per period: pos [T, 29, 2] (25 Hz, extrapolated players included),
    player_detected [T, 28] bool, ball_detected [T] bool, labels [T] (0 H, 1 A, 2 D)."""
    cache = CACHE / f"skillcorner_{mid}.npz"
    if use_cache and cache.exists():
        z = np.load(cache)
        return [{k: z[f"{k}{i}"] for k in ("pos", "player_detected", "ball_detected", "labels")} for i in range(int(z["n"]))]

    meta, home_id, away_id, team_of, frames, periods, ball, ball_det, players = _parse(mid)
    phases = pd.read_csv(ROOT / str(mid) / f"{mid}_phases_of_play.csv")
    out = []
    for period in sorted(set(periods)):
        sel = np.flatnonzero(periods == period)
        T10 = len(sel)
        fr = frames[sel]
        # slots per team from presence over the period
        ids = {0: [], 1: []}
        for pid, team in team_of.items():
            ids[team].append(pid)
        pos = np.full((T10, NUM_AGENTS, 2), np.nan, np.float32)
        det = np.zeros((T10, 2 * MAX_PER_TEAM), bool)
        for team, off in ((0, 0), (1, MAX_PER_TEAM)):
            xy = np.full((T10, len(ids[team]), 2), np.nan, np.float32)
            dd = np.zeros((T10, len(ids[team])), bool)
            col = {pid: j for j, pid in enumerate(ids[team])}
            for t, k in enumerate(sel):
                for pid, (x, y, d) in players[k].items():
                    j = col.get(pid)
                    if j is not None:
                        xy[t, j] = (x, y)
                        dd[t, j] = d
            slots = _assign_slots(~np.isnan(xy[..., 0]))
            for j, s in enumerate(slots):
                if s >= 0:
                    ok = ~np.isnan(xy[:, j, 0])
                    pos[ok, off + s] = xy[ok, j]
                    det[ok, off + s] = dd[ok, j]
        pos[:, BALL] = ball[sel]
        bdet = ball_det[sel]
        # labels from phases of play (frame ranges inclusive), outside phases = dead ball
        lab = np.full(T10, 2, np.int64)
        idx = {f: t for t, f in enumerate(fr)}
        for r in phases[phases.period == period].itertuples():
            team = 0 if r.team_in_possession_id == home_id else 1
            a = np.searchsorted(fr, r.frame_start)
            b = np.searchsorted(fr, r.frame_end, side="right")
            lab[a:b] = team
        T25 = int(T10 * FPS / SRC_HZ)
        t = np.arange(T25) * SRC_HZ / FPS
        nearest = np.clip(np.round(t).astype(int), 0, T10 - 1)
        out.append({"pos": _upsample(pos, T25).astype(np.float32), "player_detected": det[nearest],
                    "ball_detected": bdet[nearest], "labels": lab[nearest]})
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, n=len(out), **{f"{k}{i}": d[k] for i, d in enumerate(out) for k in d})
    return out


def as_observed(period, detected_only=False, ball_detected_only=True):
    """Positions with optional masking: undetected (extrapolated) players and/or undetected ball -> NaN."""
    pos = period["pos"].copy()
    if detected_only:
        pos[:, :BALL][~period["player_detected"]] = np.nan
    if ball_detected_only:
        pos[~period["ball_detected"], BALL] = np.nan
    return pos
