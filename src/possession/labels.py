"""IDSSE labels. `v1_raw`: BallStatus==0 -> D else BallPossession. `v2_control`: harmonised with CHALLENGE.md, where a
touch without control, a deflection or a lost aerial duel does not change possession.

The `v2_control` rule is fixed here, before any model is scored. T_BLIP defaults to 2 s. Nothing in this file may be
fitted to the public BAR-ATH window.
"""
from __future__ import annotations

import os
import pickle

import numpy as np

from .io_idsse import time_to_frame_index
from .paths import ARTIFACTS

FPS = 25
T_BLIP_S = 2.0
EVENT_WINDOW_S = 0.5  # slack around a candidate blip when looking for control evidence
PROX_M, PROX_S = 1.5, 1.0  # tracking evidence: same player within 1.5 m of the ball for >= 1 s
RULE_VERSION = "v2_control/1"


def load_meta(match_id):
    with open(os.path.join(ARTIFACTS, "idsse", f"{match_id}_meta.pkl"), "rb") as fh:
        return pickle.load(fh)


def v1_raw(meta):
    """(alive bool [T], home bool [T]). home is meaningful only where alive. BallPossession 1 = home, 2 = guest."""
    return meta["ball_alive"].astype(bool), meta["ball_possession"] == 1


def spells(alive, home):
    """Possession spells: maximal runs of constant team inside alive stretches -> [(start, stop, home_bool)]."""
    a = np.asarray(alive, bool)
    h = np.asarray(home, bool)
    code = np.where(a, np.where(h, 1, 0), -1)
    out, start = [], 0
    for i in range(1, len(code) + 1):
        if i == len(code) or code[i] != code[start]:
            if code[start] >= 0:
                out.append((start, i, bool(code[start])))
            start = i
    return out


def assert_mapping(meta, canon):
    """BallPossession=1 -> home at every kickoff of this match (KickOff event team vs label)."""
    idx_t0 = meta["half_t0"]
    ok, n = 0, 0
    for e in meta["events"]:
        if e["type"] != "KickOff" or not e.get("play_team"):
            continue
        i = time_to_frame_index(idx_t0, meta["frame_n"], e["time"])
        if i < 0:
            continue
        for j in range(i, min(i + 50, len(meta["ball_alive"]))):  # first alive frame at/after the kickoff
            if meta["ball_alive"][j]:
                # majority over the first 3 s of live play: the flag can lag the kick by ~1 s (see lag audit)
                seg = meta["ball_possession"][j:j + 75][meta["ball_alive"][j:j + 75]]
                n += 1
                ok += int(((seg == 1).mean() > 0.5) == (e["play_team"] == meta["home_team"]))
                break
    return ok, n


def _event_evidence(meta, team_id, a, b):
    """Completed play or possession-changing tackle by `team_id` within frames [a, b)."""
    lo = a - int(EVENT_WINDOW_S * FPS)
    hi = b + int(EVENT_WINDOW_S * FPS)
    for e in meta["_ev_frames"]:
        f = e["frame"]
        if f < lo:
            continue
        if f > hi:
            break
        if e["type"] == "Play" and e.get("play_team") == team_id and e.get("evaluation") == "successfullyCompleted":
            return "completed_play"
        if e["type"] == "TacklingGame" and e.get("possession_change") and e.get("winner_team") == team_id:
            return "tackle_possession_change"
    return None


def _track_evidence(canon, team01, a, b):
    sl = slice(max(a - int(EVENT_WINDOW_S * FPS), 0), min(b + int(EVENT_WINDOW_S * FPS), canon.T))
    pl = np.asarray(canon.players[sl][:, :, :2])
    ball = np.asarray(canon.ball[sl][:, :2])
    sel = canon.team == team01
    d = np.linalg.norm(pl[:, sel] - ball[:, None], axis=-1)
    d = np.where(np.isnan(d), np.inf, d)
    j = d.argmin(1)
    near = d[np.arange(len(j)), j] <= PROX_M
    run, best, prev = 0, 0, -1
    for t in range(len(j)):
        if near[t] and j[t] == prev:
            run += 1
        else:
            run = 1 if near[t] else 0
        prev = j[t] if near[t] else -1
        best = max(best, run)
    return "proximity" if best >= PROX_S * FPS else None


def prepare_meta(meta):
    ev = []
    for e in meta["events"]:
        i = time_to_frame_index(meta["half_t0"], meta["frame_n"], e["time"])
        if i >= 0 and e["type"] != "Delete":
            ev.append({**e, "frame": i})
    ev.sort(key=lambda e: e["frame"])
    meta["_ev_frames"] = ev
    return meta


def v2_control(canon, meta, t_blip_s=T_BLIP_S, return_changes=False):
    """Returns (alive, home[, changes]). Blips with no control evidence are relabelled to the surrounding team."""
    if "_ev_frames" not in meta:
        prepare_meta(meta)
    alive, home = v1_raw(meta)
    home = home.copy()
    team_id = {True: meta["home_team"], False: meta["guest_team"]}
    changes = []
    for _ in range(5):
        sp = spells(alive, home)
        changed = False
        for k, (a, b, h) in enumerate(sp):
            if (b - a) >= t_blip_s * FPS or k == 0:
                continue
            pa, pb, ph = sp[k - 1]
            if pb != a or ph == h:  # previous spell must be the other team and directly adjacent
                continue
            nxt_other = k + 1 < len(sp) and sp[k + 1][0] == b and sp[k + 1][2] == ph
            nxt_dead = b < len(alive) and not alive[b]
            if not (nxt_other or nxt_dead):
                continue
            ev = _event_evidence(meta, team_id[h], a, b) or _track_evidence(canon, 0 if h else 1, a, b)
            if ev:
                continue
            home[a:b] = ph
            changes.append({"start": a, "stop": b, "team_home": h, "to_home": ph, "dur_s": (b - a) / FPS, "followed_by": "other" if nxt_other else "dead"})
            changed = True
        if not changed:
            break
    return (alive, home, changes) if return_changes else (alive, home)


def event_lags(meta, alive, home):
    """Seconds between event timestamps and label transitions (lag audit): restarts vs dead->alive, tackles vs owner change."""
    prepare_meta(meta) if "_ev_frames" not in meta else None
    a = alive.astype(np.int8)
    restarts = np.flatnonzero(np.diff(a, prepend=a[0]) == 1)
    chg = np.array([s[0] for s in spells(alive, home)[1:]])
    out = {"restart": [], "tackle_change": []}
    for e in meta["_ev_frames"]:
        if e["type"] in ("ThrowIn", "GoalKick", "CornerKick", "FreeKick", "KickOff", "Penalty") and len(restarts):
            out["restart"].append((restarts[np.abs(restarts - e["frame"]).argmin()] - e["frame"]) / FPS)
        elif e["type"] == "TacklingGame" and e.get("possession_change") and len(chg):
            out["tackle_change"].append((chg[np.abs(chg - e["frame"]).argmin()] - e["frame"]) / FPS)
    return {k: np.array(v) for k, v in out.items()}
