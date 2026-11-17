"""IDSSE (Sportec/DFL) loader: local XML only, streamed line by line, no kloppy.

Deliberately avoids kloppy's Sportec loader: its `only_alive=True` default silently deletes dead frames
and it downloads instead of reading local files. Referee FrameSets are excluded.

Returns a `RawMatch` in centred metres, 25 Hz, with *all* frames kept (dead included). Identity (`person_id`,
`role`) and ball height `z` are kept in aux fields for label construction and the degradation simulator only;
they must never reach a model.
"""
from __future__ import annotations

import os
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from .paths import IDSSE_DIR, ARTIFACTS

MATCHES = ["J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY", "J03WQQ", "J03WR9"]
LOCKBOX = "J03WMX"
FPS = 25

_FS = re.compile(r'<FrameSet GameSection="(\w+)" MatchId="[^"]*" TeamId="([^"]+)" PersonId="([^"]+)"')
_FR = re.compile(r'<Frame N="(\d+)" T="([^"]+)"[^>]* X="([-\d.]+)" Y="([-\d.]+)"(?: Z="([-\d.]+)")?[^>]*?(?: BallPossession="(\d)" BallStatus="(\d)")?/>')


@dataclass
class RawMatch:
    match_id: str
    home_team: str
    guest_team: str
    pitch: tuple
    frame_n: np.ndarray  # [T] DFL frame number N
    half: np.ndarray  # [T] 1 or 2
    xy: np.ndarray  # [T,P,2] metres, NaN missing (players of both teams, no referees)
    person_id: list
    team_id: list  # team id string per person
    role: list  # PlayingPosition if known else ""
    ball: np.ndarray  # [T,2] NaN missing
    ball_z: np.ndarray  # [T]
    ball_possession: np.ndarray  # [T] 1 home, 2 guest, 0 unknown
    ball_alive: np.ndarray  # [T] bool
    kickoff_utc: str = ""
    events: list = field(default_factory=list)
    half_t0: dict = field(default_factory=dict)  # half -> (first frame N, epoch seconds of that frame)


def parse_match_info(match_id):
    path = _find(f"DFL_02_01_matchinformation_*{match_id}.xml")
    root = ET.parse(path).getroot()
    g = root.find(".//General")
    env = root.find(".//Environment")
    roles = {}
    for p in root.iter("Player"):
        roles[p.get("PersonId")] = p.get("PlayingPosition", "")
    return {
        "home": g.get("HomeTeamId"), "guest": g.get("GuestTeamId"),
        "home_name": g.get("HomeTeamName"), "guest_name": g.get("GuestTeamName"),
        "pitch": (float(env.get("PitchX")), float(env.get("PitchY"))), "roles": roles,
    }


def _find(pattern):
    import glob
    hits = glob.glob(os.path.join(IDSSE_DIR, pattern))
    if len(hits) != 1:
        raise FileNotFoundError(f"{pattern}: {len(hits)} matches in {IDSSE_DIR}")
    return hits[0]


def _parse_time(s):
    return datetime.fromisoformat(s).timestamp()


def load_raw(match_id: str) -> RawMatch:
    info = parse_match_info(match_id)
    path = _find(f"DFL_04_03_positions_raw_observed_*{match_id}.xml")
    persons, ball_rows, half_t0 = {}, {"firstHalf": {}, "secondHalf": {}}, {}
    cur = None
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("<FrameSet"):
                m = _FS.match(line)
                half, team, pid = m.groups()
                cur = None if team == "referee" else (half, team, pid)
                continue
            if cur is None or not line.startswith("<Frame "):
                continue
            m = _FR.match(line)
            n, tt, x, y, z, bp, bs = m.groups()
            half, team, pid = cur
            if team == "BALL":
                if half not in half_t0:
                    half_t0[half] = (int(n), _parse_time(tt))
                ball_rows[half][int(n)] = (float(x), float(y), float(z or 0), int(bp or 0), int(bs or 0))
            else:
                persons.setdefault(pid, {"team": team, "rows": {"firstHalf": {}, "secondHalf": {}}})["rows"][half][int(n)] = (float(x), float(y))
    frames, halves = [], []
    for hi, h in enumerate(("firstHalf", "secondHalf"), 1):
        ns = sorted(ball_rows[h])
        frames += ns
        halves += [hi] * len(ns)
    frames, halves = np.array(frames), np.array(halves)
    T = len(frames)
    # frame numbers are unique across halves in DFL files; index by N
    index = {n: i for i, n in enumerate(frames)}
    pids = sorted(persons, key=lambda p: (persons[p]["team"] != info["home"], p))
    xy = np.full((T, len(pids), 2), np.nan, np.float32)
    for j, pid in enumerate(pids):
        for h in ("firstHalf", "secondHalf"):
            for n, (x, y) in persons[pid]["rows"][h].items():
                i = index.get(n)
                if i is not None:
                    xy[i, j] = (x, y)
    ball = np.full((T, 2), np.nan, np.float32)
    z = np.zeros(T, np.float32)
    bpos = np.zeros(T, np.int8)
    alive = np.zeros(T, bool)
    for h in ("firstHalf", "secondHalf"):
        for n, (x, y, zz, bp, bs) in ball_rows[h].items():
            i = index[n]
            ball[i], z[i], bpos[i], alive[i] = (x, y), zz, bp, bs == 1
    return RawMatch(
        match_id=match_id, home_team=info["home"], guest_team=info["guest"], pitch=info["pitch"],
        frame_n=frames, half=halves, xy=xy, person_id=pids, team_id=[persons[p]["team"] for p in pids],
        role=[info["roles"].get(p, "") for p in pids], ball=ball, ball_z=z, ball_possession=bpos, ball_alive=alive,
        events=parse_events(match_id),
        half_t0={1: half_t0['firstHalf'], 2: half_t0['secondHalf']},
    )


def parse_events(match_id):
    """Flat list of dicts: time (epoch s), type, team, player, plus the attributes label rules need."""
    path = _find(f"DFL_03_02_events_raw_*{match_id}.xml")
    out = []
    for _, el in ET.iterparse(path, events=("end",)):
        if el.tag != "Event":
            continue
        child = next((c for c in el if c.tag not in ("Play",) or True), None)
        if child is None:
            el.clear()
            continue
        d = {"time": _parse_time(el.get("EventTime")), "type": child.tag, "id": el.get("EventId")}
        d["team"] = child.get("Team") or child.get("WinnerTeam")
        if child.tag == "TacklingGame":
            d.update(possession_change=child.get("PossessionChange") == "true", winner_team=child.get("WinnerTeam"),
                     loser_team=child.get("LoserTeam"), winner=child.get("Winner"), loser=child.get("Loser"),
                     winner_role=child.get("WinnerRole"), loser_role=child.get("LoserRole"))
        plays = list(child.iter("Play"))
        if plays:
            p = plays[0]
            d.update(player=p.get("Player"), play_team=p.get("Team"), evaluation=p.get("Evaluation"), recipient=p.get("Recipient"))
        else:
            d["player"] = child.get("Player")
        out.append(d)
        el.clear()
    out.sort(key=lambda e: e["time"])
    return out


def cache_path(match_id):
    return os.path.join(ARTIFACTS, "idsse", f"{match_id}_raw.npz")


def time_to_frame_index(half_t0, frame_n, t, tol=75):
    """Epoch seconds -> row index in the concatenated arrays. Event clocks can precede the first tracking frame of a
    half by ~1 s, so the nearest frame within `tol` frames (3 s) is accepted; otherwise -1."""
    h = 2 if t >= half_t0[2][1] - 3 else 1
    n0, t0 = half_t0[h]
    n = n0 + int(round((t - t0) * FPS))
    i = int(np.clip(np.searchsorted(frame_n, n), 0, len(frame_n) - 1))
    if i > 0 and abs(int(frame_n[i - 1]) - n) < abs(int(frame_n[i]) - n):
        i -= 1
    return i if abs(int(frame_n[i]) - n) <= tol else -1
