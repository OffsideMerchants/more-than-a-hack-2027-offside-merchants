"""Minimal IDSSE loader for S3 (stand-in for the S0 canonical layer).

Parses the DFL positions XML directly (no kloppy: its only_alive default would drop dead frames),
returns 5 Hz arrays with home attacking +x. Ball/identity/Z are used for labels only.
"""
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
IDSSE = ROOT / "idsse-data"
CACHE = ROOT / "artifacts" / "idsse"
MATCHES = ["J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY", "J03WQQ", "J03WR9"]
STEP = 5  # 25 Hz -> 5 Hz

_fs = re.compile(r'<FrameSet GameSection="(\w+)" MatchId="[^"]+" TeamId="([^"]+)" PersonId="([^"]+)"')
_fr = re.compile(r'<Frame N="(\d+)"[^>]* X="([-\d.]+)" Y="([-\d.]+)"(?: Z="[-\d.]+")?[^>]*?(?: BallPossession="(\d)" BallStatus="(\d)")?/>')


def _path(kind, mid):
    return next(IDSSE.glob(f"{kind}*{mid}.xml"))


def _teams(mid):
    s = _path("DFL_02_01", mid).read_text(encoding="utf-8")
    return re.search(r'HomeTeamId="([^"]+)"', s).group(1), re.search(r'GuestTeamId="([^"]+)"', s).group(1)


def parse_match(mid):
    """-> dict of 25 Hz arrays: xy[T,P,2] (nan missing), team[P], ball[T,2], poss[T], alive[T], half[T]."""
    home, guest = _teams(mid)
    tracks, ball = {}, {}
    cur = None
    with open(_path("DFL_04_03", mid), encoding="utf-8") as f:
        for line in f:
            if line.startswith("<FrameSet"):
                m = _fs.match(line)
                half, team, pid = m.groups()
                cur = None if team == "referee" else (half, team, pid)
                if cur and team == "BALL":
                    cur = (half, "BALL", "BALL")
                store = {} if cur else None
                if cur:
                    (ball if team == "BALL" else tracks)[cur] = store
            elif cur and line.startswith("<Frame"):
                m = _fr.match(line)
                n, x, y, bp, bs = m.groups()
                store[int(n)] = (float(x), float(y), int(bp) if bp else 0, int(bs) if bs else 0)
    out = {}
    halves = []
    for half in ("firstHalf", "secondHalf"):
        b = ball[(half, "BALL", "BALL")]
        fr = np.array(sorted(b))
        halves.append((half, fr))
    frames = np.concatenate([fr for _, fr in halves])
    half_id = np.concatenate([np.full(len(fr), i) for i, (_, fr) in enumerate(halves)])
    T = len(frames)
    slots = sorted(tracks, key=lambda k: (k[1] != home, k[1], k[0], k[2]))
    # one slot per (team, person) across halves
    pids = sorted({(k[1], k[2]) for k in slots}, key=lambda k: (k[0] != home, k[0], k[1]))
    pidx = {p: i for i, p in enumerate(pids)}
    xy = np.full((T, len(pids), 2), np.nan, np.float32)
    off = 0
    for half, fr in halves:
        pos = {n: i + off for i, n in enumerate(fr)}
        for (h, team, pid), d in tracks.items():
            if h != half:
                continue
            j = pidx[(team, pid)]
            for n, (x, y, _, _) in d.items():
                if n in pos:
                    xy[pos[n], j] = (x, y)
        off += len(fr)
    b = np.array([[ball[(h, "BALL", "BALL")][n][0], ball[(h, "BALL", "BALL")][n][1],
                   ball[(h, "BALL", "BALL")][n][2], ball[(h, "BALL", "BALL")][n][3]]
                  for h, fr in halves for n in fr], np.float32)
    team = np.array([0 if t == home else 1 for t, _ in pids], np.int8)
    out.update(xy=xy, team=team, ball=b[:, :2], poss=b[:, 2].astype(np.int8),
               alive=b[:, 3].astype(np.int8), half=half_id.astype(np.int8), frame=frames)
    return out


def _orient(d):
    """Flip so home attacks +x in each half; direction from home centroid around the first live frames."""
    xy, team, half = d["xy"].copy(), d["team"], d["half"]
    ball = d["ball"].copy()
    for h in (0, 1):
        idx = np.where((half == h) & (d["alive"] == 1))[0][:250]
        hx = np.nanmean(xy[idx][:, team == 0, 0])
        if hx > 0:  # home sits on +x side at kickoff -> attacks -x -> flip
            m = half == h
            xy[m] *= -1
            ball[m] *= -1
    d["xy"], d["ball"] = xy, ball
    return d


def load(mid, step=STEP):
    CACHE.mkdir(parents=True, exist_ok=True)
    p = CACHE / f"{mid}_{step}hz.npz"
    if p.exists():
        return dict(np.load(p))
    d = _orient(parse_match(mid))
    # label: D=2 if dead else H=0 (poss 1) / A=1 (poss 2); full-rate copy kept for scoring at 25 Hz
    d["y"] = np.where(d["alive"] == 0, 2, d["poss"] - 1).astype(np.int8)
    full = {"y25": d["y"], "half25": d["half"]}
    sel = slice(0, None, step)
    d = {k: (v[sel] if k not in ("team",) else v) for k, v in d.items()}
    d.update(full)
    np.savez_compressed(p, **d)
    return d
