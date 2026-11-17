"""BAR-ATH loader from the raw ATD `.txt` + `.xml` (never the team-made CSVs).

RED TIER: this module's functions return match data. Only call them from scripts wrapped in `guard.quiet`
that print PASS/FAIL, counts and distances. Nothing here prints values.
"""
from __future__ import annotations

import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

from . import canonical
from .paths import ROOT

sys.path.insert(0, ROOT)
import games  # noqa: E402

GAME = games.GAME
GRADED = [(11185, 90652), (110808, 190198)]  # half-open video-frame blocks, published in DATA.md / FAQ.md
N_FRAMES = 199319
GRADED_BALL_SEEN, GRADED_FRAMES = 94214, 158857


def team_of_slots(meta_path):
    root = ET.parse(meta_path).getroot()
    team_ids = [t.get("id") for t in root.iter("Team")]
    assert len(team_ids) == 2
    return np.array([0 if p.get("teamId") == team_ids[0] else 1 for p in root.iter("Player")], np.int8)


def read_raw(raw_path, n_slots, pitch):
    """Stream the ATD text into xy [T,N,2] metres (centred) and ball [T,2]; NaN where missing."""
    length, width = pitch
    xy = np.full((N_FRAMES, n_slots, 2), np.nan, np.float32)
    ball = np.full((N_FRAMES, 2), np.nan, np.float32)
    ids = np.zeros(N_FRAMES, np.int64)
    n = 0
    with open(raw_path) as fh:
        for line in fh:
            parts = line.split(":")
            if len(parts) != 3:
                continue
            i = int(parts[0]) - 1
            ids[n] = i
            n += 1
            for s, g in enumerate(parts[1].split(";")):
                if s < n_slots and g and g[0] != "N":
                    a = g.split(",")
                    xy[i, s] = ((float(a[0]) - 0.5) * length, (float(a[1]) - 0.5) * width)
            b = parts[2]
            if b and b[0] != "N":
                a = b.split(",")
                ball[i] = ((float(a[0]) - 0.5) * length, (float(a[1]) - 0.5) * width)
    return xy, ball, n


def halves_from_counts(count):
    """Half label per frame from the longest low-occupancy run (halftime), found algorithmically."""
    low = count < 5
    best, start, cur = (0, 0), None, None
    for i, v in enumerate(low):
        if v and start is None:
            start = i
        if (not v or i == len(low) - 1) and start is not None:
            end = i if not v else i + 1
            if end - start > best[1] - best[0] and start > 1000 and end < len(low) - 1000:
                best = (start, end)
            start = None
    mid = (best[0] + best[1]) // 2
    return np.where(np.arange(len(count)) < mid, 1, 2).astype(np.int8), best


def load_canon(data_dir=None, source="tactical") -> tuple[canonical.Canon, dict]:
    """Returns (Canon, checks) where checks are booleans/ints safe to print."""
    meta, raw = games.require(["atd_meta", "atd_raw"], GAME, data_dir, source)
    pitch = games.GAMES[GAME]["pitch"]
    team = team_of_slots(meta)
    xy, ball, n = read_raw(raw, len(team), pitch)
    count = np.isfinite(xy[..., 0]).sum(1)
    half, gap = halves_from_counts(count)
    c = canonical.build(GAME, "atd", xy, team, ball, half, np.arange(N_FRAMES))
    g = np.zeros(N_FRAMES, bool)
    for a, b in GRADED:
        g[a:b] = True
    seen = int((c.ball[:, 4] > 0)[g].sum())
    checks = {
        "rows": n, "rows_ok": n == N_FRAMES, "graded_frames": int(g.sum()), "graded_ok": int(g.sum()) == GRADED_FRAMES,
        "ball_seen_graded": seen, "ball_ok": seen == GRADED_BALL_SEEN,
        "tracked_mean": float(count[g].mean()), "slots": len(team),
    }
    return c, checks


def graded_mask():
    g = np.zeros(N_FRAMES, bool)
    for a, b in GRADED:
        g[a:b] = True
    return g
