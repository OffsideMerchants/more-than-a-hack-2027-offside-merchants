"""Per-frame possession labels derived from Metrica event data.

Labels: 0 = Home, 1 = Away, 2 = DeadBall, -1 = ignore (loose ball, uncertain timing, unlabelled).
There is no Unknown class. At inference, X comes from model confidence (see footy.possession).
"""
from pathlib import Path

import numpy as np
import pandas as pd

from .data import load_metrica_game

HOME, AWAY, DEAD, IGNORE = 0, 1, 2, -1
ON_BALL = {"PASS", "RECOVERY", "SHOT", "SET PIECE", "BALL LOST", "BALL OUT", "CARRY"}  # CARRY only in game 3 (JSON events)


def _is_stop(ev):
    t, s = ev.Type, ev.Subtype if isinstance(ev.Subtype, str) else ""
    if t in ("BALL OUT", "FAULT RECEIVED"):
        return True
    if t == "SHOT" and ("GOAL" in s or s.endswith("OUT")):
        return True
    if t == "CHALLENGE" and "FAULT" in s and "ADVANTAGE" not in s:
        return True
    return t == "BALL LOST" and ("OFFSIDE" in s or "END HALF" in s)


def _end(ev):
    return max(int(ev["Start Frame"]), int(ev["End Frame"]))


def period_labels(events, first_frame, n_frames, margin=10):
    """Labels for one period at 25 Hz. `events` holds only that period's events. Frame numbers are absolute."""
    lab = np.full(n_frames, IGNORE, dtype=np.int64)
    rel = lambda f: int(np.clip(f - first_frame, 0, n_frames))
    ev = events.sort_values("Start Frame").reset_index(drop=True)

    # Possession: from each on-ball event until the next one.
    on = ev[ev.Type.isin(ON_BALL)].reset_index(drop=True)
    teams, starts, ends = on["Team"].to_numpy(), on["Start Frame"].to_numpy(), on["End Frame"].to_numpy()
    for i in range(len(on) - 1):
        team = HOME if teams[i] == "Home" else AWAY
        stop = starts[i + 1] if teams[i] == teams[i + 1] else max(starts[i], ends[i])
        lab[rel(starts[i]):rel(stop)] = team

    # Dead ball: from a stoppage until the restart set piece, unless play continued in between.
    live_starts = on[on.Type != "SET PIECE"]["Start Frame"].to_numpy()
    restarts = on[on.Type == "SET PIECE"]["Start Frame"].to_numpy()
    for _, e in ev.iterrows():
        if not _is_stop(e):
            continue
        s = _end(e)
        nxt = restarts[restarts >= s]
        if len(nxt) and not ((live_starts > e["Start Frame"]) & (live_starts < nxt[0])).any():
            lab[rel(s):rel(nxt[0])] = DEAD
    if len(restarts):  # before the kick-off of the period
        lab[:rel(restarts[0])] = DEAD

    # Event timestamps are imprecise, so leave the frames around every transition unlabelled.
    change = np.flatnonzero(lab[1:] != lab[:-1]) + 1
    keep = lab.copy()
    for c in change:
        if keep[c] != IGNORE and keep[c - 1] != IGNORE:
            lab[max(0, c - margin):c + margin] = IGNORE
    return lab


def load_labelled_game(data_dir, game, hz=5):
    """Return a list of (positions[T, A, 2] in metres, labels[T]) pairs, one per period, at `hz`."""
    data_dir = Path(data_dir)
    events = pd.read_csv(data_dir / f"Sample_Game_{game}_RawEventsData.csv")
    frames = pd.read_csv(data_dir / f"Sample_Game_{game}_RawTrackingData_Home_Team.csv", skiprows=2,
                         usecols=["Period", "Frame"])
    step = 25 // hz
    out = []
    for period, pos in sorted(load_metrica_game(data_dir, game).items()):
        first = int(frames.Frame[frames.Period == period].iloc[0])
        lab = period_labels(events[events.Period == period], first, len(pos))
        out.append((pos[::step], lab[::step]))
    return out
