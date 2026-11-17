"""Load open tracking data via kloppy into the 29-slot format of footy.data.

Sources:
  sportec / IDSSE  7 Bundesliga matches (DFL, 25 Hz), via kloppy.sportec.load_open_*. Every frame
                   carries ball_state (alive/dead) and ball_owning_team, giving per-frame
                   reference labels: Home / Away / DeadBall.
  metrica          via kloppy.metrica.load_open_data (games 1-3). Game 1 and 2 labels still come
                   from footy.labels (derived from events).

Slots: each team has MAX_PER_TEAM slots. A substitute takes over the slot of a player who has
left the pitch (interval scheduling on first and last appearance), so 14 slots are enough even
with 5 substitutions. Positions are in metres with the pitch centred at (0, 0); NaN = missing.
Converted matches are cached as .npz files.
"""
from pathlib import Path

import numpy as np

from .data import BALL, MAX_PER_TEAM, NUM_AGENTS

SPORTEC_MATCHES = ("J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY", "J03WQQ", "J03WR9")
CACHE = Path("data/cache")


def _configure_kloppy(cache_dir="data/kloppy_cache"):
    from kloppy.config import set_config
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    set_config("cache", cache_dir)


def _assign_slots(present):
    """present: [T, n] bool for one team's players -> slot index per player (-1 if never present).

    Greedy interval scheduling: players sorted by first appearance get the first slot whose
    previous occupant's last appearance lies before their first one.
    """
    T, n = present.shape
    first = np.where(present.any(0), present.argmax(0), -1)
    last = np.where(present.any(0), T - 1 - present[::-1].argmax(0), -1)
    slot_free_from = []          # per slot: first frame after its occupant's last appearance
    slots = np.full(n, -1)
    for j in np.argsort(np.where(first < 0, T, first)):
        if first[j] < 0:
            continue
        for s, free in enumerate(slot_free_from):
            if free <= first[j]:
                slots[j], slot_free_from[s] = s, last[j] + 1
                break
        else:
            slots[j] = len(slot_free_from)
            slot_free_from.append(last[j] + 1)
    if slots.max() >= MAX_PER_TEAM:
        raise ValueError(f"needs {slots.max() + 1} slots, more than MAX_PER_TEAM={MAX_PER_TEAM}")
    return slots


def tracking_to_arrays(ds):
    """kloppy TrackingDataset -> (period ids [T], positions [T, NUM_AGENTS, 2] in metres, labels [T]).

    Labels: 0 Home, 1 Away, 2 DeadBall, -1 unknown (taken from ball_state / ball_owning_team).
    """
    from kloppy.domain import BallState, Ground

    df = ds.to_df(engine="pandas")
    L, W = ds.metadata.pitch_dimensions.pitch_length, ds.metadata.pitch_dimensions.pitch_width
    T = len(df)
    pos = np.full((T, NUM_AGENTS, 2), np.nan, dtype=np.float32)
    for team, offset in ((t, 0 if t.ground == Ground.HOME else MAX_PER_TEAM) for t in ds.metadata.teams):
        cols = [p.player_id for p in team.players if f"{p.player_id}_x" in df.columns]
        if not cols:
            continue
        xy = np.stack([df[[f"{c}_x", f"{c}_y"]].to_numpy(dtype=np.float32) for c in cols], 1)   # [T, n, 2]
        slots = _assign_slots(~np.isnan(xy).any(-1))
        for j, s in enumerate(slots):
            if s >= 0:
                ok = ~np.isnan(xy[:, j]).any(-1)
                pos[ok, offset + s] = xy[ok, j]
    pos[:, BALL] = df[["ball_x", "ball_y"]].to_numpy(dtype=np.float32)
    # kloppy's default coordinates are normalised to [0, 1] with y pointing down
    pos[..., 0] = (pos[..., 0] - 0.5) * L
    pos[..., 1] = -(pos[..., 1] - 0.5) * W

    home_id = next(t.team_id for t in ds.metadata.teams if t.ground == Ground.HOME)
    away_id = next(t.team_id for t in ds.metadata.teams if t.ground == Ground.AWAY)
    lab = np.full(T, -1, dtype=np.int64)
    if "ball_owning_team_id" in df.columns:
        owner = df["ball_owning_team_id"].astype(object).to_numpy()
        lab[owner == home_id] = 0
        lab[owner == away_id] = 1
    if "ball_state" in df.columns:
        state = df["ball_state"].astype(object).to_numpy()
        lab[(state == BallState.DEAD) | (state == "dead")] = 2
    return df["period_id"].to_numpy(), pos, lab


def load_sportec_match(match_id, limit=None, use_cache=True):
    """IDSSE / Sportec open match -> list of (positions [T, 29, 2] in metres at 25 Hz, labels [T]) per period."""
    cache = CACHE / f"sportec_{match_id}{'' if limit is None else f'_limit{limit}'}.npz"
    if use_cache and cache.exists():
        z = np.load(cache)
        return [(z[f"pos{i}"], z[f"lab{i}"]) for i in range(int(z["n"]))]
    _configure_kloppy()
    from kloppy import sportec
    ds = sportec.load_open_tracking_data(match_id, limit=limit)
    period, pos, lab = tracking_to_arrays(ds)
    out = [(pos[period == p], lab[period == p]) for p in np.unique(period)]
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, n=len(out), **{f"pos{i}": p for i, (p, _) in enumerate(out)},
                        **{f"lab{i}": l for i, (_, l) in enumerate(out)})
    return out
