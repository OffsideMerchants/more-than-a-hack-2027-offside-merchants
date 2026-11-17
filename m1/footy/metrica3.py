"""Metrica Sports sample game 3: FIFA-EPTS tracking (same format as the BAR-ATH ATD) + JSON events.

Tracking is read with kloppy and converted to 29 slots like real ATD (footy.atd). The JSON events
are converted to the column layout of the CSV events of games 1-2, so footy.labels.period_labels
derives the per-frame labels with the same rules (HOME / AWAY / DEAD / IGNORE).
Files are expected in data/metrica/Sample_Game_3_{metadata.xml, tracking.txt, events.json}.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .atd import tracks_to_positions
from .kloppy_data import CACHE
from .labels import period_labels

ROOT = Path("data/metrica")


def _subtype(e):
    st = e.get("subtypes")
    if not st:
        return np.nan
    st = st if isinstance(st, list) else [st]
    return "-".join(s["name"] for s in st)


def events_frame(home_team_id):
    """JSON events -> DataFrame with the CSV columns used by footy.labels."""
    data = json.load(open(ROOT / "Sample_Game_3_events.json"))["data"]
    rows = [{"Team": "Home" if e["team"]["id"] == home_team_id else "Away", "Type": e["type"]["name"],
             "Subtype": _subtype(e), "Period": e["period"], "Start Frame": e["start"]["frame"],
             "End Frame": e["end"]["frame"] if e.get("end") and e["end"].get("frame") is not None else e["start"]["frame"]}
            for e in data]
    return pd.DataFrame(rows)


def load_metrica_game3(use_cache=True):
    """List of (positions [T, 29, 2] metres at 25 Hz, labels [T]) per period."""
    cache = CACHE / "metrica_game3.npz"
    if use_cache and cache.exists():
        z = np.load(cache)
        return [(z[f"pos{i}"], z[f"lab{i}"]) for i in range(int(z["n"]))]
    from kloppy import metrica
    ds = metrica.load_tracking_epts(meta_data=str(ROOT / "Sample_Game_3_metadata.xml"),
                                    raw_data=str(ROOT / "Sample_Game_3_tracking.txt"))
    df = ds.to_df(engine="pandas")
    L, W = ds.metadata.pitch_dimensions.pitch_length or 105.0, ds.metadata.pitch_dimensions.pitch_width or 68.0
    to_m = lambda x, y: np.stack([(x - 0.5) * L, -(y - 0.5) * W], -1).astype(np.float32)
    teams = list(ds.metadata.teams)
    home = next(t for t in teams if t.ground.name == "HOME")
    away = next(t for t in teams if t.ground.name == "AWAY")
    tracks = {}
    for team in (home, away):
        ids = [p.player_id for p in team.players if f"{p.player_id}_x" in df.columns]
        tracks[team.team_id] = np.stack([to_m(df[f"{i}_x"].to_numpy(float), df[f"{i}_y"].to_numpy(float)) for i in ids], 1)
    ball = to_m(df["ball_x"].to_numpy(float), df["ball_y"].to_numpy(float))
    events = events_frame(home.team_id)
    out = []
    for period in sorted(df["period_id"].unique()):
        sel = (df["period_id"] == period).to_numpy()
        pos, _ = tracks_to_positions(tracks[home.team_id][sel], tracks[away.team_id][sel], ball[sel])
        frames = df["frame_id"].to_numpy()[sel]
        lab = period_labels(events[events.Period == period], int(frames[0]), len(frames))
        out.append((pos, lab))
    np.savez_compressed(cache, n=len(out), **{f"pos{i}": p for i, (p, _) in enumerate(out)},
                        **{f"lab{i}": l for i, (_, l) in enumerate(out)})
    return out
