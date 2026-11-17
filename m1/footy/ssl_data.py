"""Unlabelled tracking segments from all sources, for self-supervised pretraining (v2).

Every segment is positions [T, 29, 2] in metres at 5 Hz with NaN for missing values. No labels
are used. Sources:
  idsse_clean   IDSSE / DFL, 7 matches, clean optical tracking
  idsse_sim     the same matches as simulated ATD (default and shifted simulation, footy.pipeline; built and cached
                on first use)
  skillcorner   20 A-League matches, broadcast tracking: ball as detected, extrapolated players
                kept; frames without any player tracking split the segment
  metrica       sample games 1-3
A fixed held-out set (HELD_OUT) never enters pretraining, so reconstruction can be evaluated
on unseen matches.
"""
import numpy as np

from .data import BALL, load_metrica_game
from .kloppy_data import SPORTEC_MATCHES, load_sportec_match
from .pipeline import SPLITS

STEP = 5  # 25 Hz -> 5 Hz
HELD_OUT = {src: {m for part in ("test", "val") for m in SPLITS[part].get(src, [])}       # splits.json: test + val
            for src in ("idsse", "skillcorner", "metrica")}


def _split_untracked(pos, min_len):
    """Split a [T, 29, 2] array at frames without any player position; keep pieces >= min_len."""
    tracked = ~np.isnan(pos[:, :BALL, 0]).all(1)
    edges = np.flatnonzero(np.diff(np.r_[0, tracked.astype(int), 0]))
    return [pos[a:b] for a, b in zip(edges[::2], edges[1::2]) if b - a >= min_len]


def _sim_cache(mid, tag):
    from .pipeline import shifted_match, simulated_match
    return [p for p, *_ in (simulated_match(mid, 0) if tag == "s0" else shifted_match(mid))]


def collect(held_out=False, min_len=80, sources=("idsse_clean", "idsse_sim", "skillcorner", "metrica"), step=STEP):
    """Dict source -> list of segments at 25 / step Hz (default 5 Hz; step=1 keeps 25 Hz for patch pretraining).
    min_len is in frames at that rate. held_out=True returns only the held-out matches."""
    keep = lambda kind, mid: (mid in HELD_OUT[kind]) == held_out
    out = {s: [] for s in sources}
    for mid in SPORTEC_MATCHES:
        if not keep("idsse", mid):
            continue
        if "idsse_clean" in out:
            out["idsse_clean"] += [p[::step] for p, _ in load_sportec_match(mid)]
        if "idsse_sim" in out:
            out["idsse_sim"] += [p[::step] for tag in ("s0", "B") for p in _sim_cache(mid, tag)]
    if "skillcorner" in out:
        from .skillcorner_data import as_observed, load_skillcorner_match, match_ids
        for mid in match_ids():
            if keep("skillcorner", mid):
                for h in load_skillcorner_match(mid):
                    out["skillcorner"] += _split_untracked(as_observed(h)[::step], min_len)
    if "metrica" in out:
        for g in (1, 2):
            if keep("metrica", g):
                out["metrica"] += [p[::step] for _, p in sorted(load_metrica_game("data/metrica", g).items())]
        if keep("metrica", 3):
            from .metrica3 import load_metrica_game3
            out["metrica"] += [p[::step] for p, _ in load_metrica_game3()]
    return {s: [x.astype(np.float32) for x in segs if len(x) >= min_len] for s, segs in out.items()}
