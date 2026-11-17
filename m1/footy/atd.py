"""Automatic tracking data (ATD) -> 29-slot positions.

ATD tracks are anonymous and noisy: they appear and vanish, a track can continue on a different
person after an occlusion, and the tracker sometimes duplicates someone. Our models do not use
identities, only continuity over time, so the conversion is:

  1. split every track into segments wherever it has a gap or a physically impossible jump;
  2. assign segments to MAX_PER_TEAM slots per team (interval scheduling). A new segment prefers
     the free slot whose last position is closest to its first position, which re-links a person
     who was briefly lost. Segments that find no free slot (more than 14 concurrent objects in a
     team, i.e. duplicates) are dropped, and the drop count is reported.

The same code runs on the real ATD (load_atd_epts) and on simulated ATD (footy.atd_sim).
"""
import numpy as np

from .data import BALL, MAX_PER_TEAM, NUM_AGENTS


def segment_tracks(tracks, fps=25, max_speed=12.0, jump_margin=1.0, max_gap_s=1.0):
    """tracks: [T, n, 2] metres (NaN = missing) -> list of (start, stop, xy[stop-start, 2]).

    A track is cut where it is missing for more than `max_gap_s` or where consecutive valid
    positions are further apart than max_speed * dt + jump_margin. Short gaps inside a segment
    stay NaN.
    """
    out = []
    T, n, _ = tracks.shape
    max_gap = int(max_gap_s * fps)
    for j in range(n):
        xy = tracks[:, j]
        valid = np.flatnonzero(~np.isnan(xy[:, 0]))
        if not len(valid):
            continue
        gaps = np.diff(valid)
        dist = np.linalg.norm(np.diff(xy[valid], axis=0), axis=1)
        cut = (gaps > max_gap + 1) | (dist > max_speed * gaps / fps + jump_margin)
        bounds = np.r_[0, np.flatnonzero(cut) + 1, len(valid)]
        for a, b in zip(bounds[:-1], bounds[1:]):
            s, e = valid[a], valid[b - 1] + 1
            out.append((int(s), int(e), xy[s:e].copy()))
    return out


def assign_segments(segments, T, n_slots=MAX_PER_TEAM, fps=25, max_speed=12.0, relink_s=3.0):
    """Place segments into n_slots slots. Returns (positions [T, n_slots, 2], n_dropped, n_relinked)."""
    pos = np.full((T, n_slots, 2), np.nan, dtype=np.float32)
    free_from = np.zeros(n_slots, dtype=np.int64)          # first frame the slot is free again
    last_xy = np.full((n_slots, 2), np.nan)
    dropped = relinked = 0
    for s, e, xy in sorted(segments, key=lambda seg: (seg[0], -(seg[1] - seg[0]))):
        free = np.flatnonzero(free_from <= s)
        if not len(free):
            dropped += 1
            continue
        first = xy[np.flatnonzero(~np.isnan(xy[:, 0]))[0]]
        gap = (s - free_from[free]) / fps
        reach = max_speed * gap + 2.0
        d = np.linalg.norm(last_xy[free] - first, axis=1)
        ok = ~np.isnan(d) & (d <= reach) & (gap <= relink_s)
        if ok.any():                                         # likely the same person, re-acquired
            k = free[ok][np.argmin(d[ok])]
            relinked += 1
        else:                                                # longest-free slot
            k = free[np.argmin(free_from[free])]
        pos[s:e, k] = xy
        free_from[k] = e
        valid = np.flatnonzero(~np.isnan(xy[:, 0]))
        last_xy[k] = xy[valid[-1]]
    return pos, dropped, relinked


def tracks_to_positions(home_tracks, away_tracks, ball, fps=25, **kw):
    """Raw team tracks [T, n_h, 2] / [T, n_a, 2] and ball [T, 2] (metres) -> ([T, 29, 2], stats)."""
    T = len(ball)
    out = np.full((T, NUM_AGENTS, 2), np.nan, dtype=np.float32)
    stats = {}
    for name, tracks, off in (("home", home_tracks, 0), ("away", away_tracks, MAX_PER_TEAM)):
        segs = segment_tracks(tracks, fps=fps, **kw)
        slots, dropped, relinked = assign_segments(segs, T, fps=fps)
        out[:, off:off + MAX_PER_TEAM] = slots
        stats[name] = {"tracks": int((~np.isnan(tracks[..., 0])).any(0).sum()), "segments": len(segs),
                       "dropped": dropped, "relinked": relinked}
    out[:, BALL] = ball
    return out, stats
