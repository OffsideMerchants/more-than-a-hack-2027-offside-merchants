"""Make clean tracking data (e.g. IDSSE) look like automatic tracking data (ATD).

Targets come from the challenge's DATA.md (no ground truth involved):
  - ball missing on ~41% of graded frames, "disproportionately on the frames that matter":
    gaps are more likely when players crowd the ball, in the penalty areas, when the ball is
    fast (in the air), and when it is outside or at the edge of the pitch;
  - ~20.5 of 22 players tracked on average, with occlusion gaps more likely in crowds;
  - tracks are anonymous and fragmented: after an occlusion a person often continues in a new
    track, and two close teammates can swap tracks;
  - the tracker occasionally duplicates someone (more than 22 objects).
Base rates are calibrated by bisection so the overall missing fractions hit the targets; the
situation weights and gap-length distributions are assumptions, to be checked against the
published baseline scores.
"""
import numpy as np

from .data import BALL, MAX_PER_TEAM, PITCH_LENGTH, PITCH_WIDTH

FPS = 25


def _gaps(hazard, rng, median_s, sigma):
    """Sequential on/off process: a gap starts with probability hazard[t] when not in a gap,
    and lasts a log-normal number of frames. Returns a bool mask (True = missing)."""
    T = len(hazard)
    starts = rng.random(T) < hazard
    lengths = np.maximum(1, rng.lognormal(np.log(median_s * FPS), sigma, T).astype(int))
    miss = np.zeros(T, bool)
    t = 0
    idx = np.flatnonzero(starts)
    for s in idx:
        if s < t:
            continue
        t = min(T, s + lengths[s])
        miss[s:t] = True
    return miss


def _calibrate(weight, target, rng_seed, median_s, sigma, forced=None):
    """Find a base rate so that _gaps(base * weight), together with `forced`, misses `target` of the frames."""
    forced = np.zeros(len(weight), bool) if forced is None else forced
    lo, hi = 1e-6, 1.0
    for _ in range(25):
        mid = np.sqrt(lo * hi)
        frac = (forced | _gaps(np.clip(mid * weight, 0, 1), np.random.default_rng(rng_seed), median_s, sigma)).mean()
        lo, hi = (mid, hi) if frac < target else (lo, mid)
    return np.sqrt(lo * hi)


def simulate_atd(pos, seed=0, ball_missing=0.41, players_visible=20.5 / 22, ball_gap_median_s=1.0,
                 ball_gap_sigma=1.0, player_gap_median_s=0.8, player_gap_sigma=0.8, p_new_track=0.7,
                 p_swap=0.3, duplicate_rate=1 / 3000, jitter_m=0.1, dead=None, dead_weight=0.0,
                 dead_whole_p=0.7, dead_min_s=1.0):
    """pos: clean [T, 29, 2] metres at 25 Hz -> (home_tracks [T, n_h, 2], away_tracks [T, n_a, 2], ball [T, 2]).

    The output has ATD-like structure and goes through footy.atd.tracks_to_positions like real ATD.
    `dead` (bool [T], from the source labels) with `dead_weight` makes ball gaps more likely during
    stoppages (ball out of the tactical camera, behind the goal, picked up): the overall missing
    rate stays at `ball_missing`, only its split between live and dead play changes.
    With `dead_whole_p`, each stoppage of at least `dead_min_s` loses the ball for its whole
    duration with that probability (out of camera until the restart).
    """
    rng = np.random.default_rng(seed)
    T = len(pos)
    players, ball = pos[:, :BALL].astype(np.float64), pos[:, BALL].astype(np.float64)
    ok_p = ~np.isnan(players[..., 0])

    # --- ball gaps, situation dependent
    d_ball = np.linalg.norm(players - ball[:, None], axis=-1)
    crowd = np.nansum(d_ball < 3.0, axis=1)
    bx, by = np.abs(ball[:, 0]), np.abs(ball[:, 1])
    in_box = (bx > PITCH_LENGTH / 2 - 16.5) & (by < 20.15)
    edge = (bx > PITCH_LENGTH / 2 - 1) | (by > PITCH_WIDTH / 2 - 1)
    speed = np.r_[0, np.linalg.norm(np.diff(ball, axis=0), axis=1) * FPS]
    w_ball = 1 + 0.5 * crowd + 1.0 * in_box + 1.5 * (np.nan_to_num(speed) > 15) + 3.0 * edge
    if dead is not None:
        w_ball = w_ball + dead_weight * np.asarray(dead, float)
    w_ball = np.nan_to_num(w_ball, nan=1.0)
    forced = np.zeros(T, bool)
    if dead is not None and dead_whole_p > 0:
        dd = np.asarray(dead, bool)
        edges = np.flatnonzero(np.diff(np.r_[0, dd.astype(int), 0]))
        for a, b in zip(edges[::2], edges[1::2]):
            if b - a >= dead_min_s * FPS and rng.random() < dead_whole_p:
                forced[a:b] = True
    base = _calibrate(w_ball, ball_missing, seed, ball_gap_median_s, ball_gap_sigma, forced)
    ball_miss = forced | _gaps(np.clip(base * w_ball, 0, 1), rng, ball_gap_median_s, ball_gap_sigma)
    ball_out = np.where(ball_miss[:, None], np.nan, ball + rng.normal(0, jitter_m, ball.shape))

    # --- player occlusions, more likely in crowds
    d_pp = np.linalg.norm(players[:, :, None] - players[:, None], axis=-1)
    near = np.nansum(d_pp < 2.0, axis=2) - 1                                  # [T, 28]
    w_pl = 1 + 0.7 * np.clip(near, 0, None)
    target = 1 - players_visible
    base_p = _calibrate(w_pl[ok_p[:, 0]][:, 0] if ok_p[:, 0].any() else w_pl[:, 0], target, seed + 1,
                        player_gap_median_s, player_gap_sigma)
    miss = np.zeros_like(ok_p)
    for j in range(BALL):
        miss[:, j] = _gaps(np.clip(base_p * w_pl[:, j], 0, 1), rng, player_gap_median_s, player_gap_sigma)
    noisy = np.where((ok_p & ~miss)[..., None], players + rng.normal(0, jitter_m, players.shape), np.nan)

    # --- swaps: when two teammates are close, with probability p_swap they exchange their futures
    for off in (0, MAX_PER_TEAM):
        team = noisy[:, off:off + MAX_PER_TEAM]
        close = np.argwhere(np.triu(d_pp[:, off:off + MAX_PER_TEAM, off:off + MAX_PER_TEAM] < 1.0, 1))
        last = {}
        for t, a, b in close:
            if t - last.get((a, b), -10 ** 9) < FPS:  # one decision per encounter
                last[(a, b)] = t
                continue
            last[(a, b)] = t
            if rng.random() < p_swap:
                team[t:, [a, b]] = team[t:, [b, a]]

    # --- fragmentation: after each gap, continue in a new track with probability p_new_track
    def fragment(team):
        tracks = []
        for j in range(team.shape[1]):
            xy = team[:, j]
            valid = ~np.isnan(xy[:, 0])
            if not valid.any():
                continue
            starts = np.flatnonzero(valid & ~np.r_[False, valid[:-1]])
            cur = np.full((T, 2), np.nan)
            for k, s in enumerate(starts):
                e = s + np.argmax(~valid[s:]) if (~valid[s:]).any() else T
                if k and rng.random() < p_new_track:
                    tracks.append(cur)
                    cur = np.full((T, 2), np.nan)
                cur[s:e] = xy[s:e]
            tracks.append(cur)
        return tracks

    out = []
    for off in (0, MAX_PER_TEAM):
        tracks = fragment(noisy[:, off:off + MAX_PER_TEAM])
        # --- duplicates: copy a crowded player with an offset for 0.5-2 s
        for s in np.flatnonzero(rng.random(T) < duplicate_rate):
            j = rng.integers(MAX_PER_TEAM)
            src = noisy[s:, off + j]
            if np.isnan(src[0, 0]):
                continue
            dur = min(len(src), int(rng.uniform(0.5, 2.0) * FPS))
            dup = np.full((T, 2), np.nan)
            dup[s:s + dur] = src[:dur] + rng.normal(0, 0.6, 2)
            tracks.append(dup)
        out.append(np.stack(tracks, 1).astype(np.float32) if tracks else np.full((T, 0, 2), np.nan, np.float32))
    return out[0], out[1], ball_out.astype(np.float32)
