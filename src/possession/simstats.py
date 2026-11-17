"""Summary statistics for simulator calibration and window features for the domain classifier.

Everything here is label-free and works on any Canon. When fed BAR-ATH the results are RED/AMBER: callers must only
print distances, AUCs and PASS/FAIL.
"""
from __future__ import annotations

import numpy as np

FPS = 25
GAP_BINS = np.array([1, 5, 13, 26, 51, 101, 251, 501, 1001, 10**9])  # frames: ~0.04, .2, .5, 1, 2, 4, 10, 20, 40 s
FRAG_BINS = np.array([1, 26, 126, 376, 751, 1501, 3751, 7501, 10**9])
SPEED_Q = [50, 90, 99, 99.9]
BALL_SPEED_Q = [50, 90, 99, 99.9]


def _runs(mask):
    d = np.diff(np.r_[0, mask.astype(np.int8), 0])
    s, e = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    return s, e


def _hist_frac(lengths, bins):
    if len(lengths) == 0:
        return np.zeros(len(bins) - 1)
    h = np.histogram(lengths, bins)[0] * 1.0
    w = np.histogram(lengths, bins, weights=lengths)[0]
    return w / max(w.sum(), 1)


def summary(c, mask=None, sample_every=25):
    """Dict of label-free statistics over frames where `mask` is True (default all). Never crosses mask holes."""
    T = c.T
    mask = np.ones(T, bool) if mask is None else mask
    pl = np.asarray(c.players)
    ball = np.asarray(c.ball)
    out = {}
    # ball gaps: runs of absent ball inside each masked block
    bpres = ball[:, 4] > 0
    gaps = []
    for a, b in zip(*_runs(mask)):
        s, e = _runs(~bpres[a:b])
        gaps += list(e - s)
    gaps = np.array(gaps)
    nb = mask.sum()
    out["ball_missing_share"] = np.array([1 - bpres[mask].mean()])
    out["ball_gap_hist"] = _hist_frac(gaps, GAP_BINS)
    out["ball_gap_count_per_min"] = np.array([len(gaps) / max(nb / FPS / 60, 1e-9)])
    both = bpres[1:] & bpres[:-1] & mask[1:] & mask[:-1]
    disp = np.linalg.norm(ball[1:, :2] - ball[:-1, :2], axis=1)[both] * FPS
    out["ball_speed_q"] = np.percentile(disp, BALL_SPEED_Q) if len(disp) else np.zeros(4)
    out["ball_jump_rate"] = np.array([(disp > 60).mean() if len(disp) else 0.0])
    # tracked counts
    cnt = (pl[..., 5] > 0).sum(1)[mask]
    out["tracked_hist"] = np.bincount(np.clip(cnt, 0, 31), minlength=32)[:32] / max(len(cnt), 1)
    out["tracked_mean"] = np.array([cnt.mean()])
    # fragments (present runs per slot)
    fr = []
    starts = 0
    pres = pl[..., 5] > 0
    for a, b in zip(*_runs(mask)):
        for j in range(pres.shape[1]):
            s, e = _runs(pres[a:b, j])
            fr += list(e - s)
            starts += len(s)
    out["frag_hist"] = _hist_frac(np.array(fr), FRAG_BINS)
    out["frag_starts_per_min"] = np.array([starts / max(nb / FPS / 60, 1e-9)])
    spd = pl[..., 4][mask][pres[mask]]
    spd = spd[np.isfinite(spd)]
    out["player_speed_q"] = np.percentile(spd, SPEED_Q) if len(spd) else np.zeros(4)
    # mean nearest-teammate distance on a subsample
    idx = np.flatnonzero(mask)[::sample_every]
    nn = []
    for t in idx:
        for tm in (0, 1):
            P = pl[t, c.team == tm][:, :2][pres[t, c.team == tm]]
            if len(P) > 2:
                d = np.linalg.norm(P[:, None] - P[None], axis=-1) + np.eye(len(P)) * 1e9
                nn.append(d.min(1).mean())
    out["nn_teammate_mean"] = np.array([np.mean(nn) if nn else 0.0])
    return out


WEIGHTS = {"ball_missing_share": 4.0, "ball_gap_hist": 2.0, "ball_gap_count_per_min": 1.0, "ball_speed_q": 1.0, "ball_jump_rate": 1.0,
           "tracked_hist": 2.0, "tracked_mean": 3.0, "frag_hist": 2.0, "frag_starts_per_min": 1.0, "player_speed_q": 1.0,
           "nn_teammate_mean": 1.0}


def distance(a: dict, b: dict):
    """Weighted scale-free distance between two summaries (relative for scalars/quantiles, L1 for histograms)."""
    d = 0.0
    for k, w in WEIGHTS.items():
        x, y = np.asarray(a[k], float), np.asarray(b[k], float)
        if k.endswith("hist"):
            d += w * 0.5 * np.abs(x - y).sum()
        else:
            d += w * np.mean(np.minimum(np.abs(x - y) / (np.abs(y) + 1e-3 + 0.05 * np.abs(y).max()), 3.0))
    return d / sum(WEIGHTS.values())


# ----------------------------------------------------------------------------------------------- window features
GROUPS = {
    "ball_presence": ["ball_present", "ball_gaps", "ball_longest_gap"],
    "ball_kinematics": ["ball_speed_med", "ball_speed_p90", "ball_speed_max", "ball_jumps"],
    "tracked": ["tracked_mean", "tracked_std", "tracked_min", "tracked_max"],
    "fragments": ["frag_starts", "frag_ends"],
    "player_kinematics": ["pl_speed_mean", "pl_speed_p90", "pl_speed_max", "pl_acc_p99"],
    "geometry": ["nn_dist", "spread_x", "spread_y", "ball_nearest_dist", "centroid_gap"],
}
FEATURES = [f for g in GROUPS.values() for f in g]


def window_features(c, win=10 * FPS, stride=5 * FPS, mask=None, max_windows=None, rng=None):
    """[W, len(FEATURES)] and start frames. Windows must lie inside `mask` (default all)."""
    T = c.T
    mask = np.ones(T, bool) if mask is None else mask
    cs = np.cumsum(np.r_[0, ~mask])
    starts = [s for s in range(0, T - win, stride) if cs[s + win] - cs[s] == 0]
    if max_windows and len(starts) > max_windows:
        rng = rng or np.random.default_rng(0)
        starts = sorted(rng.choice(starts, max_windows, replace=False))
    pl, ball = np.asarray(c.players), np.asarray(c.ball)
    rows = []
    for s in starts:
        P, B = pl[s:s + win], ball[s:s + win]
        pres = P[..., 5] > 0
        bp = B[:, 4] > 0
        gs, ge = _runs(~bp)
        cnt = pres.sum(1)
        sp = P[..., 4][pres]
        sp = sp[np.isfinite(sp)]
        both = bp[1:] & bp[:-1]
        bd = np.linalg.norm(B[1:, :2] - B[:-1, :2], axis=1)[both] * FPS
        dd = np.diff(pres.astype(np.int8), axis=0)
        acc = np.abs(np.diff(P[..., 4], axis=0))[pres[1:] & pres[:-1]]
        acc = acc[np.isfinite(acc)] * FPS
        mid = P[win // 2]
        nn = []
        for tm in (0, 1):
            Q = mid[c.team == tm][:, :2][pres[win // 2][c.team == tm]]
            if len(Q) > 2:
                d = np.linalg.norm(Q[:, None] - Q[None], axis=-1) + np.eye(len(Q)) * 1e9
                nn.append(d.min(1).mean())
        allp = mid[:, :2][pres[win // 2]]
        bn = []
        for t in range(0, win, 25):
            if bp[t] and pres[t].any():
                bn.append(np.linalg.norm(P[t, :, :2][pres[t]] - B[t, :2], axis=1).min())
        cen = [np.nanmean(mid[c.team == tm][:, 0][pres[win // 2][c.team == tm]]) if pres[win // 2][c.team == tm].any() else 0 for tm in (0, 1)]
        rows.append([
            bp.mean(), len(gs), (ge - gs).max() if len(gs) else 0,
            np.median(bd) if len(bd) else 0, np.percentile(bd, 90) if len(bd) else 0, bd.max() if len(bd) else 0, (bd > 60).sum() if len(bd) else 0,
            cnt.mean(), cnt.std(), cnt.min(), cnt.max(),
            (dd == 1).sum(), (dd == -1).sum(),
            sp.mean() if len(sp) else 0, np.percentile(sp, 90) if len(sp) else 0, sp.max() if len(sp) else 0, np.percentile(acc, 99) if len(acc) else 0,
            np.mean(nn) if nn else 0, allp[:, 0].std() if len(allp) > 1 else 0, allp[:, 1].std() if len(allp) > 1 else 0,
            np.mean(bn) if bn else -1, cen[1] - cen[0],
        ])
    return np.array(rows, np.float32), np.array(starts)
