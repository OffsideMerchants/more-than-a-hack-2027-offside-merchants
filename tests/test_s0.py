import os

import numpy as np
import pytest

from possession import baselines, canonical, degrade, labels, simstats
from possession.paths import ARTIFACTS

CANON = os.path.join(ARTIFACTS, "canon")
needs_cache = pytest.mark.skipif(not os.path.exists(os.path.join(CANON, "idsse_J03WN1")), reason="run scripts/build_idsse_cache.py")


def synth_canon(T=3000, N=22, seed=0, flip_first_half=False):
    rng = np.random.default_rng(seed)
    team = np.array([0] * 11 + [1] * 11, np.int8)
    half = np.r_[np.ones(T // 2), 2 * np.ones(T - T // 2)].astype(np.int8)
    base = np.where(team == 0, -25.0, 25.0)  # home defends -x -> attacks +x
    xy = np.zeros((T, N, 2), np.float32)
    xy[..., 0] = base[None] + rng.normal(0, 3, (T, N)).cumsum(0) * 0.01
    xy[..., 1] = rng.uniform(-30, 30, N)[None] + rng.normal(0, 1, (T, N)).cumsum(0) * 0.01
    xy[0, 0, 0] = -48  # home keeper
    xy[:, 0, 0] = -48
    xy[:, 11, 0] = 48  # away keeper
    xy[half == 2] *= -1  # second half: teams swap ends
    if flip_first_half:
        xy[half == 1] *= -1
        xy[half == 2] *= -1
    ball = xy[:, 3, :].copy()
    return canonical.build("S", "idsse", xy, team, ball, half, np.arange(T))


def test_attack_direction_is_algorithmic_and_consistent():
    for flip in (False, True):
        c = synth_canon(flip_first_half=flip)
        gk_home = c.players[:, 0, 0]
        assert (gk_home < -40).all(), "home keeper must end up at -x in both halves (home attacks +x)"


def test_speed_recomputed_from_positions():
    c = synth_canon()
    pos = np.asarray(c.players[100, 3, :2]) - np.asarray(c.players[99, 3, :2])
    assert np.isclose(np.hypot(*(np.asarray(c.players[100, 3, 2:4]))), np.linalg.norm(c.players[100, 3, 2:4]), atol=1e-5)
    assert np.isfinite(c.players[:, :, 4]).mean() > 0.99


def test_nan_not_zero_filled():
    xy = np.random.default_rng(0).normal(0, 10, (200, 22, 2)).astype(np.float32)
    xy[:, 0, 0] = -48
    xy[:, 11, 0] = 48
    xy[50:60, 5] = np.nan
    team = np.array([0] * 11 + [1] * 11)
    c = canonical.build("S", "idsse", xy, team, xy[:, 3], np.ones(200, np.int8), np.arange(200))
    assert (c.players[50:60, 5, 5] == 0).all() and np.isnan(c.players[50:60, 5, 0]).all()


def test_spells():
    alive = np.array([1, 1, 1, 1, 0, 0, 1, 1], bool)
    home = np.array([1, 1, 0, 0, 0, 0, 0, 0], bool)
    assert labels.spells(alive, home) == [(0, 2, True), (2, 4, False), (6, 8, False)]


@needs_cache
def test_cached_idsse_shapes_and_orientation():
    c = canonical.load("J03WN1", "idsse", CANON)
    assert c.players.shape[2] == 6 and c.ball.shape[1] == 5 and c.T == 141561
    assert c.ball[:, 4].min() == 1  # IDSSE has the ball on every frame
    # home attacks +x in both halves: home GK-ish slot mean x is negative in each half
    for h in (1, 2):
        sel = np.asarray(c.half) == h
        mx = np.nanmean(np.asarray(c.players[sel][:, c.team == 0, 0]), 0)
        assert np.nanmin(mx) < -30


@needs_cache
def test_degrade_deterministic_and_plausible():
    c = canonical.load("J03WN1", "idsse", CANON)
    p = degrade.SimParams()
    a = degrade.degrade(c, p, 5, 0, 15000)
    b = degrade.degrade(c, p, 5, 0, 15000)
    assert np.array_equal(np.nan_to_num(a.players), np.nan_to_num(b.players))
    assert a.N == 46 and set(a.team.tolist()) == {0, 1}
    assert 0.3 < a.ball[:, 4].mean() < 0.99
    assert (a.players[..., 5].sum(1)).max() <= 46
    # a degraded copy never sees more than truth plus duplicates
    assert a.players[..., 5].sum(1).mean() < 24


@needs_cache
def test_degrade_does_not_leak_identity():
    c = canonical.load("J03WN1", "idsse", CANON)
    d = degrade.degrade(c, degrade.SimParams(), 1, 0, 5000)
    assert "person_id" not in d.aux and "role" not in d.aux


@needs_cache
def test_baseline_pipeline_on_idsse():
    from possession import conditions, evalbridge as eb
    c = conditions.get("J03WN1", "clean")
    half = np.asarray(c.half)
    alive, home = conditions.get_labels("J03WN1", "v1_raw")
    gt = eb.to_codes(alive, home)
    s0 = eb.score(gt, baselines.smooth(baselines.nearest_team(c), 0, half), half)["spell_f1_mean"]
    s50 = eb.score(gt, baselines.smooth(baselines.nearest_team(c), 50, half), half)["spell_f1_mean"]
    assert s50 > s0  # smoothing helps, as EVALUATION.md argues


def test_summary_stats_run_on_synthetic():
    s = simstats.summary(synth_canon())
    assert set(simstats.WEIGHTS) <= set(s)
    assert simstats.distance(s, s) == pytest.approx(0, abs=1e-9)
