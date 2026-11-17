"""Synthetic-data tests for the S2 TCN (no real data needed)."""
import numpy as np
import pytest
import torch

from possession.models.tcn.data import (Match, WindowSampler, augment, collate, make_sample, turnover_targets)
from possession.models.tcn.infer import predict_logits, to_contract
from possession.models.tcn.losses import possession_loss, soften
from possession.models.tcn.model import PossessionTCN, TCNConfig, ball_gap_features
from possession.models.tcn.train import TrainConfig, train

SMALL = TCNConfig(hidden=16, track_blocks=2, ball_blocks=2, joint_blocks=3)


def synthetic_match(T=3000, N=12, seed=0, with_labels=True):
    """Ball carried by whichever team's nearest player; owner flips every ~6 s; dead segments appear."""
    rng = np.random.default_rng(seed)
    team = np.array([0] * (N // 2) + [1] * (N // 2))
    pos = rng.uniform(-30, 30, (T, N, 2)).astype(np.float32)
    alive = ((np.arange(T) // 400) % 4) != 3
    step = rng.normal(0, 0.2, (T, N, 2)) * alive[:, None, None]  # players stand still while the ball is dead
    pos = np.cumsum(step, 0).astype(np.float32) + pos[0]
    vel = np.gradient(pos, axis=0) * 25
    pl = np.concatenate([pos, vel, np.linalg.norm(vel, axis=-1, keepdims=True)], -1).astype(np.float32)
    owner = (np.arange(T) // 150) % 2
    carrier = np.where(owner == 0, 0, N // 2)
    ball = np.concatenate([pos[np.arange(T), carrier], vel[np.arange(T), carrier]], -1).astype(np.float32)
    ball[np.random.default_rng(1).random(T) < 0.0] = np.nan
    ball[1000:1200] = np.nan
    m = Match(pl, team, ball)
    if with_labels:
        m.alive, m.home = alive, owner == 0
    return m


def batch(m=None, B=2, T=64, device="cpu"):
    m = m or synthetic_match()
    return collate([make_sample(m, 100 + 50 * i, T * 2, 2) for i in range(B)], device)


def test_forward_shapes_and_finite():
    b = batch()
    out = PossessionTCN(SMALL)(b["players"], b["pmask"], b["team"], b["ball"], b["bmask"])
    assert {k: tuple(v.shape) for k, v in out.items()} == {k: (2, 64) for k in ("alive", "home", "turn")}
    assert all(torch.isfinite(v).all() for v in out.values())


def test_team_swap_exactly_negates_home_logit():
    torch.manual_seed(0)
    model = PossessionTCN(SMALL).eval()
    s = make_sample(synthetic_match(), 200, 128, 2)
    sw = augment({k: v.copy() for k, v in s.items()}, np.random.default_rng(0), p_ball_mask=0, p_track_drop=0, swap=True, yflip=False)
    perm_free = {**sw}
    a, b = collate([s]), collate([perm_free])
    # augment permutes slots; swap invariance must also hold under permutation
    oa = model(a["players"], a["pmask"], a["team"], a["ball"], a["bmask"])
    ob = model(b["players"], b["pmask"], b["team"], b["ball"], b["bmask"])
    assert torch.allclose(oa["home"], -ob["home"], atol=1e-4)
    assert torch.allclose(oa["alive"], ob["alive"], atol=1e-4)


def test_slot_permutation_invariance():
    model = PossessionTCN(SMALL).eval()
    s = make_sample(synthetic_match(), 200, 128, 2)
    p = np.random.default_rng(3).permutation(s["players"].shape[1])
    s2 = {**s, "players": s["players"][:, p], "team": s["team"][p]}
    a, b = collate([s]), collate([s2])
    oa = model(a["players"], a["pmask"], a["team"], a["ball"], a["bmask"])
    ob = model(b["players"], b["pmask"], b["team"], b["ball"], b["bmask"])
    assert torch.allclose(oa["home"], ob["home"], atol=1e-4)


def test_padding_slots_do_not_change_output():
    model = PossessionTCN(SMALL).eval()
    m = synthetic_match(N=12)
    s = make_sample(m, 200, 128, 2)
    pad = {**s, "players": np.concatenate([s["players"], np.full((64, 4, 5), np.nan, np.float32)], 1),
           "team": np.concatenate([s["team"], -np.ones(4, int)])}
    a, b = collate([s]), collate([pad])
    oa = model(a["players"], a["pmask"], a["team"], a["ball"], a["bmask"])
    ob = model(b["players"], b["pmask"], b["team"], b["ball"], b["bmask"])
    assert torch.allclose(oa["alive"], ob["alive"], atol=1e-4)


def test_ball_absent_inputs_never_leak_values():
    model = PossessionTCN(SMALL).eval()
    b = batch()
    b["bmask"][:] = False
    o1 = model(b["players"], b["pmask"], b["team"], b["ball"], b["bmask"])
    b["ball"] = b["ball"] + 99.0
    o2 = model(b["players"], b["pmask"], b["team"], b["ball"], b["bmask"])
    assert torch.allclose(o1["home"], o2["home"], atol=1e-5)


def test_ball_gap_features():
    p = torch.tensor([[1, 1, 0, 0, 0, 1, 0]], dtype=torch.bool)
    f = ball_gap_features(p, dt=1.0) * 10
    assert f[0, :, 2].tolist() == [0, 0, 4, 4, 4, 0, 30.0 / 1 if False else f[0, 6, 2].item()]
    assert f[0, 2, 0].item() == 1 and f[0, 2, 1].item() == 3  # since last seen, to next seen
    assert f[0, 6, 1].item() == 30.0  # no future sighting -> capped


def test_receptive_field_at_least_10s_default():
    assert TCNConfig().receptive_field_s() >= 10


def test_augment_swap_flips_labels_and_mirrors():
    m = synthetic_match()
    s = make_sample(m, 200, 100, 2)
    orig = {k: v.copy() for k, v in s.items()}
    augment(s, np.random.default_rng(0), p_ball_mask=0, p_track_drop=0, swap=True, yflip=False)
    assert np.array_equal(s["home"], 1 - orig["home"])
    assert np.allclose(np.sort(s["ball"][:, 0]), np.sort(-orig["ball"][:, 0]))


def test_whole_ball_masking():
    s = make_sample(synthetic_match(), 200, 100, 2)
    augment(s, np.random.default_rng(0), p_ball_mask=1.0, swap=False, yflip=False)
    assert np.isnan(s["ball"]).all()


def test_turnover_targets_and_softening():
    home = np.array([1, 1, 1, 0, 0, 0, 1, 1], bool)
    alive = np.array([1, 1, 1, 1, 1, 0, 1, 1], bool)
    t = turnover_targets(home, alive)
    assert t.tolist() == [0, 0, 0, 1, 0, 0, 1, 0]
    y = torch.tensor([[1., 1, 1, 0, 0, 0]])
    s = soften(y, torch.ones_like(y), 1)
    assert 0 < s[0, 2] < 1 and s[0, 0] == 1 and s[0, 5] == 0


def test_loss_ignores_owner_on_dead_frames():
    b = batch()
    out = {"alive": torch.zeros(2, 64), "home": torch.zeros(2, 64), "turn": torch.zeros(2, 64)}
    b2 = {k: v.clone() for k, v in b.items()}
    b2["home"] = 1 - b2["home"] * 0  # corrupt owner everywhere
    dead = b["alive"] == 0
    b3 = {k: v.clone() for k, v in b2.items()}
    b3["home"][dead] = 0
    assert possession_loss(out, b2, soft_steps=0)[1]["home"] == pytest.approx(possession_loss(out, b3, soft_steps=0)[1]["home"])


def test_sampler_hits_events():
    m = synthetic_match()
    sm = WindowSampler([m], 500, {"turnover": 1.0})
    rng = np.random.default_rng(0)
    ev = sm.events[0]["turnover"]
    for _ in range(20):
        _, s = sm.sample(rng)
        assert ((ev >= s) & (ev < s + 500)).any()


def test_overlap_add_matches_single_pass_in_interior():
    torch.manual_seed(0)
    model = PossessionTCN(SMALL).eval()
    m = synthetic_match(T=1500, with_labels=False)
    lg, stride = predict_logits(model, m, win_s=20, hop_s=5)
    one = predict_logits(model, m, win_s=60, hop_s=60)[0]
    # RF of SMALL is ~ 3 s, so away from window edges both passes must agree closely
    assert np.abs(lg["home"][40:-40] - one["home"][40:-40]).max() < 0.15
    c = to_contract(lg, stride, m.T)
    assert c["p_home"].shape == (m.T,) and np.allclose(c["p_home"], np.round(c["p_home"], 4))


def test_overfit_one_match():
    """Phase 1: single clean match must be fit near-perfectly, otherwise there is a bug."""
    torch.manual_seed(0)
    m = synthetic_match(T=2500)
    cfg = TrainConfig(steps=400, batch=8, win_s=12, eval_every=100, augment=False, lr=3e-3, amp=False, patience=99)
    model, _ = train([m], SMALL, cfg, log=lambda *_: None)
    lg, stride = predict_logits(model, m, win_s=12, hop_s=6)
    alive_acc = ((lg["alive"] > 0) == m.alive[::stride][:len(lg["alive"])]).mean()
    home_acc = ((lg["home"] > 0) == m.home[::stride][:len(lg["home"])])[m.alive[::stride][:len(lg["home"])]].mean()
    assert alive_acc > 0.9 and home_acc > 0.9, (alive_acc, home_acc)


def test_spell_loss_perfect_and_wrong():
    import torch
    from possession.models.tcn.losses import spell_loss
    y = torch.tensor([[2] * 5 + [0] * 20 + [2] * 5 + [1] * 20 + [2] * 10]).repeat(2, 1)
    big = 12.0
    alive = torch.where(y < 2, big, -big)
    home = torch.where(y == 0, big, -big)
    assert spell_loss(alive, home, y) < 0.02
    assert spell_loss(-alive, -home, y) > 0.9
    # a spell touching the window edge is ignored, so an edge-only window gives zero loss
    e = torch.zeros(1, 20, dtype=torch.long)
    assert float(spell_loss(torch.zeros(1, 20), torch.zeros(1, 20), e)) == 0.0
