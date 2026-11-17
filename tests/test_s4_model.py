import numpy as np
import torch
from s4_synth import batch_from, make_clip

from possession.models.possession_transformer import (LossWeights, PossessionTransformer, S4Config,
                                                      crf, decode_half, possession_loss)
from possession.models.possession_transformer.augment import shuffle_tracks, swap_teams
from possession.models.possession_transformer.inference import combine_scores, predict_scores, window_starts

SMALL = S4Config(d=32, heads=2, social_blocks=1, temporal_blocks=2, dropout=0.0, drop_path=0.0, checkpoint=False)


def _model(seed=0):
    torch.manual_seed(seed)
    return PossessionTransformer(SMALL).eval()


def _batch(gap=(30, 70)):
    return batch_from([make_clip(T=60, seed=1, gap=gap), make_clip(T=60, seed=2)])


def _swap_idx(x):
    return x[..., [1, 0, 2]]


def test_team_swap_equivariance():
    m, b = _model(), _batch()
    with torch.no_grad():
        o1 = m(b["players"], b["team"], b["ball"])
        o2 = m(b["players"], 1 - b["team"], b["ball"])
    assert torch.allclose(o1["emit"], _swap_idx(o2["emit"]), atol=1e-4)
    assert torch.allclose(o1["trans"], o2["trans"][..., [1, 0, 2], :][..., [1, 0, 2]], atol=1e-4)
    for k in ("offset", "alive_logit"):
        assert torch.allclose(o1[k], o2[k], atol=1e-4)


def test_track_order_invariance():
    m, b = _model(), _batch()
    sh = shuffle_tracks(b, torch.Generator().manual_seed(0))
    with torch.no_grad():
        o1 = m(b["players"], b["team"], b["ball"])
        o2 = m(sh["players"], sh["team"], sh["ball"])
    assert torch.allclose(o1["trans"], o2["trans"], atol=1e-4)


def test_swap_augmentation_roundtrip():
    b = _batch()
    twice = swap_teams(swap_teams(b))
    assert torch.allclose(twice["players"], b["players"]) and torch.equal(twice["y"], b["y"])


def test_nan_ball_and_missing_players_are_finite():
    m, b = _model(), _batch()
    b["players"][0, 10:20, 3, :] = float("nan")
    b["players"][0, 10:20, 3, 5] = 0
    out = m(b["players"], b["team"], b["ball"])
    assert torch.isfinite(out["trans"]).all() and torch.isfinite(out["emit"]).all()
    batch = dict(b, lengths=torch.tensor([60, 60]))
    loss = possession_loss(out, batch, LossWeights(spell=0.5, boundary=0.5), SMALL.rate_hz)
    loss["total"].backward()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)


def test_all_terms_run():
    m, b = _model(), _batch()
    T = b["y"].shape[1]
    b["lengths"] = torch.tensor([T, T - 5])
    b["hard_neg"] = torch.zeros(2, T, dtype=torch.bool)
    b["hard_neg"][:, 20:30] = True
    b["phase"] = torch.randint(0, 3, (2, T))
    b["possessor"] = torch.randint(0, 10, (2, T))
    b["ball_xy"] = torch.randn(2, T, 2) * 10
    b["ball_xy_mask"] = torch.rand(2, T) > 0.5
    b["offset_target"] = torch.zeros(2, T)
    b["offset_mask"] = torch.rand(2, T) > 0.8
    m.train()
    out = m(b["players"], b["team"], b["ball"], b["lengths"])
    w = LossWeights(spell=0.5, boundary=0.5, false_turnover=0.5, aux=0.2)
    loss = possession_loss(out, b, w, SMALL.rate_hz)
    assert {"crf", "spell", "boundary", "false_turnover", "aux"} <= set(loss)
    loss["total"].backward()


def test_overfit_one_clip():
    torch.manual_seed(0)
    clip = make_clip(T=90, seed=3, gap=(40, 55))
    b = batch_from([clip])
    m = PossessionTransformer(SMALL)
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3)
    w = LossWeights(spell=0.5)
    for _ in range(250):
        m.train()
        out = m(b["players"], b["team"], b["ball"])
        loss = possession_loss(out, b, w, SMALL.rate_hz)["total"]
        opt.zero_grad()
        loss.backward()
        opt.step()
    m.eval()
    with torch.no_grad():
        out = m(b["players"], b["team"], b["ball"])
    path = crf.viterbi(out["emit"].double(), out["trans"].double())
    assert (path == b["y"]).float().mean() > 0.95


def test_chunk_combine_is_exact_for_context_free_scores():
    rng = np.random.default_rng(0)
    n, win = 50, 20
    emit, trans, off = rng.normal(size=(n, 3)), rng.normal(size=(n, 3, 3)), rng.normal(size=n)
    chunks = [(s, emit[s:s + win], trans[s:s + win], off[s:s + win]) for s in window_starts(n, win, 10)]
    e, t, o = combine_scores(chunks, n, win)
    assert np.allclose(e, emit) and np.allclose(t, trans) and np.allclose(o, off)


def test_single_chunk_equals_single_pass():
    m, b = _model(), _batch(gap=None)
    n = b["ball"].shape[1]
    e, t, o = predict_scores(m, b["players"][0].numpy(), b["team"][0].numpy(), b["ball"][0].numpy(), win=n, stride=n)
    with torch.no_grad():
        ref = m(b["players"][:1], b["team"][:1], b["ball"][:1])
    assert np.allclose(e, ref["emit"][0].numpy(), atol=1e-5) and np.allclose(t, ref["trans"][0].numpy(), atol=1e-5)


def test_decode_half_contract_shapes_and_determinism():
    m, b = _model(), _batch()
    n = b["ball"].shape[1]
    e, t, o = predict_scores(m, b["players"][0].numpy(), b["team"][0].numpy(), b["ball"][0].numpy(), win=30, stride=15)
    f = n * 4
    a = decode_half(e, t, o, SMALL.rate_hz * 0 + 6.25, f)
    c = decode_half(e, t, o, 6.25, f)
    assert a["native_labels"].shape == (f,) and set(a["native_labels"]) <= {"H", "A", "D"}
    for k in ("p_alive", "p_home", "p_turnover", "p_restart"):
        assert a[k].shape == (f,) and ((a[k] >= 0) & (a[k] <= 1)).all()
        assert np.array_equal(a[k], c[k])
    assert np.array_equal(a["native_labels"], c["native_labels"])
