import numpy as np

from possession import contract, evalbridge as eb
from possession.decode import decode, fit_durations
from possession.decode.viterbi import argmax_decode


def make_labels(T=20000, seed=0):
    rng = np.random.default_rng(seed)
    code = []
    s = 0
    while len(code) < T:
        s = rng.choice([x for x in (0, 1, 2) if x != s])
        d = int(rng.gamma(4, 30 if s != 2 else 120)) + 25
        code += [s] * d
    c = np.array(code[:T], np.int8)
    return c


def noisy_probs(c, noise=1.6, seed=1):
    rng = np.random.default_rng(seed)
    alive = c != 2
    ph = (c == 0)
    la = np.where(alive, 1, -1) * 1.0 + rng.normal(0, noise, len(c))
    lh = np.where(ph, 1, -1) * 1.0 + rng.normal(0, noise, len(c))
    sig = lambda z: 1 / (1 + np.exp(-z))
    return sig(la), sig(lh)


def test_perfect_prediction_and_shift_behaviour():
    c = make_labels()
    half = np.ones(len(c))
    assert eb.score(c, c, half)["spell_f1_mean"] == 1.0
    s3 = eb.score(c, np.roll(c, 3), half)["spell_f1_mean"]
    s25 = eb.score(c, np.roll(c, 25), half)["spell_f1_mean"]
    assert 0.9 < s3 <= 1.0 and s25 < s3


def test_halves_do_not_merge():
    c = np.r_[np.zeros(300), np.zeros(300)].astype(np.int8)
    half = np.r_[np.ones(300), 2 * np.ones(300)]
    assert eb.score(c, c, half)["n_gt_spells"] == 2


def test_hsmm_beats_argmax_on_noisy_probs():
    c = make_labels()
    half = np.ones(len(c))
    pa, ph = noisy_probs(c)
    dm = fit_durations([(c != 2, c == 0, half)], stride=5)
    base = eb.score(c, argmax_decode(pa, ph), half)["spell_f1_mean"]
    for m in ("viterbi", "hsmm", "hsmm+refine"):
        s = eb.score(c, decode({"p_alive": pa, "p_home": ph}, m, dm, half=half), half)["spell_f1_mean"]
        assert s > base + 0.2, (m, s, base)


def test_turnover_evidence_helps_or_is_neutral():
    c = make_labels(seed=3)
    half = np.ones(len(c))
    pa, ph = noisy_probs(c, noise=2.5, seed=4)
    pt = np.full(len(c), 0.01)
    chg = np.flatnonzero((c[1:] != c[:-1]) & (c[1:] != 2) & (c[:-1] != 2)) + 1
    pt[chg] = 0.9
    dm = fit_durations([(c != 2, c == 0, half)], stride=5)
    a = eb.score(c, decode({"p_alive": pa, "p_home": ph}, "hsmm", dm, half=half), half)["spell_f1_mean"]
    b = eb.score(c, decode({"p_alive": pa, "p_home": ph, "p_turnover": pt}, "hsmm+turn", dm, half=half), half)["spell_f1_mean"]
    assert b >= a - 0.02


def test_contract_validator():
    ok, bad = contract.validate({"frame": np.arange(5), "p_alive": np.full(5, .5), "p_home": np.full(5, .25)}, 5)
    assert ok, bad
    ok, bad = contract.validate({"frame": np.arange(5), "p_alive": np.full(5, 1.5), "p_home": np.full(5, .25)}, 5)
    assert not ok
    ok, bad = contract.validate({"frame": np.arange(5), "p_alive": np.full(5, .5)}, 5)
    assert not ok


def test_decoder_deterministic():
    c = make_labels(5000)
    half = np.ones(len(c))
    pa, ph = noisy_probs(c)
    dm = fit_durations([(c != 2, c == 0, half)])
    a = decode({"p_alive": pa, "p_home": ph}, "hsmm+refine", dm, half=half)
    b = decode({"p_alive": pa, "p_home": ph}, "hsmm+refine", dm, half=half)
    assert np.array_equal(a, b)
