import itertools

import torch

from possession.models.possession_transformer import crf


def _brute(emit, trans):
    T, S = emit.shape
    best, scores = None, []
    for path in itertools.product(range(S), repeat=T):
        s = sum(emit[t, path[t]] for t in range(T))
        s = s + sum(trans[t, path[t - 1], path[t]] for t in range(1, T))
        scores.append(s)
        if best is None or s > best[0]:
            best = (s, path)
    return torch.logsumexp(torch.stack(scores), 0), best[1], dict(zip(itertools.product(range(S), repeat=T), scores))


def test_matches_brute_force():
    g = torch.Generator().manual_seed(0)
    B, T, S = 3, 5, 3
    emit = torch.randn(B, T, S, generator=g, dtype=torch.float64)
    trans = torch.randn(B, T, S, S, generator=g, dtype=torch.float64)
    y = torch.randint(0, S, (B, T), generator=g)
    log_z, unary, pair = crf.forward_backward(emit, trans)
    path = crf.viterbi(emit, trans)
    loss = crf.nll(emit, trans, y)
    for b in range(B):
        lz, best, table = _brute(emit[b], trans[b])
        assert torch.allclose(log_z[b], lz)
        assert tuple(path[b].tolist()) == best
        assert torch.allclose(loss[b], lz - table[tuple(y[b].tolist())])
        for t in range(T):
            for s in range(S):
                m = sum(torch.exp(v - lz) for p, v in table.items() if p[t] == s)
                assert torch.allclose(unary[b, t, s], m)
        for i in range(S):
            for j in range(S):
                m = sum(torch.exp(v - lz) for p, v in table.items() if p[2] == i and p[3] == j)
                assert torch.allclose(pair[b, 3, i, j], m)


def test_variable_lengths_equal_truncation():
    g = torch.Generator().manual_seed(1)
    emit = torch.randn(2, 8, 3, generator=g, dtype=torch.float64)
    trans = torch.randn(2, 8, 3, 3, generator=g, dtype=torch.float64)
    y = torch.randint(0, 3, (2, 8), generator=g)
    lengths = torch.tensor([8, 5])
    full = crf.nll(emit, trans, y, lengths)
    cut = crf.nll(emit[1:, :5], trans[1:, :5], y[1:, :5])
    assert torch.allclose(full[1], cut[0])
    p_full = crf.viterbi(emit, trans, lengths)[1, :5]
    assert torch.equal(p_full, crf.viterbi(emit[1:, :5], trans[1:, :5])[0])
    _, u, _ = crf.forward_backward(emit, trans, lengths)
    _, u_cut, _ = crf.forward_backward(emit[1:, :5], trans[1:, :5])
    assert torch.allclose(u[1, :5], u_cut[0])
    assert (u[1, 5:] == 0).all()


def test_gradient_flows_through_marginals():
    emit = torch.randn(1, 6, 3, dtype=torch.float64, requires_grad=True)
    trans = torch.randn(1, 6, 3, 3, dtype=torch.float64, requires_grad=True)
    _, u, _ = crf.forward_backward(emit, trans)
    u[:, :, 0].sum().backward()
    assert emit.grad.abs().sum() > 0 and trans.grad.abs().sum() > 0
