"""Input-dependent linear-chain CRF over {H, A, D}.

emit[b, t, s]       score of state s at step t
trans[b, t, i, j]   score of moving from state i at t-1 to state j at t (t = 0 unused)

Start and end states are marginalised, so a window may begin or end anywhere.
"""
import torch
from torch import Tensor

H, A, D = 0, 1, 2
NUM_STATES = 3
NEG = -1e9


def _valid(lengths: Tensor | None, B: int, T: int, device) -> Tensor:
    if lengths is None:
        return torch.ones(B, T, dtype=torch.bool, device=device)
    return torch.arange(T, device=device)[None] < lengths[:, None]


def forward_backward(emit: Tensor, trans: Tensor, lengths: Tensor | None = None):
    """Returns log Z [B], unary marginals [B, T, S] and pairwise marginals [B, T, S, S].

    Differentiable in emit and trans. Pairwise [:, t, i, j] = P(z_{t-1}=i, z_t=j); [:, 0] is zero.
    """
    B, T, S = emit.shape
    valid = _valid(lengths, B, T, emit.device)
    alphas = [emit[:, 0]]
    for t in range(1, T):
        nxt = torch.logsumexp(alphas[-1].unsqueeze(2) + trans[:, t], dim=1) + emit[:, t]
        alphas.append(torch.where(valid[:, t, None], nxt, alphas[-1]))
    alpha = torch.stack(alphas, 1)
    log_z = torch.logsumexp(alpha[:, -1], dim=1)

    betas = [torch.zeros(B, S, dtype=emit.dtype, device=emit.device)]
    for t in range(T - 2, -1, -1):
        cur = torch.logsumexp(trans[:, t + 1] + (emit[:, t + 1] + betas[-1]).unsqueeze(1), dim=2)
        betas.append(torch.where(valid[:, t + 1, None], cur, torch.zeros_like(cur)))
    beta = torch.stack(betas[::-1], 1)

    unary = torch.exp(alpha + beta - log_z[:, None, None]) * valid[..., None]
    pair = alpha[:, :-1, :, None] + trans[:, 1:] + (emit[:, 1:] + beta[:, 1:]).unsqueeze(2)
    pair = torch.exp(pair - log_z[:, None, None, None]) * valid[:, 1:, None, None]
    pair = torch.cat([torch.zeros_like(pair[:, :1]), pair], 1)
    return log_z, unary, pair


def gold_score(emit: Tensor, trans: Tensor, y: Tensor, lengths: Tensor | None = None) -> Tensor:
    B, T, S = emit.shape
    valid = _valid(lengths, B, T, emit.device)
    y = y.clamp(min=0)
    e = emit.gather(2, y[..., None]).squeeze(2) * valid
    row = trans[:, 1:].gather(2, y[:, :-1, None, None].expand(-1, -1, 1, S)).squeeze(2)
    tr = row.gather(2, y[:, 1:, None]).squeeze(2) * valid[:, 1:]
    return e.sum(1) + tr.sum(1)


def nll(emit: Tensor, trans: Tensor, y: Tensor, lengths: Tensor | None = None) -> Tensor:
    """Exact per-sequence negative log-likelihood [B]."""
    B, T, S = emit.shape
    valid = _valid(lengths, B, T, emit.device)
    alpha = emit[:, 0]
    for t in range(1, T):
        nxt = torch.logsumexp(alpha.unsqueeze(2) + trans[:, t], dim=1) + emit[:, t]
        alpha = torch.where(valid[:, t, None], nxt, alpha)
    return torch.logsumexp(alpha, 1) - gold_score(emit, trans, y, lengths)


@torch.no_grad()
def viterbi(emit: Tensor, trans: Tensor, lengths: Tensor | None = None) -> Tensor:
    B, T, S = emit.shape
    valid = _valid(lengths, B, T, emit.device)
    ident = torch.arange(S, device=emit.device).expand(B, S)
    score = emit[:, 0]
    back = []
    for t in range(1, T):
        cand = score.unsqueeze(2) + trans[:, t]
        best, arg = cand.max(1)
        v = valid[:, t, None]
        back.append(torch.where(v, arg, ident))
        score = torch.where(v, best + emit[:, t], score)
    path = torch.zeros(B, T, dtype=torch.long, device=emit.device)
    path[:, -1] = score.argmax(1)
    for t in range(T - 2, -1, -1):
        path[:, t] = back[t].gather(1, path[:, t + 1, None]).squeeze(1)
    return path
