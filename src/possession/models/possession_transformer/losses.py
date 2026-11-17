"""Possession loss: CRF likelihood, soft spell IoU, boundary focal, false turnover, auxiliaries.

Batch targets (all at the model step rate):
  y [B,T] long in {0 H, 1 A, 2 D}; lengths [B] (optional)
  hard_neg [B,T] bool        steps in duels / deflections / gaps with no possession change
  offset_target, offset_mask [B,T]   sub-step boundary offset (steps) where a boundary sits
  phase [B,T] long, -1 ignore         control phase: controlled / in flight / contested
  possessor [B,T] long, -1 ignore     index of the possessing track
  ball_xy [B,T,2], ball_xy_mask [B,T] ball position (metres) during hidden gaps
A term is skipped when its weight is 0 or its targets are absent, so terms can be added one at a time.
"""
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from . import crf


@dataclass
class LossWeights:
    crf: float = 1.0
    spell: float = 0.0
    boundary: float = 0.0
    false_turnover: float = 0.0
    aux: float = 0.0
    sigma_s: float = 0.3
    focal_gamma: float = 2.0
    spell_margin: float = 1.0  # outside-nearby region, as a multiple of the spell length


def event_targets(y: Tensor) -> Tensor:
    """Hard events [B,T,3] (turnover, stop, restart) at the first step of the new state."""
    prev, cur = y[:, :-1], y[:, 1:]
    team_p, team_c = prev < 2, cur < 2
    ev = torch.stack([team_p & team_c & (prev != cur), team_p & ~team_c, ~team_p & team_c], -1).float()
    return torch.cat([torch.zeros_like(ev[:, :1]), ev], 1)


def _kernel(sigma_steps: float, device) -> Tensor:
    r = max(1, int(round(3 * sigma_steps)))
    x = torch.arange(-r, r + 1, device=device).float()
    return torch.exp(-0.5 * (x / max(sigma_steps, 1e-3)) ** 2)  # peak 1


def smooth_peak1(x: Tensor, sigma_steps: float) -> Tensor:
    """x [B,T,K] -> same shape, Gaussian-smoothed with peak 1 and clamped to [0, 1]."""
    k = _kernel(sigma_steps, x.device)
    B, T, K = x.shape
    h = x.permute(0, 2, 1).reshape(B * K, 1, T)
    out = F.conv1d(h, k[None, None], padding=len(k) // 2)
    return out.reshape(B, K, T).permute(0, 2, 1).clamp(0, 1)


def event_probs(pair: Tensor) -> Tensor:
    """Pairwise marginals [B,T,3,3] -> P(turnover, stop, restart) per step [B,T,3]."""
    H, A, D = crf.H, crf.A, crf.D
    turn = pair[..., H, A] + pair[..., A, H]
    stop = pair[..., H, D] + pair[..., A, D]
    restart = pair[..., D, H] + pair[..., D, A]
    return torch.stack([turn, stop, restart], -1)


def focal_bce(p: Tensor, t: Tensor, gamma: float) -> Tensor:
    p = p.clamp(1e-6, 1 - 1e-6)
    bce = -(t * p.log() + (1 - t) * (1 - p).log())
    return bce * (t - p).abs() ** gamma


def spell_runs(y: Tensor, lengths: Tensor | None, min_len: int = 2):
    """(batch, team, start, end) of every maximal H or A run that does not touch a window edge."""
    runs = []
    yc = y.detach().cpu().tolist()
    for b, row in enumerate(yc):
        n = len(row) if lengths is None else int(lengths[b])
        s = 0
        for t in range(1, n + 1):
            if t == n or row[t] != row[s]:
                if row[s] < 2 and s > 0 and t < n and t - s >= min_len:
                    runs.append((b, row[s], s, t))
                s = t
    return runs


def spell_loss(unary: Tensor, y: Tensor, lengths: Tensor | None, margin: float) -> Tensor:
    """1 - soft IoU per true spell, averaged over spells so each possession counts once."""
    runs = spell_runs(y, lengths)
    if not runs:
        return unary.sum() * 0
    B, T, _ = unary.shape
    cs = torch.cat([torch.zeros_like(unary[:, :1]), unary.cumsum(1)], 1)  # [B,T+1,S]
    losses = []
    for b, k, s, e in runs:
        m = max(1, int(round(margin * (e - s))))
        lo, hi = max(0, s - m), min(T, e + m)
        inside = cs[b, e, k] - cs[b, s, k]
        outside = (cs[b, s, k] - cs[b, lo, k]) + (cs[b, hi, k] - cs[b, e, k])
        losses.append(1 - inside / ((e - s) + outside))
    return torch.stack(losses).mean()


def possession_loss(out: dict, batch: dict, w: LossWeights, rate_hz: float) -> dict:
    """Returns {'total', and one entry per active term}."""
    emit, trans, y, lengths = out["emit"], out["trans"], batch["y"], batch.get("lengths")
    B, T = y.shape
    n = (lengths if lengths is not None else torch.full((B,), T, device=y.device)).float()
    valid = (torch.arange(T, device=y.device)[None] < n[:, None]).float()
    terms = {"crf": (crf.nll(emit, trans, y, lengths) / n).mean()}
    total = w.crf * terms["crf"]

    need_marg = w.spell > 0 or w.boundary > 0 or w.false_turnover > 0
    if need_marg:
        _, unary, pair = crf.forward_backward(emit, trans, lengths)
        ev = event_probs(pair)
    sigma = w.sigma_s * rate_hz

    if w.spell > 0:
        terms["spell"] = spell_loss(unary, y, lengths, w.spell_margin)
        total = total + w.spell * terms["spell"]

    if w.boundary > 0:
        tgt = smooth_peak1(event_targets(y), sigma)
        pred = smooth_peak1(ev, sigma)
        fl = focal_bce(pred, tgt, w.focal_gamma) * valid[..., None]
        n_ev = event_targets(y).sum().clamp(min=1)
        lb = fl.sum() / n_ev
        if "offset_target" in batch:
            m = batch["offset_mask"].float() * valid
            lb = lb + (F.smooth_l1_loss(out["offset"], batch["offset_target"], reduction="none") * m).sum() / m.sum().clamp(min=1)
        terms["boundary"] = lb
        total = total + w.boundary * lb

    if w.false_turnover > 0 and "hard_neg" in batch:
        hn = batch["hard_neg"].float() * valid
        turn = ev[..., 0].clamp(max=1 - 1e-6)
        terms["false_turnover"] = (-(1 - turn).log() * hn).sum() / hn.sum().clamp(min=1)
        total = total + w.false_turnover * terms["false_turnover"]

    if w.aux > 0:
        aux = out["emit"].sum() * 0
        if "phase" in batch:
            aux = aux + F.cross_entropy(out["aux_phase"].flatten(0, 1), batch["phase"].flatten(), ignore_index=-1)
        if "possessor" in batch:
            tgt = batch["possessor"].flatten()
            ok = tgt >= 0
            if ok.any():
                aux = aux + F.cross_entropy(out["aux_possessor"].flatten(0, 1)[ok], tgt[ok])
        if "ball_xy" in batch:
            m = batch["ball_xy_mask"].float() * valid
            err = (out["aux_ball"] - batch["ball_xy"]).abs().sum(-1) / 50.0
            aux = aux + (err * m).sum() / m.sum().clamp(min=1)
        terms["aux"] = aux
        total = total + w.aux * aux

    terms["total"] = total
    return terms
