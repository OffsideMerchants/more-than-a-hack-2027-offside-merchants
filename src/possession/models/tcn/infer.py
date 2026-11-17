"""Whole-match inference: Hann-weighted logit overlap-add, then 25 Hz upsampling and rounding."""
from __future__ import annotations

import numpy as np
import torch

from .data import Match, collate, make_sample


@torch.no_grad()
def predict_logits(model, m: Match, win_s=20.0, hop_s=10.0, batch=16, device="cpu"):
    """Returns (logits {alive, home, turn} at the model rate, stride). Length is ceil(T / stride)."""
    model.eval()
    stride = int(round(25 / model.cfg.rate_hz))
    n = -(-m.T // stride)
    w = min(int(win_s * model.cfg.rate_hz), n)
    hop = max(int(hop_s * model.cfg.rate_hz), 1)
    starts = list(range(0, n - w + 1, hop))
    if starts[-1] + w < n:
        starts.append(n - w)
    acc = {k: np.zeros(n) for k in ("alive", "home", "turn")}
    wsum = np.zeros(n)
    hann = np.hanning(w + 2)[1:-1]
    unlabelled = Match(m.players, m.team, m.ball)
    for i in range(0, len(starts), batch):
        chunk = starts[i:i + batch]
        b = collate([make_sample(unlabelled, s * stride, w * stride, stride) for s in chunk], device)
        out = model(b["players"], b["pmask"], b["team"], b["ball"], b["bmask"])
        for j, s in enumerate(chunk):
            for k in acc:
                acc[k][s:s + w] += hann * out[k][j].float().cpu().numpy()
            wsum[s:s + w] += hann
    return {k: v / np.maximum(wsum, 1e-9) for k, v in acc.items()}, stride


def to_contract(logits, stride, T, decimals=4):
    """Upsample logits to 25 Hz, apply sigmoid and round to 1e-4 so decoding is deterministic."""
    x_lo = np.arange(len(logits["alive"])) * stride
    x_hi = np.arange(T)
    out = {k: np.round(1 / (1 + np.exp(-np.interp(x_hi, x_lo, v))), decimals).astype(np.float32) for k, v in logits.items()}
    return {"p_alive": out["alive"], "p_home": out["home"], "p_turnover": out["turn"]}
