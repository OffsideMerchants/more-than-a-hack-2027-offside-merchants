"""Per-frame possession classifier on top of the pretrained trajectory encoder."""
import torch
import torch.nn as nn

from .data import BALL, MAX_PER_TEAM
from .model import TrajectoryMAE


class PossessionModel(nn.Module):
    """Encoder plus a per-frame head that reads the ball token and pooled home and away features."""

    def __init__(self, encoder: TrajectoryMAE, n_classes=3, dropout=0.1):
        super().__init__()
        self.encoder = encoder
        D = encoder.head.in_features
        self.head = nn.Sequential(nn.LayerNorm(3 * D), nn.Linear(3 * D, D), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(D, n_classes))

    def forward(self, pos):
        f = self.encoder.encode(pos)                                   # [B, T, A, D]
        obs = (~pos.isnan().any(-1)).float()[..., None]
        pool = lambda sl: (f[:, :, sl] * obs[:, :, sl]).sum(2) / obs[:, :, sl].sum(2).clamp(min=1)
        home, away = pool(slice(0, MAX_PER_TEAM)), pool(slice(MAX_PER_TEAM, BALL))
        return self.head(torch.cat([f[:, :, BALL], home, away], -1))  # [B, T, C]


def ball_dropout(pos, p_whole=0.15, rate=1 / 20, max_len=30):
    """Set the ball to NaN in random stretches, so a missing ball says nothing about the game state.

    Each window drops the ball entirely with probability `p_whole`. Otherwise, gaps of 1 to
    `max_len` frames start at roughly `rate` per frame. The dropped ball gets the same NaN input
    as a genuinely untracked ball, so the model cannot tell the two apart.
    """
    B, T = pos.shape[:2]
    dev = pos.device
    t = torch.arange(T, device=dev)
    start = torch.rand(B, T, device=dev) < rate
    length = torch.randint(1, max_len + 1, (B, T), device=dev)
    end = torch.where(start, t + length, torch.zeros_like(length)).cummax(1).values
    drop = (end > t) | (torch.rand(B, 1, device=dev) < p_whole)
    pos = pos.clone()
    pos[:, :, BALL][drop] = float("nan")
    return pos


@torch.no_grad()
def predict_match(model, pos, window=40, stride=10, batch=128):
    """Sliding-window inference over a whole period. pos: [T, A, 2] (already scaled). Returns probs [T, C]."""
    model.eval()
    T = pos.shape[0]
    starts = list(range(0, max(T - window, 0) + 1, stride))
    if starts[-1] + window < T:
        starts.append(T - window)
    dev = next(model.parameters()).device
    acc = torch.zeros(T, model.head[-1].out_features, device=dev)
    cnt = torch.zeros(T, 1, device=dev)
    for i in range(0, len(starts), batch):
        s = starts[i:i + batch]
        x = torch.stack([pos[j:j + window] for j in s]).to(dev)
        p = model(x).float().softmax(-1)
        for k, j in enumerate(s):
            acc[j:j + window] += p[k]
            cnt[j:j + window] += 1
    return (acc / cnt).cpu().numpy()
