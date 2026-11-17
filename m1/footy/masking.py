"""Masking strategies for masked trajectory modelling.

Each strategy returns a hidden mask [B, T, A] (True = the model must reconstruct it).
Mixing several strategies makes the model learn different things:
  random  - interpolation and short-range dynamics
  agents  - inferring a player's whole run from team shape and context
  future  - forecasting the next seconds of play
  ball    - inferring where the ball is from how players move
"""
import torch

from .data import BALL


def mask_random(observed, ratio=0.5):
    return torch.rand(observed.shape, device=observed.device) < ratio


def mask_agents(observed, k_min=1, k_max=6):
    B, T, A = observed.shape
    present = observed.any(1)                                  # [B, A]
    score = torch.rand(B, A, device=observed.device) + (~present).float() * 2
    rank = score.argsort(1).argsort(1)                         # 0 = lowest score
    k = torch.randint(k_min, k_max + 1, (B, 1), device=observed.device)
    return (rank < k)[:, None, :].expand(B, T, A)


def mask_future(observed, frac=0.3):
    B, T, A = observed.shape
    t = torch.arange(T, device=observed.device)
    return (t >= T - max(1, int(T * frac)))[None, :, None].expand(B, T, A)


def mask_ball(observed):
    m = torch.zeros_like(observed)
    m[..., BALL] = True
    return m


def make_mask(observed, mode):
    return {"random": mask_random, "agents": mask_agents, "future": mask_future, "ball": mask_ball}[mode](observed)


def nearest_visible_baseline(pos, hidden):
    """Non-learned baseline. A hidden token copies the same agent's nearest visible frame in time.
    If that agent is hidden in every frame, it uses the mean of the visible agents in that frame.
    """
    B, T, A, _ = pos.shape
    visible = ~pos.isnan().any(-1) & ~hidden
    t = torch.arange(T, device=pos.device, dtype=pos.dtype)
    dist = (t[:, None] - t[None, :]).abs()                           # [T, T']
    dist = dist[None, None] + (~visible).transpose(1, 2)[:, :, None, :] * 1e6   # [B, A, T, T']
    src = dist.argmin(-1)                                            # [B, A, T]
    p = torch.nan_to_num(pos).transpose(1, 2)                        # [B, A, T, 2]
    pred = torch.gather(p, 2, src[..., None].expand(-1, -1, -1, 2)).transpose(1, 2)
    has_any = visible.any(1)[:, None, :, None]                       # [B, 1, A, 1]
    v = visible[..., None].float()
    frame_mean = (p.transpose(1, 2) * v).sum(2, keepdim=True) / v.sum(2, keepdim=True).clamp(min=1)
    return torch.where(has_any, pred, frame_mean.expand_as(pred))


def mask_ball_gaps(observed, n_gaps=2, median_frames=7, sigma=0.8, max_frames=50):
    """Hide the ball in `n_gaps` random gaps per window, with ATD-like log-normal lengths
    (median ~1.5 s at 5 Hz). Targets the real use: reconstructing the ball inside tracking gaps."""
    B, T, A = observed.shape
    dev = observed.device
    t = torch.arange(T, device=dev)
    m = torch.zeros(B, T, A, dtype=torch.bool, device=dev)
    for _ in range(n_gaps):
        length = torch.exp(torch.randn(B, device=dev) * sigma) * median_frames
        length = length.clamp(1, max_frames).long()
        start = (torch.rand(B, device=dev) * T).long()
        gap = (t[None] >= start[:, None]) & (t[None] < (start + length)[:, None])   # [B, T]
        m[..., BALL] |= gap
    return m


MODES_V2 = ("random", "agents", "future", "ball", "ball_gaps")


def make_mask_v2(observed, mode):
    return mask_ball_gaps(observed) if mode == "ball_gaps" else make_mask(observed, mode)


def mixed_mask_v2(observed, probs=(0.3, 0.2, 0.2, 0.1, 0.2)):
    """One strategy per sample; 30% of samples are ball tasks (whole ball 10%, ball gaps 20%)."""
    masks = torch.stack([make_mask_v2(observed, m) for m in MODES_V2])
    choice = torch.multinomial(torch.tensor(probs), observed.shape[0], replacement=True).to(observed.device)
    return masks[choice, torch.arange(observed.shape[0], device=observed.device)]
