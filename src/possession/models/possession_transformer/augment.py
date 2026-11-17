import torch
from torch import Tensor


def swap_teams(batch: dict) -> dict:
    """Team swap as a 180 degree pitch rotation (home attacks +x in both). Targets swap H and A."""
    out = dict(batch)
    sign = torch.tensor([-1, -1, -1, -1, 1, 1], dtype=batch["players"].dtype, device=batch["players"].device)
    out["players"] = batch["players"] * sign
    out["ball"] = batch["ball"] * sign[[0, 1, 2, 3, 5]]
    out["team"] = 1 - batch["team"]
    if "y" in batch:
        y = batch["y"]
        out["y"] = torch.where(y < 2, 1 - y, y)
    if "ball_xy" in batch:
        out["ball_xy"] = -batch["ball_xy"]
    return out


def flip_y(batch: dict) -> dict:
    out = dict(batch)
    sign = torch.tensor([1, -1, 1, -1, 1, 1], dtype=batch["players"].dtype, device=batch["players"].device)
    out["players"] = batch["players"] * sign
    out["ball"] = batch["ball"] * sign[[0, 1, 2, 3, 5]]
    if "ball_xy" in batch:
        out["ball_xy"] = batch["ball_xy"] * torch.tensor([1.0, -1.0], device=batch["ball_xy"].device)
    return out


def shuffle_tracks(batch: dict, gen: torch.Generator | None = None) -> dict:
    """Random track order, the same for every step of a window. Possessor indices are remapped."""
    B, T, N, _ = batch["players"].shape
    out = dict(batch)
    perm = torch.stack([torch.randperm(N, generator=gen) for _ in range(B)]).to(batch["players"].device)
    out["players"] = batch["players"].gather(2, perm[:, None, :, None].expand(-1, T, -1, batch["players"].shape[-1]))
    out["team"] = batch["team"].gather(1, perm)
    if "possessor" in batch:
        inv = perm.argsort(1)
        p = batch["possessor"]
        out["possessor"] = torch.where(p >= 0, inv.gather(1, p.clamp(min=0)), p)
    return out
