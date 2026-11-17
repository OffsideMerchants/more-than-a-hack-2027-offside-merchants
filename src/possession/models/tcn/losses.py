import torch
import torch.nn.functional as F


def soften(y, mask, half_width):
    """Boundary-neighbour label softening: box-average of y over +-half_width steps, normalised on `mask`."""
    if half_width <= 0:
        return y
    k = torch.ones(1, 1, 2 * half_width + 1, device=y.device)
    num = F.conv1d((y * mask).unsqueeze(1), k, padding=half_width).squeeze(1)
    den = F.conv1d(mask.unsqueeze(1), k, padding=half_width).squeeze(1)
    return torch.where(den > 0, num / den.clamp(min=1e-6), y)


def possession_loss(out, b, lam_turn=0.3, soft_steps=2):
    """BCE(alive) + BCE(home | alive) + lam * BCE(turnover). Owner loss is masked on dead frames.
    soft_steps=2 is about +-0.2 s at 12.5 Hz."""
    alive = b["alive"]
    l_alive = F.binary_cross_entropy_with_logits(out["alive"].float(), soften(alive, torch.ones_like(alive), soft_steps))
    tgt_home = soften(b["home"], alive, soft_steps)
    l_home = (F.binary_cross_entropy_with_logits(out["home"].float(), tgt_home, reduction="none") * alive).sum() / alive.sum().clamp(min=1)
    l_turn = F.binary_cross_entropy_with_logits(out["turn"].float(), b["turn"], pos_weight=torch.tensor(5.0, device=alive.device))
    return l_alive + l_home + lam_turn * l_turn, {"alive": l_alive.item(), "home": l_home.item(), "turn": l_turn.item()}


def spell_loss(alive_logit, home_logit, y, near=25, smax=32):
    """Soft IoU per true possession spell (each spell weighted once), 1 - IoU averaged over spells.

    alive_logit/home_logit [B,T] float, y [B,T] in {0 H, 1 A, 2 D}. Spells touching a window edge are skipped.
    soft_IoU = sum_inside p / (|spell| + sum_{nearby, outside} p), p = P(alive and the spell's team).
    """
    B, T = y.shape
    change = torch.zeros_like(y, dtype=torch.long)
    change[:, 1:] = (y[:, 1:] != y[:, :-1]).long()
    sid = change.cumsum(1).clamp(max=smax - 1)
    m = F.one_hot(sid, smax).float()  # [B,T,S]
    cnt = m.sum(1)
    seg_y = (m * y[..., None].float()).sum(1) / cnt.clamp(min=1)  # [B,S]
    valid = (cnt > 0) & (seg_y < 1.5) & (m[:, 0] == 0) & (m[:, -1] == 0)
    if not valid.any():
        return alive_logit.sum() * 0
    pa, ph = torch.sigmoid(alive_logit), torch.sigmoid(home_logit)
    team_p = torch.where(seg_y[:, None, :] < 0.5, ph[..., None], 1 - ph[..., None]) * pa[..., None]  # [B,T,S]
    nb = F.max_pool1d(m.permute(0, 2, 1), 2 * near + 1, 1, near).permute(0, 2, 1)
    inter = (m * team_p).sum(1)
    union = cnt + ((nb - m).clamp(min=0) * team_p).sum(1)
    iou = inter / union.clamp(min=1)
    return (1 - iou)[valid].mean()
