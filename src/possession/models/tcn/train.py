"""Training loop for the S2 TCN. Model selection is by a caller-supplied decoded-spellF1 callback, never by loss."""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch

from .data import WindowSampler, augment, collate, make_sample
from .losses import possession_loss
from .model import PossessionTCN, TCNConfig


@dataclass
class TrainConfig:
    steps: int = 2000
    batch: int = 32
    lr: float = 2e-3
    wd: float = 1e-2
    clip: float = 1.0
    ema: float = 0.999
    win_s: float = 20.0
    lam_turn: float = 0.3
    soft_steps: int = 2
    eval_every: int = 250
    patience: int = 4
    amp: bool = True
    augment: bool = True
    seed: int = 0


def draw_batch(sampler, cfg: TrainConfig, rate_hz, rng, degrade: Callable | None = None, device="cpu"):
    stride = int(round(25 / rate_hz))
    win = int(cfg.win_s * 25)
    samples = []
    for _ in range(cfg.batch):
        mi, s0 = sampler.sample(rng)
        m = sampler.matches[mi]
        if degrade is not None:  # fresh degradation per sample (S0 simulator)
            m = degrade(m, s0, win, rng)
            s0 = 0
        s = make_sample(m, s0, win, stride, int(rng.integers(stride)))
        samples.append(augment(s, rng) if cfg.augment else s)
    n = min(len(s["alive"]) for s in samples)
    return collate([{k: (v[:n] if k != "team" else v) for k, v in s.items()} for s in samples], device)


class EMA:
    def __init__(self, model, decay):
        self.decay, self.shadow = decay, copy.deepcopy(model).eval()

    @torch.no_grad()
    def update(self, model):
        for e, p in zip(self.shadow.state_dict().values(), model.state_dict().values()):
            e.copy_(e * self.decay + p * (1 - self.decay)) if e.dtype.is_floating_point else e.copy_(p)


def train(train_matches, model_cfg: TCNConfig, cfg: TrainConfig, val_fn: Callable | None = None,
          degrade: Callable | None = None, device="cpu", log=print):
    """Returns (best EMA model, history). val_fn(model) -> decoded spellF1 on deg_A for the held-out match."""
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    model = PossessionTCN(model_cfg).to(device)
    ema = EMA(model, cfg.ema)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=cfg.steps)
    use_amp = cfg.amp and device != "cpu"
    scaler = torch.amp.GradScaler(enabled=use_amp)
    sampler = WindowSampler(train_matches, int(cfg.win_s * 25))
    best, best_state, bad, hist = -np.inf, None, 0, []
    for step in range(1, cfg.steps + 1):
        model.train()
        b = draw_batch(sampler, cfg, model_cfg.rate_hz, rng, degrade, device)
        with torch.autocast(device_type=torch.device(device).type, enabled=use_amp):
            out = model(b["players"], b["pmask"], b["team"], b["ball"], b["bmask"])
        loss, parts = possession_loss(out, b, cfg.lam_turn, cfg.soft_steps)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip)
        scaler.step(opt)
        scaler.update()
        sched.step()
        ema.update(model)
        if step % cfg.eval_every == 0 or step == cfg.steps:
            score = val_fn(ema.shadow) if val_fn else -loss.item()
            hist.append({"step": step, "loss": loss.item(), "val": score, **parts})
            log(f"step {step} loss {loss.item():.4f} val {score:.4f}")
            if score > best:
                best, best_state, bad = score, copy.deepcopy(ema.shadow.state_dict()), 0
            else:
                bad += 1
                if bad >= cfg.patience:
                    break
    ema.shadow.load_state_dict(best_state)
    return ema.shadow, hist
