"""Self-supervised pretraining of M1: masked trajectory modelling on unlabelled open tracking data.

Data (footy.ssl_data): IDSSE clean + two simulated-ATD versions, SkillCorner (broadcast), Metrica 1-3, about 52 h;
a fixed set of held-out matches never enters training. 16 s windows (80 frames at 5 Hz), a window every 1 s
(--stride 5). Masking: random / agents / future / whole ball / ATD-like ball gaps (30% ball tasks). The model
(footy.model.TrajectoryMAE) reconstructs the hidden positions. Resumable: --resume continues from last.pt (seeded per
epoch, so a resume at an epoch boundary draws the same random numbers).
Validation per epoch on held-out windows: mean error in metres per masking mode, model vs the nearest-visible
baseline; the best epoch is saved as runs/<name>/best.pt.

    python scripts/pretrain.py --name ssl_full_mae --stride 5 --epochs 10 --resume
"""
import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import sys

M1 = Path(__file__).resolve().parents[1]   # m1/: footy, data/ and runs/ are relative to it
sys.path.insert(0, str(M1))
os.chdir(M1)

from footy.data import POS_SCALE, TrackingWindows
from footy.kloppy_data import _configure_kloppy
from footy.masking import MODES_V2, make_mask_v2, mixed_mask_v2, nearest_visible_baseline
from footy.model import TrajectoryMAE
from footy.ssl_data import collect

SEED = 42


def epoch_batches(n, bs, epoch):
    """Shuffled, drop-last batches of window indices; deterministic per epoch so a run can resume mid-epoch."""
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(SEED + epoch)).tolist()
    return [perm[i:i + bs] for i in range(0, n - bs + 1, bs)]


def save_atomic(obj, path, keep_prev=False):
    """torch.save through a temp file; a crash mid-write never corrupts `path`. keep_prev keeps the last good copy."""
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    if keep_prev and path.exists():
        os.replace(path, path.with_name(path.stem + ".prev.pt"))
    os.replace(tmp, path)


def load_latest(dirs, device):
    """Most advanced loadable checkpoint among last.pt / last.prev.pt in `dirs` (the run directory)."""
    best = None
    for d in dirs:
        for name in ("last.pt", "last.prev.pt"):
            path = d / name
            if not path.exists():
                continue
            try:
                ck = torch.load(path, map_location=device)
            except Exception as err:               # truncated write after a crash: fall back to the other copy
                print(f"skipping unreadable checkpoint {path}: {err}", flush=True)
                continue
            ck.setdefault("next_epoch", ck["epoch"] + 1)   # checkpoints from before mid-epoch saving
            ck.setdefault("next_batch", 0)
            progress = (ck["next_epoch"], ck["next_batch"])
            if best is None or progress > best[0]:
                best = (progress, ck, path)
    return (best[1], best[2]) if best else (None, None)


def masked_mse(pred, pos, hidden):
    ok = hidden & ~pos.isnan().any(-1)
    err = ((pred - torch.nan_to_num(pos)) ** 2).sum(-1)
    return (err * ok).sum() / ok.sum().clamp(min=1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    sums = {m: [0.0, 0.0, 0, 0.0, 0] for m in MODES_V2}
    with torch.random.fork_rng(devices=[device]):
        torch.manual_seed(0)
        for pos in loader:
            pos = pos.to(device)
            observed = ~pos.isnan().any(-1)
            for m in MODES_V2:
                hidden = make_mask_v2(observed, m)
                with torch.autocast("cuda", dtype=torch.float16):
                    pred = model(pos, hidden).float()
                base = nearest_visible_baseline(pos, hidden)
                ok = hidden & observed
                truth = torch.nan_to_num(pos)
                sums[m][1] += ((base - truth).norm(dim=-1) * ok).sum().item()
                sums[m][0] += ((pred - truth).norm(dim=-1) * ok).sum().item()
                sums[m][2] += ok.sum().item()
    model.train()
    torch.cuda.empty_cache()
    res = {m: {"model_m": s[0] / max(s[2], 1) * POS_SCALE, "baseline_m": s[1] / max(s[2], 1) * POS_SCALE} for m, s in sums.items()}
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="ssl_v2_small")
    p.add_argument("--window", type=int, default=80)
    p.add_argument("--stride", type=int, default=10)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--bs", type=int, default=64)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--max-steps", type=int, default=0, help="stop after this many steps (smoke test)")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--workers", type=int, default=4, help="DataLoader workers")
    p.add_argument("--ckpt-every", type=int, default=0,
                   help="also save last.pt every N optimiser steps, resumable mid-epoch (0 = epoch end only)")
    args = p.parse_args()

    torch.manual_seed(SEED)
    device = torch.device("cuda")
    out = Path("runs") / args.name
    out.mkdir(parents=True, exist_ok=True)
    frames = args.window
    _configure_kloppy()
    train_segs = [s for segs in collect(held_out=False, min_len=frames).values() for s in segs]
    val_segs = [s for segs in collect(held_out=True, min_len=frames).values() for s in segs]
    train_ds = TrackingWindows(train_segs, frames, args.stride, augment=True)
    val_ds = TrackingWindows(val_segs, frames, stride=frames)
    del train_segs, val_segs                                     # TrackingWindows keeps its own scaled copy
    steps_per_epoch = len(train_ds) // args.bs
    val_dl = DataLoader(val_ds, 32, num_workers=args.workers)
    print(f"train windows {len(train_ds)}, val windows {len(val_ds)}", flush=True)

    model = TrajectoryMAE(args.dim, args.depth, args.heads, max_len=args.window, dropout=args.dropout)
    model = model.to(device)
    decay = [q for n, q in model.named_parameters() if q.ndim >= 2 and "emb" not in n]
    no_decay = [q for n, q in model.named_parameters() if not (q.ndim >= 2 and "emb" not in n)]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.05}, {"params": no_decay, "weight_decay": 0.0}],
                            lr=args.lr, betas=(0.9, 0.95))
    total = args.epochs * steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, (s + 1) / args.warmup) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
    scaler = torch.amp.GradScaler()
    start_epoch, start_batch, resumed_run, best = 0, 0, 0.0, float("inf")
    if args.resume:
        ck, ck_path = load_latest([out], device)
        if ck is not None:
            model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
            scaler.load_state_dict(ck["scaler"]); best = ck["best"]
            start_epoch, start_batch, resumed_run = ck["next_epoch"], ck["next_batch"], ck.get("run_loss", 0.0)
            print(f"resumed from {ck_path}: epoch {start_epoch}, batch {start_batch}/{steps_per_epoch}", flush=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=2))

    def checkpoint(epoch_done, next_epoch, next_batch, run_loss):
        save_atomic({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                     "scaler": scaler.state_dict(), "epoch": epoch_done, "next_epoch": next_epoch,
                     "next_batch": next_batch, "run_loss": run_loss, "best": best, "args": vars(args)}, out / "last.pt", keep_prev=True)

    log = open(out / "log.jsonl", "a")
    for epoch in range(start_epoch, args.epochs):
        # per-epoch seed (CPU, CUDA, DataLoader workers): a run resumed at an epoch boundary draws the same random numbers
        torch.manual_seed(SEED * 1000 + epoch)
        first = start_batch if epoch == start_epoch else 0
        batches = epoch_batches(len(train_ds), args.bs, epoch)[first:]
        train_dl = DataLoader(train_ds, batch_sampler=batches, num_workers=args.workers, pin_memory=True)
        t0, run = time.time(), (resumed_run if epoch == start_epoch else 0.0)
        step = epoch * steps_per_epoch + first
        for i, pos in enumerate(train_dl, start=first):
            pos = pos.to(device, non_blocking=True)
            hidden = mixed_mask_v2(~pos.isnan().any(-1))
            with torch.autocast("cuda", dtype=torch.float16):
                pred = model(pos, hidden)
            loss = masked_mse(pred.float(), pos, hidden)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            run += loss.detach()                  # stays on the GPU: no host sync every step
            step += 1
            if i == first + 50:                   # speed is timed after warm-up (cudnn, DataLoader)
                torch.cuda.synchronize(); t_ref = time.time()
            if i == first + 250:
                torch.cuda.synchronize()
                rate = 200 * args.bs / (time.time() - t_ref)
                mem = torch.cuda.memory_stats()
                print(f"epoch {epoch}: {rate:.0f} windows/s -> ~{len(train_ds) / rate / 60:.0f} min per epoch"
                      f" (GPU reserved {mem['reserved_bytes.all.current'] / 2**30:.1f} GB, alloc retries {mem['num_alloc_retries']})",
                      flush=True)
            if args.max_steps and step >= args.max_steps:
                print(f"smoke test stopped at step {step}, loss {float(run) / (i + 1 - first):.4f}", flush=True)
                save_atomic({"model": model.state_dict(), "args": vars(args)}, out / "smoke.pt")
                return
            if args.ckpt_every and step % args.ckpt_every == 0 and i + 1 < len(batches) + first:
                checkpoint(epoch - 1, epoch, i + 1, float(run))
        train_sec = time.time() - t0
        val = evaluate(model, val_dl, device)
        score = sum(v["model_m"] for v in val.values()) / len(val)
        rec = {"epoch": epoch, "step": step, "train_loss": float(run) / steps_per_epoch, "val": val, "val_mean_m": score,
               "sec": time.time() - t0, "train_sec": train_sec}
        log.write(json.dumps(rec) + "\n"); log.flush()
        print(f"ep {epoch} loss {rec['train_loss']:.4f} | val error m (model/baseline): "
              + "  ".join(f"{m} {v['model_m']:.2f}/{v['baseline_m']:.2f}" for m, v in val.items()) + f" | {rec['sec']:.0f}s (train {train_sec:.0f}s)", flush=True)
        if score < best:
            best = score
            save_atomic({"model": model.state_dict(), "args": vars(args), "epoch": epoch, "val": val}, out / "best.pt")
        checkpoint(epoch, epoch + 1, 0, 0.0)
    (out / "done").write_text("finished\n")


if __name__ == "__main__":
    main()
