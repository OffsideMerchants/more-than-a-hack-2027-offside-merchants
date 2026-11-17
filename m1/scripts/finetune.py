"""Fine-tune the pretrained M1 encoder for per-frame possession (H / A / D).

Default: fine-tune on 25 labelled matches, choose the HMM emission scale on 3 validation matches and score
2 held-out test matches (IDSSE J03WR9, SkillCorner 1874553) with the official scorer. --all-data trains the
final model on all 30 matches (no held-out score; an in-sample check only). Labels are controlled possession.
The fine-tuned model goes to runs/final_eval/<name>.pt, results to runs/final_eval/<name>.json; a checkpoint is
written after every epoch (<name>.ckpt.pt) and --resume continues from it. The HMM parameters for scripts/predict.py
(class prior and transitions of the fine-tuning labels, boost beta 16, emission scale 0.8 chosen on the validation
matches) go to runs/bar_ath/decoder.npz, or decoder_all.npz with --all-data; --decoder-only writes just that (CPU).

    python scripts/finetune.py --make-cache --data-cache data/finetune_cache.pkl      # CPU, needs the raw open data
    python scripts/finetune.py --name M1_transformer     --ckpt runs/ssl_full_mae/best.pt --data-cache data/finetune_cache.pkl --resume
    python scripts/finetune.py --name M1_transformer_all --ckpt runs/ssl_full_mae/best.pt --data-cache data/finetune_cache.pkl --resume --all-data
    python scripts/finetune.py --name x --data-cache data/finetune_cache.pkl --all-data --decoder-only
"""
import argparse
import json
import os
import pickle
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import sys

M1 = Path(__file__).resolve().parents[1]   # m1/: footy, data/ and runs/ are relative to it
sys.path.insert(0, str(M1))
os.chdir(M1)

from footy.data import BALL, MAX_PER_TEAM, POS_SCALE
from footy.hmm import fit_hmm_transitions
from footy.kloppy_data import _configure_kloppy
from footy.labels import IGNORE
from footy.model import TrajectoryMAE
from footy.pipeline import EVAL, SCALES, VAL, all_matches, boost, featurise, halves_of, score
from footy.possession import PossessionModel, ball_dropout, predict_match

BETA, SCALE = 16, 0.8      # HMM possession-change boost and emission scale of M1, chosen on the validation matches
SEED = 0


class LabelledWindows(Dataset):
    def __init__(self, periods, window=40, stride=4):
        self.data = [(torch.from_numpy(p / POS_SCALE), torch.from_numpy(l)) for p, l in periods]
        self.window = window
        self.index = [(i, s) for i, (p, _) in enumerate(self.data) for s in range(0, len(p) - window + 1, stride)]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        i, s = self.index[idx]
        x = self.data[i][0][s:s + self.window].clone()
        y = self.data[i][1][s:s + self.window].clone()
        if torch.rand(()) < 0.5:
            x[..., 0] = -x[..., 0]
        if torch.rand(()) < 0.5:
            x[..., 1] = -x[..., 1]
        if torch.rand(()) < 0.5:  # swap the teams, and the labels with them
            x = torch.cat([x[:, MAX_PER_TEAM:BALL], x[:, :MAX_PER_TEAM], x[:, BALL:]], dim=1)
            y = torch.where(y == 0, 1, torch.where(y == 1, 0, y))
        return x, y


def train_model(model, periods, window, device, epochs=12, bs=64, lr=1e-3, enc_lr_mult=0.2, ball_drop=True, log=print,
                resume=None, on_epoch=None):
    """Fine-tune on (positions, labels) periods. Labels of -1 are ignored, so partial pseudo-labels work as is.

    on_epoch(state) is called after every epoch with model / optimiser / scheduler / scaler state and the epoch index;
    pass that dict back as `resume` to continue after a crash (both default to off)."""
    dl = DataLoader(LabelledWindows(periods, window), bs, shuffle=True, drop_last=True, num_workers=4)
    if hasattr(model, "encoder"):  # pretrained encoder gets a lower learning rate than the head
        enc_lr = lr * (0.5 if getattr(model, "from_scratch", False) else enc_lr_mult)
        groups, lrs = [{"params": model.encoder.parameters(), "lr": enc_lr},
                       {"params": model.head.parameters(), "lr": lr}], [enc_lr, lr]
    else:
        groups, lrs = [{"params": model.parameters(), "lr": lr}], [lr]
    opt = torch.optim.AdamW(groups, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lrs, total_steps=epochs * len(dl), pct_start=0.1)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    start = 0
    if resume is not None:
        model.load_state_dict(resume["model"]); opt.load_state_dict(resume["opt"])
        sched.load_state_dict(resume["sched"]); scaler.load_state_dict(resume["scaler"])
        start = resume["epoch"] + 1
        log(f"resumed after epoch {resume['epoch']}")
    for epoch in range(start, epochs):
        torch.manual_seed(SEED * 1000 + epoch)   # per-epoch seed: a resumed run draws the same random numbers
        model.train()
        tot = 0.0
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            if ball_drop:
                x = ball_dropout(x)
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(x)
            loss = F.cross_entropy(logits.float().flatten(0, 1), y.flatten(), ignore_index=IGNORE)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item()
        log(f"ep {epoch:2d} loss {tot / len(dl):.4f}")
        if on_epoch is not None:
            on_epoch({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                      "scaler": scaler.state_dict(), "epoch": epoch})
    return model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--ckpt", default="runs/ssl_full_mae/best.pt", help="pretrained M1 encoder (scripts/pretrain.py)")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--smoke", action="store_true", help="2 training matches, 1 epoch")
    ap.add_argument("--resume", action="store_true", help="continue from runs/final_eval/<name>.ckpt.pt (written after every epoch)")
    ap.add_argument("--make-cache", action="store_true", help="featurise all matches, write them to --data-cache and stop (CPU only)")
    ap.add_argument("--data-cache", help="pickle of featurised matches: skips the raw-data loaders (e.g. on Colab)")
    ap.add_argument("--all-data", action="store_true", help="final model: fine-tune on all 30 matches (validation and evaluation "
                    "matches included), no held-out score; the HMM scale is fixed by --scale")
    ap.add_argument("--scale", type=float, default=SCALE, help="--all-data: HMM emission scale for the in-sample check")
    ap.add_argument("--decoder-only", action="store_true", help="write the HMM parameters for predict.py and stop (CPU)")
    args = ap.parse_args()
    _configure_kloppy()
    t0 = time.time()
    cache = Path(args.data_cache) if args.data_cache else None
    allp = pickle.loads(cache.read_bytes()) if cache is not None and cache.exists() and not args.make_cache else None
    keys = list(allp) if allp is not None else all_matches()   # the cache was written as fit + VAL + EVAL
    fit = [k for k in keys if k not in EVAL + VAL]
    if args.all_data:
        fit = fit + VAL + EVAL
    if args.smoke:
        fit = fit[:2]
    use = list(dict.fromkeys(fit + VAL + EVAL))
    if allp is not None:
        data = {k: allp[k] for k in use}
    else:
        data = {k: [featurise(p, l) for p, l in halves_of(*k)] for k in use}
        if args.make_cache:
            assert cache is not None and not args.smoke, "--make-cache needs --data-cache and no --smoke"
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_bytes(pickle.dumps(data, protocol=4))
            print(f"cache written: {len(data)} matches -> {cache} ({cache.stat().st_size / 2**20:.0f} MB)", flush=True)
            return
    print(f"data ready: {len(fit)} fine-tuning, {len(VAL)} validation, {len(EVAL)} evaluation matches ({time.time() - t0:.0f}s)", flush=True)
    lab_fit = np.concatenate([h["lab5"] for k in fit for h in data[k]])
    prior = np.bincount(lab_fit[lab_fit >= 0], minlength=3) / (lab_fit >= 0).sum()
    base_trans = fit_hmm_transitions([h["lab5"] for k in fit for h in data[k]])
    lt = boost(base_trans, BETA)
    dec = (Path("runs/smoke") / f"{args.name}_decoder.npz" if args.smoke      # never overwrite the real decoder
           else Path("runs/bar_ath") / ("decoder_all.npz" if args.all_data else "decoder.npz"))
    dec.parent.mkdir(parents=True, exist_ok=True)
    np.savez(dec, prior=prior, base_trans=base_trans, log_trans=lt, methods=np.array(["M1"]), betas=np.array([BETA], float),
             scales=np.array([SCALE], float))
    print(f"HMM parameters ({len(fit)} matches, prior H/A/D {np.round(prior, 3).tolist()}) -> {dec}", flush=True)
    if args.decoder_only:
        return
    out = Path("runs/final_eval")
    out.mkdir(parents=True, exist_ok=True)

    ck = torch.load(args.ckpt, map_location="cpu")
    a = ck["args"]
    enc = TrajectoryMAE(a["dim"], a["depth"], a["heads"], max_len=a["window"], dropout=a["dropout"])
    enc.load_state_dict(ck["model"])
    torch.manual_seed(SEED)                     # initialises the possession head
    device = torch.device("cuda")
    net = PossessionModel(enc).to(device)
    ck_path = out / f"{args.name}.ckpt.pt"

    def save_epoch(state):                      # atomic: a crash mid-write never corrupts the checkpoint
        tmp = ck_path.with_name(ck_path.name + ".tmp")
        torch.save(state, tmp)
        os.replace(tmp, ck_path)

    resume = torch.load(ck_path, map_location=device) if args.resume and ck_path.exists() else None
    train_model(net, [(h["pos5"], h["lab5"]) for k in fit for h in data[k]], a["window"], device,
                epochs=1 if args.smoke else args.epochs, bs=args.bs, lr=1e-3, log=lambda s: print(f"  {s} ({time.time() - t0:.0f}s)", flush=True),
                resume=resume, on_epoch=save_epoch)
    torch.save({"model": net.state_dict(), "pretrained": args.ckpt, "arch": "mae", "window": a["window"]}, out / f"{args.name}.pt")
    probs = {k: [predict_match(net, torch.from_numpy(h["pos5"] / POS_SCALE).float(), a["window"], stride=20, batch=32)
                 for h in data[k]] for k in VAL + EVAL}

    if args.all_data:     # the 5 matches were trained on: an in-sample check that training worked, not a score
        r = score(data, VAL + EVAL, probs, prior, lt, args.scale)
        summary = {"name": args.name, "ckpt": args.ckpt, "all_data": True, "n_matches": len(fit), "hmm_scale": args.scale,
                   "in_sample_spellF1": {f"{k[0]}_{k[1]}": x["spell_f1_mean"] for k, x in zip(VAL + EVAL, r)},
                   "minutes": (time.time() - t0) / 60}
        (out / f"{args.name}{'_smoke' if args.smoke else ''}.json").write_text(json.dumps(summary, indent=2, default=float))
        print(json.dumps(summary, default=float), flush=True)
        return
    best = max(SCALES, key=lambda s: np.mean([r["spell_f1_mean"] for r in score(data, VAL, probs, prior, lt, s)]))
    res = {}
    for k, r in zip(EVAL, score(data, EVAL, probs, prior, lt, best)):
        res[f"{k[0]}_{k[1]}"] = {m: r[m] for m in ("spell_f1_mean", "macro_f1", "alive_dead_f1", "team_acc_both_alive",
                                                   "n_pred_spells", "n_gt_spells")}
    summary = {"name": args.name, "ckpt": args.ckpt, "hmm_scale": best, "per_match": res,
               "mean_spellF1": float(np.mean([v["spell_f1_mean"] for v in res.values()])),
               "mean_macroF1": float(np.mean([v["macro_f1"] for v in res.values()])), "minutes": (time.time() - t0) / 60}
    (out / f"{args.name}{'_smoke' if args.smoke else ''}.json").write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps({k: summary[k] for k in ("name", "hmm_scale", "mean_spellF1", "mean_macroF1")}), flush=True)
    for k, v in res.items():
        print(f"  {k}: spellF1 {v['spell_f1_mean']:.3f} macroF1 {v['macro_f1']:.3f} possessions {v['n_pred_spells']}/{v['n_gt_spells']}", flush=True)


if __name__ == "__main__":
    main()
