"""Train/evaluate one fold:  python -m possession_gnn.run --model gnn --ball blind --test J03WN1 --val J03WOY

Selection uses the val match only. The test match is scored once at the end on two fixed degraded copies
(deg_A seed 101, deg_B seed 202). Writes arsh-research/results/runs/<tag>.json.
"""
import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import data, degrade, evalbridge as eb
from .models import PossessionNet

ROOT = Path(__file__).resolve().parents[2]
DEV = ["J03WN1", "J03WOH", "J03WOY", "J03WPY", "J03WQQ", "J03WR9"]  # J03WMX is the lockbox
DEV_T = 96   # 19.2 s window at 5 Hz
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build(d, seed, ball, clean=False):
    rng = np.random.default_rng(seed)
    xy, b = degrade.clean(d) if clean else degrade.degrade(d, rng)
    f, p = degrade.node_features(xy, b, d["team"], ball_blind=(ball == "blind"))
    return torch.from_numpy(f), torch.from_numpy(p), np.isnan(b[:, 0])


def augment(nodes, y):
    """Team swap coupled with an x mirror (keeps 'home attacks +x'), independent y flip."""
    B = nodes.shape[0]
    sw = torch.rand(B, device=nodes.device) < .5
    fy = torch.rand(B, device=nodes.device) < .5
    n = nodes.clone()
    s = sw[:, None, None]
    n[..., 0] = torch.where(s, -n[..., 0], n[..., 0])
    n[..., 2] = torch.where(s, -n[..., 2], n[..., 2])
    n[..., 5], n[..., 6] = torch.where(s, n[..., 6], n[..., 5]), torch.where(s, n[..., 5], n[..., 6])
    f = fy[:, None, None]
    n[..., 1] = torch.where(f, -n[..., 1], n[..., 1])
    n[..., 3] = torch.where(f, -n[..., 3], n[..., 3])
    # zero-pad slots stay zero: mirrored zero is zero
    y = torch.where(sw[:, None] & (y < 2), 1 - y, y)
    return n, y


def losses(model, nodes, pres, y):
    la, lh = model(nodes, pres)
    alive = (y < 2)
    l_a = F.binary_cross_entropy_with_logits(la, alive.float())
    l_h = (F.binary_cross_entropy_with_logits(lh, (y == 0).float(), reduction="none") * alive).sum() / alive.sum().clamp_min(1)
    return l_a + l_h, l_a.item(), l_h.item()


@torch.no_grad()
def infer(model, nodes, pres, T=DEV_T, stride=48):
    model.eval()
    n = nodes.shape[0]
    starts = list(range(0, max(1, n - T + 1), stride))
    if starts[-1] + T < n:
        starts.append(n - T)
    pa, ph, cnt = np.zeros(n), np.zeros(n), np.zeros(n)
    w = np.ones(T)
    w[:8] = w[-8:] = 0.05
    for i in range(0, len(starts), 16):
        ss = starts[i:i + 16]
        x = torch.stack([nodes[s:s + T] for s in ss]).to(dev)
        p = torch.stack([pres[s:s + T] for s in ss]).to(dev)
        with torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
            la, lh = model(x, p)
        la, lh = torch.sigmoid(la.float()).cpu().numpy(), torch.sigmoid(lh.float()).cpu().numpy()
        for k, s in enumerate(ss):
            pa[s:s + T] += la[k] * w
            ph[s:s + T] += lh[k] * w
            cnt[s:s + T] += w
    return pa / cnt, ph / cnt


def evaluate_match(model, d, seed, ball, bootstrap_n=0, clean=False):
    nodes, pres, bmiss = build(d, seed, ball, clean)
    pa, ph = infer(model, nodes, pres)
    m = eb.score(d, pa, ph, bootstrap_n=bootstrap_n)
    keep = ["spell_f1_mean", "macro_f1", "alive_dead_f1", "team_acc_both_alive", "n_gt_spells", "n_pred_spells",
            "bootstrap_low", "bootstrap_high", "spell_bootstrap_low", "spell_bootstrap_high"]
    out = {k: m[k] for k in keep}
    out["transition_f1"] = m["transitions"]
    out["raw_argmax_spell_f1"] = eb.score(d, pa, ph, decode="argmax")["spell_f1_mean"]
    out["gaps"] = eb.gap_report(d, bmiss if ball != "blind" else np.ones(len(bmiss), bool), ph)
    out["gaps_vs_real_ball_gaps"] = eb.gap_report(d, bmiss, ph)
    out["val_logits"] = None
    return out, pa, ph


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gnn")
    ap.add_argument("--ball", default="blind", choices=["blind", "real"])
    ap.add_argument("--test", required=True)
    ap.add_argument("--val", default=None)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--draws", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--final", action="store_true", help="train on all dev matches (for the lockbox)")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    tag = a.tag or f"{a.model}_{a.ball}_{a.test}_s{a.seed}"
    D = {m: data.load(m) for m in data.MATCHES}
    if a.final:
        train_m, val = list(DEV), a.val or "J03WOY"
    else:
        val = a.val or [m for m in DEV if m != a.test][0]
        train_m = [m for m in DEV if m not in (a.test, val)]
    assert a.test not in train_m and (a.final or val not in train_m)

    pool = []
    for m in train_m:
        for k in range(a.draws):
            n, p, _ = build(D[m], int(rng.integers(1 << 30)), a.ball)
            pool.append((n.to(dev), p.to(dev), torch.from_numpy(D[m]["y"].astype(np.int64)).to(dev)))
    vn, vp, _ = build(D[val], 303, a.ball)
    vy = torch.from_numpy(D[val]["y"].astype(np.int64))

    model = PossessionNet(a.model).to(dev)
    npar = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), 2e-3, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=a.steps, pct_start=0.1)
    scaler = torch.amp.GradScaler(enabled=dev.type == "cuda")
    best, best_state, t0, log = 1e9, None, time.time(), []
    for step in range(1, a.steps + 1):
        model.train()
        xs, ps, ys = [], [], []
        for _ in range(a.batch):
            n, p, y = pool[rng.integers(len(pool))]
            s = int(rng.integers(0, len(y) - DEV_T))
            xs.append(n[s:s + DEV_T]); ps.append(p[s:s + DEV_T]); ys.append(y[s:s + DEV_T])
        x, p, y = torch.stack(xs), torch.stack(ps), torch.stack(ys)
        x, y = augment(x, y)
        if a.ball == "real" and rng.random() < 0.2:  # ball-dropout curriculum: some fully ball-blind windows
            x = x.clone(); p = p.clone(); x[:, :, -1] = 0; p[:, :, -1] = False
        with torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
            loss, la, lh = losses(model, x, p, y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); sched.step()
        if step % 500 == 0 or step == a.steps:
            pa, ph = infer(model, vn, vp)
            ok = vy.numpy() < 2
            vl = float(-np.mean(np.log(np.where(vy.numpy() == 0, ph, 1 - ph)[ok].clip(1e-4)))
                       + -np.mean(np.log(np.where(ok, pa, 1 - pa).clip(1e-4))))
            log.append({"step": step, "train": float(loss), "val": vl, "t": round(time.time() - t0)})
            print(tag, log[-1], flush=True)
            if vl < best:
                best, best_state = vl, {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    res = {"tag": tag, "args": vars(a), "params": npar, "git": subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip(),
           "train": train_m, "val": val, "log": log, "best_val": best, "test": {}}
    for name, seed in (("deg_A", 101), ("deg_B", 202)):
        res["test"][name], pa, ph = evaluate_match(model, D[a.test], seed, a.ball, bootstrap_n=200 if a.final else 0)
        if name == "deg_A":  # prediction contract (00_FOUNDATION s7): 25 Hz p_alive / p_home for the shared decoders
            pd_ = ROOT / "artifacts" / "preds" / "m3_gnn" / tag / "idsse"
            pd_.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(pd_ / f"{a.test}.npz", frame=D[a.test]["frame"],
                                p_alive=pa.astype(np.float32), p_home=ph.astype(np.float32), hz=5)
    res["test"]["clean"], _, _ = evaluate_match(model, D[a.test], 0, a.ball, clean=True)
    out = ROOT / "arsh-research" / "results" / "runs"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{tag}.json").write_text(json.dumps(res, indent=1, default=float))
    t = res["test"]
    print(tag, "DONE", {k: (round(v["spell_f1_mean"], 3), round(v["macro_f1"], 3)) for k, v in t.items()})


if __name__ == "__main__":
    main()
