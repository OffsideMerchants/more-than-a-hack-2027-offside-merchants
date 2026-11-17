"""S2 TCN / S4 transformer on the S3 protocol (same 5 Hz data, degradation, folds, decoder and scorer).

  python -m possession_gnn.run_s2s4 --model tcn|s4 --test J03WN1 [--val J03WOY] [--final]

Ball is real with 20% ball-blind windows (the S3 'real' setup). Writes arsh-research/results/runs/<model>_real_<test>_s0.json.
"""
import argparse
import json
import subprocess
import time

import numpy as np
import torch

from . import data, data_sc, degrade, diagnostics as dg, evalbridge as eb
from .run import DEV, ROOT, dev
from possession.models.tcn.data import turnover_targets
from possession.models.tcn.losses import possession_loss as tcn_loss, spell_loss
from possession.models.tcn.model import PossessionTCN, TCNConfig
from possession.models.possession_transformer import crf, losses as s4l
from possession.models.possession_transformer.model import PossessionTransformer, S4Config

HZ = 5
NSLOT = 32
SC_ALL = data_sc.matches()
SC_TEST, SC_VAL, SC_TRAIN = [], [], []


def set_shared_split():
    """splits.json (shared with M1): -> (IDSSE test, IDSSE val, IDSSE training matches); sets the SkillCorner lists."""
    global SC_TEST, SC_VAL, SC_TRAIN
    sp = json.loads((ROOT / "splits.json").read_text())
    SC_TEST = [str(m) for m in sp["test"]["skillcorner"]]
    SC_VAL = [str(m) for m in sp["val"]["skillcorner"]]
    SC_TRAIN = [m for m in SC_ALL if m not in SC_TEST + SC_VAL]
    (test,), (val,) = sp["test"]["idsse"], sp["val"]["idsse"]
    return test, val, [m for m in data.MATCHES if m not in (test, val)]


def set_fold(k):
    """SkillCorner 5-fold split: test = every 5th match from k, validation = the match after the first test match."""
    global SC_TEST, SC_VAL, SC_TRAIN
    SC_TEST = SC_ALL[k::5]
    SC_VAL = [SC_ALL[k + 1]]
    SC_TRAIN = [m for m in SC_ALL if m not in SC_TEST + SC_VAL]


if SC_ALL:  # training folds need the SkillCorner data; prediction does not
    set_fold(0)
WIN = {"tcn": 100, "tfm": 300, "s4": 300}
BATCH = {"tcn": 16, "tfm": 8, "s4": 4}


def raw(d, seed, clean=False):
    """-> players [T,N,5] (x y vx vy speed, NaN missing), ball [T,4], from the degraded positions only."""
    xy, b = degrade.clean(d) if (clean or "native" in d) else degrade.degrade(d, np.random.default_rng(seed))
    allp = np.concatenate([xy, b[:, None]], 1)
    pres = ~np.isnan(allp[..., 0])
    v = np.zeros_like(allp)
    ok = pres[1:] & pres[:-1]
    v[1:] = np.where(ok[..., None], (allp[1:] - allp[:-1]) * HZ, 0.0)
    v[:-1] = np.where(ok[..., None], v[1:], v[:-1])
    v = np.clip(np.nan_to_num(v), -12, 12)
    f = np.concatenate([allp, v, np.linalg.norm(v, axis=-1, keepdims=True)], -1).astype(np.float32)
    f[~pres] = np.nan
    pl = f[:, :-1]
    pl = np.concatenate([pl, np.full((len(pl), NSLOT - pl.shape[1], 5), np.nan, np.float32)], 1)  # fixed slots
    return torch.from_numpy(pl), torch.from_numpy(f[:, -1, [0, 1, 2, 3]]), np.isnan(b[:, 0])


def labels(d):
    y = d["y"].astype(np.int64)
    turn = turnover_targets(y == 0, y < 2, 1)
    return torch.from_numpy(y), torch.from_numpy(turn)


def augment(pl, ball, team, y, turn):
    """Team swap coupled with an x mirror and an independent y flip (as in S3)."""
    B = pl.shape[0]
    sw = torch.rand(B, device=pl.device) < .5
    fy = torch.rand(B, device=pl.device) < .5
    sx = torch.where(sw, -1.0, 1.0)[:, None, None, None]
    sy = torch.where(fy, -1.0, 1.0)[:, None, None, None]
    pl = pl.clone(); ball = ball.clone()
    pl[..., [0, 2]] *= sx; ball[..., [0, 2]] *= sx[..., 0]
    pl[..., [1, 3]] *= sy; ball[..., [1, 3]] *= sy[..., 0]
    team = torch.where(sw[:, None], 1 - team, team)
    y = torch.where(sw[:, None] & (y < 2), 1 - y, y)
    return pl, ball, team, y, turn


def make_model(kind, hidden=64, blocks=5, attn=4):
    if kind == "tcn":
        return PossessionTCN(TCNConfig(hidden=hidden, joint_blocks=blocks, rate_hz=HZ))
    if kind == "tfm":
        return PossessionTCN(TCNConfig(hidden=hidden, joint_blocks=blocks, rate_hz=HZ, attn_blocks=attn, drop_path=0.1))
    return PossessionTransformer(S4Config(rate_hz=HZ, window_s=WIN["s4"] / HZ))


def forward(kind, model, pl, ball, team):
    pm, bm = torch.isfinite(pl[..., 0]), torch.isfinite(ball[..., 0])
    if kind != "s4":
        return model(torch.nan_to_num(pl), pm, team, torch.nan_to_num(ball), bm)
    return model(torch.cat([pl, pm[..., None].float()], -1), team, torch.cat([ball, bm[..., None].float()], -1))


LAM_TURN = 0.3
LAM_SPELL = 0.0
W4 = s4l.LossWeights(crf=1.0, spell=0.5, boundary=0.5)


def loss_fn(kind, model, pl, ball, team, y, turn):
    out = forward(kind, model, pl, ball, team)
    if kind != "s4":
        alive = (y < 2).float()
        loss = tcn_loss(out, {"alive": alive, "home": (y == 0).float(), "turn": turn}, LAM_TURN, 1)[0]
        if LAM_SPELL > 0:
            loss = loss + LAM_SPELL * spell_loss(out["alive"].float(), out["home"].float(), y)
        return loss
    out = {k: (v.float() if torch.is_tensor(v) else v) for k, v in out.items()}
    return s4l.possession_loss(out, {"y": y}, W4, HZ)["total"]


@torch.no_grad()
def infer(kind, model, pl, ball, team, T, stride=None):
    model.eval()
    n = len(pl)
    stride = stride or T // 2
    starts = list(range(0, max(1, n - T + 1), stride))
    if starts[-1] + T < n:
        starts.append(n - T)
    pa, ph, cnt = np.zeros(n), np.zeros(n), np.zeros(n)
    w = np.hanning(T + 2)[1:-1] + 0.05
    tm = team[None].to(dev)
    for i in range(0, len(starts), 8):
        ss = starts[i:i + 8]
        x = torch.stack([pl[s:s + T] for s in ss]).to(dev)
        b = torch.stack([ball[s:s + T] for s in ss]).to(dev)
        tt = tm.expand(len(ss), -1)
        with torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
            out = forward(kind, model, x, b, tt)
        if kind != "s4":
            la = torch.sigmoid(out["alive"].float()).cpu().numpy()
            lh = torch.sigmoid(out["home"].float()).cpu().numpy()
        else:
            _, un, _ = crf.forward_backward(out["emit"].float(), out["trans"].float(), None)
            un = un.cpu().numpy()
            la = 1 - un[..., crf.D]
            lh = un[..., crf.H] / np.maximum(un[..., crf.H] + un[..., crf.A], 1e-9)
        for k, s in enumerate(ss):
            pa[s:s + T] += la[k] * w; ph[s:s + T] += lh[k] * w; cnt[s:s + T] += w
    return pa / cnt, ph / cnt


def evaluate(kind, model, d, seed, team, bootstrap_n=0, clean=False):
    pl, ball, bmiss = raw(d, seed, clean)
    pa, ph = infer(kind, model, pl, ball, team, WIN[kind])
    m = eb.score(d, pa, ph, bootstrap_n=bootstrap_n)
    keep = ["spell_f1_mean", "macro_f1", "alive_dead_f1", "team_acc_both_alive", "n_gt_spells", "n_pred_spells",
            "bootstrap_low", "bootstrap_high", "spell_bootstrap_low", "spell_bootstrap_high"]
    out = {k: m[k] for k in keep}
    out["transition_f1"] = m["transitions"]
    out["raw_argmax_spell_f1"] = eb.score(d, pa, ph, decode="argmax")["spell_f1_mean"]
    out["confusion"] = dg.confusion(*eb.decode_25hz(d, pa, ph))
    out["gaps_vs_real_ball_gaps"] = eb.gap_report(d, bmiss, ph)
    return out, pa, ph


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["tcn", "tfm", "s4"])
    ap.add_argument("--split", choices=["folds", "shared"], default="folds",
                    help="folds: --sc_fold / --test / --val as before; shared: splits.json (same split as M1, J03WMX trains)")
    ap.add_argument("--test", default=None)
    ap.add_argument("--val", default=None)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--draws", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--blocks", type=int, default=5)
    ap.add_argument("--attn", type=int, default=4)
    ap.add_argument("--win", type=int, default=None, help="window steps at 5 Hz")
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--lam_spell", type=float, default=0.0)
    ap.add_argument("--sc_fold", type=int, default=0)
    ap.add_argument("--lam_turn", type=float, default=0.3)
    ap.add_argument("--sc", action="store_true", help="also train on SkillCorner open data (native degradation)")
    ap.add_argument("--sc_frac", type=float, default=0.5)
    ap.add_argument("--pseudo", default=None, help="teacher preds dir (<...>/sc) -> self-training on the unlabelled SC test matches")
    ap.add_argument("--pfrac", type=float, default=0.3, help="share of batches drawn from pseudo-labelled windows")
    ap.add_argument("--pconf", type=float, default=0.9, help="min mean teacher confidence of a pseudo-labelled window")
    a = ap.parse_args()
    kind = a.model
    if a.split == "shared":
        a.test, a.val, shared_train = set_shared_split()
    else:
        assert a.test, "--test is required with --split folds"
        set_fold(a.sc_fold)
    global LAM_TURN, LAM_SPELL
    LAM_TURN, LAM_SPELL = a.lam_turn, a.lam_spell
    WIN[kind] = a.win or WIN[kind]
    BATCH[kind] = a.batch or BATCH[kind]
    T, B = WIN[kind], BATCH[kind]
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    tag = f"{a.tag or kind + '_real'}_{a.test}_s{a.seed}"
    D = {m: data.load(m) for m in data.MATCHES}
    if a.split == "shared":
        train_m, val = shared_train, a.val
    elif a.final:
        train_m, val = list(DEV), a.val or "J03WOY"
    else:
        val = a.val or [m for m in DEV if m != a.test][0]
        train_m = [m for m in DEV if m not in (a.test, val)]
    assert a.test not in train_m
    team_of = {m: torch.from_numpy(np.concatenate([D[m]["team"], np.full(NSLOT - len(D[m]["team"]), -1)]).astype(np.int64)) for m in D}
    pool = []
    for m in train_m:
        y, turn = labels(D[m])
        for _ in range(a.draws):
            pl, ball, _ = raw(D[m], int(rng.integers(1 << 30)))
            pool.append((pl.to(dev), ball.to(dev), team_of[m].to(dev), y.to(dev), turn.to(dev)))
    pool_sc = []
    if a.sc:
        for m in SC_TRAIN + (SC_VAL if a.final else []):
            d = data_sc.load(m)
            y, turn = labels(d)
            pl, ball, _ = raw(d, 0)
            tm = torch.from_numpy(d["team"].astype(np.int64))
            pool_sc.append((pl.to(dev), ball.to(dev), tm.to(dev), y.to(dev), turn.to(dev)))
    pool_ps = []
    if a.pseudo:  # teacher predictions only; the true labels of these matches are never read here
        for m in SC_TEST:
            z = np.load(f"{a.pseudo}/{m}.npz")
            d = data_sc.load(m)
            pa_, ph_ = z["p_alive"], z["p_home"]
            path = np.concatenate([eb.viterbi(pa_[d["half"] == h], ph_[d["half"] == h]) for h in (0, 1)])
            conf = np.concatenate([np.maximum(np.where(pa_ < .5, 1 - pa_, pa_ * np.maximum(ph_, 1 - ph_))[d["half"] == h], 0) for h in (0, 1)])
            d2 = dict(d, y=path.astype(np.int8))
            y, turn = labels(d2)
            pl, ball, _ = raw(d, 0)
            tm = torch.from_numpy(d["team"].astype(np.int64))
            pool_ps.append((pl.to(dev), ball.to(dev), tm.to(dev), y.to(dev), turn.to(dev), torch.from_numpy(conf.astype(np.float32)).to(dev)))
    vpl, vball, _ = raw(D[val], 303)
    sc_val = SC_VAL if a.split == "shared" else SC_VAL[:1]       # SkillCorner matches in the validation loss
    svs = []
    for m in (sc_val if a.sc else []):
        scv = data_sc.load(m)
        svpl, svball, _ = raw(scv, 0)
        svs.append((m, svpl, svball, scv["y"].astype(np.int64), torch.from_numpy(scv["team"].astype(np.int64))))
    vy = D[val]["y"].astype(np.int64)

    model = make_model(kind, a.hidden, a.blocks, a.attn).to(dev)
    npar = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), a.lr if kind != "s4" else 5e-4, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, opt.param_groups[0]["lr"], total_steps=a.steps, pct_start=0.1)
    scaler = torch.amp.GradScaler(enabled=dev.type == "cuda")
    best, best_state, t0, log = 1e9, None, time.time(), []
    for step in range(1, a.steps + 1):
        model.train()
        parts = []
        for _ in range(B):
            if pool_ps and rng.random() < a.pfrac:
                pl, ball, team, y, turn, conf = pool_ps[rng.integers(len(pool_ps))]
                for _try in range(10):
                    s = int(rng.integers(0, len(y) - T))
                    if float(conf[s:s + T].mean()) >= a.pconf:
                        break
            else:
                src = pool_sc if (pool_sc and rng.random() < a.sc_frac) else pool
                pl, ball, team, y, turn = src[rng.integers(len(src))][:5]
                s = int(rng.integers(0, len(y) - T))
            parts.append((pl[s:s + T], ball[s:s + T], team, y[s:s + T], turn[s:s + T]))
        pl, ball, team, y, turn = (torch.stack(z) for z in zip(*parts))
        pl, ball, team, y, turn = augment(pl, ball, team, y, turn)
        if rng.random() < 0.2:  # ball-dropout curriculum
            ball = torch.full_like(ball, float("nan"))
        with torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
            loss = loss_fn(kind, model, pl, ball, team, y, turn)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); sched.step()
        if step % 500 == 0 or step == a.steps:
            pa, ph = infer(kind, model, vpl, vball, team_of[val], T)
            ok = vy < 2
            vl = float(-np.mean(np.log(np.where(vy == 0, ph, 1 - ph)[ok].clip(1e-4)))
                       + -np.mean(np.log(np.where(ok, pa, 1 - pa).clip(1e-4))))
            for _, svpl, svball, svy, svteam in svs:
                pa, ph = infer(kind, model, svpl, svball, svteam, T)
                ok = svy < 2
                vl += float(-np.mean(np.log(np.where(svy == 0, ph, 1 - ph)[ok].clip(1e-4)))
                            + -np.mean(np.log(np.where(ok, pa, 1 - pa).clip(1e-4))))
            log.append({"step": step, "train": float(loss), "val": vl, "t": round(time.time() - t0)})
            print(tag, log[-1], flush=True)
            if vl < best:
                best, best_state = vl, {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    (ROOT / "artifacts" / "models").mkdir(parents=True, exist_ok=True)
    torch.save({"state": best_state, "args": vars(a)}, ROOT / "artifacts" / "models" / f"{tag}{'_final' if a.final else ''}.pt")
    res = {"tag": tag, "args": vars(a), "params": npar, "train": train_m, "val": val, "log": log, "best_val": best,
           "git": subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip(),
           "test": {}}
    for name, seed in (("deg_A", 101), ("deg_B", 202)):
        res["test"][name], pa, ph = evaluate(kind, model, D[a.test], seed, team_of[a.test], 200 if a.final else 0)
        if name == "deg_A":
            pd_ = ROOT / "artifacts" / "preds" / f"m_{a.tag or kind}" / tag / "idsse"
            pd_.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(pd_ / f"{a.test}.npz", frame=D[a.test]["frame"],
                                p_alive=pa.astype(np.float32), p_home=ph.astype(np.float32), hz=5)
    if a.sc:  # validation-match probabilities, for decoder tuning / ensembling without touching test matches
        vd = ROOT / "artifacts" / "preds" / f"m_{a.tag or kind}" / tag / "val"
        vd.mkdir(parents=True, exist_ok=True)
        for m, svpl, svball, _, svteam in svs:
            pa, ph = infer(kind, model, svpl, svball, svteam, T)
            np.savez_compressed(vd / f"{m}.npz", p_alive=pa.astype(np.float32), p_home=ph.astype(np.float32))
        res["test_sc"] = {}
        for m in SC_TEST:
            d = data_sc.load(m)
            res["test_sc"][m], pa, ph = evaluate(kind, model, d, 0, torch.from_numpy(d["team"].astype(np.int64)), 0, clean=True)
            pd_ = ROOT / "artifacts" / "preds" / f"m_{a.tag or kind}" / tag / "sc"
            pd_.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(pd_ / f"{m}.npz", frame=d["frame"], p_alive=pa.astype(np.float32), p_home=ph.astype(np.float32), hz=5)
        print(tag, "SC", round(float(np.mean([v["spell_f1_mean"] for v in res["test_sc"].values()])), 3),
              "macro", round(float(np.mean([v["macro_f1"] for v in res["test_sc"].values()])), 3), flush=True)
    res["test"]["clean"], _, _ = evaluate(kind, model, D[a.test], 0, team_of[a.test], clean=True)
    out = ROOT / "arsh-research" / "results" / "runs"
    (out / (f"lockbox_{a.tag or kind + '_real'}.json" if a.final else f"{tag}.json")).write_text(json.dumps(res, indent=1, default=float))
    print(tag, "DONE", {k: (round(v["spell_f1_mean"], 3), round(v["macro_f1"], 3)) for k, v in res["test"].items()}, flush=True)


if __name__ == "__main__":
    main()
