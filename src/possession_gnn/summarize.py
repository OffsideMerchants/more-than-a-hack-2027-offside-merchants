"""Aggregate arsh-research/results/runs/*.json -> markdown tables (paired GNN vs pool per fold, per ball condition)."""
import json
import sys
from pathlib import Path

import numpy as np

RUNS = Path(__file__).resolve().parents[2] / "arsh-research" / "results" / "runs"


def load():
    out = {}
    for f in sorted(RUNS.glob("*.json")):
        r = json.loads(f.read_text())
        a = r["args"]
        out[(a["model"], a["ball"], a["test"], a["seed"], a.get("final", False))] = r
    return out


def table(runs, ball, cond="deg_A", final=False):
    folds = sorted({k[2] for k in runs if k[1] == ball and k[4] == final})
    lines = [f"| fold | model | params | spellF1 | macroF1 | alive/dead F1 | team acc (both alive) | spellF1 raw argmax |", "|---|---|---|---|---|---|---|---|"]
    acc = {"gnn": [], "pool": []}
    for t in folds:
        for m in ("gnn", "pool"):
            r = runs.get((m, ball, t, 0, final))
            if not r:
                continue
            x = r["test"][cond]
            acc[m].append((x["spell_f1_mean"], x["macro_f1"], x["alive_dead_f1"], x["team_acc_both_alive"]))
            lines.append(f"| {t} | {m} | {r['params']:,} | {x['spell_f1_mean']:.3f} | {x['macro_f1']:.3f} | "
                         f"{x['alive_dead_f1']:.3f} | {x['team_acc_both_alive']:.3f} | {x['raw_argmax_spell_f1']:.3f} |")
    for m, v in acc.items():
        if v:
            a = np.array(v)
            lines.append(f"| **mean** | {m} | | {a[:,0].mean():.3f} | {a[:,1].mean():.3f} | {a[:,2].mean():.3f} | {a[:,3].mean():.3f} | |")
    paired = [(runs[("gnn", ball, t, 0, final)]["test"][cond]["spell_f1_mean"] - runs[("pool", ball, t, 0, final)]["test"][cond]["spell_f1_mean"])
              for t in folds if ("gnn", ball, t, 0, final) in runs and ("pool", ball, t, 0, final) in runs]
    if paired:
        lines.append(f"\nPaired spellF1 gain (GNN - pool) over {len(paired)} folds: mean {np.mean(paired):+.3f}, "
                     f"per fold {', '.join(f'{p:+.3f}' for p in paired)}; wins {sum(p > 0 for p in paired)}/{len(paired)}.")
    return "\n".join(lines)


def gaps(runs, ball, cond="deg_A"):
    key = "gaps_vs_real_ball_gaps"
    buckets = ["visible", "<1s", "1-3s", "3-10s", ">10s"]
    lines = ["| bucket | model | owner acc (alive steps) | turnover recall +-2s | n alive steps | n turnovers |", "|---|---|---|---|---|---|"]
    for b in buckets:
        for m in ("gnn", "pool"):
            rs = [r for k, r in runs.items() if k[0] == m and k[1] == ball and not k[4]]
            n = sum(r["test"][cond][key][b]["n"] for r in rs)
            if not n:
                continue
            acc = sum(r["test"][cond][key][b]["n"] * (r["test"][cond][key][b]["owner_acc"] or 0) for r in rs) / n
            nt = sum(r["test"][cond][key][b]["turnovers"] for r in rs)
            hit = sum((r["test"][cond][key][b]["turnover_recall_2s"] or 0) * r["test"][cond][key][b]["turnovers"] for r in rs)
            lines.append(f"| {b} | {m} | {acc:.3f} | {hit / nt if nt else float('nan'):.3f} | {n} | {nt} |")
    return "\n".join(lines)


if __name__ == "__main__":
    runs = load()
    for ball in ("blind", "real"):
        if any(k[1] == ball for k in runs):
            print(f"\n### ball={ball}, deg_A\n\n{table(runs, ball)}\n\n### ball={ball}, deg_B\n\n{table(runs, ball, 'deg_B')}\n\n"
                  f"### ball={ball}: by real ball-gap bucket (deg_A, pooled over folds)\n\n{gaps(runs, ball)}")
