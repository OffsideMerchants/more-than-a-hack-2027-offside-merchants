"""Self-training rounds on the unlabelled SkillCorner test matches of each fold (teacher preds only, no test labels).

  python scripts/selftrain.py --folds 0 --rounds 3      resumable; writes arsh-research/results/selftrain.json
Round r trains a student on the labelled pool plus confident pseudo-labels from the round r-1 model (round 0 = cs<k>).
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IDS = ["J03WN1", "J03WOH", "J03WOY", "J03WPY", "J03WQQ"]
VAL = ["J03WOH", "J03WOY", "J03WPY", "J03WQQ", "J03WR9"]
OUT = ROOT / "arsh-research" / "results" / "selftrain.json"


def sc_score(tag, test):
    r = json.loads((ROOT / "arsh-research" / "results" / "runs" / f"{tag}_{test}_s{tag_seed(tag)}.json").read_text())["test_sc"]
    return sum(v["spell_f1_mean"] for v in r.values()) / len(r), sum(v["macro_f1"] for v in r.values()) / len(r)


def tag_seed(tag):
    return int(tag.split("_")[-1]) if tag.startswith("st") else int(tag[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, nargs="+", default=[0])
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--pfrac", type=float, default=0.3)
    ap.add_argument("--pconf", type=float, default=0.9)
    a = ap.parse_args()
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    for k in a.folds:
        teacher = f"cs{k}"
        res.setdefault(str(k), {})["round0"] = dict(zip(("spell", "macro"), sc_score(teacher, IDS[k])))
        tdir = ROOT / "artifacts" / "preds" / f"m_cs{k}" / f"cs{k}_{IDS[k]}_s{k}" / "sc"
        for r in range(1, a.rounds + 1):
            tag = f"st{r}_{k}"
            js = ROOT / "arsh-research" / "results" / "runs" / f"{tag}_{IDS[k]}_s{k}.json"
            if not js.exists():
                subprocess.run([str(ROOT / ".venv/Scripts/python.exe"), "-m", "possession_gnn.run_s2s4", "--model", "tcn",
                                "--hidden", "128", "--blocks", "7", "--steps", str(a.steps), "--lam_turn", "1.0", "--lam_spell", "0.5",
                                "--sc", "--sc_frac", "0.85", "--draws", "6", "--sc_fold", str(k), "--test", IDS[k], "--val", VAL[k],
                                "--seed", str(k), "--pseudo", str(tdir), "--pfrac", str(a.pfrac), "--pconf", str(a.pconf), "--tag", tag],
                               cwd=ROOT, env={**__import__("os").environ, "PYTHONPATH": "src"}, check=True)
            rr = json.loads(js.read_text())["test_sc"]
            res[str(k)][f"round{r}"] = {"spell": sum(v["spell_f1_mean"] for v in rr.values()) / len(rr),
                                       "macro": sum(v["macro_f1"] for v in rr.values()) / len(rr)}
            OUT.write_text(json.dumps(res, indent=1))
            print(k, r, res[str(k)][f"round{r}"], flush=True)
            tdir = ROOT / "artifacts" / "preds" / f"m_{tag}" / f"{tag}_{IDS[k]}_s{k}" / "sc"
            if res[str(k)][f"round{r}"]["spell"] < res[str(k)][f"round{r-1}"]["spell"] - 0.005:
                print("round got worse, stopping this fold", flush=True)
                break


if __name__ == "__main__":
    main()
