#!/bin/bash
# 5-fold SkillCorner CV (test = every 5th match) + rotating IDSSE hold-out; one model per fold.
# usage: scripts/run_cv.sh PREFIX [extra args]   e.g. scripts/run_cv.sh cv --steps 8000
# The cross-validation behind arsh-research/05_results.md; the submitted models come from scripts/train_all.sh.
cd "$(dirname "$0")/.."
export PYTHONPATH=src
P=${PY:-.venv/bin/python}
[ -x "$P" ] || P=.venv/Scripts/python.exe
PRE=$1; shift
IDS=(J03WN1 J03WOH J03WOY J03WPY J03WQQ)
VAL=(J03WOH J03WOY J03WPY J03WQQ J03WR9)
for k in 0 1 2 3 4; do
  $P -m possession_gnn.run_s2s4 --model tcn --hidden 128 --blocks 7 --steps 8000 --lam_turn 1.0 --sc --sc_frac 0.85 --draws 6 \
     --sc_fold $k --test ${IDS[$k]} --val ${VAL[$k]} --seed $k --tag ${PRE}$k "$@"
done
