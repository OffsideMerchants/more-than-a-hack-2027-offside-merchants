#!/bin/bash
# extra seeds of the fold-0 spell-loss model (tags cse1, cse2) to measure a seed-ensemble gain
cd "$(dirname "$0")/.."
export PYTHONPATH=src
for S in 1 2; do
  .venv/Scripts/python.exe -m possession_gnn.run_s2s4 --model tcn --hidden 128 --blocks 7 --steps 8000 --lam_turn 1.0 --lam_spell 0.5 \
     --sc --sc_frac 0.85 --draws 6 --sc_fold 0 --test J03WN1 --val J03WOH --seed $S --tag cse$S
done
