#!/bin/bash
# Retrain both models from scratch on the shared split (splits.json), with fixed seeds (reseeded every epoch) and the
# environment from uv.lock. GPU kernels are not forced to be deterministic, so a rerun gives close, not identical,
# weights. The TCN chain runs first, then the M1 chain (about 1.5 h + 3.5 h on an RTX 2060 SUPER). PARALLEL=1 runs
# them side by side, which needs more than 8 GB of GPU memory. Logs in artifacts/train_logs/.
#
#   bash scripts/train_all.sh           # continues where it stopped (M1 at epoch boundaries, TCN per seed)
#   bash scripts/train_all.sh --fresh   # deletes the outputs of an earlier run first
#   python scripts/ensemble_weight.py   # afterwards: M1 weight of the ensemble, chosen on the validation matches
#
# M1:  fine-tuning cache (CPU, only if missing) -> pretraining (10 epochs) -> fine-tuning (5 epochs) + HMM parameters
#      -> m1/runs/ssl_full_mae/best.pt, m1/runs/final_eval/M1_transformer.{pt,json}, m1/runs/bar_ath/decoder.npz
# TCN: five seeds (0-4) of the same model on the same split -> models/possession_ensemble.pt
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
[ -x "$PY" ] || PY=.venv/Scripts/python.exe
export PYTHONPATH=src
L=artifacts/train_logs
mkdir -p "$L"

if [ "${1:-}" = "--fresh" ]; then
  rm -f m1/runs/ssl_full_mae/{last.pt,last.prev.pt,log.jsonl,done,best.pt,args.json} \
        m1/runs/final_eval/M1_transformer.ckpt.pt artifacts/models/cs[0-4]_*.pt
fi

m1_chain() {
  [ -f m1/data/finetune_cache.pkl ] || $PY m1/scripts/finetune.py --name cache --make-cache --data-cache data/finetune_cache.pkl || return 1
  [ -f m1/runs/ssl_full_mae/done ] || $PY m1/scripts/pretrain.py --name ssl_full_mae --stride 5 --epochs 10 --resume || return 1
  $PY m1/scripts/finetune.py --name M1_transformer --ckpt runs/ssl_full_mae/best.pt --data-cache data/finetune_cache.pkl --resume
}

tcn_chain() {
  for k in 0 1 2 3 4; do
    ls artifacts/models/cs${k}_*_s${k}.pt >/dev/null 2>&1 && continue
    $PY -m possession_gnn.run_s2s4 --split shared --model tcn --hidden 128 --blocks 7 --steps 8000 --lam_turn 1.0 \
        --lam_spell 0.5 --sc --sc_frac 0.85 --draws 6 --seed $k --tag cs$k || return 1
  done
  $PY scripts/export_model.py cs0 cs1 cs2 cs3 cs4
}

echo "$(date '+%F %T') start (logs: $L/m1.log, $L/tcn.log)"
if [ "${PARALLEL:-0}" = 1 ]; then
  m1_chain > "$L/m1.log" 2>&1 &
  P1=$!
  tcn_chain > "$L/tcn.log" 2>&1 &
  P2=$!
  wait $P1; E1=$?
  echo "$(date '+%F %T') M1 chain exit $E1"
  wait $P2; E2=$?
  echo "$(date '+%F %T') TCN chain exit $E2"
else
  tcn_chain > "$L/tcn.log" 2>&1; E2=$?
  echo "$(date '+%F %T') TCN chain exit $E2"
  m1_chain > "$L/m1.log" 2>&1; E1=$?
  echo "$(date '+%F %T') M1 chain exit $E1"
fi
[ $E1 -eq 0 ] && [ $E2 -eq 0 ] && echo "ALL DONE" || { echo "FAILED"; exit 1; }
