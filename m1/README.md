# Possession model (M1): self-supervised pretraining + fine-tuning + HMM

Code for our BAR-ATH submission, self-contained in `m1/`. It labels every frame H / A / D from the
automatic tracking (ATD) alone. No BAR-ATH labels, no hand annotation and no BAR-ATH data are used for training, and none
are in this repository.

## Pipeline

1. **Pretraining without labels** (`scripts/pretrain.py`). A transformer (`footy/model.py`) learns
   football movement by reconstructing hidden parts of 16 s windows of player and ball positions at
   5 Hz (masked trajectory modelling) on ~52 h of open tracking data → `runs/ssl_full_mae/best.pt`.
2. **Fine-tuning for possession** (`scripts/finetune.py`) on labelled open matches: 7 IDSSE (DFL;
   positions degraded to look like automatic tracking, `footy/atd_sim.py`), 20 SkillCorner, 3 Metrica.
   Labels are converted to *controlled* possession (`footy/semantics.py`): a change of team counts
   only once the new team keeps the ball for 2 s. The submitted model is fine-tuned on 25 of these
   matches, with 5 held out → `runs/final_eval/M1_transformer.pt`.
3. **Decoding** (`scripts/predict.py`): probabilities from overlapping windows, averaged over four
   test-time transforms (teams swapped, pitch rotated), then an HMM (prior and transitions estimated
   on the training labels, written by `scripts/finetune.py`) turns them into possession spells. Boundaries
   are snapped to rule-based ball events (`footy/rules.py`). Frames outside the halves are X.

## Predict BAR-ATH (GPU about 1 minute, CPU about 10)

The submission combines M1 with the TCN models: `scripts/predict_ensemble.py`, see the
[repository README](../README.md). M1 alone, from the repository root in its environment (`uv sync --frozen`):

```bash
python m1/scripts/predict.py --data-dir "/path/to/dataset"    # the folder holding tactical/ (not in this repo)
# GPU if available (deterministic settings), else CPU
# -> evaluation/results/ssl_M1.csv, checked with evaluation/validate_submission.py
python evaluation/run_evaluation.py --bootstrap 1000       # public 5 minutes: pipeline check only
```

## Reproduce training

Open data, not in this repository, under `m1/data/` (git-ignored). `python scripts/get_open_data.py` (repository
root) downloads SkillCorner and Metrica into these folders; IDSSE is downloaded by kloppy itself on first use. The
training scripts run from `m1/` wherever they are started, so paths below are relative to `m1/`:

| source | where | path |
|---|---|---|
| IDSSE / DFL (7 matches) | downloaded by kloppy on first use (`kloppy.sportec.load_open_tracking_data`, from huggingface.co/datasets/pysport/idsse-data) | `data/kloppy_cache/` |
| SkillCorner (20 matches) | github.com/SkillCorner/opendata, via `scripts/get_open_data.py` (links to the TCN's copy in `artifacts/external/skillcorner/`) | `data/skillcorner/<id>/<id>_{match.json,tracking_extrapolated.jsonl,phases_of_play.csv}`, `data/skillcorner/matches.json` |
| Metrica sample games 1–3 | github.com/metrica-sports/sample-data, via `scripts/get_open_data.py` | `data/metrica/Sample_Game_*` (flat) |

`bash scripts/train_all.sh` (repository root) runs the commands below together with the TCN training. By hand:

```bash
python m1/scripts/finetune.py --name cache --make-cache --data-cache data/finetune_cache.pkl          # CPU
python m1/scripts/pretrain.py --name ssl_full_mae --stride 5 --epochs 10 --resume                    # GPU, about 2 h
python m1/scripts/finetune.py --name M1_transformer --ckpt runs/ssl_full_mae/best.pt --data-cache data/finetune_cache.pkl --resume   # about 1.3 h; submitted model + runs/bar_ath/decoder.npz
python m1/scripts/finetune.py --name M1_transformer_all --data-cache data/finetune_cache.pkl --resume --all-data  # optional: all 30 matches + decoder_all.npz
```

The held-out matches come from `splits.json` (shared with the TCN): they are excluded from pretraining, and
fine-tuning uses the remaining 25 matches. Seeds are fixed and re-applied every epoch, so a run resumed at an epoch
boundary draws the same random numbers, but GPU kernels are not bit-for-bit deterministic; the trained weights in
`runs/` reproduce the submitted CSV exactly.

## Held-out evaluation

`finetune.py` without `--all-data` fine-tunes on 25 matches, picks the HMM emission scale on 3
validation matches (IDSSE J03WQQ, SkillCorner 1886347 and 1899585) and scores 2 test matches never seen in pretraining
or fine-tuning (IDSSE J03WR9, SkillCorner 1874553) with the official scorer: **0.784 spell-F1** (J03WR9 0.786,
1874553 0.782; published floor 0.243), `runs/final_eval/M1_transformer.json`. This is the submitted model; with the
four test-time passes of `predict.py` it scores 0.789, and in the ensemble with the TCN 0.810 (see the
[repository README](../README.md)). `--all-data` trains the same recipe on all 30 matches; such a model has no
held-out score of its own (use it with
`predict.py --model runs/final_eval/M1_transformer_all.pt --decoder runs/bar_ath/decoder_all.npz`).
