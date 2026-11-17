# 4. Training and validation

Code: `src/possession_gnn/run_s2s4.py` (training and scoring), `scripts/run_cv.sh`, `scripts/selftrain.py`, `scripts/ensemble_eval.py`.

## Data splits (no leakage)
- **SkillCorner, 5-fold by match.** The 20 matches are sorted; fold k tests on matches `k, k+5, k+10, k+15` (4 matches), validates on the next match, and trains on the remaining 15.
  Every match is a test match exactly once. A split by match (never by frame) is used because neighbouring frames are nearly identical.
- **IDSSE, rotating.** The IDSSE test match changes with the fold; the validation match is the next one. The lockbox J03WMX is never in training or validation.
- **BAR-ATH** is never trained on. Its public window is scored once per gate for a correctness check and not used for any choice.

## Recipe
| Setting | Value |
|---|---|
| Steps | 8000, batch 16 windows of 100 steps (20 s) |
| Optimiser | AdamW, learning rate 2e-3 with one-cycle schedule, weight decay 1e-2, gradient clip 1.0, mixed precision |
| Data mix | each batch: 85% from SkillCorner, 15% from IDSSE (6 degraded copies per match) |
| Augmentation | team swap together with an x mirror; independent y flip; ball dropped for the whole window 20% of the time |
| Model selection | best validation loss, checked every 500 steps, on the IDSSE validation match plus the SkillCorner validation match |
| Time | about 30 minutes per model on a 6 GB laptop GPU |

Why these:
- **Team swap + mirror, y flip:** doubles the effective data and enforces the symmetries football really has.
- **Ball dropout 20%:** guarantees the model learns to answer without the ball, which it will have to do on 41% of BAR-ATH frames.
- **85% SkillCorner:** 50% gave 0.741, 85% gave 0.753, 100% gave 0.684. These comparisons were read on the fold-0 test matches, so fold 0 is slightly optimistic (see below).
- **Early stopping on two validation matches:** one from each source, so the stopping point suits both.

## How the scores are computed
`evaluation/possession_metrics.py` (the organisers' scorer, not reimplemented) at 25 Hz after upsampling. Halves are separated by an X block so they can never merge.
Each SkillCorner fold reports the mean over its 4 test matches, which were never used for training or validation. Fold 0 was also the place where design
choices were compared (rule 4 below).

## Rules that keep the numbers honest
1. Nothing is tuned on the lockbox match J03WMX. It is used a maximum of three times per model family, each logged in `results/lockbox_log.md`.
2. Nothing is tuned on BAR-ATH. The public 5 minutes were scored once for the final ensemble, and the result was not used to change anything.
3. No hand labels and no constants tied to specific frames of BAR-ATH.
4. Design choices (model size, steps, data mix, loss weights) were compared on fold 0 before the other folds were run. Fold 0 is therefore mildly optimistic, and the other four folds are the cleaner estimate. Decoder costs were never tuned on test data.
5. Self-training uses only the teacher's predictions on unlabelled test matches. The true labels of those matches are read only to score.
6. Fold 0 is the easiest split (0.79 vs 0.76 for the others), so claims are quoted over all five folds.

## Ensembling
- **Five seeds** (`cs0`..`cs4`, the same recipe trained on the 17 SkillCorner and 5 IDSSE training matches of `splits.json`) are averaged in logit space at inference. The fold models of the cross-validation were used only for the estimates in `05_results.md`.
- Measured on fold 0 with three seeds: single models 0.803, 0.806, 0.797; average 0.808. A same-architecture ensemble gives about +0.006.
  Averaging two *different* architectures (conv and hybrid attention) lost 0.01 and was not used.
