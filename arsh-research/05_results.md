# 5. Results

All numbers are on matches the model never trained on. Nothing was tuned on the lockbox (J03WMX) or on BAR-ATH.

## What spellF1 is (`evaluation/possession_metrics.py`)
1. A spell is an unbroken run of one team's possession (H or A). Dead frames (D) end a spell.
2. For IoU thresholds 0.3, 0.4, 0.5, 0.6, 0.7, predicted and true spells are matched one-to-one. A pair needs the same team and IoU at or above the threshold.
3. Matched pairs are true positives, unmatched predictions false positives, unmatched true spells false negatives. F1 is computed per threshold.
4. spellF1 is the mean of the five. macroF1 is the frame-level mean of the H, A and D F1 scores.

A false D splits a spell in two, and a boundary one second late costs a lot, so spellF1 is far below frame-level scores (0.78 against 0.94).

## Headline
| Setting | spellF1 | macroF1 |
|---|---:|---:|
| 5-fold CV over all 20 SkillCorner matches, single model (honest estimate) | **0.778** | about 0.93 |
| Fold 0 only, 3-seed ensemble (easiest split) | 0.808 | 0.942 |

The target of 0.85 has **not** been reached. The honest figure is about 0.78 to 0.79.

## Submitted models (shared split, `splits.json`)
The submission does not use the fold models. The same recipe (h128, 7 blocks, 8000 steps, spell loss 0.5, 85%
SkillCorner) is trained five times with seeds 0-4 on one split shared with M1: test IDSSE J03WR9 and SkillCorner
1874553, validation IDSSE J03WQQ and SkillCorner 1886347 and 1899585, every other match for training
(`scripts/train_all.sh`; run JSONs `results/runs/cs*_J03WR9_s*.json`). Single seeds, mean (sd) over the five, scored
against this write-up's labels (provider phases for SkillCorner, the IDSSE labels for J03WR9):

| Test match | spellF1 | macroF1 |
|---|---:|---:|
| SkillCorner 1874553 | 0.804 (0.010) | 0.931 |
| IDSSE J03WR9, clean | 0.705 (0.018) | 0.945 |
| IDSSE J03WR9, degraded copy A | 0.683 (0.007) | 0.929 |
| IDSSE J03WR9, degraded copy B | 0.702 (0.013) | 0.938 |

Scored like M1 instead (same degraded input, M1's decoder, controlled-possession labels), the five-seed ensemble
reaches 0.796 on the two test matches, M1 0.789, and the submitted M1 + TCN ensemble 0.810, with the M1 weight (0.3)
chosen on the validation matches (`scripts/ensemble_weight.py`, `results/ensemble_weight.json`).

**Caveat.** These test matches are not fully independent of the TCN's design. SkillCorner 1874553 is one of the four
fold-0 test matches on which the steps below were compared, and J03WR9 was one of the IDSSE development matches of the
earlier stages. Treat the test numbers as optimistic; the 5-fold CV mean (0.778) remains the more honest estimate of
a single TCN.

## How the score developed (SkillCorner fold 0, 4 test matches)
| Step | spellF1 |
|---|---:|
| TCN h96, IDSSE + SkillCorner 50/50, 3000 steps | 0.741 |
| 85% SkillCorner | 0.753 |
| 100% SkillCorner | 0.684 |
| TCN h128, 7 blocks, 6000 steps | 0.779 |
| Hybrid TCN + attention, h96 | 0.773 |
| 10,000 steps, turnover weight 1.0 or 0.3 | 0.783 |
| 8000 steps, CV recipe | 0.790 |
| + spell loss 0.5 | 0.803 |
| spell loss 1.5 | 0.791 |
| 3-seed ensemble of the spell-loss model | 0.808 |

## Five-fold cross-validation (all 20 matches)
| Fold | Baseline | + spell loss (0.5) |
|---|---:|---:|
| 0 | 0.790 | 0.803 |
| 1 | 0.766 | 0.783 |
| 2 | 0.761 | 0.772 |
| 3 | 0.757 | 0.764 |
| 4 | 0.763 | 0.767 |
| mean | 0.767 | **0.778** |

The spell loss helps on every fold, by about 0.011 on average.

## Confusion matrix
Frame level, 25 Hz, fold-0 test (4 matches), model before the spell loss. Rows are truth.

|  | pred H | pred A | pred D | recall |
|---|---:|---:|---:|---:|
| true H | 123,905 | 6,450 | 6,425 | 0.906 |
| true A | 6,395 | 164,055 | 5,865 | 0.930 |
| true D | 3,630 | 4,645 | 273,795 | 0.971 |
| precision | 0.925 | 0.937 | 0.957 | accuracy 0.944 |

F1 per class: H 0.915, A 0.934, D 0.964. Errors are balanced: about 12.9k frames confuse H and A and about 11k mix alive and dead.

## Where the remaining error is (fold 0, 986 true spells)
- 72% matched at IoU 0.5 or more; 7% IoU 0.3-0.5; 6% weak overlap; 8% right time but wrong team; 7% predicted dead.
- 825 spells predicted against 986 true, about 20% too few: short possessions are merged (over-smoothing).
- Unmatched true spells are short (median 1.8 s against 7.4 s overall).
- Matched boundaries: median error 0.2-0.4 s, 90th percentile 1.6-1.9 s.
- The error budget on IDSSE says perfect dead/alive would add +0.19 and perfect boundaries +0.14.
- 5 Hz is not a limit (two frames late costs about 1%).
- The labels themselves are probably a limit: provider "phases of play" and the competition labels need not agree on where a possession starts and ends.

## What was tried and did not help
| Idea | Result |
|---|---|
| Bigger model (h128 vs h96), more steps (6000 to 10000) | +0.03 once, then flat at about 0.78 |
| Turnover loss weight 1.0 vs 0.3 | same score |
| Hybrid TCN + attention | 0.773, no better than the plain TCN |
| Mixed-architecture ensemble with decoder tuned on validation | 0.768, worse |
| Decoder switching-cost tuning | no gain |
| Self-training on confident pseudo-labels (3 rounds, fold 0) | 0.798, 0.798, 0.801 against the teacher's 0.803: no gain |
| Graph network (GNN) and a CRF transformer from the earlier S3/S4 stages | IDSSE dev means 0.532 and 0.211 against 0.593 for the TCN; dropped |
| Pure SkillCorner training | 0.684, worse than mixing with IDSSE |
| Per-frame possessor ID as an auxiliary target | available on only 19% of SkillCorner frames, dropped |
| SoccerNet-GAR | gated by an NDA and has event labels, not possession |

## Older reference points (IDSSE leave-one-match-out, degraded copy A)
Nearest-player baseline with smoothing: 0.25 to 0.30. TCN: 0.593 mean over dev matches. Lockbox J03WMX, first touch each: TCN 0.504, GNN 0.449
(`results/lockbox_log.md`).

## Directions most likely to help
1. **More real labelled tracking data.** PFF FC World Cup 2022 (request pending at `fchelp@pff.com`) is the best next source.
2. **Labels that match the competition.** Re-labelling provider data to the competition's definition of a possession is where most of the remaining gap probably sits.
3. **A better degradation simulator.** A classifier still separates degraded IDSSE from BAR-ATH at AUC 0.999, so some feature of the real data is not reproduced.
4. **Fix merging of short spells.** Segment-level objectives (the spell loss) helped, and a dedicated start/end boundary head is the natural next step.
5. **A stage-2 refiner on out-of-fold probabilities.** Designed but never run.

Run-by-run JSON files, including per-match confusion matrices, are in `results/runs/`.
