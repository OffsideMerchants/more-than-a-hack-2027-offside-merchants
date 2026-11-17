# arsh-research: how the possession model was built

This folder is the full write-up of the work on the "ball possession" challenge: where the data came from,
how it was cleaned and labelled, why each step exists, how the model works, how it was trained and checked,
and what the numbers are. The repository README belongs to the organisers and is left alone.

| File | Content |
|---|---|
| `00_initial_research.md` | Literature notes and the first design, written before any model was trained |
| `01_data.md` | Every data source: what it is, how it was gathered, licence, and what is deliberately not used |
| `02_labels_and_cleaning.md` | Loading, orientation, 5 Hz resampling, labels, degradation, and why each is needed |
| `03_model.md` | Model architecture, losses and the decoder |
| `04_training_and_validation.md` | Splits, training recipe, ensembling, and the rules that keep the scores honest |
| `05_results.md` | All results, the confusion matrix, error analysis, what failed, and directions |
| `06_reproduce.md` | Commands from raw data to the submission file |
| `results/` | Per-run JSON files, lockbox log, self-training and seed-ensemble outputs |

Final model: `models/possession_ensemble.pt` (five TCN seeds on the shared split `splits.json`, 20 MB; see `05_results.md`). Prediction script: `scripts/predict_barath.py`.
