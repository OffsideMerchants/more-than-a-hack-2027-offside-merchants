# 6. Reproduce

Environment: Python 3.12, `.venv`, dependencies in `requirements.txt` / `uv.lock`. A CUDA GPU with 6 GB is enough; the whole run is about 3 hours.
Run from the repository root with `PYTHONPATH=src`.

## 0. Data
`python scripts/get_open_data.py` downloads everything below (several GB; files already present are skipped; `--dry-run` only checks the sources):
- IDSSE: the 7 matches of huggingface.co/datasets/pysport/idsse-data into `idsse-data/` (gitignored, not in the repo).
- SkillCorner: github.com/SkillCorner/opendata (MIT) into `artifacts/external/skillcorner/data/`, linked into `m1/data/skillcorner/` for M1.
- Metrica sample games 1-3 into `m1/data/metrica/` (M1 only).
- BAR-ATH: the organisers' bundle in `data/` or `More Than A Hack 2027 - Initial Challenge - Dataset/`.

## 1. Build caches (5 Hz arrays)
```
python scripts/build_idsse_cache.py          # IDSSE -> artifacts/idsse/*.npz
# SkillCorner caches are built on first use (artifacts/skillcorner/)
```

## 2. Train the submitted models (about 16 minutes per seed on an RTX 2060 SUPER)
```
bash scripts/train_all.sh --fresh             # five seeds cs0..cs4 on splits.json, then M1 (m1/README.md); exports models/possession_ensemble.pt
python scripts/ensemble_weight.py             # M1 weight of the ensemble, chosen on the validation matches
```
Each seed trains on the 17 SkillCorner and 5 IDSSE matches outside `splits.json`'s test and validation sets, picks its
checkpoint on the validation matches, and is scored on the test matches (run JSON in `arsh-research/results/runs/`,
weights in `artifacts/models/`). The TCN alone: `python -m possession_gnn.run_s2s4 --split shared --model tcn ...`
with the arguments in `scripts/train_all.sh`.

The five-fold cross-validation behind `05_results.md` (fold k: 15 SkillCorner matches plus the degraded IDSSE matches
for training, one of each for validation, 4 SkillCorner matches for test) is `bash scripts/run_cv.sh cs --lam_spell 0.5`.
Extra seeds for the ensemble check: `bash scripts/run_seeds.sh`, then
`python scripts/ensemble_eval.py --tags cs0 cse1 cse2 --no_tune --out arsh-research/results/ensemble_seeds.json`.

## 3. Optional analysis
```
python scripts/spell_errors.py cs0            # spell-level error taxonomy
python scripts/selftrain.py --folds 0 --rounds 3   # self-training experiment (no gain)
```

## 4. Bundle the model
```
python scripts/export_model.py cs0 cs1 cs2 cs3 cs4   # artifacts/models/cs*.pt -> models/possession_ensemble.pt (train_all.sh does this)
```

## 5. Predict BAR-ATH and check the file
```
python scripts/predict_barath.py --out artifacts/barath/pred_ensemble.csv
python evaluation/validate_submission.py artifacts/barath/pred_ensemble.csv
```
The script prints only counts and PASS/FAIL and writes the CSV under `artifacts/` (not opened). To score the public 5 minutes once:
copy the file into an empty folder and run `python evaluation/run_evaluation.py --results <folder>`.

## 6. Tests
```
PYTHONPATH=src python -m pytest tests          # 38 pass, 4 skipped without the optional data
```

## Layout
```
src/possession/            canonical layer (loading, orientation), TCN model and losses, guard, labels
src/possession_gnn/        5 Hz loaders (IDSSE, SkillCorner), degradation, training/eval runner, decoder bridge, diagnostics
scripts/                   cache builder, CV, seeds, self-training, ensemble eval, error taxonomy, export, BAR-ATH prediction
models/                    final ensemble weights
arsh-research/             this write-up and all result files
evaluation/, starter/      organisers' scorer and starter code (unchanged)
```
