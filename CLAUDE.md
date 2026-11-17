# Project rules for Claude sessions

A Claude Code session is a cloud-hosted LLM. Competition terms forbid making match data or extracts available to one.

- **Green (ok):** repo docs/code, the public 5-minute GT, `baseline_*.csv`, all IDSSE data (`idsse-data/`), papers.
- **Amber (only with the organisers' written confirmation):** numbers computed locally from BAR-ATH (histograms, AUCs, counts).
- **Red (never):** BAR-ATH rows, per-frame/per-spell listings, tracking/homography/Smart Tagging extracts, `df.head()`,
  plots of match data, any BAR-ATH prediction file.

Rules:
- Never Read/cat/head/grep anything under `More Than A Hack 2027 - Initial Challenge - Dataset/`, `data/`, or BAR-ATH prediction CSVs.
- Scripts touching BAR-ATH print only row counts, schema assertions, PASS/FAIL, distances. Everything else goes to `artifacts/` (gitignored); do not open it.
- Develop and debug on IDSSE. If something fails only on BAR-ATH, add an assertion that prints PASS/FAIL.
- Exceptions on BAR-ATH paths print types/shapes, never values (wrap entry points with `possession.guard.quiet`).
- Public 5 minutes (frames 11185-18684): correctness check only, at most once per gate. No tuning.
- No hand labels or frame-specific constants on BAR-ATH. Never feed Z, roles or IDs to a model.
- Do not push or publish without explicit instruction.

Env: `.venv` (Python 3.12). Run tests: `PYTHONPATH=src .venv/Scripts/python.exe -m pytest tests`.
