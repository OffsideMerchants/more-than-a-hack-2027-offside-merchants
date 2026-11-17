# 1. Data

## What the task needs
A per-frame label for one match (BAR-ATH): H (home in possession), A (away), D (dead ball), at 25 fps, 199,319 frames.
The scorer ranks on spellF1, which compares possession *segments*, not frames (see `05_results.md`).
Only a 5-minute window (frames 11185-18684, 17 possessions) has public ground truth. There are no labels for the rest of the match,
so a model has to be trained on other matches that do have labels.

## Sources used

| Source | Matches | Rate | Labels | Licence / access | Role |
|---|---:|---:|---|---|---|
| IDSSE (DFL Bundesliga, `idsse-data/`, pysport/idsse-data) | 7 | 25 Hz | `BallPossession` and `BallStatus` in the position XML | open, download separately into `idsse-data/` (gitignored) | training with synthetic degradation, leave-one-match-out checks, lockbox |
| SkillCorner open data (github.com/SkillCorner/opendata) | 20 A-League matches | 10 fps | provider "phases of play" CSV | MIT | training and held-out test, real tracking gaps |
| BAR-ATH (the challenge match) | 1 | 25 fps | public 5 min only | organiser terms | inference target; never trained on, never tuned on |

SkillCorner is a downloaded clone in `artifacts/external/skillcorner/` (gitignored). It must be listed in the technical document as an external dataset.

## Why SkillCorner was added
IDSSE tracking is complete (provider-corrected, ball almost always present). BAR-ATH's is automatic: the ball is missing on 41% of graded frames
and about 20 objects are tracked per frame. A model trained only on clean data fails on that. Two things were done:
1. IDSSE is degraded synthetically to look like BAR-ATH (`02_labels_and_cleaning.md`).
2. SkillCorner is added because its gaps are *real* (the ball is undetected on 35-47% of frames) and its possession definition is closer to the
   competition's. Spell statistics, definition-level only (nothing tuned on them):

| Source | Spells per 5 min | Mean spell | Median spell | Spells under 1 s |
|---|---:|---:|---:|---:|
| Public ground truth | 17 | 14.2 s | 7.6 s | n/a |
| SkillCorner | 12.9 | 12.4 s | 7.4 s | 1% |
| IDSSE | 22.6 | 7.3 s | 3.8 s | 10% |

A 50/50 mix of IDSSE and SkillCorner scored 0.741 on held-out SkillCorner matches, 85% SkillCorner 0.753, pure SkillCorner 0.684.
So the IDSSE data still helps.

## Sources considered and not used
- **SoccerNet-GAR** (OpenSportsLab, 64 World Cup 2022 matches): gated behind a request form and an NDA, and it labels *events* (pass, tackle) at
  30 fps, not possession. Not downloadable here and not redistributable, so it is not used.
- **Metrica sample games** (3 matches): downloaded, not used. Too small to move the score.
- **PFF FC World Cup 2022**: requires a request (`fchelp@pff.com`). Pending. It is the most promising next source.
- **Broadcast video and Smart Tagging of BAR-ATH**: dropped. Broadcast has cuts and no mapping to the tactical timeline.

## Rules that shape every script
BAR-ATH data may not be shown to a hosted LLM session. All scripts that touch it print only row counts, PASS/FAIL and aggregate scores, write
outputs under `artifacts/` (gitignored) and are wrapped in `possession.guard.quiet`, which replaces exception messages by type and location.
The lockbox match J03WMX (IDSSE) is scored a maximum of three times per model family; see `results/lockbox_log.md`.
Z (ball height), player roles and player IDs are never given to a model.
