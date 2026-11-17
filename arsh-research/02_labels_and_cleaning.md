# 2. Loading, labels, cleaning and degradation

Everything here turns raw provider files into one common shape: a 5 Hz dictionary per match
`xy[T,P,2]` (players, NaN where missing), `team[P]` (0 home, 1 away), `ball[T,2]`, `y[T]` (0 H, 1 A, 2 D), `half[T]`, plus `y25`/`half25` copies at 25 Hz
for scoring. Code: `src/possession_gnn/data.py` (IDSSE), `data_sc.py` (SkillCorner), `degrade.py`.

## IDSSE (`data.py`)
1. **Parse the DFL position XML directly.** `kloppy` was avoided because its `only_alive` default drops dead-ball frames, which are exactly the D label.
   Referee tracks are excluded.
2. **One slot per (team, person)** across halves, so a player keeps the same column all match. Missing frames are NaN, never zero-filled.
3. **Orientation.** Each half is flipped so the home team attacks +x, determined from the home centroid in the first live frames. Why: the model
   should not need to learn which way each team plays, and team-swap augmentation (below) relies on a fixed convention.
4. **Labels.** D when `BallStatus` is 0 (ball dead), otherwise H or A from `BallPossession`. These are the DFL's own labels (`v1_raw`).
5. **Resample 25 Hz to 5 Hz** (every 5th frame). Why: a 20 s window is then 100 steps and training is fast. A check that moves predictions two
   frames late costs about 1%, so 5 Hz is not what limits the score. Labels at 25 Hz are kept so scoring is done at full rate.

### A cleaner label that was tried and not used
`labels.py::v2_control` re-labels short "blips" (a team holding the ball for under 2 s without a completed pass, a won tackle, or one player within
1.5 m for a second) to the surrounding team. It cuts IDSSE spells from about 450 to about 310 per match and raises the mean from 5-8 s to 8-12 s,
closer to the competition's 14.2 s. The final models were trained on `v1_raw` for IDSSE and on the SkillCorner labels, which already have the
right style, so the extra rule was left out of the final pipeline. The audit tables are in the git history (`reports/label_audit.md`).

## SkillCorner (`data_sc.py`)
1. Read `*_match.json` (teams, periods, home side), `*_tracking_extrapolated.jsonl` (players and ball at 10 fps) and `*_phases_of_play.csv`.
2. **Labels.** Each phase of play names the team in possession (H or A). Frames between phases are D.
3. **Ball** is NaN where `is_detected` is false. This is the real missing-ball pattern.
4. **Players** keep the provider's (partly extrapolated) positions.
5. **Orientation** from the declared home side per period so home attacks +x.
6. **Resample** 10 fps to 5 Hz (every 2nd frame). Keep the 32 most-present player slots. A `native` flag tells the pipeline not to add
   synthetic degradation.

## Synthetic degradation of IDSSE (`degrade.py`)
Purpose: make clean, complete IDSSE tracking look like BAR-ATH's automatic tracking, so the model learns to cope with what it will see.
- Ball missing in alternating blocks, about 41% of frames overall, gap lengths lognormal with a mean of 3.5 s.
- Each player independently missing about 7% of frames in gaps of mean 1.5 s.
- Gaussian position noise, standard deviation 0.3 m.
- Parameters are global, never indexed by frame, and were set from public aggregate facts about BAR-ATH (41% ball missing, about 20.5 tracked
  objects), not from hand-labelling anything.
- Two fixed copies (`deg_A` seed 101 and `deg_B` seed 202) are used for evaluation; training draws six random copies of each training match.
- Velocities and speeds are computed *after* degradation, from the degraded positions only, so nothing the model sees is cleaner than the test condition.

**Honest limitation.** A classifier can still tell degraded IDSSE from BAR-ATH windows almost perfectly (AUC 0.999), so the simulator is not a
faithful copy. This is why SkillCorner, with real gaps, was brought in, and why the final numbers are quoted on SkillCorner as well as IDSSE.
A simulator that passes that test is still open work.

## Checks on the data
- Home/away mapping verified on IDSSE: the majority `BallPossession` in the 3 s after kickoff agrees with the kickoff team in 19 of 19 kickoffs.
- Spell statistics of each source compared with the public aggregates (table in `01_data.md`).
- BAR-ATH canonical layer: 199,319 rows, 158,857 graded frames, ball seen on 94,214 of them. Asserted at load time.
- At prediction time BAR-ATH has 46 player slots; the 32 most present are kept because the model has 32 slots.
