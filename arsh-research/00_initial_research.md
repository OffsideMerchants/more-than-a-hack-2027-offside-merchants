# Research notes: frame-level possession labeling

## Conclusion

Build it as a **two-stage pipeline on tactical data only**, because that is the split the one
published method on this exact task (dead/home/away from tracking) found most reliable, and because
the ranking metric scores possession segments, not frames.

1. **Stage 1, alive vs dead.** Decide `D` first and separately. Ball out of the projected pitch
   bounds (via homography), ball at rest for a sustained window, and the `SET PIECES` tags give
   three mostly independent signals. This is the easier half of the problem and isolating it keeps
   dead-ball noise out of the owner decision.
2. **Stage 2, home vs away on the alive frames.** Assign the owner from a player-ball control score
   (sphere-of-influence style, Link and Hoernig), not from raw nearest-player-to-ball. Use a
   pitch-control surface as the fallback owner prior on the frames where the ball is missing.
3. **Handle the airborne ball explicitly.** Segment the ball track into touch and flight intervals,
   assign one owner per interval, and hold the owner constant through flight. This is what stops a
   long pass or cross from flipping the label mid-air, which is the single failure the challenge
   definition calls out.
4. **Fill the missing ball by physics, not by straight lines.** Constant-acceleration Kalman filter
   over short gaps; abstain (`X`) through long occlusions that sit inside one possession, since the
   ranking metric forward-fills an `X` and that is free mid-possession but costly across a handover.
5. **Decode the sequence, do not smooth it.** A Viterbi or HMM pass with a minimum-possession
   duration turns frame scores into clean segments. The metric is temporal-IoU over segments, so a
   segmentation objective beats a majority-vote window, which is a frame objective.

Starting point for the score to beat is the published floor, `baseline_smooth50` at spellF1 0.2433.

**Two decisions I changed after a second pass, both noted below:** pitch control is the fallback and
not the base, and a learned model (TCN) is an upgrade path rather than the first build.

---

## What I reconsidered

My first draft put pitch control at the center and treated the problem as mostly geometric. Two
things pushed against that after reading the directly-on-task literature.

- **Pitch control is the wrong primary tool for possession.** Spearman and Fernandez build it to
  value space and off-ball movement, not to answer who is controlling the ball right now. The
  method that production systems actually use for possession from tracking is player-ball control
  (Link and Hoernig [3]): how long the ball stays inside a player's sphere of influence, from
  player-ball distance, relative direction, speed and ball acceleration. That is the base signal.
  Pitch control earns its place only as the team-level fallback when the ball itself is missing,
  which is 41% of graded frames, so it still matters, just not as the main decision.
- **There is a published method for this precise task, and it is two-stage.** Borghesi et al. [2]
  classify every frame as dead, home or away straight from tracking, and report that splitting it
  into dead-vs-alive first and owner second beats a single three-way decision. That is a free
  structural win and it is why the pipeline above is staged that way rather than scoring `H`/`A`/`D`
  in one shot.

---

## Why, point by point

### Why tactical only, broadcast dropped

- Tactical is the fixed wide camera: 199,319 frames, 100% usable homography, every frame a normal
  moment of play, and row *i* of the submission is frame *i* of this video. Broadcast is on its own
  timeline, with cuts and replays, 65.8% usable homography, and no mapping to tactical is provided.
- Using broadcast means first solving two unsolved problems: off-screen-player imputation and
  tactical-to-broadcast frame alignment. Choi [1] measures the first cost directly: ignoring the
  6 to 12 players a broadcast camera hides inflates hidden-zone pitch-control error by 25 to 27
  points, and the best training-free fix only halves that. Alignment is on top of that and has no
  provided answer. Not worth it for a first working system.

### Why not nearest-player-to-ball

The challenge definition states that while the ball is contested or loose, possession stays with
the team that last had it. During a long pass or a cross the ball is airborne and physically
closest to the receiving team's defenders, so a nearest-player rule flips the label in mid-flight,
which is the exact case the rules say to get right. This is also why the four committed baselines,
which are nearest-player plus a smoothing window, top out at spellF1 0.2433.

### Why player-ball control, with a duel rule

Link and Hoernig [3] model individual ball control with a Bayesian network over the spatio-temporal
configuration of the controlling player, the ball and the nearest opponents, validated on 60
Bundesliga matches (F around 0.88 for ball actions, 0.83 for possession). Vidal-Codina et al. [4]
add the practical refinement that matters here: a possession zone around each player, plus a larger
duel zone around the ball where two or more opponents contest it, and the duel case takes
precedence. That duel rule is how you keep a contested ball from oscillating between teams frame to
frame.

### Why segment the ball track into touches and flight

A kicked ball moves close to ballistically between touches: roughly constant deceleration on the
ground, a parabola in the air, until the next touch changes its velocity sharply. Detecting those
velocity and acceleration discontinuities gives touch and flight intervals. Assigning one owner per
interval, and holding it through the flight, directly removes the mid-air label flip above. This is
the same discontinuity logic the event-detection work [4] uses to find on-ball actions, applied
here only to stabilise the owner label rather than to name events.

### Why Kalman filtering for the missing ball, and `X` for the rest

The ball is the most informative signal and the least reliable one, missing on 41% of graded
frames, worst exactly in crowded boxes and aerial duels. Straight-line interpolation is acceptable
over very short gaps but cuts straight through the arc of a long ball, which would corrupt the
touch detector. A constant-velocity or constant-acceleration Kalman filter keeps a physically
plausible curve. For the gaps too long to trust, abstain: the scorer forward-fills an `X` with the
previous label, so going quiet inside a possession costs nothing while a wrong guess there can
split one possession into two. Choose the interpolation-versus-abstain cutoff from the actual
gap-length distribution in the data, not from a guess.

### Why a sequence decoder, not a smoothing window

The evaluation doc shows the two headline metrics disagreeing across a four-window sweep of one
baseline: macroF1 moves 0.04 and prefers `smooth25`, spellF1 moves fivefold and prefers `smooth50`.
spellF1 is temporal-IoU segment matching averaged over five overlap thresholds, so it scores where
possessions start and stop. A majority-vote window optimises per-frame agreement, which is the
wrong target. A Viterbi or HMM decode with a transition cost and a minimum-segment-duration prior
optimises the boundaries, which is what is scored. Borghesi [2] reaches the same conclusion from the
model side: their best variant adds self-attention over the sequence rather than classifying frames
independently.

### Why the public 5 minutes is only a check

The public window is 5 minutes and 17 possessions, and the docs prove it cannot rank: the smoothing
window that wins there loses on the full match, and one missed possession moves its spellF1 by about
six points. So it is used only to confirm the pipeline is correct, frames aligned and labels meaning
what they should. It is never training data and never a tuning set, which is a rule, not a
preference. Any threshold or prior is set on other games or on held-out synthetic sequences, then
checked once here.

---

## Learned model as an upgrade, not the first build

Borghesi et al. [2] beat rule-based possession by about 7% with a TCN that embeds each object's
trajectory and combines them with permutation-invariant layers. It is the strongest reported result
on this task, so it is the obvious ceiling to aim for. It is the second build and not the first for
one reason: training labels. The provided game cannot be hand-labeled or tuned on, so a learned
model has to be trained on other games' tracking with possession labels, which is allowed but
introduces domain shift and a data-sourcing step. The geometry-plus-rules pipeline above needs no
labels at all and is fully reproducible, so it ships first and sets the baseline the TCN has to
beat before it replaces anything.

---

## Eval-aware details worth locking early

- The second half is graded from frame 110,808 but does not kick off until 113,742. Those 2,934
  frames are graded and are all `D`. Set them directly.
- Every output is exactly 199,319 rows. Ungraded frames (before kickoff, halftime, after the final
  whistle) can be `X` at no cost. A short file is the most common way to score zero.
- `Team` is `N/A` on all 36 throw-ins in the tagging, so a set piece that restarts possession often
  will not tell you which team restarts. Do not treat the tag as the owner; use it as the dead-ball
  boundary and recover the owner from the control signal on the first alive frames after it.
- Smart Tagging intervals are phases, not possessions, and are automatic, so copying an interval
  across as a label imports its errors. Use it as a prior, cross-checked, never as the answer.

---

## Where to spend first

1. Load tactical ATD and homography, confirm 199,319 rows and the ball-missing rate near 41% on
   graded frames, and plot the gap-length distribution. Everything else depends on that number.
2. Build and check Stage 1 (dead-ball) alone against the public window first. It is the separable,
   high-value half and it is where the fixed `D` stretches above already give you free correct
   frames.
3. Add the control signal and the touch/flight segmentation for Stage 2.
4. Add the Kalman fill and the abstain policy.
5. Add the Viterbi decode last, and tune its duration prior and transition cost on other games
   only. Validate once on the public window with `--bootstrap 1000` and confirm intervals overlap
   before locking anything.

---

## Measured: which learned model wins (IDSSE, leave-one-match-out)

Updated 2026-10-31 after running S2 (TCN), S3 (GNN, plus a pooled-set control) and S4 (possession transformer)
on the same protocol: six IDSSE development folds, 5 Hz, real ball with 20% ball-blind windows, two fixed
degraded copies (`deg_A`, `deg_B`), the same Viterbi decoder and the same scorer. S1 was not run.

| model | params | spellF1 deg_A | spellF1 deg_B | spellF1 clean | macroF1 deg_A | alive/dead F1 | team acc |
|---|---|---|---|---|---|---|---|
| **S2 TCN** | 223k | **0.593** | **0.605** | **0.657** | 0.904 | 0.976 | 0.895 |
| S3 GNN | 411k | 0.532 | 0.535 | 0.500 | 0.879 | 0.961 | 0.879 |
| S3 pooled-set control | 418k | 0.394 | 0.380 | 0.093 | 0.815 | 0.915 | 0.848 |
| S4 transformer | 913k | 0.211 | 0.213 | 0.189 | 0.696 | 0.900 | 0.666 |

Lockbox (J03WMX, trained on all six dev matches, scored once per model): TCN 0.526 / 0.535 on
`deg_A` / `deg_B` (bootstrap 5-95% on `deg_A` 0.474 to 0.563), GNN 0.488 / 0.467 (0.444 to 0.532),
pooled 0.329 / 0.338, S4 0.242 / 0.251. The TCN-over-GNN margin on the lockbox is smaller than the dev margin
and the intervals overlap, so treat the TCN lead there as likely, not proven.

**Result: the TCN is best.** It beats the GNN on all 6 folds on both degraded copies (paired gap
+0.045 to +0.080 on `deg_A`, mean +0.061), with fewer parameters, and it is the only model that scores
better on clean input than degraded. The GNN is second and beats the pooled control on every fold
(+0.10 to +0.18). S4 is far behind on every fold.

Caveats, stated plainly:
- **S4 is undertrained, not shown to be worse in principle.** It got 1,200 steps, 60 s windows, batch 4, the
  CRF, spell and boundary losses only (no auxiliary heads, no hard-negative labels, no pretraining), against 2,000
  steps for the TCN. Its low team accuracy (0.67) says it has not yet learned who owns the ball. Treat this as
  "not competitive at this compute budget", and do not drop the architecture on it alone.
- Single seed per fold, and the fold-to-fold spread is about 0.04, so only gaps above that are firm. The TCN-GNN gap
  clears it on 4 of 6 folds.
- Absolute numbers do not transfer to BAR-ATH (floor 0.2433): IDSSE `v1_raw` spells are about half as long, the
  simulator is uncalibrated against it, and nothing here touched BAR-ATH.
- The models share the 5 Hz stand-in data layer from S3, not the S0 canonical 25 Hz arrays, and the TCN's receptive field is
  about 12 s at that rate. Selection used the validation match only.

Practical reading for the pipeline above: the learned upgrade path is worth taking, and the TCN is the one to build on.
It is cheaper than the GNN and ahead of it. Next steps are more S4 training or a larger window sweep only if time allows,
S5 stacking of TCN and GNN out-of-fold predictions (saved in `artifacts/preds/`), and the BAR-ATH-side checks
under the compliance rules. Per-fold JSON is in `reports/gnn/runs/`; runner is `src/possession_gnn/run_s2s4.py`.

---

## Sources

[1] Choi, S. (2026). *Training-Free Off-Screen Player Imputation for Broadcast-Based Spatial
Football Analytics.* arXiv:2607.11548.

[2] Borghesi, M., Fiandrino, G., Cavagnino, D. et al. (2023). *Using Temporal Convolutional Networks
to estimate ball possession in soccer games.* Expert Systems with Applications 223, 119780.
doi:10.1016/j.eswa.2023.119780.

[3] Link, D. & Hoernig, M. (2017). *Individual ball possession in soccer.* PLOS ONE 12(7): e0179953.

[4] Vidal-Codina, F., Evans, N., El Fakir, B. & Billingham, J. (2022). *Automatic event detection in
football using tracking data.* Sports Engineering 25, 1. arXiv:2202.00804.

[5] Spearman, W. (2018). *Beyond Expected Goals.* MIT Sloan Sports Analytics Conference.

[6] Fernandez, J. & Bornn, L. (2018). *Wide Open Spaces: A statistical technique for measuring space
creation in professional soccer.* MIT Sloan Sports Analytics Conference.

[7] This repository: `CHALLENGE.md`, `DATA.md`, `EVALUATION.md`, `FAQ.md`, `RULES.md`, `OVERVIEW.md`,
and `evaluation/possession_metrics.py` for the metric definitions.
