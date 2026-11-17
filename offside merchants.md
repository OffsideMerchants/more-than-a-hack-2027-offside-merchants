# offside merchants: ball possession on BAR-ATH from automatic tracking

More than a Hack 2027, Phase 1. Team: offside merchants (Arsh Chowhan, Till Schaland, Johar Rizvi, Ibtsam sadiq, Ahmad Khalifeh).
Code: (https://github.com/OffsideMerchants/more-than-a-hack-2027-offside-merchants), branch `main`. Submission file: `offside merchants.csv`.

## 1. Summary

We label every frame of BAR-ATH as H, A or D from the automatic tracking (ATD) alone. Two models trained only on
open, labelled tracking data from other matches each give per-frame H/A/D probabilities; these are pooled and decoded
into possession spells by one hidden Markov model (HMM):

- **M1**, a transformer pretrained without labels on about 52 hours of open tracking (masked trajectory
  reconstruction), then fine-tuned to predict possession;
- **TCN**, a temporal convolutional network designed for missing balls and home/away symmetry, five seeds averaged.

On two held-out open matches the ensemble reaches spellF1 0.810 (M1 alone 0.789, TCN alone 0.796), with the M1
weight chosen on three separate validation matches. That figure is optimistic for reasons given in section 6.
No BAR-ATH labels, no hand annotation and no other human-made data about BAR-ATH are used anywhere.

## 2. Approach, and why

**The problem as we read it.** The scorer ranks on spellF1, which matches whole possession spells by IoU, so a
system has to get the *segments* right: one flickering frame splits a spell, and a boundary a second late costs a lot.
The input is hard: on the graded frames the ball is missing 41% of the time, players are tracked anonymously and in
fragments, and only five minutes of BAR-ATH have public labels, which the rules reserve for pipeline checks.
So the models have to learn possession on other matches whose tracking can be made to look like BAR-ATH's, and the
decoding has to produce clean spells.

**Why learned models rather than rules.** A rule-based baseline (nearest player to the ball, hold rules when the ball
is missing, dead-ball rules from ball position and player speeds; `m1/footy/rules.py`) is part of the pipeline, but
only to place boundaries precisely (below). On its own it cannot bridge long ball gaps: the context that tells you who
has the ball while it is invisible is the movement of all 22 players over several seconds, which is what both models
learn.

**M1: pretrained transformer** (`m1/`). Each (frame, agent) pair is a token; blocks alternate attention over each
agent's trajectory and across all agents in a frame. Agents carry only a home/away/ball type, no identity, so the model
is permutation-equivariant within a team. Pretraining hides parts of 16 s windows at 5 Hz (random tokens, whole
agents, the future, the whole ball, ball gaps shaped like the ATD's) and reconstructs them, on IDSSE (clean and two
simulated-ATD versions), SkillCorner and Metrica tracking. Possession is learned in a second stage: a head reads the
ball token and pooled home and away features per frame. Why: labelled tracking is scarce (30 matches), unlabelled
tracking is plentiful, and reconstructing a hidden ball from player movement is close to the skill the task needs.

**TCN** (`src/possession/`, `src/possession_gnn/`). The label is split into two questions, *is the ball alive?* and
*which team has it?*, each with its own head, plus an auxiliary head for possession changes. Players pass through a
shared dilated-convolution encoder (order-invariant team pooling), the ball through its own encoder with gap features
(time since last seen, time to next seen), and a joint stack of seven dilated blocks sees several seconds each way.
Every window is encoded from both teams' points of view, so swapping the teams exactly swaps the answer. A spell
loss (1 - soft IoU per true spell) targets the metric's main failure, merging short possessions. Why a second model:
it is built differently (convolutions, explicit gap features, two-question output) and trained with different labels
and losses, so its errors are only partly shared with M1's.

**Ensemble and decoding** (`scripts/predict_ensemble.py`). Per 5 Hz frame, M1's probabilities (averaged over four
test-time passes: teams swapped and/or pitch rotated) and the TCN's (five seeds averaged in logit space, turned into
H = p_alive p_home, A = p_alive (1 - p_home), D = 1 - p_alive) are pooled log-linearly, p ∝ p_M1^0.3 p_TCN^0.7. An HMM
decodes the pooled probabilities into a state sequence: class prior and transitions estimated on the training labels,
home <-> away changes made 16 times more likely (beta), emissions scaled by 0.8 because window-averaged probabilities
are strongly correlated in time. The 5 Hz path is upsampled to 25 Hz and each boundary is snapped to a transition of
the rule-based baseline with the same new label within ±4 frames. Frames outside the graded halves are X.

## 3. Assets: used and dropped

| asset | use |
|---|---|
| ATD tracking (`tactical/BAR-ATH_tactical_atd.{xml,txt}`) | **used**: the only model input. Player and ball x/y; no ball height, roles or IDs are given to any model. Tracks are cut into segments and assigned to 14 + 14 slots (`m1/footy/atd.py`); for the TCN, each half is oriented so home attacks +x from the goalkeepers' positions (`src/possession/canonical.py`). |
| public 5-minute ground truth | **pipeline check only**, never for training, tuning or model choice |
| graded-frame blocks, frame alignment | published constants from DATA.md / `games.py` |
| broadcast video, tactical video | **dropped**: the broadcast has cuts and no mapping to the tactical timeline; pixel models were out of scope for the time available |
| Smart Tagging | **not used**: we kept both models to the tracking alone, so that no tagging of the provided game enters training, validation or prediction |
| homography | **not needed**: the ATD is already in pitch coordinates |

## 4. Training data

None of it comes from the provided game. All sources are public, and download instructions are in the repository.

| dataset | matches | labels: source and who made them | how we use the labels |
|---|---:|---|---|
| IDSSE (DFL Bundesliga, pysport/idsse-data; Bassek et al., Scientific Data 2025; CC BY 4.0) | 7 (5 train, 1 val, 1 test) | DFL/Sportec's official per-frame `BallPossession` and `BallStatus` flags, produced by the league's data provider | D where the ball is dead, else the possessing team. Tracking degraded to look like ATD: ball and players removed in situation-dependent gaps, fragmented anonymous tracks, noise (`m1/footy/atd_sim.py`, `src/possession_gnn/degrade.py`); targets from DATA.md's published aggregates, not from labels |
| SkillCorner open data (github.com/SkillCorner/opendata; MIT) | 20 A-League (17 train, 2 val, 1 test) | SkillCorner's "phases of play", produced by SkillCorner from broadcast tracking | team in possession during a phase is H/A, frames between phases D. Real tracking gaps (ball undetected on 35-47% of frames) |
| Metrica Sports sample data (github.com/metrica-sports/sample-data) | 3 (M1 training only) | event data tagged by Metrica Sports | our rules turn events into frame labels: possession from each on-ball event (pass, recovery, shot, carry, set piece, ball lost/out) to the next; dead from a stoppage (ball out, foul, goal, offside, end of half) to the restart set piece; ±10 frames around each transition left unlabelled because event times are imprecise (`m1/footy/labels.py`) |

Total: 30 labelled matches; about 52 h of tracking for M1's pretraining, which uses no labels. One split is shared
by both models (`splits.json`): test IDSSE J03WR9 and SkillCorner 1874553 (never trained on), validation IDSSE
J03WQQ and SkillCorner 1886347, 1899585 (model selection, HMM scale, ensemble weight), everything else for training.
Test and validation matches are also excluded from pretraining.

## 5. Possession rules

The challenge asks for *controlled* possession. We fixed one rule from the challenge text before scoring any model
and applied it to all training labels for M1 and the decoder (`m1/footy/semantics.py`):

- **Change of possession needs control.** Within a stretch of live play, the possessing team only changes when the
  other team's run lasts at least **2 s**. Shorter runs (a touch without control, a deflection, a challenge won
  without keeping the ball) are absorbed into the current possession.
- **Loose and contested balls** stay with the last team in control.
- **Restarts.** After a dead ball the first possessor is kept whatever the length of its run: a restart always has a
  clear possessor.
- **Dead ball** comes from the providers' ball-status information (DFL `BallStatus`; frames outside SkillCorner's
  phases; Metrica stoppage-to-restart). In the rule-based baseline used for snapping, a missing ball on its own never
  means dead ball; it also needs slow players.

At prediction time the same choices are enforced by the decoder: the HMM's transition costs suppress short spells and
flicker, and boundaries snap to the rule-based baseline, which switches team only after sustained control and keeps
the last team while the ball is in flight or briefly missing (`m1/footy/rules.py`). An alternative relabelling that
looks for evidence of control (completed pass, won tackle, a player within 1.5 m for 1 s; `src/possession/labels.py`)
was built and not used in the final pipeline. <!-- TODO team: if you have the held-out comparison of the 2 s rule
against the evidence labels, add the two numbers here; argue only from held-out open matches, not the public window. -->

## 6. Results on open data

Official scorer, controlled-possession labels, both models on the same degraded input (`scripts/ensemble_weight.py`):

| | validation spellF1 (3 matches) | test spellF1 (2 matches) |
|---|---:|---:|
| M1 alone | 0.761 | 0.789 |
| TCN alone (5 seeds) | 0.769 | 0.796 |
| **ensemble, M1 weight 0.3** | **0.786** | **0.810** |

Every M1 weight from 0.1 to 0.9 beats both single models on validation; 0.3 is the best point of a fairly flat
plateau. The test matches were scored once, after the weight was fixed. Frame-level macroF1 is about 0.92-0.94 for
both models; spellF1 is far lower because it punishes boundaries and split spells.

**Why 0.810 is optimistic.** The test matches are not fully independent of the TCN's design: SkillCorner 1874553 was
a test match of the cross-validation fold on which the TCN recipe was compared, and J03WR9 was among the IDSSE matches
used to compare earlier TCN variants. The TCN's 5-fold cross-validation over all 20 SkillCorner matches (0.778 for a
single model) is the more honest estimate of that component. BAR-ATH's tracking also differs from both the simulated
and the SkillCorner tracking, so we expect a lower score on the full match.

## 7. What we tried and dropped

| idea | outcome |
|---|---|
| graph network and a CRF transformer (earlier TCN stages) | clearly worse on IDSSE development matches (0.53 and 0.21 against 0.59 for the TCN) |
| TCN with attention blocks | no better than the plain TCN (0.773 against 0.779) |
| bigger TCN, more steps | +0.03 once, then flat |
| pure SkillCorner training for the TCN | worse (0.684) than 85% SkillCorner + 15% degraded IDSSE (0.753) |
| self-training on confident pseudo-labels | no gain over the teacher |
| mixing two TCN architectures; tuning decoder switching costs | no gain or worse |
| per-frame possessor identity as an auxiliary target | available on too few frames, dropped |
| evidence-based control labels instead of the 2 s rule | built, not used (section 5) |
| SoccerNet-GAR, PFF World Cup 2022 | not accessible in time (NDA / request pending); SoccerNet labels events, not possession |
<!-- TODO team: add M1 variants that were tried and dropped, if any. -->

## 8. Limitations, and what we would do with more time

- **Simulated versus real automatic tracking.** A classifier still separates our simulated ATD from real ATD easily,
  so the simulator misses some properties of the real tracker. A simulator fitted to pass that test, or more real ATD
  with labels, is the most direct fix.
- **Labels do not match the competition's definition exactly.** DFL flags, SkillCorner phases and Metrica events
  each draw spell boundaries differently; the 2 s rule harmonises only the change-of-possession part. Relabelling the
  training data to the competition's definition is probably where most of the remaining gap is.
- **Short possessions get merged.** Both models predict too few spells (about 10-30% fewer than true); the unmatched
  true spells are short (median under 2 s). A dedicated boundary head or a segment-level decoder trained on the metric
  would target this.
- **Dead ball while the ball is missing** is the hardest call: with no ball, a stoppage has to be read from player
  speeds and positions alone.
- **Small validation set.** The ensemble weight and HMM scale are chosen on three matches; the curve over weights is
  noisy at the ±0.005 level.
- **Hardware dependence.** Prediction is deterministic on the same GPU, driver and library versions; on other hardware
  floating-point differences can change a few labels.

## 9. Reproduction

From a clone of the repository, with Python 3.12 and the organisers' data bundle:

```bash
uv sync --frozen        # or: python3.12 -m venv .venv && .venv/bin/pip install -r requirements-lock.txt
.venv/bin/python scripts/predict_ensemble.py --data-dir "/path/to/dataset" --out "offside merchants.csv"
```

This uses the committed weights and validates the file. The submitted file was produced on an NVIDIA RTX 2060 SUPER
(8 GB), driver 617.14, CUDA 13.2, PyTorch 2.15.0, Python 3.12.3. To retrain both models from the open data (about 5 h
on that GPU) and choose the ensemble weight again:

```bash
bash scripts/train_all.sh --fresh
python scripts/ensemble_weight.py
```

Seeds are fixed but GPU kernels are not forced to be deterministic, so a retrain gives close, not identical, weights.
Data downloads and per-model details: `m1/README.md`, `arsh-research/06_reproduce.md`.
