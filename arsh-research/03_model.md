# 3. The model

Code: `src/possession/models/tcn/model.py` (network), `losses.py` (losses), `src/possession_gnn/evalbridge.py` (decoder).

## Why this shape of model
The label is three states per frame: H, A, D. It is split into two questions, as in the literature (`00_initial_research.md`):
**is the ball alive?** and **if so, which team has it?** Two reasons: alive/dead and owner depend on different evidence, and the error budget
showed that perfect dead/alive would be worth +0.19 spellF1 and perfect boundaries +0.14, so each gets its own head.

The input is a set of players that arrives in an arbitrary order, with the ball missing for long stretches. So the network must
(a) not depend on player order or identity, (b) work with a missing ball, and (c) treat home and away symmetrically.

## Architecture (a temporal convolution network, TCN, "tcn128")
Input at 5 Hz, windows of 100 steps (20 s). Per player and step: x, y, vx, vy, speed, position relative to the ball, distance to the ball, present flag.
Per step for the ball: x, y, vx, vy, present flag, and three ball-gap features (time since last seen, time to next seen, gap length).

1. **Track encoder.** Every player's time series passes through a stack of residual dilated 1-D convolution blocks (hidden size 128). The same weights
   are applied to every player, so the model does not depend on the slot a player is in.
2. **Team pooling.** Player features are pooled per step into "own team" and "opponent" summaries using mean, max and learned attention, with
   missing players masked out. Pooling is order-invariant by construction.
3. **Ball encoder.** The ball series plus its gap features passes through its own conv stack. When the ball is missing the model still knows how long it has been
   missing, and can rely on where the players are going.
4. **Fusion and joint stack.** The three streams are concatenated and passed through 7 residual dilated blocks (dilations 1, 2, 4, ..., 64, kernel 3,
   non-causal), which gives a receptive field of several seconds each way, so a decision at one step uses the context before and after it.
5. **Heads.** `alive` (ball in play), `owner` (own team has the ball) and `turn` (a possession change happens here, used only as an auxiliary signal).

Size: about one million parameters (4 MB per model). Hybrid variants with attention blocks (rotary-embedding self-attention after the conv stack) were also built
(`attn_blocks`, `heads`, `drop_path`) and reached about the same score (0.773 vs 0.779), so the plain TCN was kept.

## Exact home/away symmetry
Every window is encoded twice: once from the home team's point of view, once from the away team's, flipping x so "own team attacks +x".
The final home logit is `owner(home view) - owner(away view)`, which is exactly antisymmetric: swapping the teams swaps the answer.
This removes a whole class of bias (a model that quietly prefers the home side) without needing data to fix it.
Z (ball height), player roles and IDs are never inputs.

## Losses (`losses.py`)
Total = alive loss + owner loss + 1.0 x turnover loss + 0.5 x spell loss.
- **Alive:** binary cross-entropy, labels softened over +-2 steps (0.4 s) so a boundary one frame off is not punished like a full error.
- **Owner:** binary cross-entropy on alive frames only. Dead frames have no owner.
- **Turnover:** a head that predicts the moment possession changes, positive-weighted 5x. Weight 1.0 (0.3 gave the same score, so it is not a lever).
- **Spell loss (new, +0.011 spellF1 on average):** the metric rewards whole segments, so for every true possession the loss is `1 - soft IoU` between the
  predicted probability of "alive and this team" and the true spell, counting nearby false mass. It specifically fights *merging* short possessions, the main error.

## Decoder (`evalbridge.viterbi`)
Per-step probabilities become a state sequence with a 3-state Viterbi: emission = negative log probability of H, A or D; switching cost 4 between H and A, 6 into or out of D.
A minimum cost for switching is what removes one-frame flicker. The costs are fixed, not tuned on any test data. Tuning them on a validation match gave no gain.
The 5 Hz path is repeated 5 times to 25 Hz for scoring.

## Inference (`scripts/predict_barath.py`)
- Windows of 100 steps with stride 50, blended with a Hann window so window edges do not show.
- Five models (`models/possession_ensemble.pt`, keys `cs0`..`cs4`: seeds 0-4 on the shared split `splits.json`) are averaged in logit space for both alive and home; Viterbi then runs on the averaged probabilities.
- The two halves of BAR-ATH are run separately. Frames outside the published graded blocks are written as X.
