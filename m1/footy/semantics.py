"""Convert raw per-frame ownership flags into the challenge's "controlled possession" semantics.

Provider flags such as DFL's ball_owning_team change on every touch, deflection and contested ball.
The MTAH challenge asks for *controlled* possession:
  - a change of possession needs controlled possession by the other team (proxy from the rules:
    "a player keeps the ball for a few seconds");
  - a touch without control, a deflection or a won duel without control does not count;
  - while the ball is contested or loose, possession stays with the last team;
  - after a restart there is always a clear possessor.

Rule implemented here, derived from that text (not fitted to any ground truth): within each
alive stretch (between dead-ball frames), the possessing team only changes when the other team's
run lasts at least `min_control_s`. Shorter runs are absorbed into the current possession.
The first run after a restart is kept whatever its length. Dead-ball frames are unchanged.
"""
import numpy as np

HOME, AWAY, DEAD = 0, 1, 2


def _runs(lab):
    b = np.r_[0, np.flatnonzero(np.diff(lab)) + 1, len(lab)]
    return [(int(lab[s]), int(s), int(e)) for s, e in zip(b[:-1], b[1:])]


def controlled_possession(lab, min_control_s=2.0, fps=25):
    """lab: per-frame 0/1/2 (-1 allowed, treated as 'unknown' and kept). Returns a relabelled copy."""
    lab = np.asarray(lab)
    out = lab.copy()
    min_len = int(round(min_control_s * fps))
    current = None
    for c, s, e in _runs(lab):
        if c == DEAD or c < 0:
            current = None if c == DEAD else current  # a stoppage ends the possession; unknown keeps it
            continue
        if current is None:           # first run after a restart (or at the start): clear possessor
            current = c
        elif c != current and e - s >= min_len:
            current = c               # controlled possession by the other team
        out[s:e] = current
    return out
