"""Rule-based possession baseline: no learning, thresholds set from football knowledge.

Rules, applied at 25 Hz:
  DeadBall  - ball visible and outside the pitch, or
            - ball visible and stationary for >= `still_s` while players are slow, or
            - ball missing and the mean player speed (smoothed over ±2 s) < `dead_speed`.
              A missing ball on its own never means DeadBall.
  Possession (live play):
            - ball visible: team of the nearest player if within `control_m`. Switching to the
              other team needs `switch_s` of continuous control. In flight, the last team keeps it.
            - ball missing: the last team keeps it for `hold_s`, then the clearly wider team
              (width difference > `width_margin` m, smoothed) takes it.
  Finally, spells shorter than `min_spell_s` are merged into the preceding spell.
"""
import warnings

import numpy as np

from .data import BALL, MAX_PER_TEAM, PITCH_LENGTH, PITCH_WIDTH
from .hmm import segments

HOME, AWAY, DEAD = 0, 1, 2


def _smooth(x, k):
    """Centred moving average with edge padding, ignoring NaN."""
    ok = ~np.isnan(x)
    kern = np.ones(2 * k + 1)
    num = np.convolve(np.pad(np.where(ok, x, 0), k, mode="edge"), kern, "valid")
    den = np.convolve(np.pad(ok.astype(float), k, mode="edge"), kern, "valid")
    with np.errstate(invalid="ignore", divide="ignore"):
        return num / den


def _signals(pos25, fps=25):
    """Quantities shared by the rules and the pseudo-labeller."""
    vel = np.full_like(pos25, np.nan)
    vel[3:-3] = (pos25[6:] - pos25[:-6]) / (6 / fps)
    spd = np.linalg.norm(vel, axis=-1)
    ball = pos25[:, BALL]
    seen = ~np.isnan(ball[:, 0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        width = lambda s: np.nanmax(pos25[:, s, 1], 1) - np.nanmin(pos25[:, s, 1], 1)
        dist = np.where(np.isnan(np.linalg.norm(pos25[:, :BALL] - ball[:, None], axis=-1)), np.inf,
                        np.linalg.norm(pos25[:, :BALL] - ball[:, None], axis=-1))
        return {
            "speed": spd, "ball": ball, "seen": seen,
            "player_speed": _smooth(np.nanmean(spd[:, :BALL], 1), 2 * fps),
            "width_diff": _smooth(width(slice(0, MAX_PER_TEAM)) - width(slice(MAX_PER_TEAM, BALL)), 2 * fps),
            "nearest": np.where(seen, dist.argmin(1), -1),
            "d_min": np.where(seen, dist.min(1), np.inf),
        }


def rule_possession(pos25, control_m=1.0, switch_s=0.4, hold_s=3.0, dead_speed=1.4, still_s=1.5,
                    still_speed=0.5, out_margin=0.5, width_margin=3.0, min_spell_s=1.0, fps=25, team_prior=None):
    """pos25: [T, A, 2] in metres at 25 Hz (NaN = missing). Returns labels [T] (0 Home, 1 Away, 2 Dead).

    team_prior: optional per-frame team (0/1, -1 = none), e.g. from Smart Tagging. It replaces the
    team-width guess when the ball has been missing for more than `hold_s`, and sets the team at a
    restart until a player is seen controlling the ball.
    """
    T = len(pos25)
    sig = _signals(pos25, fps)
    spd, ball, seen, player_speed = sig["speed"], sig["ball"], sig["seen"], sig["player_speed"]
    width_diff, nearest, d_min = sig["width_diff"], sig["nearest"], sig["d_min"]

    # Dead ball
    out = seen & ((np.abs(ball[:, 0]) > PITCH_LENGTH / 2 + out_margin) | (np.abs(ball[:, 1]) > PITCH_WIDTH / 2 + out_margin))
    still = seen & (np.nan_to_num(spd[:, BALL], nan=99) < still_speed)
    run = np.zeros(T, int)  # length of the current stationary-ball run
    for t in range(T):
        run[t] = run[t - 1] + 1 if still[t] and t else int(still[t])
    still_long = np.zeros(T, bool)
    for t in np.flatnonzero(run >= still_s * fps):  # mark the whole run, not only its tail
        still_long[t - run[t] + 1:t + 1] = True
    dead = out | (still_long & (player_speed < dead_speed + 0.5)) | (~seen & (player_speed < dead_speed))

    # Possession state machine
    control = np.where(d_min < control_m, np.where(nearest < MAX_PER_TEAM, HOME, AWAY), -1)
    lab = np.empty(T, int)
    team, cand, cand_run, last_seen = (HOME if np.nanmean(width_diff) > 0 else AWAY), -1, 0, -10 ** 9
    for t in range(T):
        if seen[t]:
            last_seen = t
            c = control[t]
            if c >= 0 and c != team:
                cand_run = cand_run + 1 if c == cand else 1
                cand = c
                if cand_run >= switch_s * fps:
                    team, cand, cand_run = c, -1, 0
            elif c == team:
                cand, cand_run = -1, 0
        elif t - last_seen > hold_s * fps:
            if team_prior is not None and team_prior[t] >= 0:
                team = int(team_prior[t])
            elif abs(width_diff[t]) > width_margin:
                team = HOME if width_diff[t] > 0 else AWAY
        if team_prior is not None and t and dead[t - 1] and not dead[t] and team_prior[t] >= 0 \
                and not (seen[t] and control[t] >= 0):
            team = int(team_prior[t])  # restart: trust the tagged team until someone is seen on the ball
        lab[t] = DEAD if dead[t] else team

    # Merge spells shorter than min_spell_s into the previous spell
    min_len = int(min_spell_s * fps)
    for c, s, e in segments(lab):
        if e - s < min_len and s > 0:
            lab[s:e] = lab[s - 1]
    return lab
