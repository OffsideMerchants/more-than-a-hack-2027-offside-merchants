"""Loading Metrica tracking data into fixed-slot arrays and windowed datasets.

Every frame is an array of shape (A, 2) with A = 2 * MAX_PER_TEAM + 1 agent slots:
home players in slots [0, 14), away players in [14, 28), and the ball in slot 28.
Positions are in metres, with the pitch centred at (0, 0), x in [-52.5, 52.5] and y in
[-34, 34]. When a player is off the pitch (substitute or sent off), or a slot is unused
padding, its position is NaN.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0
MAX_PER_TEAM = 14
NUM_AGENTS = 2 * MAX_PER_TEAM + 1
BALL = NUM_AGENTS - 1
# Agent type ids, used for the type embedding: 0 = home, 1 = away, 2 = ball.
AGENT_TYPES = np.array([0] * MAX_PER_TEAM + [1] * MAX_PER_TEAM + [2])
# Positions are divided by this before entering the model.
POS_SCALE = 50.0


def _read_team_csv(path):
    """Return (period, players[T, n, 2], ball[T, 2]) in Metrica's normalised coordinates."""
    df = pd.read_csv(path, skiprows=2)
    period = df["Period"].to_numpy()
    xy = df.iloc[:, 3:].to_numpy(dtype=np.float32).reshape(len(df), -1, 2)
    return period, xy[:, :-1], xy[:, -1]


def _to_metres(xy):
    out = np.empty_like(xy)
    out[..., 0] = (xy[..., 0] - 0.5) * PITCH_LENGTH
    out[..., 1] = -(xy[..., 1] - 0.5) * PITCH_WIDTH  # Metrica's y axis points down
    return out


def load_metrica_game(data_dir, game):
    """Load one game and return {period: positions[T, NUM_AGENTS, 2]} at 25 Hz, in metres."""
    data_dir = Path(data_dir)
    period, home, ball = _read_team_csv(data_dir / f"Sample_Game_{game}_RawTrackingData_Home_Team.csv")
    period_a, away, _ = _read_team_csv(data_dir / f"Sample_Game_{game}_RawTrackingData_Away_Team.csv")
    assert np.array_equal(period, period_a), "home/away files are not frame-aligned"
    assert home.shape[1] <= MAX_PER_TEAM and away.shape[1] <= MAX_PER_TEAM

    pos = np.full((len(period), NUM_AGENTS, 2), np.nan, dtype=np.float32)
    pos[:, : home.shape[1]] = home
    pos[:, MAX_PER_TEAM : MAX_PER_TEAM + away.shape[1]] = away
    pos[:, BALL] = ball
    pos = _to_metres(pos)
    return {int(p): pos[period == p] for p in np.unique(period)}


class TrackingWindows(Dataset):
    """Fixed-length windows cut from continuous tracking segments.

    Returns positions[T, A, 2], scaled by POS_SCALE and NaN where unobserved.
    With `augment`, applies random pitch flips and swaps the home and away slots.
    These are symmetries of the game, so the representation should not depend on them.
    """

    def __init__(self, segments, window=40, stride=4, augment=False):
        self.segments = [torch.from_numpy(s / POS_SCALE) for s in segments]
        self.window = window
        self.augment = augment
        self.index = [
            (i, s)
            for i, seg in enumerate(self.segments)
            for s in range(0, len(seg) - window + 1, stride)
        ]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        i, s = self.index[idx]
        x = self.segments[i][s : s + self.window].clone()
        if self.augment:
            if torch.rand(()) < 0.5:
                x[..., 0] = -x[..., 0]
            if torch.rand(()) < 0.5:
                x[..., 1] = -x[..., 1]
            if torch.rand(()) < 0.5:
                x = torch.cat([x[:, MAX_PER_TEAM:BALL], x[:, :MAX_PER_TEAM], x[:, BALL:]], dim=1)
        return x
