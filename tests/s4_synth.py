"""Toy match for S4 tests: the carrier of the possessing team holds the ball, dead balls sit still."""
import numpy as np
import torch


def make_clip(T=120, n_per_team=5, seed=0, gap=None):
    rng = np.random.default_rng(seed)
    N = 2 * n_per_team
    team = np.array([0] * n_per_team + [1] * n_per_team)
    pos = np.stack([rng.uniform(-40, 40, N), rng.uniform(-25, 25, N)], 1)
    vel = rng.normal(0, 1.0, (N, 2))
    y = np.zeros(T, dtype=np.int64)
    t, state = 0, rng.integers(0, 2)
    while t < T:
        dur = int(rng.integers(15, 40))
        y[t:t + dur] = state
        t += dur
        state = 2 if state < 2 and rng.random() < 0.4 else int(rng.integers(0, 2))
    players = np.zeros((T, N, 6), dtype=np.float32)
    ball = np.zeros((T, 5), dtype=np.float32)
    carrier = {0: 0, 1: n_per_team}
    for i in range(T):
        vel += rng.normal(0, 0.3, vel.shape)
        pos = np.clip(pos + vel * 0.16, [-52, -34], [52, 34])
        players[i, :, :2] = pos
        players[i, :, 2:4] = vel
        players[i, :, 4] = np.linalg.norm(vel, axis=1)
        players[i, :, 5] = 1
        if y[i] < 2:
            if i == 0 or y[i - 1] != y[i] or rng.random() < 0.05:
                carrier[y[i]] = int(rng.integers(0, n_per_team)) + y[i] * n_per_team
            ball[i, :2] = pos[carrier[y[i]]]
        else:
            ball[i, :2] = ball[i - 1, :2]
        ball[i, 4] = 1
    if gap is not None:
        ball[gap[0]:gap[1], :] = np.nan
        ball[gap[0]:gap[1], 4] = 0
    return players, team, ball, y


def batch_from(clips):
    return dict(
        players=torch.as_tensor(np.stack([c[0] for c in clips])),
        team=torch.as_tensor(np.stack([c[1] for c in clips])),
        ball=torch.as_tensor(np.stack([c[2] for c in clips])),
        y=torch.as_tensor(np.stack([c[3] for c in clips])),
    )
