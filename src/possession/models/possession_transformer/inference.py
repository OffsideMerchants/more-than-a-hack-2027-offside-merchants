"""Whole-half inference: chunk, Hann-combine the CRF scores, then one exact decode over the half."""
import numpy as np
import torch

from . import crf

LABELS = np.array(["H", "A", "D"])


def window_starts(n: int, win: int, stride: int) -> list[int]:
    if n <= win:
        return [0]
    starts = list(range(0, n - win + 1, stride))
    if starts[-1] != n - win:
        starts.append(n - win)
    return starts


def combine_scores(chunks, n: int, win: int):
    """chunks: list of (start, emit [w,S], trans [w,S,S], offset [w]). Hann-weighted average."""
    S = chunks[0][1].shape[-1]
    emit, trans, off = np.zeros((n, S)), np.zeros((n, S, S)), np.zeros(n)
    wsum = np.zeros(n)
    for start, e, tr, o in chunks:
        w = np.hanning(len(e) + 2)[1:-1] if len(e) > 1 else np.ones(1)
        sl = slice(start, start + len(e))
        emit[sl] += w[:, None] * e
        trans[sl] += w[:, None, None] * tr
        off[sl] += w * o
        wsum[sl] += w
    wsum = np.maximum(wsum, 1e-12)
    return emit / wsum[:, None], trans / wsum[:, None, None], off / wsum


@torch.no_grad()
def predict_scores(model, players, team, ball, win: int | None = None, stride: int | None = None, device="cpu"):
    """players [T,N,6], team [N], ball [T,5] as numpy at the model rate."""
    win = win or model.cfg.window_steps
    stride = stride or win // 2
    n = len(ball)
    model.eval().to(device)
    team_t = torch.as_tensor(team, device=device)[None]
    chunks = []
    for s in window_starts(n, win, stride):
        e = min(n, s + win)
        p = torch.as_tensor(players[s:e], dtype=torch.float32, device=device)[None]
        b = torch.as_tensor(ball[s:e], dtype=torch.float32, device=device)[None]
        out = model(p, team_t, b)
        chunks.append((s, out["emit"][0].double().cpu().numpy(), out["trans"][0].double().cpu().numpy(),
                       out["offset"][0].double().cpu().numpy()))
    return combine_scores(chunks, n, win)


def round_scores(emit, trans, off, decimals: int = 4):
    return np.round(emit, decimals), np.round(trans, decimals), np.round(off, decimals)


def decode_half(emit, trans, offset, rate_hz: float, n_frames: int, out_hz: float = 25.0):
    """Exact Viterbi and forward-backward over the whole half; outputs at out_hz.

    Returns the contract arrays p_alive, p_home, p_turnover, p_restart and native_labels ('H','A','D').
    """
    e = torch.as_tensor(emit, dtype=torch.float64)[None]
    t = torch.as_tensor(trans, dtype=torch.float64)[None]
    path = crf.viterbi(e, t)[0].numpy()
    _, unary, pair = crf.forward_backward(e, t)
    unary, pair = unary[0].numpy(), pair[0].numpy()

    ratio = out_hz / rate_hz
    centres = np.arange(len(path)) * ratio
    frames = np.arange(n_frames)

    def up(v):
        return np.interp(frames, centres, v)

    p_alive = up(1.0 - unary[:, crf.D])
    p_home = up(unary[:, crf.H] / np.maximum(unary[:, crf.H] + unary[:, crf.A], 1e-9))
    turn = pair[:, crf.H, crf.A] + pair[:, crf.A, crf.H]
    restart = pair[:, crf.D, crf.H] + pair[:, crf.D, crf.A]

    def spread(v):
        out = np.zeros(n_frames)
        idx = np.clip(np.round((np.arange(len(v)) - 0.5 + offset) * ratio).astype(int), 0, n_frames - 1)
        np.add.at(out, idx, v)
        return np.clip(out, 0, 1)

    labels = np.empty(n_frames, dtype="U1")
    change = np.flatnonzero(path[1:] != path[:-1]) + 1
    bounds = np.clip(np.round((change - 0.5 + offset[change]) * ratio).astype(int), 0, n_frames)
    starts = np.concatenate([[0], bounds])
    ends = np.concatenate([bounds, [n_frames]])
    states = np.concatenate([[path[0]], path[change]])
    for s, en, st in zip(starts, ends, states):
        labels[s:max(s, en)] = LABELS[st]
    return dict(p_alive=p_alive, p_home=p_home, p_turnover=spread(turn), p_restart=spread(restart),
                native_labels=labels)
