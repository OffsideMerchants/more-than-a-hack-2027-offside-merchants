"""HMM utilities for decoding per-frame class probabilities into possession spells.

fit_hmm_transitions estimates the transition matrix from training labels; hmm_viterbi decodes hybrid NN/HMM
emissions (log p(c | x_t) - log prior(c)), scaled by `scale` because window-averaged frame probabilities are
strongly correlated in time.
"""
import numpy as np


def segments(lab):
    """(class, start, end) runs of a label sequence without -1."""
    b = np.r_[0, np.flatnonzero(np.diff(lab)) + 1, len(lab)]
    return [(int(lab[s]), int(s), int(e)) for s, e in zip(b[:-1], b[1:])]


def hmm_viterbi(loglik, log_trans, log_init=None, scale=1.0, enter_bonus=None):
    """Most likely state sequence of a plain HMM. loglik [T, C] (log p(obs | state)), log_trans [C, C].

    enter_bonus: optional [T, C] log-bonus added to every transition *into* state c at frame t
    from a different state (time-varying evidence for a boundary, e.g. a transition tag).
    """
    T, C = loglik.shape
    e = scale * loglik
    delta = (np.log(np.full(C, 1 / C)) if log_init is None else log_init) + e[0]
    back = np.zeros((T, C), int)
    off = ~np.eye(C, dtype=bool)
    for t in range(1, T):
        cand = delta[:, None] + log_trans                     # [prev, next]
        if enter_bonus is not None and enter_bonus[t].any():
            cand = cand + off * enter_bonus[t][None, :]
        back[t] = cand.argmax(0)
        delta = cand.max(0) + e[t]
    out = np.empty(T, int)
    out[-1] = int(delta.argmax())
    for t in range(T - 1, 0, -1):
        out[t - 1] = back[t, out[t]]
    return out


def fit_hmm_transitions(label_seqs, n_states=3, alpha=1.0):
    """Log transition matrix (with self-transitions) from label sequences; -1 frames are skipped."""
    counts = np.full((n_states, n_states), alpha)
    for lab in label_seqs:
        a, b = lab[:-1], lab[1:]
        m = (a >= 0) & (b >= 0)
        np.add.at(counts, (a[m], b[m]), 1)
    return np.log(counts / counts.sum(1, keepdims=True))
