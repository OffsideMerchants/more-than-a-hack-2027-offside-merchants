"""Shared pieces of the M1 pipeline: training data per half, simulated ATD, input features, HMM decoding, scoring.

Collected verbatim from the experiment scripts the submission was developed with (calibrate_sim.py,
robustness_sim.py, finetune_v2.py, postprocess_sim.py, postprocess_tune.py), so cached files and
results stay identical.

Labelled training matches (all open data):
  idsse        7 Bundesliga matches (DFL), positions degraded by footy.atd_sim to look like automatic
               tracking (ATD); labels from DFL's ball status and owning team
  skillcorner  20 A-League matches, broadcast tracking; labels from SkillCorner's phases of play
  metrica      Metrica Sports sample games 1-3; labels derived from their event data (footy.labels)
All labels are converted to controlled possession (footy.semantics.controlled_possession, 2 s rule).
"""
import json
from pathlib import Path

import numpy as np

from . import official
from .atd import tracks_to_positions
from .atd_sim import simulate_atd
from .data import load_metrica_game
from .hmm import hmm_viterbi
from .kloppy_data import CACHE, SPORTEC_MATCHES, load_sportec_match
from .labels import load_labelled_game
from .metrica3 import load_metrica_game3
from .rules import rule_possession
from .semantics import controlled_possession
from .skillcorner_data import as_observed, load_skillcorner_match, match_ids

STEP, SNAP = 5, 4                 # 25 Hz -> 5 Hz model rate; boundary snapping radius in frames
SCALES = (0.2, 0.4, 0.8)          # HMM emission scales tried on the validation matches
SPLITS = json.loads((Path(__file__).resolve().parents[2] / "splits.json").read_text())   # shared with the TCN
EVAL = [(src, m) for src in ("idsse", "skillcorner") for m in SPLITS["test"][src]]      # held-out test matches
VAL = [(src, m) for src in ("idsse", "skillcorner") for m in SPLITS["val"][src]]        # validation matches
HOME, AWAY, NOBODY = 0, 1, -1
# shifted ATD simulation (second pretraining variant of the IDSSE matches)
SHIFT = dict(dead_whole_p=0.5, ball_gap_median_s=2.0, player_gap_median_s=1.2, p_new_track=0.9, p_swap=0.5,
             duplicate_rate=1 / 1500)


def all_matches():
    """(source, match id) of the 30 labelled matches, in the order of the data caches."""
    return ([("idsse", m) for m in SPORTEC_MATCHES] + [("skillcorner", m) for m in sorted(match_ids())]
            + [("metrica", g) for g in (1, 2, 3)])


# ------------------------------------------------------------------------------ simulated ATD

def nearest_codes(home_tracks, away_tracks, ball):
    """Team of the nearest raw track to the ball per frame, NOBODY where the ball is missing."""
    tracks = np.concatenate([home_tracks, away_tracks], 1)
    team = np.r_[np.zeros(home_tracks.shape[1], np.int8), np.ones(away_tracks.shape[1], np.int8)]
    d = np.linalg.norm(tracks - ball[:, None], axis=-1)
    d = np.where(np.isnan(d), np.inf, d)
    best = d.argmin(1)
    codes = team[best].astype(np.int8)
    codes[np.isnan(ball[:, 0]) | ~np.isfinite(d.min(1))] = NOBODY
    return codes


def simulated_match(mid, seed=0):
    """List of (positions [T, 29, 2], baseline codes [T], labels [T]) per period, cached."""
    path = CACHE / f"sim_{mid}_s{seed}.npz"
    if path.exists():
        z = np.load(path)
        return [(z[f"pos{i}"], z[f"near{i}"], z[f"lab{i}"]) for i in range(int(z["n"]))]
    out = []
    for i, (pos, lab) in enumerate(load_sportec_match(mid)):
        h, a, b = simulate_atd(pos, seed=seed * 100 + i, dead=(lab == 2))
        p, _ = tracks_to_positions(h, a, b)
        out.append((p, nearest_codes(h, a, b), lab))
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, n=len(out), **{f"pos{i}": o[0] for i, o in enumerate(out)},
                        **{f"near{i}": o[1] for i, o in enumerate(out)}, **{f"lab{i}": o[2] for i, o in enumerate(out)})
    return out


def shifted_match(mid):
    """Simulated ATD with shifted parameters: list of (positions, labels) per half, cached."""
    path = CACHE / f"sim_{mid}_B.npz"
    if path.exists():
        z = np.load(path)
        return [(z[f"pos{i}"], z[f"lab{i}"]) for i in range(int(z["n"]))]
    out = []
    for i, (pos, lab) in enumerate(load_sportec_match(mid)):
        h, a, b = simulate_atd(pos, seed=1000 + i, dead=(lab == 2), **SHIFT)
        out.append((tracks_to_positions(h, a, b)[0], lab))
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, n=len(out), **{f"pos{i}": p for i, (p, _) in enumerate(out)},
                        **{f"lab{i}": l for i, (_, l) in enumerate(out)})
    return out


# ------------------------------------------------------------------------------ labelled data

def halves_of(source, mid):
    """List of (positions [T, 29, 2] at 25 Hz, controlled-possession labels [T]) per half."""
    if source == "idsse":
        raw = [(p, l) for p, _, l in simulated_match(mid)]
    elif source == "skillcorner":
        raw = [(as_observed(h), h["labels"]) for h in load_skillcorner_match(mid)]
    elif mid == 3:
        raw = load_metrica_game3()
    else:
        pos25 = [p for _, p in sorted(load_metrica_game("data/metrica", mid).items())]
        raw = list(zip(pos25, [l for _, l in load_labelled_game("data/metrica", mid, hz=25)]))
    return [(p, controlled_possession(l)) for p, l in raw]


def featurise(pos, lab):
    """One half -> model input at 5 Hz, labels, and the rule-based possession used to snap boundaries."""
    rules = rule_possession(pos)
    n = len(pos[::STEP])
    return {"pos5": pos[::STEP][:n], "lab5": lab[::STEP][:n], "rules": rules, "gt": lab, "T": len(pos)}


# ------------------------------------------------------------------------------ decoding

def boost(log_trans, beta):
    """Make home <-> away changes beta times more likely in the HMM transitions."""
    lt = log_trans.copy()
    lt[0, 1] += np.log(beta)
    lt[1, 0] += np.log(beta)
    return lt - np.logaddexp.reduce(lt, axis=1, keepdims=True)


def snap(pred, ref, radius=SNAP):
    """Move each transition of `pred` to a transition of `ref` with the same new label within +/- radius."""
    pred = pred.copy()
    ref_t = np.flatnonzero(np.diff(ref)) + 1
    for t in np.flatnonzero(np.diff(pred)) + 1:
        cand = ref_t[(np.abs(ref_t - t) <= radius) & (ref[np.clip(ref_t, 0, len(ref) - 1)] == pred[t])]
        if len(cand):
            c = cand[np.argmin(np.abs(cand - t))]
            if c < t:
                pred[c:t] = pred[t]
            else:
                pred[t:c] = pred[t - 1]
    return pred


def hmm_decode(h, probs, prior, lt, scale):
    ll = np.log(np.clip(probs, 1e-6, 1)) - np.log(prior)
    return snap(np.repeat(hmm_viterbi(ll, lt, scale=scale), STEP)[:h["T"]], h["rules"])


def score(data, keys, probs, prior, lt, scale):
    return [official.evaluate([h["gt"] for h in data[k]], [hmm_decode(h, p, prior, lt, scale) for h, p in zip(data[k], probs[k])])
            for k in keys]
