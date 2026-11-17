"""Predict BAR-ATH possession with M1 and write the submission CSV. Uses only the ATD (no video, no Smart Tagging).

Pipeline: ATD -> 29 slots per frame (footy.atd: anonymous tracks cut into segments and assigned to 14 + 14 slots;
ball as tracked) -> per half: 16 s windows every 1 s at 5 Hz; M1 with test-time augmentation (teams swapped and/or
pitch rotated by 180 degrees, outputs mapped back and averaged); window probabilities averaged per frame -> HMM
(prior and transitions from the fine-tuning labels, possession-change boost beta, emission scale; from
scripts/finetune.py) -> 25 Hz, boundaries snapped to rule-based transitions (+/- 4 frames).
Every graded frame gets H, A or D; frames outside the two halves get X.
Runs on the GPU (CUDA, about 1 min) with PyTorch's deterministic settings, so repeated runs give the same file;
falls back to the CPU (about 10 min) when no GPU is available.

Alignment (verified by the organisers, starter/load_atd.py): the ATD has one record per video frame and
video_frame = frame_id - 1. Home (H) is the first team in the ATD metadata (Barcelona).

    python scripts/predict.py --data-dir "/path/to/dataset"     # folder holding tactical/ (default $MTAH_DATA_DIR or data/)
    python evaluation/run_evaluation.py --bootstrap 1000          # public 5 minutes: pipeline check only, never for tuning
"""
import argparse
import os
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")   # deterministic cuBLAS; must be set before CUDA starts
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]   # m1/: footy and the model files
REPO = ROOT.parent                           # repository root: games.py and evaluation/ of the challenge
sys.path[:0] = [str(ROOT), str(REPO)]
from footy.atd import tracks_to_positions  # noqa: E402
from footy.data import BALL, MAX_PER_TEAM, POS_SCALE  # noqa: E402
from footy.hmm import hmm_viterbi  # noqa: E402
from footy.model import TrajectoryMAE  # noqa: E402
from footy.pipeline import STEP, boost, featurise, snap  # noqa: E402
from footy.possession import PossessionModel  # noqa: E402

MODEL = "runs/final_eval/M1_transformer.pt"       # fine-tuned on 25 matches, held-out spell-F1 0.784
DECODER = "runs/bar_ath/decoder.npz"              # HMM parameters from the same 25 matches
STRIDE = 5                         # window starts every 5 frames at 5 Hz = 1 s
PASSES = {"id": (False, (1.0, 1.0)), "swap": (True, (1.0, 1.0)),            # test-time transforms: (swap teams, mirror)
          "rot": (False, (-1.0, -1.0)), "swaprot": (True, (-1.0, -1.0))}
LETTERS = np.array(["H", "A", "D"])


def setup_device(name=None):
    """Torch device (default: cuda if available, else cpu) with deterministic kernels."""
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    return torch.device(name or ("cuda" if torch.cuda.is_available() else "cpu"))


# ----------------------------------------------------------------------------------------- input

def read_atd(meta_path, raw_path, n_frames, pitch):
    """FIFA/EPTS ATD -> (slot_team [n] 0 = home / 1 = away, players [n_frames, n, 2], ball [n_frames, 2]).

    Row i is video frame i (frame_id - 1). Metres, pitch centred at (0, 0), y up: the same
    convention as footy.atd.load_atd_epts and the training data. NaN = not tracked. Parsed with the
    standard library like the organisers' starter/baseline.py: slot j of a raw line is the j-th
    <Player> of the metadata, and the first <Team> is the home side.
    """
    root = ET.parse(meta_path).getroot()
    team_ids = [t.get("id") for t in root.iter("Team")]
    if len(team_ids) != 2:
        raise ValueError(f"{meta_path}: expected 2 teams, found {len(team_ids)}")
    slot_team = np.array([0 if p.get("teamId") == team_ids[0] else 1 for p in root.iter("Player")], np.int8)
    n = len(slot_team)
    L, W = pitch
    players = np.full((n_frames, n, 2), np.nan, np.float32)
    ball = np.full((n_frames, 2), np.nan, np.float32)
    bad_groups = outside = 0
    with open(raw_path) as f:
        for line in f:
            parts = line.strip().split(":")
            if len(parts) != 3:
                continue
            i = int(parts[0]) - 1
            if not 0 <= i < n_frames:
                outside += 1
                continue
            groups = [g for g in parts[1].split(";") if g]
            if len(groups) != n:
                bad_groups += 1
            for j, g in enumerate(groups[:n]):
                if g[0] != "N":
                    x, y = g.split(",")[:2]
                    players[i, j] = (float(x) - 0.5) * L, -(float(y) - 0.5) * W
            b = parts[2].strip(";")
            if b and b[0] != "N":
                x, y = b.split(",")[:2]
                ball[i] = (float(x) - 0.5) * L, -(float(y) - 0.5) * W
    print(f"ATD: {len(team_ids)} teams, {n} track slots", flush=True)
    if bad_groups or outside:
        print(f"  WARNING: {bad_groups} lines with a slot count other than {n}, {outside} lines outside "
              f"0..{n_frames - 1}", flush=True)
    return slot_team, players, ball


def load_net(model_path, device):
    """Fine-tuned possession model -> (network in eval mode, window length)."""
    ck = torch.load(ROOT / model_path, map_location="cpu")
    a = torch.load(ROOT / ck["pretrained"], map_location="cpu")["args"]
    net = PossessionModel(TrajectoryMAE(a["dim"], a["depth"], a["heads"], max_len=ck["window"], dropout=a["dropout"]))
    net.load_state_dict(ck["model"])
    return net.to(device).eval(), ck["window"]


@torch.no_grad()
def frame_probs(net, pos5, window, stride, swap, mirror, device, batch=32):
    """Sliding windows over one half -> per-frame class probabilities [T, 3] (uniform mean over windows).

    swap / mirror: test-time transform of the input (teams swapped, coordinates multiplied by
    mirror); a swapped prediction is mapped back by exchanging H and A.
    """
    pos = torch.from_numpy(pos5 / POS_SCALE).float()
    T = len(pos)
    starts = list(range(0, max(T - window, 0) + 1, stride))
    if starts[-1] + window < T:
        starts.append(T - window)
    acc, cnt = np.zeros((T, 3)), np.zeros((T, 1))
    for i in range(0, len(starts), batch):
        s = starts[i:i + batch]
        x = torch.stack([pos[j:j + window] for j in s]).to(device) * torch.tensor(mirror, device=device)
        if swap:
            x = torch.cat([x[:, :, MAX_PER_TEAM:BALL], x[:, :, :MAX_PER_TEAM], x[:, :, BALL:]], 2)
        p = net(x).float().softmax(-1)
        if swap:
            p = p[..., [1, 0, 2]]
        p = p.cpu().numpy()
        for k, j in enumerate(s):
            acc[j:j + window] += p[k]
            cnt[j:j + window] += 1
    return acc / cnt


# ---------------------------------------------------------------------------------------- decode

def decode(probs5, rules, T, prior, log_trans, scale):
    """5 Hz probabilities -> 25 Hz labels (0 H, 1 A, 2 D) for one half."""
    ll = np.log(np.clip(probs5, 1e-6, 1)) - np.log(prior)
    d25 = np.repeat(hmm_viterbi(ll, log_trans, scale=scale), STEP)[:T]
    if len(d25) < T:                                   # the 5 Hz grid can end a few frames early
        d25 = np.r_[d25, np.full(T - len(d25), d25[-1])]
    return snap(d25, rules)


def m1_probs(net, window, pos5, device):
    """Mean of the four test-time passes -> class probabilities at 5 Hz [n, 3]."""
    return np.mean([frame_probs(net, pos5, window, STRIDE, swap, mirror, device) for swap, mirror in (PASSES[k] for k in sorted(PASSES))], 0)


def load_decoder(decoder=DECODER):
    """HMM parameters written by scripts/finetune.py -> (prior, boosted log transitions, emission scale)."""
    z = np.load(ROOT / decoder)
    beta, scale = dict(zip(z["methods"].tolist(), zip(z["betas"].tolist(), z["scales"].tolist())))["M1"]
    print(f"decoder {decoder} (beta {beta:g}, scale {scale:g})", flush=True)
    return z["prior"], boost(z["base_trans"], beta), scale


def half_probs(meta, raw, blocks, n_frames, pitch, model=MODEL, device=None):
    """M1 on each graded half -> [(period, first, last, probs [n5, 3] at 5 Hz, rule-based labels [T], T)].

    blocks: [(period, first_video_frame, last_video_frame)]. 5 Hz sample k is video frame first + STEP * k.
    """
    t0 = time.time()
    device = device or setup_device()
    print(f"model {model}, device {device}", flush=True)
    slot_team, players, ball = read_atd(meta, raw, n_frames, pitch)
    net, window = load_net(model, device)
    out = []
    for period, first, last in blocks:
        sl = slice(first, last + 1)
        pos, _ = tracks_to_positions(players[sl][:, slot_team == 0], players[sl][:, slot_team == 1], ball[sl])
        h = featurise(pos, np.full(len(pos), -1))
        out.append((period, first, last, m1_probs(net, window, h["pos5"], device), h["rules"], h["T"]))
        print(f"half {period}: {len(pos)} frames done ({time.time() - t0:.0f}s)", flush=True)
    return out


def write_csv(labels, out_path):
    """One row per video frame: frame,label."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("frame,label\n")
        f.writelines(f"{i},{c}\n" for i, c in enumerate(labels))
    print(f"wrote {len(labels)} rows -> {out_path}", flush=True)


def run(meta, raw, blocks, n_frames, pitch, out_path, model=MODEL, decoder=DECODER, device=None):
    """Predict and write the submission CSV. blocks: [(period, first_video_frame, last_video_frame)] of the graded halves."""
    prior, log_trans, scale = load_decoder(decoder)
    labels = np.full(n_frames, "X", dtype="<U1")
    for period, first, last, probs5, rules, T in half_probs(meta, raw, blocks, n_frames, pitch, model, device):
        labels[first:last + 1] = LETTERS[decode(probs5, rules, T, prior, log_trans, scale)]
    write_csv(labels, out_path)
    return labels


def main():
    from games import GAME, GAMES, half_video_blocks, require, source_frames

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data-dir", help="data bundle root holding tactical/ (default: $MTAH_DATA_DIR or <repository root>/data)")
    p.add_argument("--out", default=str(REPO / "evaluation" / "results" / "ssl_M1.csv"))
    p.add_argument("--model", default=MODEL)
    p.add_argument("--decoder", default=DECODER)
    p.add_argument("--device", default=None, help="cuda or cpu (default: cuda if available)")
    p.add_argument("--no-validate", action="store_true")
    args = p.parse_args()

    data_dir = args.data_dir
    if data_dir and os.path.basename(os.path.normpath(data_dir)) == "tactical" and not os.path.isdir(os.path.join(data_dir, "tactical")):
        data_dir = os.path.dirname(os.path.normpath(data_dir))      # accept the tactical/ folder itself as well
    meta, raw = require(["atd_meta", "atd_raw"], GAME, data_dir)
    run(meta, raw, half_video_blocks(GAME), source_frames("tactical", GAME), GAMES[GAME]["pitch"], args.out, args.model, args.decoder, setup_device(args.device))
    if not args.no_validate:
        r = subprocess.run([sys.executable, str(REPO / "evaluation" / "validate_submission.py"), args.out], capture_output=True, text=True)
        print(f"validate: {'OK' if r.returncode == 0 else 'FAILED  ' + (r.stdout + r.stderr).strip()}")


if __name__ == "__main__":
    main()
