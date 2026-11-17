"""Parse the 7 IDSSE matches into canonical memmapped arrays under artifacts/canon/ (green tier, prints freely)."""
import os
import pickle
import sys

from possession import canonical, io_idsse
from possession.paths import ARTIFACTS

root = os.path.join(ARTIFACTS, "canon")
for mid in (sys.argv[1:] or io_idsse.MATCHES):
    raw = io_idsse.load_raw(mid)
    c = canonical.from_idsse(raw)
    canonical.save(c, root)
    os.makedirs(os.path.join(ARTIFACTS, "idsse"), exist_ok=True)
    with open(os.path.join(ARTIFACTS, "idsse", f"{mid}_meta.pkl"), "wb") as fh:
        pickle.dump({"events": raw.events, "ball_possession": raw.ball_possession, "ball_alive": raw.ball_alive,
                     "home_team": raw.home_team, "guest_team": raw.guest_team, "person_id": raw.person_id,
                     "team_id": raw.team_id, "frame_n": raw.frame_n, "half": raw.half, "role": raw.role, "half_t0": raw.half_t0}, fh)
    print(mid, c.players.shape, c.aux["attack_signs"], "frames", c.T, "N", c.N)
