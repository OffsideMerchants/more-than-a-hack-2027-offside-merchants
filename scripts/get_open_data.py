"""Download the open training data (IDSSE, SkillCorner, Metrica) into the folders the training code reads.

  python scripts/get_open_data.py                      # everything (several GB), skips files that are already there
  python scripts/get_open_data.py --only skillcorner   # one source (idsse, skillcorner, metrica)
  python scripts/get_open_data.py --dry-run            # only check that every source URL answers

Only needed for retraining (scripts/train_all.sh). Predicting BAR-ATH uses the committed weights and needs none of it.
No git or git-lfs: file URLs are built as the pinned kloppy (3.19.1) builds them, including GitHub's large-file
redirects. Every file goes through a temporary name, so an interrupted run never leaves a partial file behind.

Layout written (paths relative to the repository root):
  idsse-data/DFL_{02_01_matchinformation,03_02_events_raw,04_03_positions_raw_observed}_*.xml
      7 IDSSE matches from huggingface.co/datasets/pysport/idsse-data (CC BY 4.0), read by the TCN
      (src/possession_gnn/data.py). M1 loads IDSSE through kloppy, which downloads the same files itself.
  artifacts/external/skillcorner/data/matches.json, .../data/matches/<id>/<id>_{match.json,
      tracking_extrapolated.jsonl,phases_of_play.csv}
      github.com/SkillCorner/opendata (MIT), read by the TCN (src/possession_gnn/data_sc.py).
  m1/data/skillcorner/matches.json, m1/data/skillcorner/<id>/
      the same SkillCorner files for M1 (m1/footy/skillcorner_data.py): links to the copy above, or a copy where
      links are not available.
  m1/data/metrica/Sample_Game_{1,2}_{RawTrackingData_Home_Team,RawTrackingData_Away_Team,RawEventsData}.csv,
  m1/data/metrica/Sample_Game_3_{tracking.txt,metadata.xml,events.json}
      github.com/metrica-sports/sample-data, read by M1 (m1/footy/data.py, labels.py, metrica3.py).
"""
import argparse
import json
import os
import shutil
import sys
import urllib.request
from pathlib import Path

from kloppy._providers.sportec import get_IDSSE_url   # pinned kloppy: the official IDSSE file URLs
from kloppy.utils import github_resolve_raw_data_url   # resolves GitHub raw and large-file (LFS) URLs

REPO = Path(__file__).resolve().parents[1]
IDSSE_MATCHES = ("J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY", "J03WQQ", "J03WR9")
IDSSE_KINDS = {"meta": "DFL_02_01", "event": "DFL_03_02", "tracking": "DFL_04_03"}
SC_REPO, SC_FILES = "SkillCorner/opendata", ("match.json", "tracking_extrapolated.jsonl", "phases_of_play.csv")
METRICA_REPO = "metrica-sports/sample-data"
METRICA_FILES = [f"Sample_Game_{g}/Sample_Game_{g}_{k}.csv" for g in (1, 2)
                 for k in ("RawTrackingData_Home_Team", "RawTrackingData_Away_Team", "RawEventsData")] + \
                [f"Sample_Game_3/Sample_Game_3_{k}" for k in ("tracking.txt", "metadata.xml", "events.json")]
HEADERS = {"User-Agent": "get_open_data.py"}


def github(repo, path):
    return github_resolve_raw_data_url(repository=repo, branch="master", file=path)


def fetch(url, dest, dry_run):
    """Download url to dest unless dest exists. dry_run: only check that the URL answers."""
    if dry_run:
        req = urllib.request.Request(url, headers=HEADERS, method="HEAD")
        with urllib.request.urlopen(req, timeout=60) as r:
            size = r.headers.get("Content-Length")
        print(f"  ok   {dest.relative_to(REPO)}  ({int(size) / 2**20:.1f} MB)" if size else f"  ok   {dest.relative_to(REPO)}")
        return
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  have {dest.relative_to(REPO)}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=600) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, length=1 << 20)
    os.replace(tmp, dest)
    print(f"  got  {dest.relative_to(REPO)}  ({dest.stat().st_size / 2**20:.1f} MB)", flush=True)


def idsse(dry_run):
    out = REPO / "idsse-data"
    print("IDSSE -> idsse-data/")
    for mid in IDSSE_MATCHES:
        for kind, prefix in IDSSE_KINDS.items():
            have = sorted(out.glob(f"{prefix}*{mid}.xml"))       # the TCN finds files by prefix and match id
            if have and not dry_run:
                print(f"  have {have[0].relative_to(REPO)}")
                continue
            url = get_IDSSE_url(mid, kind)
            fetch(url, out / url.rsplit("/", 1)[1], dry_run)


def skillcorner(dry_run):
    base = REPO / "artifacts" / "external" / "skillcorner" / "data"
    print("SkillCorner -> artifacts/external/skillcorner/data/ (TCN) and m1/data/skillcorner/ (M1)")
    fetch(github(SC_REPO, "data/matches.json"), base / "matches.json", dry_run)
    ids = [int(m["id"]) for m in json.loads((base / "matches.json").read_text(encoding="utf-8"))] if not dry_run else []
    if dry_run:                                               # one match is enough to check the URLs
        with urllib.request.urlopen(urllib.request.Request(github(SC_REPO, "data/matches.json"), headers=HEADERS)) as r:
            ids = [int(json.load(r)[0]["id"])]
    for mid in ids:
        for name in SC_FILES:
            fetch(github(SC_REPO, f"data/matches/{mid}/{mid}_{name}"), base / "matches" / str(mid) / f"{mid}_{name}", dry_run)
    if dry_run:
        return
    m1 = REPO / "m1" / "data" / "skillcorner"
    m1.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(base / "matches.json", m1 / "matches.json")
    for mid in ids:
        src, dst = base / "matches" / str(mid), m1 / str(mid)
        if dst.exists():
            continue
        try:
            dst.symlink_to(src.resolve(), target_is_directory=True)
        except OSError:                                       # e.g. Windows without symlink rights
            shutil.copytree(src, dst)
    splits = json.loads((REPO / "splits.json").read_text())
    needed = {int(m) for part in ("test", "val") for m in splits[part]["skillcorner"]}
    missing = needed - set(ids)
    assert not missing, f"splits.json SkillCorner matches missing from the download: {sorted(missing)}"
    print(f"  {len(ids)} matches; M1 folder m1/data/skillcorner/ ready")


def metrica(dry_run):
    out = REPO / "m1" / "data" / "metrica"
    print("Metrica sample games 1-3 -> m1/data/metrica/")
    for path in METRICA_FILES:
        fetch(github(METRICA_REPO, f"data/{path}"), out / path.split("/", 1)[1], dry_run)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", nargs="+", choices=["idsse", "skillcorner", "metrica"], default=["idsse", "skillcorner", "metrica"])
    ap.add_argument("--dry-run", action="store_true", help="check that every source answers; download nothing")
    a = ap.parse_args()
    for name in a.only:
        {"idsse": idsse, "skillcorner": skillcorner, "metrica": metrica}[name](a.dry_run)
    print("dry run: all sources reachable" if a.dry_run else "done")


if __name__ == "__main__":
    sys.exit(main())
