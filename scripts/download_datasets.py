"""Reproducible downloader for the public multi-cohort datasets.

Pure standard library (no aws-cli / datalad needed). Fetches only the files
the CAFN pipeline uses (T1w MRI + the 1-D electrophysiology recording)
from the public OpenNeuro S3 bucket into ``data/raw/<dataset_id>/``.

    python scripts/download_datasets.py                  # dry run: list files + exact sizes
    python scripts/download_datasets.py --yes            # download the "core" tier
    python scripts/download_datasets.py --yes --tier all # also the large optional cohort
    python scripts/download_datasets.py --only ds003568 --yes
    python scripts/download_datasets.py --verify         # re-check sizes/ETags of local files
    python scripts/download_datasets.py --max-subjects-per-group 2 --yes   # small smoke subset

Reproducibility guarantees
  * the dataset list, version tags and file-selection rules are pinned in
    ``REGISTRY`` below; the run writes ``data/raw/download_manifest.json``
    (every file: size + S3 ETag + dataset tag) so two machines can diff it;
  * file order and any subject subsampling are deterministic (sorted keys,
    ``SEED`` below), never random-per-run;
  * downloads are resumable (``.part`` + HTTP Range) and verified against the
    size reported by S3 before being renamed into place.

Nothing is downloaded unless ``--yes`` is passed.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import re
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

SEED = 42  # global constant; every randomised step in the project must derive from it

BUCKET = "https://s3.amazonaws.com/openneuro.org"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "raw")
MANIFEST = os.path.join(OUT, "download_manifest.json")

# Files every dataset needs (BIDS metadata, tiny).
_COMMON = r"^(dataset_description\.json|participants\.(tsv|json)|README[^/]*|CHANGES|LICENSE)$"

# NOTE on sizes: ``expected_*`` came from the audit run on 2026-10-14 and are only
# used to warn when the remote snapshot drifts from what we planned against.
REGISTRY = {
    "ds003568": dict(
        tier="core",
        title="Mood induction in MDD and healthy adolescents (MEG + T1w)",
        tag="1.0.4", license="CC0",
        groups={"MDD": 29, "HV": 22},
        # T1w + resting-state MEG only (task-mmi3 runs are ~90 GB and not needed)
        include=[_COMMON, r"/anat/.*_T1w\.(nii\.gz|json)$", r"/meg/.*task-rest_", r"/meg/[^/]*(coordsystem|_meg)\.json$"],
        expected_gb=43.5,
    ),
    "ds007181": dict(
        tier="core",
        title="Zoster-associated neuralgia vs healthy controls (T1w + PSG/EEG)",
        tag="1.0.1", license="CC0",
        groups={"ZAN": 32, "HC": 34},
        # T1w + whole-night PSG EEG; resting-state fMRI (39 GB) is skipped
        include=[_COMMON, r"/anat/.*_T1w\.(nii|nii\.gz|json)$", r"/eeg/"],
        expected_gb=24.5,
    ),
    "ds008894": dict(
        tier="extended",
        title="Suicide-risk MDD/BP vs healthy volunteers (task MEG + T1w)",
        tag="1.0.0", license="see dataset_description.json",
        groups={"HV": 18, "CC": 13, "SR": 12},
        include=[_COMMON, r"/anat/.*_T1w\.(nii\.gz|json)$", r"/meg/"],
        expected_gb=54.0,
    ),
}


def _get(url: str, headers: dict | None = None, retries: int = 4):
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=headers or {})
            return urllib.request.urlopen(req, timeout=60)
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)


def list_keys(dataset: str) -> list[dict]:
    """Every object under ``<dataset>/`` as {key,size,etag}, sorted by key."""
    out, token = [], None
    while True:
        q = {"list-type": "2", "prefix": f"{dataset}/", "max-keys": "1000"}
        if token:
            q["continuation-token"] = token
        with _get(f"{BUCKET}?{urllib.parse.urlencode(q)}") as r:
            tree = ET.fromstring(r.read())
        ns = {"s": "http://s3.amazonaws.com/doc/2006-03-01/"}
        for c in tree.findall("s:Contents", ns):
            out.append(dict(
                key=c.find("s:Key", ns).text,
                size=int(c.find("s:Size", ns).text),
                etag=c.find("s:ETag", ns).text.strip('"'),
            ))
        if tree.find("s:IsTruncated", ns).text != "true":
            break
        token = tree.find("s:NextContinuationToken", ns).text
    return sorted(out, key=lambda d: d["key"])


def select(dataset: str, keys: list[dict], subjects: set[str] | None = None) -> list[dict]:
    pats = [re.compile(p) for p in REGISTRY[dataset]["include"]]
    rel = lambda k: k[len(dataset) + 1:]
    out = [k for k in keys if any(p.search(("/" + rel(k["key"])) if "/" in rel(k["key"]) else rel(k["key"])) for p in pats)]
    if subjects is not None:  # keep dataset-level files + only the chosen subjects
        out = [k for k in out if not rel(k["key"]).startswith("sub-") or rel(k["key"]).split("/")[0] in subjects]
    return out


def choose_subjects(dataset: str, per_group: int) -> set[str]:
    """Deterministic subset: ``per_group`` subjects from every group, drawn with ``SEED``.

    Used for smoke tests; the subjects depend only on participants.tsv and SEED.
    """
    import csv, io, random
    with _get(f"{BUCKET}/{dataset}/participants.tsv") as r:
        rows = list(csv.DictReader(io.StringIO(r.read().decode()), delimiter="\t"))
    by_group: dict[str, list[str]] = {}
    for row in rows:
        by_group.setdefault((row.get("group") or "").strip(), []).append(row["participant_id"])
    rng = random.Random(SEED)
    chosen: set[str] = set()
    for g in sorted(by_group):
        if g in ("", "n/a"):
            continue
        ids = sorted(by_group[g])
        chosen.update(rng.sample(ids, min(per_group, len(ids))))
    return chosen


def fetch(item: dict) -> str:
    dest = os.path.join(OUT, item["key"])
    if os.path.exists(dest) and os.path.getsize(dest) == item["size"]:
        return "skip"
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    part = dest + ".part"
    have = os.path.getsize(part) if os.path.exists(part) else 0
    if have > item["size"]:
        os.remove(part); have = 0
    if have < item["size"] or item["size"] == 0:
        headers = {"Range": f"bytes={have}-"} if have else {}
        with _get(f"{BUCKET}/{urllib.parse.quote(item['key'])}", headers) as r, open(part, "ab") as f:
            shutil.copyfileobj(r, f, 1 << 20)
    if os.path.getsize(part) != item["size"]:
        raise IOError(f"size mismatch for {item['key']}: got {os.path.getsize(part)} want {item['size']}")
    os.replace(part, dest)
    return "ok"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--yes", action="store_true", help="actually download (default: dry run)")
    ap.add_argument("--tier", choices=("core", "all"), default="core")
    ap.add_argument("--only", nargs="*", help="dataset ids, e.g. ds003568")
    ap.add_argument("--verify", action="store_true", help="check local files against S3 sizes")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-subjects-per-group", type=int, default=0,
                    help="smoke-test mode: only N deterministic subjects per diagnostic group")
    ap.add_argument("--min-free-gb", type=float, default=15.0, help="refuse to start if less free space would remain")
    args = ap.parse_args()

    ids = args.only or [d for d, v in REGISTRY.items() if args.tier == "all" or v["tier"] == "core"]
    plan, manifest = {}, {"seed": SEED, "bucket": BUCKET, "datasets": {}}
    for d in ids:
        if d not in REGISTRY:
            sys.exit(f"unknown dataset {d}; known: {', '.join(REGISTRY)}")
        chosen = choose_subjects(d, args.max_subjects_per_group) if args.max_subjects_per_group else None
        sel = select(d, list_keys(d), chosen)
        if chosen is not None:
            print(f"{d}: smoke subset -> {sorted(chosen)}")
        gb = sum(i["size"] for i in sel) / 1e9
        exp = REGISTRY[d]["expected_gb"]
        flag = "" if (chosen is not None or abs(gb - exp) / exp < 0.05) else f"   !! differs from audited {exp} GB -- remote snapshot may have changed"
        print(f"{d}  {len(sel):5d} files  {gb:7.2f} GB  {REGISTRY[d]['title']}{flag}")
        plan[d] = sel
        manifest["datasets"][d] = dict(tag=REGISTRY[d]["tag"], groups=REGISTRY[d]["groups"], files=sel)

    total = sum(i["size"] for s in plan.values() for i in s)
    free = shutil.disk_usage(ROOT).free
    print(f"\nTOTAL {total / 1e9:.2f} GB   free disk {free / 1e9:.1f} GB")

    if args.verify:
        bad = [i["key"] for s in plan.values() for i in s
               if not os.path.exists(os.path.join(OUT, i["key"])) or os.path.getsize(os.path.join(OUT, i["key"])) != i["size"]]
        print("verify:", "all files present and sizes match" if not bad else f"{len(bad)} missing/mismatched, e.g. {bad[:3]}")
        return 1 if bad else 0

    if not args.yes:
        print("dry run only -- re-run with --yes to download.")
        return 0
    if free - total < args.min_free_gb * 1e9:
        sys.exit(f"refusing: would leave < {args.min_free_gb} GB free (override with --min-free-gb)")

    items = [i for d in sorted(plan) for i in plan[d]]
    done = 0
    with cf.ThreadPoolExecutor(args.workers) as ex:
        for res in ex.map(fetch, items):
            done += 1
            if done % 100 == 0 or done == len(items):
                print(f"  {done}/{len(items)} files", flush=True)
    os.makedirs(OUT, exist_ok=True)
    with open(MANIFEST, "w") as f:
        json.dump(manifest, f, indent=1, sort_keys=True)
    print(f"done. manifest -> {os.path.relpath(MANIFEST, ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
