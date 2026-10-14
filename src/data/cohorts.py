"""Real multi-cohort data: BIDS discovery, 1-D spectrum extraction, manifest.

The public cohorts we can actually download (see ``scripts/download_datasets.py``)
pair a T1-weighted MRI with a *resting* MEG/EEG recording, not a VEP sweep.
Each recording is reduced to a fixed-length 1-D vector -- the channel-averaged
log power spectrum, 1-45 Hz resampled to ``VEPConfig.length`` points -- which
plays the role of the "temporal" stream (``VEPConfig.signal_mode="spectrum"``).
The T1w volume is standardised once (Otsu mask + foreground z-score, 64^3) and
cached as ``.npy`` so repeated training runs do not redo the NIfTI pipeline.

Label scheme: ``0 = control``, ``1 = patient`` (pan-disorder). What "patient"
means differs per cohort (MDD, zoster-associated neuralgia, ...), which is a
deliberate, documented limitation of pooling heterogeneous cohorts.

CLI::

    python -m src.data.cohorts --raw data/raw --out data/processed
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import sys
import time
import traceback
from dataclasses import dataclass

import numpy as np
from scipy.signal import welch

from src.config import MRIConfig

CLASS_NAMES_BINARY = ("Control", "Patient")
FMIN, FMAX = 1.0, 45.0
SPECTRUM_LENGTH = 500
MRI_CONFIG = MRIConfig(mask_method="otsu", normalization="zscore", fit="bbox_resize")


@dataclass(frozen=True)
class CohortSpec:
    dataset: str
    recording: str                     # "ctf_meg" | "edf_eeg"
    control: tuple[str, ...]           # values of the group column mapped to label 0
    patient: tuple[str, ...]           # ... mapped to label 1
    group_column: str = "group"
    recording_glob: str = ""           # glob (relative to the subject dir) for the recording
    description: str = ""


COHORTS: dict[str, CohortSpec] = {
    "ds003568": CohortSpec(
        "ds003568", "ctf_meg", control=("HV",), patient=("MDD",),
        recording_glob="**/meg/*task-rest*_meg.ds",
        description="MDD vs healthy adolescents; resting-state CTF MEG",
    ),
    "ds007181": CohortSpec(
        "ds007181", "edf_eeg", control=("HC",), patient=("ZAN",),
        recording_glob="**/eeg/*_eeg.edf",
        description="Zoster-associated neuralgia vs healthy controls; whole-night PSG EEG",
    ),
    "ds008894": CohortSpec(
        "ds008894", "ctf_meg", control=("HV",), patient=("CC", "SR"),
        recording_glob="**/meg/*_meg.ds",
        description="Mood disorder / suicide-risk vs healthy volunteers; task MEG",
    ),
}


# ---------------------------------------------------------------------------
# 1-D spectrum from an electrophysiology recording
# ---------------------------------------------------------------------------

def log_psd_vector(
    data: np.ndarray,
    sfreq: float,
    length: int = SPECTRUM_LENGTH,
    fmin: float = FMIN,
    fmax: float = FMAX,
    segment_s: float = 4.0,
) -> np.ndarray:
    """Channel-averaged log10 Welch PSD, ``fmin``-``fmax`` Hz, resampled to ``length``.

    ``data`` is ``(n_channels, n_samples)``. Flat/non-finite channels are
    dropped. The scale of the output is arbitrary (units differ between MEG
    and EEG); the dataset layer z-scores it per recording.
    """
    data = np.asarray(data, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError(f"expected (channels, samples), got {data.shape}")
    keep = np.isfinite(data).all(axis=1) & (data.std(axis=1) > 0)
    if not keep.any():
        raise ValueError("no usable (finite, non-flat) channels in recording")
    data = data[keep]
    nperseg = int(min(max(8, round(segment_s * sfreq)), data.shape[1]))
    freqs, psd = welch(data, fs=sfreq, nperseg=nperseg, noverlap=nperseg // 2, axis=-1)
    return _psd_to_vector(freqs, psd, length, fmin, fmax)


def _psd_to_vector(freqs, psd, length, fmin, fmax) -> np.ndarray:
    band = (freqs >= fmin) & (freqs <= fmax)
    if band.sum() < 4:
        raise ValueError(f"recording too short/low-rate for a {fmin}-{fmax} Hz spectrum")
    log_psd = np.log10(psd[:, band] + 1e-300).mean(axis=0)
    grid = np.linspace(fmin, fmax, length)
    return np.interp(grid, freqs[band], log_psd)


def window_starts(n_samples: int, win: int, n_windows: int) -> list[int]:
    """Deterministic, evenly spaced window starts covering the recording."""
    if n_samples <= win:
        return [0]
    n = min(n_windows, max(1, n_samples // win))
    return [int(s) for s in np.linspace(0, n_samples - win, n)]


def spectrum_from_ctf(ds_dir: str, length: int = SPECTRUM_LENGTH,
                      skip_s: float = 10.0, use_s: float = 120.0) -> np.ndarray:
    """Resting CTF-MEG ``*.ds`` directory -> 1-D log-PSD vector (first ``use_s`` s after ``skip_s``)."""
    import mne

    raw = mne.io.read_raw_ctf(ds_dir, preload=False, verbose="ERROR", system_clock="ignore")
    picks = mne.pick_types(raw.info, meg=True, ref_meg=False, eeg=False, exclude="bads")
    if len(picks) == 0:
        raise ValueError(f"no MEG channels in {ds_dir}")
    sf = float(raw.info["sfreq"])
    start = int(min(skip_s, max(0.0, raw.times[-1] - use_s)) * sf)
    stop = min(raw.n_times, start + int(use_s * sf))
    data = raw.get_data(picks=picks, start=start, stop=stop)
    return log_psd_vector(data, sf, length)


_NON_EEG = re.compile(
    r"(eog|emg|ecg|ekg|chin|leg|abd|thor|tho|resp|flow|airflow|snor|spo2|sao2|pulse|"
    r"pos|light|stim|trig|status|marker|event|hr|pleth|effort|sum|dc\d?)", re.I)


def pick_eeg_channels(ch_names: list[str], channels_tsv: str | None) -> list[str]:
    """EEG-only channel names: BIDS ``channels.tsv`` types when present, else a name filter."""
    if channels_tsv and os.path.exists(channels_tsv):
        with open(channels_tsv, newline="") as f:
            rows = list(csv.DictReader(f, delimiter="\t"))
        eeg = [r["name"] for r in rows if str(r.get("type", "")).upper() == "EEG"]
        eeg = [c for c in eeg if c in ch_names]
        if eeg:
            return eeg
    return [c for c in ch_names if not _NON_EEG.search(c)]


def spectrum_from_edf(edf_path: str, length: int = SPECTRUM_LENGTH,
                      n_windows: int = 120, win_s: float = 30.0) -> np.ndarray:
    """Whole-night EDF -> 1-D log-PSD from ``n_windows`` evenly spaced ``win_s``-s windows."""
    import mne

    base = edf_path[: -len("_eeg.edf")] if edf_path.endswith("_eeg.edf") else os.path.splitext(edf_path)[0]
    header = mne.io.read_raw_edf(edf_path, preload=False, verbose="ERROR")
    names = pick_eeg_channels(list(header.ch_names), base + "_channels.tsv")
    if not names:
        raise ValueError(f"no EEG channels found in {edf_path}")
    # Re-open with only the EEG channels: PSG files mix sampling rates across
    # channels, and MNE resamples mixed-rate files with edge artefacts.
    raw = mne.io.read_raw_edf(edf_path, preload=False, include=names, verbose="ERROR")
    sf = float(raw.info["sfreq"])
    win = int(win_s * sf)
    acc, count = None, 0
    nperseg = int(min(4.0 * sf, win))
    for s in window_starts(raw.n_times, win, n_windows):
        seg = raw.get_data(picks=names, start=s, stop=min(raw.n_times, s + win))
        keep = np.isfinite(seg).all(axis=1) & (seg.std(axis=1) > 0)
        if not keep.any():
            continue
        freqs, psd = welch(seg[keep], fs=sf, nperseg=nperseg, noverlap=nperseg // 2, axis=-1)
        # Average *power* over windows per channel (log is taken afterwards).
        acc = (psd.mean(axis=0) if acc is None else acc + psd.mean(axis=0))
        count += 1
    if acc is None:
        raise ValueError(f"no usable EEG windows in {edf_path}")
    mean_psd = (acc / count)[None, :]
    return _psd_to_vector(freqs, mean_psd, length, FMIN, FMAX)


def extract_spectrum(kind: str, path: str, length: int = SPECTRUM_LENGTH) -> np.ndarray:
    if kind == "ctf_meg":
        return spectrum_from_ctf(path, length)
    if kind == "edf_eeg":
        return spectrum_from_edf(path, length)
    raise ValueError(f"unknown recording kind {kind!r}")


# ---------------------------------------------------------------------------
# BIDS discovery
# ---------------------------------------------------------------------------

def read_participants(dataset_dir: str) -> list[dict]:
    path = os.path.join(dataset_dir, "participants.tsv")
    with open(path, newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def find_subject_files(dataset_dir: str, sub: str, spec: CohortSpec) -> tuple[str | None, str | None]:
    """First T1w (prefer .nii.gz) and the first recording for a subject, deterministic."""
    sdir = os.path.join(dataset_dir, sub)
    t1 = sorted(glob.glob(os.path.join(sdir, "**", "anat", "*T1w.nii*"), recursive=True))
    t1 = sorted(t1, key=lambda p: (not p.endswith(".nii.gz"), p))
    rec = sorted(glob.glob(os.path.join(sdir, spec.recording_glob), recursive=True))
    return (t1[0] if t1 else None), (rec[0] if rec else None)


def label_for(row: dict, spec: CohortSpec) -> int | None:
    g = (row.get(spec.group_column) or "").strip()
    if g in spec.control:
        return 0
    if g in spec.patient:
        return 1
    return None


def discover(raw_root: str, datasets: list[str] | None = None) -> list[dict]:
    """Subjects that have a label, a T1w and a recording on disk (sorted, deterministic)."""
    out: list[dict] = []
    for ds in datasets or sorted(COHORTS):
        spec = COHORTS[ds]
        ddir = os.path.join(raw_root, ds)
        if not os.path.exists(os.path.join(ddir, "participants.tsv")):
            continue
        for row in read_participants(ddir):
            sub = row["participant_id"]
            label = label_for(row, spec)
            if label is None:
                continue
            t1, rec = find_subject_files(ddir, sub, spec)
            if t1 and rec:
                out.append(dict(dataset=ds, sub=sub, group=row[spec.group_column].strip(),
                                label=label, t1w=t1, recording=rec, kind=spec.recording))
    return sorted(out, key=lambda r: (r["dataset"], r["sub"]))


# ---------------------------------------------------------------------------
# Preprocess: T1w -> .npy, recording -> spectrum CSV, then manifest
# ---------------------------------------------------------------------------

def _process_one(job: dict) -> dict:
    """Worker: build (or reuse) the cached MRI .npy and spectrum .csv for one subject."""
    from src.preprocessing.mri_transforms import preprocess_mri_file

    mri_p, vep_p = job["mri_out"], job["vep_out"]
    t0 = time.time()
    try:
        if not os.path.exists(mri_p):
            vol = preprocess_mri_file(job["t1w"], MRI_CONFIG)
            if not np.isfinite(vol).all() or float(vol.std()) < 1e-6:
                raise ValueError("preprocessed MRI is empty or non-finite")
            os.makedirs(os.path.dirname(mri_p), exist_ok=True)
            tmp = mri_p + ".tmp.npy"
            np.save(tmp, vol.astype(np.float32))
            os.replace(tmp, mri_p)
        if not os.path.exists(vep_p):
            spec = extract_spectrum(job["kind"], job["recording"], SPECTRUM_LENGTH)
            if not np.isfinite(spec).all():
                raise ValueError("spectrum has non-finite values")
            np.savetxt(vep_p + ".tmp", spec, header="log_psd", comments="", fmt="%.8f")
            os.replace(vep_p + ".tmp", vep_p)
        return dict(job=job["id"], ok=True, seconds=round(time.time() - t0, 1))
    except Exception as e:  # keep going; the report lists every failure
        return dict(job=job["id"], ok=False, error=f"{type(e).__name__}: {e}",
                    trace=traceback.format_exc(limit=3))


def build_manifest(raw_root: str, out_root: str, datasets: list[str] | None = None,
                   workers: int = 2, log=print) -> str:
    """Preprocess every discoverable subject (resumable) and write ``manifest.csv``."""
    subjects = discover(raw_root, datasets)
    if not subjects:
        raise SystemExit(f"no usable subjects under {raw_root}; run scripts/download_datasets.py first")
    jobs = []
    for s in subjects:
        sid = f"{s['dataset']}_{s['sub']}"
        base = os.path.join(out_root, s["dataset"], s["sub"])
        jobs.append(dict(s, id=sid, mri_out=base + "_mri.npy", vep_out=base + "_spectrum.csv"))
    log(f"preprocessing {len(jobs)} subjects with {workers} worker(s) "
        f"(cached files are reused, so this step is resumable)")
    results = []
    if workers > 1:
        import concurrent.futures as cf
        with cf.ProcessPoolExecutor(workers) as ex:
            for i, r in enumerate(ex.map(_process_one, jobs), 1):
                results.append(r)
                if i % 10 == 0 or i == len(jobs):
                    log(f"  {i}/{len(jobs)}")
    else:
        for i, j in enumerate(jobs, 1):
            results.append(_process_one(j))
            if i % 10 == 0 or i == len(jobs):
                log(f"  {i}/{len(jobs)}")
    failed = [r for r in results if not r["ok"]]
    ok_ids = {r["job"] for r in results if r["ok"]}
    rows = []
    for j in jobs:
        if j["id"] not in ok_ids:
            continue
        rows.append(dict(
            subject_id=j["id"], dataset=j["dataset"], group=j["group"], label=j["label"],
            mri_path=os.path.relpath(j["mri_out"], out_root),
            vep_path=os.path.relpath(j["vep_out"], out_root), fs=1.0,
        ))
    os.makedirs(out_root, exist_ok=True)
    manifest = os.path.join(out_root, "manifest.csv")
    with open(manifest, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["subject_id", "dataset", "group", "label", "mri_path", "vep_path", "fs"])
        w.writeheader(); w.writerows(rows)
    with open(os.path.join(out_root, "preprocess_report.json"), "w") as f:
        json.dump(dict(n_subjects=len(jobs), n_ok=len(rows), failed=failed,
                       summary=summarize_rows(rows)), f, indent=1)
    log(f"manifest: {manifest}  ({len(rows)}/{len(jobs)} subjects ok)")
    for r in failed:
        log(f"  FAILED {r['job']}: {r['error']}")
    log(summary_table(rows))
    if not rows:
        raise SystemExit("every subject failed preprocessing; see preprocess_report.json")
    return manifest


def summarize_rows(rows: list[dict]) -> dict:
    out: dict = {}
    for r in rows:
        d = out.setdefault(r["dataset"], {"control": 0, "patient": 0})
        d["patient" if int(r["label"]) == 1 else "control"] += 1
    return out


def summary_table(rows: list[dict]) -> str:
    lines = ["dataset      control  patient"]
    for ds, c in sorted(summarize_rows(rows).items()):
        lines.append(f"{ds:<12} {c['control']:>7}  {c['patient']:>7}")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args(argv)
    build_manifest(args.raw, args.out, args.datasets, args.workers)
    return 0


if __name__ == "__main__":
    sys.exit(main())
