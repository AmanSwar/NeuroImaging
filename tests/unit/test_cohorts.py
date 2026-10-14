"""Tests for src/data/cohorts.py: spectrum extraction, BIDS discovery, manifest build."""

import csv
import os

import nibabel as nib
import numpy as np
import pytest

from src.config import Config, ModelConfig, VEPConfig
from src.data import cohorts as C
from src.data.dataset import PairedNeuroDataset


def test_log_psd_vector_peaks_at_signal_frequency():
    sf = 250.0
    t = np.arange(int(60 * sf)) / sf
    rng = np.random.default_rng(0)
    data = np.stack([np.sin(2 * np.pi * 10 * t) + 0.1 * rng.standard_normal(t.size) for _ in range(3)])
    v = C.log_psd_vector(data, sf)
    assert v.shape == (C.SPECTRUM_LENGTH,)
    freqs = np.linspace(C.FMIN, C.FMAX, C.SPECTRUM_LENGTH)
    assert abs(freqs[int(np.argmax(v))] - 10.0) < 0.6


def test_log_psd_vector_drops_flat_channels_and_rejects_all_flat():
    sf = 200.0
    rng = np.random.default_rng(1)
    good = rng.standard_normal((1, 4000))
    flat = np.zeros((2, 4000))
    np.testing.assert_allclose(C.log_psd_vector(np.vstack([good, flat]), sf), C.log_psd_vector(good, sf))
    with pytest.raises(ValueError, match="usable"):
        C.log_psd_vector(flat, sf)
    with pytest.raises(ValueError, match="channels, samples"):
        C.log_psd_vector(np.zeros(10), sf)


def test_log_psd_vector_too_short_or_slow_recording_raises():
    with pytest.raises(ValueError, match="too short"):
        C.log_psd_vector(np.random.default_rng(0).standard_normal((1, 6)), 20.0)


def test_window_starts_are_deterministic_and_in_range():
    s = C.window_starts(100_000, 6_000, 10)
    assert s == C.window_starts(100_000, 6_000, 10)
    assert len(s) == 10 and s[0] == 0 and s[-1] == 94_000
    assert C.window_starts(100, 6_000, 10) == [0]
    assert len(C.window_starts(30_000, 6_000, 120)) == 5  # never more windows than fit


def test_pick_eeg_channels_uses_channels_tsv_then_name_filter(tmp_path):
    tsv = tmp_path / "x_channels.tsv"
    tsv.write_text("name\ttype\nC3\tEEG\nO1\tEEG\nEOG1\tEOG\nEMG\tEMG\n")
    assert C.pick_eeg_channels(["C3", "O1", "EOG1", "EMG"], str(tsv)) == ["C3", "O1"]
    assert C.pick_eeg_channels(["C3-A2", "EOG-L", "ECG", "SpO2", "O2-A1"], None) == ["C3-A2", "O2-A1"]


def _write_nifti(path, seed, shape=(28, 30, 26)):
    rng = np.random.default_rng(seed)
    g = np.indices(shape).astype(float)
    c = np.array(shape)[:, None, None, None] / 2
    r = np.sqrt(((g - c) ** 2).sum(0))
    vol = np.where(r < min(shape) * 0.38, 100.0 + 5 * rng.standard_normal(shape), 2.0 * rng.random(shape))
    nib.save(nib.Nifti1Image(vol.astype(np.float32), np.eye(4)), str(path))


def _fake_bids(root, ds, groups):
    d = root / ds
    d.mkdir(parents=True)
    rows = [("participant_id", "group")]
    for i, g in enumerate(groups):
        sub = f"sub-{i:02d}"
        rows.append((sub, g))
        (d / sub / "anat").mkdir(parents=True)
        _write_nifti(d / sub / "anat" / f"{sub}_T1w.nii.gz", i)
        rec = d / sub / ("meg" if ds == "ds003568" else "eeg")
        rec.mkdir(parents=True)
        if ds == "ds003568":
            (rec / f"{sub}_task-rest_run-1_meg.ds").mkdir()
        else:
            (rec / f"{sub}_task-sleep_acq-PSG_eeg.edf").write_bytes(b"")
    (d / "participants.tsv").write_text("\n".join("\t".join(r) for r in rows) + "\n")


def test_label_for_and_discover_skip_unlabelled_and_incomplete(tmp_path):
    _fake_bids(tmp_path, "ds003568", ["HV", "MDD", "OTHER"])
    (tmp_path / "ds003568" / "sub-01" / "anat" / "sub-01_T1w.nii.gz").unlink()  # missing T1w
    found = C.discover(str(tmp_path), ["ds003568"])
    assert [(f["sub"], f["label"]) for f in found] == [("sub-00", 0)]
    assert C.label_for({"group": " MDD "}, C.COHORTS["ds003568"]) == 1
    assert C.label_for({"group": "zzz"}, C.COHORTS["ds003568"]) is None


def test_build_manifest_roundtrips_into_paired_dataset(tmp_path, monkeypatch):
    raw, out = tmp_path / "raw", tmp_path / "proc"
    _fake_bids(raw, "ds003568", ["HV", "MDD", "HV", "MDD"])
    _fake_bids(raw, "ds007181", ["HC", "ZAN", "HC", "ZAN"])
    calls = []

    def fake_extract(kind, path, length=C.SPECTRUM_LENGTH):
        calls.append(kind)
        return np.linspace(-1, 1, length) + (0.5 if "sub-01" in path or "sub-03" in path else 0.0)

    monkeypatch.setattr(C, "extract_spectrum", fake_extract)
    manifest = C.build_manifest(str(raw), str(out), workers=1, log=lambda *_: None)
    rows = list(csv.DictReader(open(manifest)))
    assert len(rows) == 8 and {r["dataset"] for r in rows} == {"ds003568", "ds007181"}
    assert sorted(set(calls)) == ["ctf_meg", "edf_eeg"]
    assert all(os.path.exists(out / r["mri_path"]) and os.path.exists(out / r["vep_path"]) for r in rows)

    cfg = Config(vep=VEPConfig(signal_mode="spectrum"), model=ModelConfig(num_classes=2))
    ds = PairedNeuroDataset(manifest, cfg=cfg)
    item = ds[0]
    assert item["mri"].shape == (1, 64, 64, 64) and item["vep"].shape == (1, 500)
    assert abs(float(item["vep"].mean())) < 1e-5 and abs(float(item["vep"].std()) - 1) < 1e-2
    assert ds.cohorts[0] == "ds003568" and sorted(set(ds.labels)) == [0, 1]

    # resumable: a second build reuses cached files and never re-extracts
    calls.clear()
    C.build_manifest(str(raw), str(out), workers=1, log=lambda *_: None)
    assert calls == []


def test_build_manifest_reports_failures_but_keeps_going(tmp_path, monkeypatch):
    raw, out = tmp_path / "raw", tmp_path / "proc"
    _fake_bids(raw, "ds003568", ["HV", "MDD", "HV"])

    def flaky(kind, path, length=C.SPECTRUM_LENGTH):
        if "sub-01" in path:
            raise ValueError("corrupt recording")
        return np.linspace(0, 1, length)

    monkeypatch.setattr(C, "extract_spectrum", flaky)
    manifest = C.build_manifest(str(raw), str(out), workers=1, log=lambda *_: None)
    assert len(list(csv.DictReader(open(manifest)))) == 2
    import json
    rep = json.load(open(out / "preprocess_report.json"))
    assert rep["n_ok"] == 2 and "corrupt recording" in rep["failed"][0]["error"]


def test_extract_spectrum_rejects_unknown_kind():
    with pytest.raises(ValueError):
        C.extract_spectrum("nope", "x")


def test_spectrum_mode_skips_time_domain_filtering(tmp_path):
    """A bandpass/notch would distort a spectrum; spectrum mode only resamples + z-scores."""
    spec = np.sin(np.linspace(0, 20, 300)) + 3.0
    vep = tmp_path / "s.csv"; np.savetxt(vep, spec, header="log_psd", comments="")
    mri = tmp_path / "m.npy"; np.save(mri, np.zeros((1, 64, 64, 64), np.float32))
    man = tmp_path / "manifest.csv"
    man.write_text(f"subject_id,dataset,label,mri_path,vep_path,fs\ns,d,1,{mri},{vep},1.0\n")
    ds = PairedNeuroDataset(str(man), cfg=Config(vep=VEPConfig(signal_mode="spectrum"), model=ModelConfig(num_classes=2)))
    got = ds[0]["vep"][0].numpy()
    ref = np.interp(np.linspace(0, 1, 500), np.linspace(0, 1, 300), spec)
    ref = (ref - ref.mean()) / (ref.std() + 1e-8)
    np.testing.assert_allclose(got, ref, atol=1e-4)
