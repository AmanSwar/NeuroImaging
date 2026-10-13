"""Tests for P100 latency estimation on cleaned VEP sweeps."""

import numpy as np
import pytest

from src.data.synthetic import VEP_CLASS_PARAMS, synthetic_vep_signal
from src.explainability.p100 import (
    P100_WINDOW_S,
    estimate_p100,
    estimate_p100_after_cleaning,
    sweep_duration,
)
from src.preprocessing.signal_cleaner import clean_vep_signal


class TestSyntheticGroundTruth:
    """Synthetic sweeps draw latency ~ N(class mean, sd) per sample; average
    several sweeps per class so the jitter variance does not dominate."""

    FS = 250.0
    N_SWEEPS = 5

    @pytest.mark.parametrize("label", [0, 1, 2])
    def test_recovers_class_latency_within_tolerance(self, label, rng):
        params = VEP_CLASS_PARAMS[label]
        tolerances_ms = 4.0 * params["latency_sd"] + 1000.0 / self.FS + 2.0
        latencies = []
        for _ in range(self.N_SWEEPS):
            raw = synthetic_vep_signal(label, fs=self.FS, rng=rng)
            cleaned = clean_vep_signal(raw, self.FS)
            latencies.append(estimate_p100(cleaned, self.FS).latency_ms)
        mean_latency = float(np.mean(latencies))
        assert abs(mean_latency - params["latency_ms"]) <= tolerances_ms

    def test_neuritis_latency_exceeds_healthy(self, rng):
        def mean_latency(label):
            values = []
            for _ in range(self.N_SWEEPS):
                cleaned = clean_vep_signal(
                    synthetic_vep_signal(label, fs=self.FS, rng=rng), self.FS
                )
                values.append(estimate_p100(cleaned, self.FS).latency_ms)
            return float(np.mean(values))

        healthy = mean_latency(0)
        neuritis = mean_latency(2)
        assert neuritis - healthy > 0.6 * (
            VEP_CLASS_PARAMS[2]["latency_ms"] - VEP_CLASS_PARAMS[0]["latency_ms"]
        )


class TestEstimateP100:
    def test_pure_gaussian_peak(self):
        fs = 500.0
        t = np.arange(0, 1.0, 1 / fs)
        signal = np.exp(-((t - 0.12) ** 2) / (2 * 0.01**2))
        meas = estimate_p100(signal, fs)
        assert meas.latency_ms == pytest.approx(120.0, abs=2.5)
        assert meas.amplitude == pytest.approx(1.0, abs=0.05)
        assert meas.peak_index == int(round(0.12 * fs))

    def test_ignores_huge_off_window_artifact(self):
        fs = 250.0
        t = np.arange(0, 2.0, 1 / fs)
        signal = np.exp(-((t - 0.12) ** 2) / (2 * 0.01**2))
        signal += 50 * np.exp(-((t - 0.6) ** 2) / (2 * 0.02**2))  # outside window
        meas = estimate_p100(signal, fs)
        assert meas.latency_ms == pytest.approx(120.0, abs=4.5)

    def test_custom_window(self):
        fs = 250.0
        t = np.arange(0, 2.0, 1 / fs)
        signal = np.exp(-((t - 0.25) ** 2) / (2 * 0.01**2))
        meas = estimate_p100(signal, fs, window_s=(0.2, 0.3))
        assert meas.latency_ms == pytest.approx(250.0, abs=4.5)
        assert meas.window_used == (0.2, 0.3)

    def test_window_too_narrow_raises(self):
        with pytest.raises(ValueError):
            estimate_p100(np.zeros(16), 4.0, window_s=(0.0, 0.1))

    def test_rejects_bad_inputs(self):
        with pytest.raises(ValueError):
            estimate_p100(np.zeros(100), fs=0)
        with pytest.raises(ValueError):
            estimate_p100(np.zeros(4), fs=250.0)
        with pytest.raises(ValueError):
            estimate_p100(np.full(100, np.nan), fs=250.0)

    def test_as_dict_serialisable(self):
        fs = 250.0
        t = np.arange(0, 2.0, 1 / fs)
        meas = estimate_p100(np.exp(-((t - 0.1) ** 2) / (2 * 0.01**2)), fs)
        d = meas.as_dict()
        assert set(d) == {
            "latency_ms", "amplitude", "peak_index",
            "window_used_s", "half_width_ms",
        }
        assert d["latency_ms"] == pytest.approx(100.0, abs=4.5)

    def test_default_window_matches_iscev(self):
        assert P100_WINDOW_S[0] <= 0.075 and P100_WINDOW_S[1] >= 0.145


class TestAfterCleaning:
    def test_effective_rate_correction_for_downsampled_source(self, rng):
        """A 500 Hz acquisition resampled to 500 points must not be read at
        250 Hz -- latency would come out 2x too large."""
        fs = 500.0
        t = np.arange(0, 2.0, 1 / fs)  # 1000 samples @ 500 Hz
        raw = np.exp(-((t - 0.14) ** 2) / (2 * 0.01**2)) + 0.02 * rng.standard_normal(t.size)
        cleaned = clean_vep_signal(raw, fs)  # now exactly 500 samples
        naive = estimate_p100(cleaned, fs)  # WRONG: assumes 250 Hz
        corrected = estimate_p100_after_cleaning(cleaned, source_samples=1000, source_fs=fs)
        assert corrected.latency_ms == pytest.approx(140.0, abs=4.0)
        assert abs(naive.latency_ms - 140.0) > abs(corrected.latency_ms - 140.0)

    def test_identity_when_rates_match(self, rng):
        fs = 250.0
        raw = synthetic_vep_signal(0, fs=fs, rng=rng)
        cleaned = clean_vep_signal(raw, fs)
        via_helper = estimate_p100_after_cleaning(cleaned, source_samples=len(raw), source_fs=fs)
        direct = estimate_p100(cleaned, fs)
        assert via_helper.latency_ms == direct.latency_ms

    def test_sweep_duration(self):
        assert sweep_duration(500, 250.0) == pytest.approx(2.0)
        with pytest.raises(ValueError):
            sweep_duration(500, 0)
        with pytest.raises(ValueError):
            sweep_duration(0, 250.0)
