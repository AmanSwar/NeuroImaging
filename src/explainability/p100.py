"""P100 peak latency and amplitude estimation for cleaned VEP sweeps.

The P100 is the dominant positive deflection of the pattern-reversal VEP
roughly 70-180 ms post-stimulus (ISCEV 2016 standard); its latency is the
clinical hallmark separating demyelinating optic-neuritis delays
(~140 ms) from healthy responses (~100 ms).

The estimator works on an already-filtered signal (the output of
``clean_vep_signal``) so the zero-phase filtering guarantee upstream is
what makes peak timing trustworthy here. Detection:
1. z-score the input defensively (idempotent for already-normalised input),
2. search only the plausible P100 window,
3. take the maximum-amplitude sample as the peak
   (half-max crossing for latency refinement is unnecessary at the
   sample resolutions used here, but the width is reported for QC).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ISCEV-plausible P100 search window, seconds post-stimulus.
P100_WINDOW_S = (0.070, 0.180)


@dataclass(frozen=True)
class P100Measurement:
    latency_ms: float
    amplitude: float       # in the units of the input signal (z-units if cleaned)
    peak_index: int
    window_used: tuple[float, float]  # seconds
    half_width_ms: float   # rough QC width at half peak amplitude

    def as_dict(self) -> dict:
        return {
            "latency_ms": self.latency_ms,
            "amplitude": self.amplitude,
            "peak_index": self.peak_index,
            "window_used_s": list(self.window_used),
            "half_width_ms": self.half_width_ms,
        }


def _validate(signal: np.ndarray, fs: float) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float64).ravel()
    if fs <= 0:
        raise ValueError(f"fs must be positive, got {fs}")
    if signal.size < 8:
        raise ValueError(f"signal too short ({signal.size} samples) for P100 estimation")
    if not np.all(np.isfinite(signal)):
        raise ValueError("signal contains NaN or Inf values")
    return signal


def estimate_p100(
    signal: np.ndarray,
    fs: float,
    window_s: tuple[float, float] = P100_WINDOW_S,
) -> P100Measurement:
    """Estimate P100 latency (ms), amplitude and QC width from a sweep.

    Parameters
    ----------
    signal: 1D VEP sweep, ideally the normalised output of
        ``clean_vep_signal`` (whose ``length`` spans ``duration = L/fs_in``
        seconds at the model's nominal rate).
    fs: effective sample rate of ``signal`` in Hz.
    window_s: inclusive latency search window in seconds.
    """
    signal = _validate(signal, fs)
    n = signal.size
    lo = max(0, int(np.floor(window_s[0] * fs)))
    hi = min(n, int(np.ceil(window_s[1] * fs)) + 1)
    if hi - lo < 3:
        raise ValueError(
            f"search window {window_s} holds only {hi - lo} samples at fs={fs}; "
            "window too narrow or signal too short"
        )

    std = signal.std()
    search = (signal - signal.mean()) / (std + 1e-12)
    seg = search[lo:hi]
    peak_rel = int(np.argmax(seg))
    peak_idx = lo + peak_rel
    peak_val = float(search[peak_idx]) * (std + 1e-12)

    # Rough half-max width around the peak within the search segment (QC).
    half = peak_rel + seg.max() / 2.0
    left = peak_rel
    while left > 0 and seg[left] > half:
        left -= 1
    right = peak_rel
    while right < len(seg) - 1 and seg[right] > half:
        right += 1
    half_width_ms = 1000.0 * (right - left) / fs

    return P100Measurement(
        latency_ms=1000.0 * peak_idx / fs,
        amplitude=peak_val,
        peak_index=peak_idx,
        window_used=tuple(window_s),
        half_width_ms=half_width_ms,
    )


def sweep_duration(source_samples: int, source_fs: float) -> float:
    """Duration in seconds of an acquisition with ``source_samples`` points."""
    if source_fs <= 0:
        raise ValueError(f"source_fs must be positive, got {source_fs}")
    if source_samples <= 0:
        raise ValueError(f"source_samples must be positive, got {source_samples}")
    return source_samples / source_fs


def estimate_p100_after_cleaning(
    cleaned_signal: np.ndarray,
    source_samples: int,
    source_fs: float,
) -> P100Measurement:
    """P100 estimation on the output of ``clean_vep_signal``.

    ``clean_vep_signal`` resamples arbitrary acquisitions to a fixed number
    of points, so the cleaned array's *effective* sample rate differs from
    the acquisition rate whenever lengths differ. The cleaned sweep spans
    the acquisition duration exactly, so:

        fs_effective = len(cleaned) / duration_source

    and latency in milliseconds stays physically correct regardless of the
    resampling.
    """
    cleaned = np.asarray(cleaned_signal, dtype=np.float64).ravel()
    duration_s = sweep_duration(source_samples, source_fs)
    fs_effective = cleaned.size / duration_s
    return estimate_p100(cleaned, fs_effective)
