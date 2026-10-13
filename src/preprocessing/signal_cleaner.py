"""1D VEP signal conditioning: zero-phase Butterworth bandpass + mains notch.

Filtering order matters for VEP work: bandpass first removes drift and
high-frequency EMG jitter, then the notch removes the narrowband 50Hz mains
hum. Both filters use zero-phase (filtfilt / sosfiltfilt) application so the
P100 peak latency -- the clinically meaningful measurement -- is not
shifted in time by the filter's group delay.
"""

import numpy as np
from scipy.signal import butter, iirnotch, sosfiltfilt, filtfilt

from src.config import VEPConfig, DEFAULT_CONFIG


def _validate_signal(signal: np.ndarray) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float64)
    if signal.ndim != 1:
        raise ValueError(f"Expected a 1D signal, got shape {signal.shape}")
    if signal.size == 0:
        raise ValueError("Signal is empty")
    if not np.all(np.isfinite(signal)):
        raise ValueError("Signal contains NaN or Inf values")
    return signal


def butterworth_bandpass(
    signal: np.ndarray,
    fs: float,
    lowcut: float = 1.0,
    highcut: float = 100.0,
    order: int = 4,
) -> np.ndarray:
    """Zero-phase 4th-order Butterworth bandpass filter."""
    signal = _validate_signal(signal)
    nyquist = fs / 2.0
    if lowcut <= 0:
        raise ValueError("lowcut must be > 0")
    if highcut >= nyquist:
        raise ValueError(
            f"highcut ({highcut} Hz) must be below the Nyquist frequency "
            f"({nyquist} Hz) for fs={fs} Hz"
        )
    if lowcut >= highcut:
        raise ValueError("lowcut must be < highcut")

    sos = butter(order, [lowcut, highcut], btype="bandpass", fs=fs, output="sos")
    # Zero-phase filtering needs enough samples relative to the filter's
    # padding requirement; short synthetic signals can trip this up.
    min_len = 3 * (2 * len(sos) + 1)
    if signal.size <= min_len:
        raise ValueError(
            f"Signal too short ({signal.size} samples) for zero-phase "
            f"filtering; need > {min_len} samples"
        )
    return sosfiltfilt(sos, signal)


def notch_filter(
    signal: np.ndarray,
    fs: float,
    freq: float = 50.0,
    quality: float = 30.0,
) -> np.ndarray:
    """Zero-phase IIR notch filter to remove AC power-line hum."""
    signal = _validate_signal(signal)
    nyquist = fs / 2.0
    if freq >= nyquist:
        raise ValueError(
            f"notch freq ({freq} Hz) must be below Nyquist ({nyquist} Hz)"
        )
    b, a = iirnotch(freq, quality, fs)
    return filtfilt(b, a, signal)


def zscore_normalize(signal: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Zero mean, unit variance normalization."""
    signal = _validate_signal(signal)
    std = signal.std()
    return (signal - signal.mean()) / (std + eps)


def baseline_correct(signal: np.ndarray, fs: float, window_ms: float) -> np.ndarray:
    """Subtract the mean of the first `window_ms` of the sweep.

    Standard electrophysiology practice: the pre-stimulus interval carries no
    evoked response, so its mean estimates the DC offset of that recording,
    which is removed sweep-by-sweep (rather than globally) so trial-to-trial
    drift does not contaminate latency measurements.
    """
    signal = _validate_signal(signal)
    if window_ms <= 0:
        raise ValueError(f"baseline window must be > 0 ms, got {window_ms}")
    n_baseline = min(signal.size, max(1, int(round(fs * window_ms / 1000.0))))
    return signal - signal[:n_baseline].mean()


def resample_to_length(signal: np.ndarray, target_length: int) -> np.ndarray:
    """Linear-interpolation resample to a fixed number of samples."""
    signal = _validate_signal(signal)
    if signal.size == target_length:
        return signal
    x_old = np.linspace(0.0, 1.0, num=signal.size)
    x_new = np.linspace(0.0, 1.0, num=target_length)
    return np.interp(x_new, x_old, signal)


def clean_vep_signal(
    raw_signal: np.ndarray,
    fs: float,
    cfg: VEPConfig = DEFAULT_CONFIG.vep,
) -> np.ndarray:
    """Full 1D conditioning pipeline:
    baseline correction (optional) -> bandpass -> notch -> resample -> z-score.

    Baseline correction runs first, on the raw sweep, because that is where
    the pre-stimulus offset is defined; filtering after removal of the DC
    offset also avoids filter ringing from the step discontinuity at either
    end of the recording.

    Returns a float32 array of shape (cfg.length,).
    """
    signal = _validate_signal(raw_signal)
    if cfg.baseline_window_ms:
        signal = baseline_correct(signal, fs, cfg.baseline_window_ms)
    filtered = butterworth_bandpass(
        signal, fs, lowcut=cfg.bandpass_low, highcut=cfg.bandpass_high
    )
    denotched = notch_filter(filtered, fs, freq=cfg.notch_freq, quality=cfg.notch_quality)
    resampled = resample_to_length(denotched, cfg.length)
    normalized = zscore_normalize(resampled)
    return normalized.astype(np.float32)
