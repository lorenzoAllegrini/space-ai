"""Utility functions for feature extraction from time series segments."""

from __future__ import annotations
from functools import partial
from typing import (
    Any,
    Callable,
    Dict,
    Optional,
)
import numpy as np
from scipy import signal as sig  # type: ignore
from scipy.stats import (  # type: ignore
    kurtosis,
    skew,
)
import stumpy
from joblib import Parallel, delayed

def apply_statistic_to_batch(
    segments: np.ndarray,
    func: Callable,
    preprocess: Callable | None = None,
    func_kwargs: Dict[str, Any] | None = None,
) -> np.ndarray:
    """
    Apply a statistic function to a batch of segments.

    Args:
        segments: 2D array of segments (n_segments, window_size).
        func: Function to apply to the segments. Must accept a 2D array.
        preprocess: Optional function to apply to segments BEFORE statistic.
        func_kwargs: Optional keyword arguments to pass to func (e.g., axis=1).
    """
    if preprocess is not None:
        segments = preprocess(segments)

    kwargs = func_kwargs or {}
    with np.errstate(divide="ignore", invalid="ignore"):
        return func(segments, **kwargs)

def count_peaks_vectorized(windows: np.ndarray) -> np.ndarray:
    """
    Vectorized peak counting in pure NumPy.
    """
    if windows.shape[1] < 3:
        return np.zeros(windows.shape[0], dtype=int)

    left = windows[:, :-2]
    center = windows[:, 1:-1]
    right = windows[:, 2:]
    
    is_peak = (center > left) & (center > right)
    
    prominences = 0.1 * np.ptp(windows, axis=1, keepdims=True)
    
    is_prominent = ((center - left) >= prominences) | ((center - right) >= prominences)
    
    valid_peaks = is_peak & is_prominent
    return np.sum(valid_peaks, axis=1)


def dominant_frequency_energy(windows: np.ndarray) -> np.ndarray:
    """
    Calculate the energy (FFT power) of the dominant frequency.
    
    Args:
        windows: 2D array (batch_size, window_len).
        
    Returns:
        1D array with the power of the main frequency peak.
    """
    if windows.shape[1] < 2:
        return np.zeros(windows.shape[0], dtype=float)
        
    fft_vals = np.abs(np.fft.rfft(windows, axis=1))
    power_spectrum = (fft_vals ** 2) / windows.shape[1]
    
    if power_spectrum.shape[1] > 1:
        return np.max(power_spectrum[:, 1:], axis=1)
    return np.zeros(windows.shape[0], dtype=float)


def dominant_frequency_index(windows: np.ndarray) -> np.ndarray:
    """Return the index (bin) of the frequency with maximum power."""
    if windows.shape[1] < 2:
        return np.zeros(windows.shape[0], dtype=float)
    fft_vals = np.abs(np.fft.rfft(windows, axis=1))
    if fft_vals.shape[1] > 1:
        return np.argmax(fft_vals[:, 1:], axis=1) + 1
    return np.zeros(windows.shape[0], dtype=float)


def spectral_entropy(windows: np.ndarray) -> np.ndarray:
    """Calculate the normalized spectral entropy (0-1)."""
    if windows.shape[1] < 2:
        return np.zeros(windows.shape[0], dtype=float)
    fft_vals = np.abs(np.fft.rfft(windows, axis=1))
    psd = (fft_vals ** 2) / windows.shape[1]
    psd_norm = psd / (np.sum(psd, axis=1, keepdims=True) + 1e-12)
    entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-12), axis=1)
    return entropy / np.log2(psd.shape[1])


def spectral_centroid(windows: np.ndarray) -> np.ndarray:
    """Calculate the power-weighted average frequency (centroid)."""
    if windows.shape[1] < 2:
        return np.zeros(windows.shape[0], dtype=float)
    fft_vals = np.abs(np.fft.rfft(windows, axis=1))
    psd = (fft_vals ** 2) / windows.shape[1]
    freqs = np.arange(psd.shape[1])
    centroid = np.sum(psd * freqs, axis=1) / (np.sum(psd, axis=1) + 1e-12)
    return centroid


def spectral_spread(windows: np.ndarray) -> np.ndarray:
    """Calculate the spectral spread (variance) around the centroid."""
    if windows.shape[1] < 2:
        return np.zeros(windows.shape[0], dtype=float)
    fft_vals = np.abs(np.fft.rfft(windows, axis=1))
    psd = (fft_vals ** 2) / windows.shape[1]
    freqs = np.arange(psd.shape[1])
    centroid = np.sum(psd * freqs, axis=1, keepdims=True) / (np.sum(psd, axis=1, keepdims=True) + 1e-12)
    spread = np.sqrt(np.sum(psd * (freqs - centroid)**2, axis=1) / (np.sum(psd, axis=1) + 1e-12))
    return spread


def spectral_rolloff(windows: np.ndarray, roll_percent: float = 0.85) -> np.ndarray:
    """Calculate the frequency below which a specified percentage of energy resides."""
    if windows.shape[1] < 2:
        return np.zeros(windows.shape[0], dtype=float)
    fft_vals = np.abs(np.fft.rfft(windows, axis=1))
    psd = (fft_vals ** 2) / windows.shape[1]
    total_energy = np.sum(psd, axis=1, keepdims=True)
    cumulative_energy = np.cumsum(psd, axis=1)
    threshold = roll_percent * total_energy
    rolloff = np.argmax(cumulative_energy >= threshold, axis=1)
    return rolloff.astype(float)


def _safe_savgol(x, target_window, polyorder=2):
    """
    Safely apply the Savitzky-Golay filter.
    """
    L = x.shape[-1]
    if L < polyorder + 1:
        return x
    if L < target_window:
        window_length = L if L % 2 == 1 else L - 1
    else:
        window_length = target_window
    return sig.savgol_filter(x, window_length=window_length, polyorder=polyorder, axis=-1)


def smooth_10(x):
    """Apply Savitzky-Golay smoothing with target window size 11."""
    return _safe_savgol(x, target_window=11)


def smooth_20(x):
    """Apply Savitzky-Golay smoothing with target window size 21."""
    return _safe_savgol(x, target_window=21)


def diff1(x):
    """Compute the first-order temporal difference."""
    d = np.diff(x, axis=-1)
    if x.ndim == 1:
        return np.concatenate(([0], d))
    return np.column_stack((np.zeros(x.shape[0]), d))


def peak_to_peak_amplitude(windows: np.ndarray) -> np.ndarray:
    """
    Calculate the Peak-to-Peak amplitude (Max - Min) for each segment.
    """
    if windows.shape[1] < 1:
        return np.zeros(windows.shape[0], dtype=float)
    return np.ptp(windows, axis=1)


def internal_matrix_profile_max(windows: np.ndarray, m: Optional[int] = None) -> np.ndarray:
    """
    Calculate the maximum morphological anomaly (Discord) within each window.
    """
    if windows.shape[1] < 3:
        return np.zeros(windows.shape[0])
    if m is None:
        m = max(3, int(windows.shape[1] * 0.2))
    if windows.shape[1] <= m * 2:
        return np.zeros(windows.shape[0], dtype=float)

    def _compute_single_mp_max(w, m_val):
        mp = stumpy.stump(w.astype(np.float64), m_val)
        return np.max(mp[:, 0])

    discord_scores = Parallel(n_jobs=-1)(
        delayed(_compute_single_mp_max)(windows[i], m) 
        for i in range(windows.shape[0])
    )
    return np.array(discord_scores)


def diff2(x):
    """Compute the second-order temporal difference."""
    d = np.diff(x, n=2, axis=-1)
    if x.ndim == 1:
        return np.concatenate(([0, 0], d))
    return np.column_stack((np.zeros((x.shape[0], 2)), d))


def mean_crossing_rate(windows: np.ndarray) -> np.ndarray:
    """Calculate the frequency of signal crossings through the local mean."""
    if windows.shape[1] < 2:
        return np.zeros(windows.shape[0], dtype=float)
    means = np.mean(windows, axis=1, keepdims=True)
    centered = windows - means
    crossings = np.sum(np.diff(np.signbit(centered), axis=1), axis=1)
    return crossings / windows.shape[1]


def waveform_length(windows: np.ndarray) -> np.ndarray:
    """Measure the cumulative path length of the signal (complexity/vibration)."""
    if windows.shape[1] < 2:
        return np.zeros(windows.shape[0], dtype=float)
    return np.sum(np.abs(np.diff(windows, axis=1)), axis=1)


def lag1_autocorrelation(windows: np.ndarray) -> np.ndarray:
    """Measure short-term signal predictability (smoothness)."""
    if windows.shape[1] < 2:
        return np.zeros(windows.shape[0], dtype=float)
    means = np.mean(windows, axis=1, keepdims=True)
    centered = windows - means
    var = np.sum(centered**2, axis=1)
    var = np.where(var == 0, 1e-12, var)
    autocov = np.sum(centered[:, :-1] * centered[:, 1:], axis=1)
    return autocov / var


def safe_kill_switch_function(windows: np.ndarray) -> np.ndarray:
    """
    Emergency kill-switch function providing a near-constant signal.
    """
    if windows.shape[0] == 0:
        return np.array([])
    res = np.ones(windows.shape[0], dtype=float)
    if len(res) > 1:
        res[-1] = 1.00001 
    return res


def crest_factor(windows: np.ndarray) -> np.ndarray:
    """Ratio of absolute peak to root mean square of the centered signal."""
    return np.full(windows.shape[0], 1e-6)


def iqr_robust(windows: np.ndarray) -> np.ndarray:
    """Measure dispersion while ignoring extreme outliers."""
    if windows.shape[1] < 2:
        return np.zeros(windows.shape[0], dtype=float)
    q75 = np.percentile(windows, 75, axis=1)
    q25 = np.percentile(windows, 25, axis=1)
    return q75 - q25


def hjorth_mobility(windows: np.ndarray) -> np.ndarray:
    """Estimate apparent average frequency in the time domain."""
    if windows.shape[1] < 2:
        return np.zeros(windows.shape[0], dtype=float)
    var_signal = np.var(windows, axis=1)
    var_diff = np.var(np.diff(windows, axis=1), axis=1)
    return np.sqrt(var_diff / (var_signal + 1e-12))


def local_median(windows: np.ndarray) -> np.ndarray:
    """
    Capture baseline structural drops while ignoring asymmetric micro-peaks.
    """
    if windows.shape[1] == 0:
        return np.zeros(windows.shape[0], dtype=float)
    return np.median(windows, axis=1)


def trend_slope(windows: np.ndarray) -> np.ndarray:
    """
    Calculate the linear regression slope for each window.
    """
    if windows.shape[1] < 2:
        return np.zeros(windows.shape[0], dtype=float)
    w_size = windows.shape[1]
    x = np.arange(w_size)
    x_centered = x - np.mean(x)
    ss_x = np.sum(x_centered**2)
    y_mean = np.mean(windows, axis=1, keepdims=True)
    y_centered = windows - y_mean
    slope = np.sum(y_centered * x_centered, axis=1) / ss_x
    return slope


def lag5_autocorrelation(windows: np.ndarray) -> np.ndarray:
    """
    Calculate Pearson autocorrelation with a 5-step lag.
    """
    if windows.shape[1] <= 5:
        return np.zeros(windows.shape[0], dtype=float)
    x_t = windows[:, 5:]
    x_t_minus_5 = windows[:, :-5]
    mean_t = np.mean(x_t, axis=1, keepdims=True)
    mean_t_minus_5 = np.mean(x_t_minus_5, axis=1, keepdims=True)
    x_t_centered = x_t - mean_t
    x_t_minus_5_centered = x_t_minus_5 - mean_t_minus_5
    numerator = np.sum(x_t_centered * x_t_minus_5_centered, axis=1)
    var_t = np.sum(x_t_centered**2, axis=1)
    var_t_minus_5 = np.sum(x_t_centered**2, axis=1)
    denominator = np.sqrt(var_t * var_t_minus_5)
    with np.errstate(divide='ignore', invalid='ignore'):
        corr = np.where(denominator == 0, 0.0, numerator / denominator)
    return corr


def diff_zcr(windows: np.ndarray) -> np.ndarray:
    """
    Count zero crossings of the first derivative.
    """
    if windows.shape[1] < 3:
        return np.zeros(windows.shape[0], dtype=float)
    diffs = np.diff(windows, axis=1)
    sign_changes = (diffs[:, :-1] * diffs[:, 1:]) < 0
    return np.sum(sign_changes, axis=1)


def ewma_difference(windows: np.ndarray) -> np.ndarray:
    """
    Calculate the difference between current value and Exponentially Weighted Moving Average.
    """
    if windows.shape[1] < 2:
        return np.zeros(windows.shape[0], dtype=float)
    alpha = 0.3
    w_size = windows.shape[1]
    powers = np.arange(w_size - 1, -1, -1)
    weights = (1 - alpha) ** powers
    weights = weights / np.sum(weights)
    expected_trend = np.sum(windows * weights, axis=1)
    actual_values = windows[:, -1]
    return actual_values - expected_trend


def tkeo_mean(windows: np.ndarray) -> np.ndarray:
    """
    Calculate the average Teager-Kaiser Energy Operator (TKEO) for the window.
    """
    if windows.shape[1] < 3:
        return np.zeros(windows.shape[0], dtype=float)
    x_sq = windows[:, 1:-1] ** 2
    x_adj = windows[:, :-2] * windows[:, 2:]
    tkeo = x_sq - x_adj
    return np.mean(tkeo, axis=1)


def fractal_roughness(windows: np.ndarray) -> np.ndarray:
    """
    Calculate the ratio of total signal path length to net endpoint displacement.
    """
    if windows.shape[1] < 2:
        return np.zeros(windows.shape[0], dtype=float)
    path_length = np.sum(np.abs(np.diff(windows, axis=1)), axis=1)
    net_distance = np.abs(windows[:, -1] - windows[:, 0])
    return path_length / (net_distance + 1e-6)


def signal_monotonicity(windows: np.ndarray) -> np.ndarray:
    """
    Measure the directional consistency of the signal movement.
    """
    if windows.shape[1] < 2:
        return np.zeros(windows.shape[0], dtype=float)
    diffs = np.diff(windows, axis=1)
    steps = np.sign(diffs)
    net_direction = np.abs(np.sum(steps, axis=1))
    max_possible_steps = windows.shape[1] - 1
    return net_direction / max_possible_steps


def cusum_deviation(windows: np.ndarray) -> np.ndarray:
    """
    Calculate the maximum excursion of the Cumulative Sum (CUSUM).
    """
    if windows.shape[1] < 2:
        return np.zeros(windows.shape[0], dtype=float)
    window_means = np.mean(windows, axis=1, keepdims=True)
    centered = windows - window_means
    cumulative_sum = np.cumsum(centered, axis=1)
    return np.max(np.abs(cumulative_sum), axis=1)


def root_mean_square(windows: np.ndarray) -> np.ndarray:
    """
    Calculate the Root Mean Square (RMS) of the window.
    """
    if windows.shape[1] == 0:
        return np.zeros(windows.shape[0], dtype=float)
    return np.sqrt(np.mean(windows**2, axis=1))


def time_reversibility(windows: np.ndarray) -> np.ndarray:
    """
    Measure temporal asymmetry using the 3rd moment of the derivative.
    """
    if windows.shape[1] < 2:
        return np.zeros(windows.shape[0], dtype=float)
    diffs = np.diff(windows, axis=1)
    return np.mean(diffs**3, axis=1)


def poincare_ratio(windows: np.ndarray) -> np.ndarray:
    """
    Calculate the ratio of short-term (SD1) to long-term (SD2) variability.
    """
    if windows.shape[1] < 2:
        return np.zeros(windows.shape[0], dtype=float)
    x_t = windows[:, :-1]
    x_t_plus_1 = windows[:, 1:]
    sd1_sq = np.var(x_t - x_t_plus_1, axis=1)
    sd2_sq = np.var(x_t + x_t_plus_1, axis=1)
    return np.sqrt(sd1_sq / (sd2_sq + 1e-6))


def _make_stat(func, preprocess=None, **kwargs):
    return partial(
        apply_statistic_to_batch,
        func=func,
        preprocess=preprocess,
        func_kwargs=kwargs,
    )


FEATURE_MAP = {
    #"mean": _make_stat(np.mean, axis=1),
    "var": _make_stat(np.var, axis=1),
    #std": _make_stat(np.std, axis=1),
    "n_peaks": _make_stat(count_peaks_vectorized),
    "smooth10_n_peaks": _make_stat(count_peaks_vectorized, preprocess=smooth_10),
    #"smooth20_n_peaks": _make_stat(count_peaks_vectorized, preprocess=smooth_20),
    "diff_peaks": _make_stat(count_peaks_vectorized, preprocess=diff1),
    #"diff2_peaks": _make_stat(count_peaks_vectorized, preprocess=diff2),
    "diff_var": _make_stat(np.var, preprocess=diff1, axis=1),
    "diff2_var": _make_stat(np.var, preprocess=diff2, axis=1),
    "kurtosis": _make_stat(kurtosis, axis=1, fisher=True, bias=False),
    "peak_to_peak_amplitude": _make_stat(peak_to_peak_amplitude),
    #"internal_matrix_profile_max": _make_stat(internal_matrix_profile_max),
    "skew": _make_stat(skew, axis=1, bias=False),
    "dom_freq_energy": _make_stat(dominant_frequency_energy),
    #"dom_freq_index": _make_stat(dominant_frequency_index),
    "spectral_entropy": _make_stat(spectral_entropy),
    "spectral_centroid": _make_stat(spectral_centroid),
    "spectral_spread": _make_stat(spectral_spread),
    "spectral_rolloff": _make_stat(spectral_rolloff),
    "mean_crossing_rate": _make_stat(mean_crossing_rate),
    "waveform_length": _make_stat(waveform_length),
    "lag1_autocorrelation": _make_stat(lag1_autocorrelation),
    "crest_factor": _make_stat(crest_factor),
    #"iqr_robust": _make_stat(iqr_robust),
    #"hjorth_mobility": _make_stat(hjorth_mobility),
    #"local_median": _make_stat(local_median),
    "trend_slope": _make_stat(trend_slope),
    #"lag5_autocorrelation": _make_stat(lag5_autocorrelation),
    "ewma_difference": _make_stat(ewma_difference),
    "diff_zcr": _make_stat(diff_zcr),
    "tkeo_mean": _make_stat(tkeo_mean),
    "fractal_roughness": _make_stat(fractal_roughness),
    "signal_monotonicity": _make_stat(signal_monotonicity),
    "cusum_deviation": _make_stat(cusum_deviation),
    "root_mean_square": _make_stat(root_mean_square),
    "time_reversibility": _make_stat(time_reversibility),
    "poincare_ratio": _make_stat(poincare_ratio),
    "safe_kill_switch_function": _make_stat(safe_kill_switch_function),
}



def apply_statistic_to_segments(
    data: np.ndarray,
    func: Callable,
    window_shape: int,
    step_duration: int,
    preprocess: Callable | None = None,
    func_kwargs: Dict[str, Any] | None = None,
) -> np.ndarray:
    """
    Apply a statistic function to sliding windows of the data.
    """
    if preprocess is not None:
        data = preprocess(data)

    windows = np.lib.stride_tricks.sliding_window_view(
        data,
        window_shape=window_shape,
    )[::step_duration]

    kwargs = func_kwargs or {}
    with np.errstate(divide="ignore", invalid="ignore"):
        return func(windows, **kwargs)

__all__ = ["FEATURE_MAP", "apply_statistic_to_batch", "apply_statistic_to_segments"]
