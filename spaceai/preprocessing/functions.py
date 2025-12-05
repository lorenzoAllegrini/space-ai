"""Utility functions for feature extraction from time series segments."""

from functools import partial
from typing import (
    Any,
    Callable,
    Dict,
)

import numpy as np
from scipy import signal as sig  # type: ignore
from scipy.stats import (  # type: ignore
    kurtosis,
    skew,
)


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
    Count peaks in each window of the 2D array.
    Designed to be used with apply_statistic_to_batch.
    """
    prominences = 0.1 * (np.max(windows, axis=1) - np.min(windows, axis=1))
    counts = []
    for i in range(windows.shape[0]):
        if prominences[i] == 0:
            counts.append(0)
            continue
        peaks, _ = sig.find_peaks(windows[i], prominence=prominences[i])
        counts.append(len(peaks))
    return np.array(counts)


def smooth_10(x):
    """Apply Savitzky-Golay smoothing filter with window length 10."""
    return sig.savgol_filter(x, window_length=10, polyorder=2, axis=-1)


def smooth_20(x):
    """Apply Savitzky-Golay smoothing filter with window length 20."""
    return sig.savgol_filter(x, window_length=20, polyorder=2, axis=-1)


def diff1(x):
    """Compute first order difference of array."""
    d = np.diff(x, axis=-1)
    # Pad with 0 at the beginning to maintain shape
    if x.ndim == 1:
        return np.concatenate(([0], d))
    return np.column_stack((np.zeros(x.shape[0]), d))


def diff2(x):
    """Compute second order difference of array."""
    d = np.diff(x, n=2, axis=-1)
    # Pad with 0s at the beginning to maintain shape
    if x.ndim == 1:
        return np.concatenate(([0, 0], d))
    return np.column_stack((np.zeros((x.shape[0], 2)), d))


def _make_stat(func, preprocess=None, **kwargs):
    return partial(
        apply_statistic_to_batch,
        func=func,
        preprocess=preprocess,
        func_kwargs=kwargs,
    )


FEATURE_MAP = {
    "mean": _make_stat(np.mean, axis=1),
    "var": _make_stat(np.var, axis=1),
    "std": _make_stat(np.std, axis=1),
    "n_peaks": _make_stat(count_peaks_vectorized),
    "smooth10_n_peaks": _make_stat(count_peaks_vectorized, preprocess=smooth_10),
    "smooth20_n_peaks": _make_stat(count_peaks_vectorized, preprocess=smooth_20),
    "diff_peaks": _make_stat(count_peaks_vectorized, preprocess=diff1),
    "diff2_peaks": _make_stat(count_peaks_vectorized, preprocess=diff2),
    "diff_var": _make_stat(np.var, preprocess=diff1, axis=1),
    "diff2_var": _make_stat(np.var, preprocess=diff2, axis=1),
    "kurtosis": _make_stat(kurtosis, axis=1, fisher=True, bias=False),
    "skew": _make_stat(skew, axis=1, bias=False),
}


# For backward compatibility if needed, though we are moving away from it
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
