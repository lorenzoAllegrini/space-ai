"""Utility functions for feature extraction from time series segments."""

from __future__ import annotations
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
    Conteggio dei picchi 100x più veloce in puro NumPy (senza cicli for).
    """
    if windows.shape[1] < 3:
        return np.zeros(windows.shape[0], dtype=int)

    # 1. Slicing vettoriale: confrontiamo il centro con sinistra e destra
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
    Calcola l'energia (potenza FFT normalizzata) della frequenza dominante 
    in ogni segmento di dati, escludendo la componente DC (frequenza 0).
    
    Args:
        windows: Array 2D (batch_size, window_len)
        
    Returns:
        Array 1D (batch_size,) con la potenza del picco principale di frequenza
    """
    if windows.shape[1] < 2:
        return np.zeros(windows.shape[0], dtype=float)
        
    # Calcola la FFT reale: abs(FFT)
    fft_vals = np.abs(np.fft.rfft(windows, axis=1))
    
    # Calcola la POTENZA: (abs(FFT)^2) / N
    # Ciò normalizza per la finestra di tempo (window size)
    power_spectrum = (fft_vals ** 2) / windows.shape[1]
    
    # Ignora la componente DC (indice 0)
    if power_spectrum.shape[1] > 1:
        # Prende il valore massimo dello spettro per ogni segmento
        return np.max(power_spectrum[:, 1:], axis=1)
    return np.zeros(windows.shape[0], dtype=float)


def dominant_frequency_index(windows: np.ndarray) -> np.ndarray:
    """Ritorna l'indice (bin) della frequenza con massima potenza."""
    if windows.shape[1] < 2:
        return np.zeros(windows.shape[0], dtype=float)
    fft_vals = np.abs(np.fft.rfft(windows, axis=1))
    if fft_vals.shape[1] > 1:
        return np.argmax(fft_vals[:, 1:], axis=1) + 1
    return np.zeros(windows.shape[0], dtype=float)


def spectral_entropy(windows: np.ndarray) -> np.ndarray:
    """Calcola l'entropia dello spettro di potenza (normalizzata 0-1)."""
    if windows.shape[1] < 2:
        return np.zeros(windows.shape[0], dtype=float)
    fft_vals = np.abs(np.fft.rfft(windows, axis=1))
    psd = (fft_vals ** 2) / windows.shape[1]
    psd_norm = psd / (np.sum(psd, axis=1, keepdims=True) + 1e-12)
    entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-12), axis=1)
    # Normalizza per log2(numero di bin)
    return entropy / np.log2(psd.shape[1])


def spectral_centroid(windows: np.ndarray) -> np.ndarray:
    """Calcola il baricentro delle frequenze pesato sulla potenza."""
    if windows.shape[1] < 2:
        return np.zeros(windows.shape[0], dtype=float)
    fft_vals = np.abs(np.fft.rfft(windows, axis=1))
    psd = (fft_vals ** 2) / windows.shape[1]
    freqs = np.arange(psd.shape[1])
    centroid = np.sum(psd * freqs, axis=1) / (np.sum(psd, axis=1) + 1e-12)
    return centroid


def spectral_spread(windows: np.ndarray) -> np.ndarray:
    """Calcola lo spread (varianza) dello spettro attorno al centroide."""
    if windows.shape[1] < 2:
        return np.zeros(windows.shape[0], dtype=float)
    fft_vals = np.abs(np.fft.rfft(windows, axis=1))
    psd = (fft_vals ** 2) / windows.shape[1]
    freqs = np.arange(psd.shape[1])
    
    centroid = np.sum(psd * freqs, axis=1, keepdims=True) / (np.sum(psd, axis=1, keepdims=True) + 1e-12)
    spread = np.sqrt(np.sum(psd * (freqs - centroid)**2, axis=1) / (np.sum(psd, axis=1) + 1e-12))
    return spread


def spectral_rolloff(windows: np.ndarray, roll_percent: float = 0.85) -> np.ndarray:
    """Calcola la frequenza sotto la quale risiede l'85% dell'energia spettrale."""
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
    Helper function to apply Savitzky-Golay safely.
    Reduces window size if input data is too short.
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
    """Apply Savitzky-Golay smoothing filter (target window 10)."""
    return _safe_savgol(x, target_window=11)

def smooth_20(x):
    """Apply Savitzky-Golay smoothing filter (target window 20)."""
    return _safe_savgol(x, target_window=21)

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
    #"mean": _make_stat(np.mean, axis=1),
    "var": _make_stat(np.var, axis=1),
    "std": _make_stat(np.std, axis=1),
    #"n_peaks": _make_stat(count_peaks_vectorized),
    #"smooth10_n_peaks": _make_stat(count_peaks_vectorized, preprocess=smooth_10),
    #"smooth20_n_peaks": _make_stat(count_peaks_vectorized, preprocess=smooth_20),
    #"diff_peaks": _make_stat(count_peaks_vectorized, preprocess=diff1),
    #"diff2_peaks": _make_stat(count_peaks_vectorized, preprocess=diff2),
    #"diff_var": _make_stat(np.var, preprocess=diff1, axis=1),
    "diff2_var": _make_stat(np.var, preprocess=diff2, axis=1),
    "kurtosis": _make_stat(kurtosis, axis=1, fisher=True, bias=False),
    "skew": _make_stat(skew, axis=1, bias=False),
    "dom_freq_energy": _make_stat(dominant_frequency_energy),
    "dom_freq_index": _make_stat(dominant_frequency_index),
    "spectral_entropy": _make_stat(spectral_entropy),
    "spectral_centroid": _make_stat(spectral_centroid),
    "spectral_spread": _make_stat(spectral_spread),
    "spectral_rolloff": _make_stat(spectral_rolloff),
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
