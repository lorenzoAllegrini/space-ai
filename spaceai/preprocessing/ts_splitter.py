"""Time-series splitter for SpaceAI datasets."""

from __future__ import annotations
import os
import logging
import csv
from dataclasses import dataclass
from typing import Union, Dict, List, Optional, Any, Tuple, Callable, TYPE_CHECKING
import numpy as np
import pandas as pd
import more_itertools as mit
import statsmodels.api as sm
from scipy.signal import find_peaks, detrend
from scipy.fft import rfft, rfftfreq
import matplotlib.pyplot as plt
from spaceai.benchmark.callbacks.mixin import CallbackMixin
from spaceai.data.anomaly_dataset import AnomalyDataset, AnomalyDatasetSubset

if TYPE_CHECKING:
    from spaceai.benchmark.callbacks.handler import CallbackHandler


def default_apply_func(x: np.ndarray) -> np.ndarray:
    """Default application function for segmentation (detrending)."""
    return detrend(x, type='linear', axis=-1)


@dataclass
class SegmentationResult:
    """Packaging for segmentation results."""
    segments: Union[np.ndarray, List[AnomalyDatasetSubset]]
    labels: np.ndarray
    segment_indices: np.ndarray
    intervals: List[List[int]]


class TimeSeriesSplitter(CallbackMixin):
    """
    Unified time-series splitter for SpaceAI datasets (ESA, NASA, OPS-SAT).
    """
    def __init__(
        self,
        window_size: Optional[Union[int, str, pd.Timedelta]] = None,
        step_size: Optional[Union[int, str, pd.Timedelta]] = None,
        perc_step_size: float = 0.5,
        min_window: int = 50,
        max_window: int = 500,
        callback_handler: Optional[CallbackHandler] = None,
        apply_func: Optional[Callable[[np.ndarray], np.ndarray]] = default_apply_func,
        **kwargs
    ) -> None:
        self.window_size_raw = window_size
        self.perc_step_size = perc_step_size if perc_step_size is not None else 0.5
        self.step_size_raw = int(window_size * self.perc_step_size) if step_size is None and window_size is not None else step_size
        self.min_window = min_window or 50
        self.max_window = max_window or 500
        self.apply_func = apply_func
        super().__init__(callback_handler=callback_handler, **kwargs)

    def fit(self, *messages: "PipelineMessage", sampling_period: Optional[float] = None, **kwargs) -> TimeSeriesSplitter:
        """
        Fit the splitter to the data contained in the first message.
        """
        if not messages:
            return self
        message = messages[0]
        X = message.data
        data = X.data[:, 0] if isinstance(X, AnomalyDataset) else X
        self._ensure_sizes(data, sampling_period)
        return self

    def transform(
        self, 
        message: "PipelineMessage", 
        sampling_period: Optional[float] = None, 
        mode: str = "anomaly",
        **kwargs
    ) -> "PipelineMessage":
        """
        Transform the input in the message into segments/windows.
        """
        X = message.data
        results = message.results
        save_dir = message.save_dir
        suffix = "test" if "test" in getattr(message, "mode", "") else "train"

        if isinstance(X, AnomalyDataset):
            res = self.segment_dataset(X, mode=mode, results=results, save_dir=save_dir, suffix=suffix)
            message.data, message.labels = res.segments, res.labels
            message.original_indices, message.true_intervals = res.segment_indices, res.intervals
        else:
            message.data = self.split(X, sampling_period=sampling_period, results=results)
            if message.labels is not None:
                message.labels = self.split_labels(message.labels, sampling_period=sampling_period, results=results)

            idxs = np.arange(len(message.data)) * self.step_size
            message.original_indices = np.column_stack((idxs, idxs + self.window_size - 1))
        
        return message

    def fit_transform(self, message: "PipelineMessage", **kwargs) -> "PipelineMessage":
        """
        Fit to data, then transform it.
        """
        return self.fit(message, **kwargs).transform(message, **kwargs)

    def _resolve_samples(self, size: Union[int, str, pd.Timedelta, None], sampling_period: Optional[float] = None) -> Optional[int]:
        """Convert durations or strings to sample counts."""
        if size is None:
            return None
        if isinstance(size, int):
            return size
        try:
            return int(size)
        except (ValueError, TypeError):
            pass
        duration = pd.Timedelta(size)
        if pd.isna(duration):
            return None
        return int(duration.total_seconds() / (sampling_period or 1.0))

    def _ensure_sizes(self, data: np.ndarray, sampling_period: Optional[float] = None) -> None:
        """Resolve window and step sizes dynamically or statically."""
        if getattr(self, "window_size", None) is None:
            self.window_size = self._resolve_samples(self.window_size_raw, sampling_period) if self.window_size_raw else self.find_window_size(data, sampling_period)
        if getattr(self, "step_size", None) is None:
            self.step_size = self._resolve_samples(self.step_size_raw, sampling_period) if self.step_size_raw else max(1, int(self.window_size * self.perc_step_size))

    def split(self, data: np.ndarray, sampling_period: Optional[float] = None, results: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """Segment data into windows."""
        with self._callback_context("segmentation_split", results):
            self._ensure_sizes(data, sampling_period)
            if len(data) < self.window_size:
                return np.empty((0, self.window_size))
            starts = np.arange(0, len(data) - self.window_size + 1, self.step_size)
            return np.lib.stride_tricks.sliding_window_view(data, window_shape=self.window_size)[starts]

    def split_labels(self, labels: np.ndarray, sampling_period: Optional[float] = None, results: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """Segment pointwise labels into window-level binary labels."""
        with self._callback_context("segmentation_split_labels", results):
            self._ensure_sizes(labels, sampling_period)
            if len(labels) < self.window_size:
                return np.array([], dtype=int)
            starts = np.arange(0, len(labels) - self.window_size + 1, self.step_size)
            windows = np.lib.stride_tricks.sliding_window_view(labels, window_shape=self.window_size)[starts]
            return (np.max(windows, axis=1) > 0).astype(int)

    def segment_dataset(self, dataset_channel: AnomalyDataset, mode: str = "anomaly", results: Optional[Dict[str, Any]] = None, save_dir: Optional[str] = None, suffix: str = "") -> SegmentationResult:
        """Segment an AnomalyDataset channel into windows or sub-datasets."""
        with self._callback_context("segmentation", results):
            sampling_period = getattr(dataset_channel, "sampling_period", None)
            data = dataset_channel.data[:, 0]
            if save_dir:
                self.save_timeseries_csv(dataset_channel, save_dir, suffix=suffix)
            self._ensure_sizes(data, sampling_period)
            
            p_labels = np.zeros(len(data), dtype=int)
            if getattr(dataset_channel, "anomalies", None) is not None:
                for s, e in dataset_channel.anomalies:
                    p_labels[max(0, s):min(len(data), e)] = 1

            all_segments, all_labels, all_indices = [], [], []
            for s, e in getattr(dataset_channel, "block_intervals", [(0, len(data))]):
                b_data = data[s:e]
                if self.apply_func is not None:
                    b_data = self.apply_func(b_data)
                segs = self.split(b_data, sampling_period, results=results)
                if not len(segs):
                    continue
                lbls = self.split(p_labels[s:e], sampling_period, results=results) if mode == "experience" else self.split_labels(p_labels[s:e], sampling_period, results=results)
                all_segments.append(segs)
                all_labels.append(lbls)
                idxs = np.arange(len(segs)) * self.step_size + s
                all_indices.append(np.column_stack((idxs, idxs + self.window_size - 1)))

            if not all_segments:
                return SegmentationResult(np.empty((0, self.window_size)), np.array([]), np.empty((0, 2)), [])

            final_segs, final_indices = np.vstack(all_segments), np.vstack(all_indices)
            if mode == "experience":
                return SegmentationResult([AnomalyDatasetSubset(dataset_channel, int(s), int(e)) for s, e in final_indices], np.vstack(all_labels), final_indices, [])
            
            f_labels = np.concatenate(all_labels)
            idx = np.where(f_labels == 1)[0]
            intervals = [[g[0], g[-1]] for g in [list(group) for group in mit.consecutive_groups(idx)]] if idx.size > 0 else []
            return SegmentationResult(final_segs, f_labels, final_indices, intervals)

    def get_timestamp_intervals(self, dataset_channel: AnomalyDataset, window_intervals: List[Tuple[int, int]]) -> List[Tuple[Any, Any]]:
        """Map window anomaly intervals back to timestamps or absolute indices."""
        res = self.segment_dataset(dataset_channel, mode="experience")
        timestamps = getattr(dataset_channel, "timestamps", None)
        limit = len(timestamps) if timestamps is not None else len(dataset_channel.data)
        
        intervals = []
        for ws, we in window_intervals:
            if ws >= len(res.segment_indices):
                continue
            s, e = res.segment_indices[ws][0], res.segment_indices[min(we, len(res.segment_indices)-1)][1]
            if s >= limit:
                continue
            e = min(e, limit - 1)
            intervals.append((timestamps[s], timestamps[e]) if timestamps is not None else (s + getattr(dataset_channel, "start_idx", 0), e + getattr(dataset_channel, "start_idx", 0)))
        return intervals

    def find_window_size(self, data: np.ndarray, sampling_period: Optional[float] = None) -> int:
        """Estimate optimal window size using Auto-Correlation Function (ACF)."""
        d = np.diff(np.asarray(data).ravel())
        if len(d) < 200 or np.var(d) < 1e-6:
            return 100
        acf = sm.tsa.acf(d, nlags=min(self.max_window, len(d) // 2), fft=True)
        peaks, _ = find_peaks(acf, prominence=0.02, distance=10)
        peaks = [p for p in peaks if self.min_window < p < self.max_window]
        return int(peaks[0]) if peaks else self.find_window_size_fft(data)

    def find_window_size_fft(self, data: np.ndarray) -> int:
        """Estimate optimal window size using Fast Fourier Transform (FFT)."""
        d = np.diff(np.asarray(data).ravel())
        if len(d) < 200 or np.var(d) < 1e-6:
            return 100
        xf, yf = rfftfreq(len(d), d=1.0), np.abs(rfft(d))
        valid = np.where((xf >= 1.0/self.max_window) & (xf <= 1.0/self.min_window))[0]
        return int(round(1.0 / xf[valid[np.argmax(yf[valid])]])) if valid.size > 0 else 100

    def save_timeseries_csv(self, dataset_channel: AnomalyDataset, save_dir: str, suffix: str = "") -> None:
        """Save OXI-compatible raw timeseries CSV."""
        data = dataset_channel.data[:, 0]
        p_labels = np.zeros(len(data), dtype=int)
        for s, e in getattr(dataset_channel, "anomalies", []):
            p_labels[max(0, s):min(len(data), e)] = 1
        
        idx = np.arange(0, len(data), 10)
        channel_id = getattr(dataset_channel, "channel_id", "Unknown")
        df_dict = {"series": [channel_id] * len(idx), "value": data[idx].tolist(), "label": ["anomaly" if x == 1 else "" for x in p_labels[idx]]}
        
        ts = getattr(dataset_channel, "timestamps", None)
        if ts is not None:
            ts_v = ts.values[idx] if hasattr(ts, "values") else ts[idx]
            df_dict["timestamp"] = pd.to_datetime(ts_v).strftime('%Y-%m-%dT%H:%M:%S.000Z').tolist()
            cols = ["series", "timestamp", "value", "label"]
        else:
            cols = ["series", "value", "label"]
            
        os.makedirs(save_dir, exist_ok=True)
        filename = f"timeseries_export_{channel_id}{'_'+suffix if suffix else ''}.csv"
        pd.DataFrame(df_dict)[cols].to_csv(os.path.join(save_dir, filename), index=False, quoting=csv.QUOTE_MINIMAL)

__all__ = ["TimeSeriesSplitter", "SegmentationResult"]