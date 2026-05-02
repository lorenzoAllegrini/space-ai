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
from scipy.signal import find_peaks, detrend
from scipy.fft import rfft, rfftfreq

from spaceai.benchmark.callbacks.mixin import CallbackMixin
from spaceai.data.anomaly_dataset import AnomalyDataset, AnomalyDatasetSubset

if TYPE_CHECKING:
    from spaceai.benchmark.callbacks.handler import CallbackHandler

def linear_detrend_func(x: np.ndarray) -> np.ndarray:
    """Named function for linear detrending (pickleable)."""
    return detrend(x, type='linear', axis=-1)

from spaceai.models.anomaly_pipeline.anomaly_classifier import PipelineState


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
        apply_func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        include_remainder: bool = True,
        ignore_gaps: bool = False,
        **kwargs
    ) -> None:
        self.window_size_raw = window_size
        self.perc_step_size = perc_step_size if perc_step_size is not None else 0.5
        self.step_size_raw = int(window_size * self.perc_step_size) if step_size is None and window_size is not None else step_size
        self.min_window = min_window or 50
        self.max_window = max_window or 500
        self.apply_func = apply_func
        self.include_remainder = include_remainder
        self.ignore_gaps = ignore_gaps
        super().__init__(callback_handler=callback_handler, **kwargs)

    def fit(self, X: Union[np.ndarray, AnomalyDataset], y: Optional[np.ndarray] = None, sampling_period: Optional[float] = None, **kwargs) -> TimeSeriesSplitter:
        """
        Fit the splitter to the data, resolving window and step sizes.
        """
        data = X.data[:, 0] if isinstance(X, AnomalyDataset) else X
        self._ensure_sizes(data, sampling_period)
        return self

    def transform(
        self, 
        X: Union[np.ndarray, AnomalyDataset, Any], 
        y: Optional[np.ndarray] = None, 
        sampling_period: Optional[float] = None, 
        results: Optional[Dict[str, Any]] = None,
        return_subsets: bool = False,
        save_dir: Optional[str] = None,
        suffix: str = "",
        **kwargs
    ) -> Union[np.ndarray, Any, Tuple[np.ndarray, np.ndarray]]:
        """
        Transform the input into segments/windows. Supports PipelineMessage.
        """
        if hasattr(X, "data") and not isinstance(X, (np.ndarray, AnomalyDataset)):
            msg = X
            if isinstance(msg.data, AnomalyDataset):
                res = self.segment_dataset(msg.data, return_subsets=return_subsets, results=results, save_dir=save_dir, suffix=suffix)
                msg.data, msg.labels = res.segments, res.labels
                msg.original_indices, msg.true_intervals = res.segment_indices, res.intervals
            else:
                msg.data = self.split(msg.data, sampling_period=sampling_period, results=results)
                if msg.labels is not None:
                    msg.labels = self.split_labels(msg.labels, sampling_period=sampling_period, results=results)
                idxs = np.arange(len(msg.data)) * self.step_size
                msg.original_indices = np.column_stack((idxs, idxs + self.window_size - 1))
            return msg

        if isinstance(X, AnomalyDataset):
            return self.segment_dataset(X, return_subsets=return_subsets, results=results, save_dir=save_dir, suffix=suffix)
        
        segments = self.split(X, sampling_period=sampling_period, results=results)
        if y is not None:
            labels = self.split_labels(y, sampling_period=sampling_period, results=results)
            return segments, labels
        return segments, None

    def fit_transform(self, X: Union[np.ndarray, AnomalyDataset], y: Optional[np.ndarray] = None, **kwargs) -> Any:
        """
        Fit to data, then transform it.
        """
        return self.fit(X, y, **kwargs).transform(X, y, **kwargs)

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
            return np.lib.stride_tricks.sliding_window_view(data, window_shape=self.window_size, axis=0)[starts]

    def split_labels(self, labels: np.ndarray, sampling_period: Optional[float] = None, results: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """Segment pointwise labels into window-level binary labels."""
        with self._callback_context("segmentation_split_labels", results):
            self._ensure_sizes(labels, sampling_period)
            if len(labels) < self.window_size:
                return np.array([], dtype=int)
            starts = np.arange(0, len(labels) - self.window_size + 1, self.step_size)
            windows = np.lib.stride_tricks.sliding_window_view(labels, window_shape=self.window_size, axis=0)[starts]
            return (np.max(windows, axis=1) > 0).astype(int)

    def segment_dataset(self, dataset_channel: AnomalyDataset, return_subsets: bool = False, results: Optional[Dict[str, Any]] = None, save_dir: Optional[str] = None, suffix: str = "") -> PipelineState:
        """Segment an AnomalyDataset channel into windows or sub-datasets."""
        sampling_period = getattr(dataset_channel, "sampling_period", None)
        data = dataset_channel.data[:, 0]
        if save_dir:
            self.save_timeseries_csv(dataset_channel, save_dir, suffix=suffix)
        self._ensure_sizes(data, sampling_period)
        
        p_labels = np.zeros(len(data), dtype=int)
        if getattr(dataset_channel, "anomalies", None) is not None:
            for s, e in dataset_channel.anomalies:
                p_labels[max(0, s):min(len(data), e)] = 1

        intervals = getattr(dataset_channel, "block_intervals", [(0, len(data))])
        
        all_segments, all_labels, all_indices = [], [], []
        
        # Override intervals if ignore_gaps is True
        if self.ignore_gaps:
            intervals = [(0, len(data))]

        with self._callback_context("total_segmentation", results):
            for i, (s, e) in enumerate(intervals):
                b_data = data[s:e]
                if self.apply_func is not None:
                    b_data = self.apply_func(b_data)
                
                if len(b_data) < self.window_size:
                    continue
                
                segs = self.split(b_data, sampling_period, results=None)
                if segs.size == 0:
                    continue
                
                lbl_data = p_labels[s:e]
                lbls = self.split(lbl_data, sampling_period, results=None) if return_subsets else self.split_labels(lbl_data, sampling_period, results=None)
                
                # Global handling of remainder via sliding back
                n_segs = len(segs)
                last_end = (n_segs - 1) * self.step_size + self.window_size
                remainder = len(b_data) - last_end
                
                if remainder > 0 and self.include_remainder:
                    # Last window slides back to cover the very end of the block/data
                    r_start = len(b_data) - self.window_size
                    r_window = b_data[r_start:].reshape(1, -1)
                    segs = np.vstack([segs, r_window])
                    
                    r_lbl_window = lbl_data[r_start:]
                    if return_subsets:
                        lbls = np.vstack([lbls, self.split(r_lbl_window, sampling_period, results=None)])
                    else:
                        r_lbl = 1 if np.any(r_lbl_window > 0) else 0
                        lbls = np.concatenate([lbls, [r_lbl]])
                    
                    new_idx = np.array([[r_start + s, len(b_data) - 1 + s]])
                    idxs = np.arange(n_segs) * self.step_size + s
                    block_indices = np.vstack([np.column_stack((idxs, idxs + self.window_size - 1)), new_idx])
                else:
                    idxs = np.arange(len(segs)) * self.step_size + s
                    block_indices = np.column_stack((idxs, idxs + self.window_size - 1))

                all_segments.append(segs)
                all_labels.append(lbls)
                all_indices.append(block_indices)
        

        if not all_segments:
            return PipelineState(data=np.empty((0, self.window_size)), labels=np.array([]), indices=np.empty((0, 2)), intervals=[])

        final_segs, final_indices = np.vstack(all_segments), np.vstack(all_indices)
        if return_subsets:
            subsets = [AnomalyDatasetSubset(dataset_channel, int(s), int(e)) for s, e in final_indices]
            return PipelineState(data=subsets, labels=np.vstack(all_labels), indices=final_indices, intervals=[])
        
        f_labels = np.concatenate(all_labels)
        idx = np.where(f_labels == 1)[0]
        intervals = [[g[0], g[-1]] for g in [list(group) for group in mit.consecutive_groups(idx)]] if idx.size > 0 else []
        return PipelineState(data=final_segs, labels=f_labels, indices=final_indices, intervals=intervals)

    def get_timestamp_intervals(self, dataset_channel: AnomalyDataset, window_intervals: List[Tuple[int, int]]) -> List[Tuple[Any, Any]]:
        """Map window anomaly intervals back to timestamps or absolute indices."""
        res_state = self.segment_dataset(dataset_channel, return_subsets=True)
        timestamps = getattr(dataset_channel, "timestamps", None)
        limit = len(timestamps) if timestamps is not None else len(dataset_channel.data)
        
        intervals = []
        for ws, we in window_intervals:
            if ws >= len(res_state.indices):
                continue
            s, e = res_state.indices[ws][0], res_state.indices[min(we, len(res_state.indices)-1)][1]
            if s >= limit:
                continue
            e = min(e, limit - 1)
            intervals.append((timestamps[s], timestamps[e]) if timestamps is not None else (s + getattr(dataset_channel, "start_idx", 0), e + getattr(dataset_channel, "start_idx", 0)))
        return intervals

    def find_window_size(self, data: np.ndarray, sampling_period: Optional[float] = None) -> int:
        """Estimate optimal window size using Auto-Correlation Function (ACF)."""
        try:
            import statsmodels.api as sm
            d = np.diff(np.asarray(data).ravel()[:100000])
            if len(d) < 200 or np.var(d) < 1e-6:
                return 100
            acf = sm.tsa.acf(d, nlags=min(self.max_window, len(d) // 2), fft=True)
            peaks, _ = find_peaks(acf, prominence=0.02, distance=10)
            peaks = [p for p in peaks if self.min_window < p < self.max_window]
            res = int(peaks[0]) if peaks else self.find_window_size_fft(data)
            return res
        except (ImportError, ModuleNotFoundError):
            logging.warning("statsmodels not found. Falling back to FFT estimation for window_size.")
            return self.find_window_size_fft(data)

    def find_window_size_fft(self, data: np.ndarray) -> int:
        """Estimate optimal window size using Fast Fourier Transform (FFT)."""
        d = np.diff(np.asarray(data).ravel()[:100000])
        if len(d) < 200 or np.var(d) < 1e-6:
            return 100
        xf, yf = rfftfreq(len(d), d=1.0), np.abs(rfft(d))
        valid = np.where((xf >= 1.0/self.max_window) & (xf <= 1.0/self.min_window))[0]
        res = int(round(1.0 / xf[valid[np.argmax(yf[valid])]])) if valid.size > 0 else 100
        return res

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