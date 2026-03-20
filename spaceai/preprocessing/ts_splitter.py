"""SpaceAI segmentator module."""

import more_itertools as mit
import numpy as np
import pandas as pd
from typing import Union, Dict, List, Optional, Any, Tuple

from spaceai.data.anomaly_dataset import AnomalyDataset, AnomalyDatasetSubset

from dataclasses import dataclass

@dataclass
class SegmentationResult:
    segments: Union[np.ndarray, List[Any]]
    labels: np.ndarray
    segment_indices: np.ndarray
    intervals: List[Tuple[int, int]]

class TimeSeriesSplitter:
    """
    Unified time-series splitter for SpaceAI datasets (ESA, NASA, OPS-SAT).
    Inherits vectorized segmentation logic and adds:
    - CSV Caching (persistence)
    - Feature Pooling
    - Telecommand handling
    - Anomaly interval extraction
    """

    def __init__(
        self,
        window_size: Union[int, str, pd.Timedelta] = 100,
        step_size: Union[int, str, pd.Timedelta] = 50,
        stride: Optional[Union[int, str, pd.Timedelta]] = None,
    ) -> None:
        self.window_size_raw = window_size
        self.step_size_raw = stride if stride is not None else step_size

    def _resolve_samples(self, size: Union[int, str, pd.Timedelta], sampling_period: Optional[float] = None) -> int:
        """Helper to convert timed durations to sample counts given a sampling_period."""
        if isinstance(size, int):
            return size
        
        duration = pd.Timedelta(size)
        
        if sampling_period is None:
            return int(duration.total_seconds())
            
        samples = duration.total_seconds() / sampling_period
        return int(samples)

    def split(self, data: np.ndarray, sampling_period: Optional[float] = None) -> np.ndarray:
        """
        Segment the dataset channel into windows.
        
        Args:
            data (np.ndarray): 1D array of telemetry.
            sampling_period (float): Optional sampling period in seconds for timed conversions.
        """
        window_size = self._resolve_samples(self.window_size_raw, sampling_period)
        step_size = self._resolve_samples(self.step_size_raw, sampling_period)

        if len(data) < window_size:
            return np.empty((0, window_size))

        starts = np.arange(0, len(data) - window_size + 1, step_size)
        return np.lib.stride_tricks.sliding_window_view(data, window_shape=window_size)[starts]

    def split_labels(self, labels: np.ndarray, sampling_period: Optional[float] = None) -> np.ndarray:
        """
        Segment pointwise labels into window-level binary labels.
        (1 if any point in the window is anomalous).
        
        Args:
            labels (np.ndarray): 1D array of pointwise labels.
            sampling_period (float): Optional sampling period.
        """
        window_size = self._resolve_samples(self.window_size_raw, sampling_period)
        step_size = self._resolve_samples(self.step_size_raw, sampling_period)

        if len(labels) < window_size:
            return np.array([], dtype=int)

        starts = np.arange(0, len(labels) - window_size + 1, step_size)
        label_windows = np.lib.stride_tricks.sliding_window_view(labels, window_shape=window_size)[starts]
        return (label_windows.max(axis=1) > 0).astype(int)

    def segment_dataset(self, dataset_channel: AnomalyDataset, mode: str = "anomaly") -> SegmentationResult:
        """
        High-level method to segment an AnomalyDataset channel.
        Uses split() internally but handles masks, blocks, and labels.

        Args:
            dataset_channel (AnomalyDataset): The dataset channel.
            mode (str): 'anomaly' (binary window labels) or 'experience' (pointwise window labels).
        """
        if mode not in ["anomaly", "experience"]:
            raise ValueError("mode must be 'anomaly' or 'experience'")

        sampling_period = getattr(dataset_channel, "sampling_period", None)
        window_size = self._resolve_samples(self.window_size_raw, sampling_period)
        step_size = self._resolve_samples(self.step_size_raw, sampling_period)

        data = dataset_channel.data[:, 0] # type: ignore[attr-defined]

        pointwise_labels = np.zeros(len(data), dtype=int)
        if getattr(dataset_channel, "anomalies", None) is not None:
            for start, end in dataset_channel.anomalies:
                pointwise_labels[max(0, start):min(len(data), end)] = 1

        block_intervals = getattr(dataset_channel, "block_intervals", [(0, len(data))])
        
        all_segments = []
        all_labels = []
        all_indices = []

        for start_idx, end_idx in block_intervals:
            block_data = data[start_idx:end_idx]
            block_pointwise_labels = pointwise_labels[start_idx:end_idx]
            
            segments = self.split(block_data, sampling_period)
            if len(segments) == 0:
                continue
            
            if mode == "experience":
                labels = self.split(block_pointwise_labels, sampling_period)
            else:
                labels = self.split_labels(block_pointwise_labels, sampling_period)
            
            all_segments.append(segments)
            all_labels.append(labels)
            
            starts = np.arange(len(segments)) * step_size
            global_starts = starts + start_idx
            global_ends = global_starts + window_size - 1
            all_indices.append(np.column_stack((global_starts, global_ends)))

        if not all_segments:
            return SegmentationResult(
                segments=np.empty((0, window_size)),
                labels=np.array([]),
                intervals=[],
                segment_indices=np.empty((0, 2)),
            )

        segments = np.vstack(all_segments)
        segment_indices = np.vstack(all_indices)

        if mode == "experience":
            segment_labels = np.vstack(all_labels)
            
            subset_segments = []
            for s, e in segment_indices:
                subset_segments.append(AnomalyDatasetSubset(dataset_channel, int(s), int(e)))
                
            return SegmentationResult(
                segments=subset_segments,
                labels=segment_labels, 
                segment_indices=segment_indices,
                intervals=[],
            )
        else:
            final_labels = np.concatenate(all_labels)
            indices = np.where(final_labels == 1)[0]
            if indices.size == 0:
                anomalies_intervals = []
            else:
                groups = [list(group) for group in mit.consecutive_groups(indices)]
                anomalies_intervals = [[group[0], group[-1]] for group in groups]

            return SegmentationResult(
                segments=segments,
                labels=final_labels,
                intervals=anomalies_intervals,
                segment_indices=segment_indices,
            )

    def get_timestamp_intervals(self, dataset_channel: AnomalyDataset, window_intervals: List[Tuple[int, int]]) -> List[Tuple[Any, Any]]:
        """Map window anomaly intervals back to timestamp intervals or absolute indices."""
        has_timestamps = hasattr(dataset_channel, "timestamps") and dataset_channel.timestamps is not None and len(dataset_channel.timestamps) > 0
        
        result = self.segment_dataset(dataset_channel, mode="experience")
        segment_indices = result.segment_indices
        
        limit = float('inf')
        if has_timestamps:
            timestamps = dataset_channel.timestamps
            limit = len(timestamps)
        elif hasattr(dataset_channel, "data") and dataset_channel.data is not None:
            limit = len(dataset_channel.data)
        
        time_intervals = []
        offset = getattr(dataset_channel, "start_idx", 0)
        
        for w_start, w_end in window_intervals:
            if w_start >= len(segment_indices):
                continue
            if w_end >= len(segment_indices):
                w_end = len(segment_indices) - 1
                
            s_idx = segment_indices[w_start][0]
            e_idx = segment_indices[w_end][1]
            
            if s_idx >= limit:
                continue
            if e_idx >= limit:
                e_idx = limit - 1
                
            if has_timestamps:
                time_intervals.append((timestamps[s_idx], timestamps[e_idx]))
            else:
                time_intervals.append((s_idx + offset, e_idx + offset))
                
        return time_intervals