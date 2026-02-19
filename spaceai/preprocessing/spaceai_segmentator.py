"""SpaceAI segmentator module."""

import more_itertools as mit
import numpy as np

from spaceai.data.anomaly_dataset import AnomalyDataset
from spaceai.data.nasa import NASA


class SpaceAISegmentator:
    """
    Unified segmentator for SpaceAI datasets (ESA, NASA, OPS-SAT).
    Inherits vectorized segmentation logic and adds:
    - CSV Caching (persistence)
    - Feature Pooling
    - Telecommand handling
    - Anomaly interval extraction
    """

    def __init__(
        self,
        window_size: int = 100,
        step_size: int = 50,
        max_gap_sigma: float = 3.0,
    ) -> None:

        self.window_size = window_size
        self.step_size = step_size
        self.max_gap_sigma = max_gap_sigma

    def segment(self, dataset_channel: AnomalyDataset, time_aware: bool = True):
        """
        Segment the dataset channel into windows.

        Args:
            dataset_channel (AnomalyDataset): The dataset channel to segment.

        Returns:
            Dict:
                - segments (np.ndarray): Array of data segments (windows).
                - labels (np.ndarray): Binary labels for each segment (1 if anomalous, 0 otherwise).
                - intervals (List[List[int]]): List of [start_seg_idx, end_seg_idx] for anomalous segment ranges.
                - segment_indices (np.ndarray): Nx2 array mapping each segment to [start_idx, end_idx] in the original time series.
        """
        if isinstance(dataset_channel, NASA):
            time_aware = False

        data = dataset_channel.data[:, 0]  # type: ignore[attr-defined]

        if hasattr(dataset_channel, "block_intervals"):
            block_intervals = dataset_channel.block_intervals
        else:
            block_intervals = [(0, len(data))]


        if len(data) < self.window_size:
            return {
                "segments": np.array([]),
                "labels": np.array([]),
                "intervals": [],
                "segment_indices": np.array([]),
            }

        start = 0 
        starts = []
        for start_idx, end_idx in block_intervals:
            if end_idx - start_idx >= self.window_size:
                starts.append(np.arange(start_idx, end_idx - self.window_size + 1, self.step_size))
    
        starts = np.concatenate(starts) if starts else np.array([], dtype=int)
        ends = starts + self.window_size - 1
        segment_indices = np.column_stack((starts, ends))

        if starts.size > 0:
            segments = np.lib.stride_tricks.sliding_window_view(data, window_shape=self.window_size)[starts]
        else:
            segments = np.empty((0, self.window_size))

        n_segments = len(segments)
        anomaly_labels = np.zeros(n_segments, dtype=int)

        if dataset_channel.anomalies is not None:  # type: ignore[attr-defined]
            for start, end in dataset_channel.anomalies:  # type: ignore[attr-defined]
                lower_bound_start = max(0, start - self.window_size + 1)
                upper_bound_start = end
                
                # Find the range of segment start indices that satisfy the overlap condition
                i_min = np.searchsorted(starts, lower_bound_start, side='left')
                i_max = np.searchsorted(starts, upper_bound_start, side='right') - 1

                if i_min <= i_max:
                    anomaly_labels[i_min : i_max + 1] = 1

        indices = np.where(anomaly_labels == 1)[0]
        if indices.size == 0:
            anomalies_intervals = []
        else:
            groups = [list(group) for group in mit.consecutive_groups(indices)]
            anomalies_intervals = [[group[0], group[-1]] for group in groups]

        print("------------------------------------------------")
        print([(dataset_channel.timestamps[segment_indices[s][0]], dataset_channel.timestamps[segment_indices[e][1]]) for s, e in anomalies_intervals])
        print("-----------------------------------------------------")
        return {
            "segments": segments,
            "labels": anomaly_labels,
            "intervals": anomalies_intervals,
            "segment_indices": segment_indices,
        }


