"""SpaceAI segmentator module."""

import more_itertools as mit
import numpy as np
import pandas as pd  # type: ignore

from spaceai.data.anomaly_dataset import AnomalyDataset


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
        window_size: int = 50,
        step_size: int = 50,
    ) -> None:

        self.window_size = window_size
        self.step_size = step_size

    def segment(self, dataset_channel: AnomalyDataset):
        """
        Segment the dataset channel into windows and extract features.

        Args:
            dataset_channel (AnomalyDataset): The dataset channel to segment.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Segments and anomaly labels (mask).
        """
        data = dataset_channel.data[:, 0]  # type: ignore[attr-defined]

        segments = np.lib.stride_tricks.sliding_window_view(
            data,
            window_shape=self.window_size,
        )[:: self.step_size]

        n_segments = len(segments)
        anomaly_labels = np.zeros(n_segments, dtype=int)

        if dataset_channel.anomalies is not None:  # type: ignore[attr-defined]
            for start, end in dataset_channel.anomalies:  # type: ignore[attr-defined]
                i_min = max(0, (start - self.window_size) // self.step_size + 1)
                i_max = min(n_segments - 1, end // self.step_size)

                if i_min <= i_max:
                    anomaly_labels[int(i_min) : int(i_max) + 1] = 1

        indices = np.where(anomaly_labels == 1)[0]
        if indices.size == 0:
            anomalies_intervals = []
        else:
            groups = [list(group) for group in mit.consecutive_groups(indices)]
            anomalies_intervals = [[group[0], group[-1]] for group in groups]

        return {
            "segments": segments,
            "labels": anomaly_labels,
            "intervals": anomalies_intervals,
        }
