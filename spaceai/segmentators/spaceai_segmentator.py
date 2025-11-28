"""SpaceAI segmentator module."""

from typing import (
    Callable,
    Dict,
)

import more_itertools as mit
import numpy as np
import pandas as pd  # type: ignore

from spaceai.data.anomaly_dataset import AnomalyDataset
from spaceai.segmentators.functions import apply_statistic_to_segments


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
        transformations: Dict[str, Callable],
        window_size: int = 50,
        step_size: int = 50,
        telecommands: bool = False,
        extract_features: bool = True,
        run_id: str = "esa_segments",
        exp_dir: str = "experiments",
    ) -> None:

        self.transformations = transformations
        self.window_size = window_size
        self.step_size = step_size
        self.telecommands = telecommands
        self.run_id = run_id
        self.exp_dir = exp_dir
        self.extract_features = extract_features

    def _get_anomaly_mask(self, anomalies, data_len):
        if anomalies is None or len(anomalies) == 0:
            return np.zeros(data_len, dtype=int)

        anomalies = np.array(anomalies)

        if anomalies.ndim == 1 and len(anomalies) == data_len:
            return anomalies.astype(int)

        mask = np.zeros(data_len, dtype=int)
        if anomalies.ndim == 2 and anomalies.shape[1] == 2:
            for start, end in anomalies:
                mask[int(start) : int(end) + 1] = 1
        return mask

    def segment(self, dataset_channel: AnomalyDataset):
        """
        Segment the dataset channel into windows and extract features.

        Args:
            dataset_channel (AnomalyDataset): The dataset channel to segment.

        Returns:
            Tuple[np.ndarray, List[List[int]]]: Segments and anomaly intervals.
        """
        data = dataset_channel.data[:, 0]  # type: ignore[attr-defined]

        anomalies_mask = self._get_anomaly_mask(
            dataset_channel.anomalies, len(data)  # type: ignore[attr-defined]
        )

        anomalies = apply_statistic_to_segments(
            data=anomalies_mask,
            func=np.max,
            window_shape=self.window_size,
            step_duration=self.step_size,
            func_kwargs={"axis": 1},
        )

        indices = np.where(anomalies == 1)[0]
        if indices.size == 0:
            anomalies_intervals = []
        else:
            groups = [list(group) for group in mit.consecutive_groups(indices)]
            anomalies_intervals = [[group[0], group[-1]] for group in groups]

        if self.extract_features:
            feature_list = [
                func(
                    data=data,
                    window_shape=self.window_size,
                    step_duration=self.step_size,
                )
                for func in self.transformations.values()
            ]
            segments = np.column_stack(feature_list)
        else:
            segments = np.lib.stride_tricks.sliding_window_view(
                data,
                window_shape=self.window_size,
            )[:: self.step_size]

        if self.extract_features:
            columns = list(self.transformations.keys())

            segments_df = pd.DataFrame(segments, columns=columns)
            segments = segments_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        return segments, anomalies_intervals
