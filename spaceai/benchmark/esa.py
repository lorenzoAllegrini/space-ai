"""ESA benchmark module for anomaly detection on ESA telemetry data."""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    List,
    Dict,
)

from spaceai.data import (
    ESA,
    ESAMission,
)

from .benchmark import Benchmark

if TYPE_CHECKING:
    from spaceai.models.predictors import SequenceModel
    from spaceai.models.anomaly import AnomalyDetector
    from .callbacks import Callback

import pandas as pd

class ESABenchmark(Benchmark):
    """Benchmark for ESA telemetry anomaly detection datasets."""

    def __init__(
        self,
        run_id: str,
        exp_dir: str,
        segmentator: Any = None,
        mission: Optional[ESAMission] = None,
        feature_extractor: Optional[Any] = None,
        seq_length: int = 250,
        n_predictions: int = 1,
        data_root: str = "datasets",
    ):
        """Initializes a new benchmark run.

        Args:
            run_id (str): A unique identifier for this run.
            exp_dir (str): The directory where the results of this run are stored.
            mission (Optional[ESAMission]): the ESA mission to use.
            seq_length (int): The length of the sequences used for training and testing.
        """
        super().__init__(run_id, exp_dir, segmentator, feature_extractor, seq_length, n_predictions, data_root)
        self.mission = mission

    def get_default_channels(self) -> List[str]:
        """Get the default list of channels for the benchmark."""
        if self.mission is None:
            raise ValueError("Mission must be set for ESABenchmark")
        return self.mission.target_channels

    def get_global_temporal_params(self, channels: List[str]) -> Tuple[pd.Timestamp, float]:
        """Get global start time and period for event-level aggregation."""
        if self.mission is None:
            raise ValueError("Mission must be set for ESABenchmark")
            
        min_start_time = self.mission.train_test_split.tz_localize(None) # Use train_test_split as reference
        min_period = self.mission.resampling_rule.total_seconds()
        return min_start_time, min_period


    def load_channel(
        self, channel_id: str, overlapping_train: bool = True
    ) -> Tuple[ESA, ESA]:
        """Load the training and testing datasets for a given channel.

        Args:
            channel_id (str): the ID of the channel to be used
            overlapping_train (bool): whether to use overlapping sequences for the training dataset

        Returns:
            Tuple[ESA, ESA]: training and testing datasets
        """
        if self.mission is None:
            raise ValueError("Mission must be set for ESABenchmark")
        train_channel = ESA(
            root=self.data_root,
            mission=self.mission,
            channel_id=channel_id,
            mode="prediction",
            overlapping=overlapping_train,
            seq_length=self.seq_length,
            n_predictions=self.n_predictions,
        )

        test_channel = ESA(
            root=self.data_root,
            mission=self.mission,
            channel_id=channel_id,
            mode="anomaly",
            overlapping=False,
            seq_length=self.seq_length,
            train=False,
            drop_last=False,
            n_predictions=1,
        )

        return train_channel, test_channel
