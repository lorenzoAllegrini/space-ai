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
        mission: Optional[ESAMission] = None,
        data_root: str = "datasets",
    ):
        """Initializes a new benchmark run.

        Args:
            run_id (str): A unique identifier for this run.
            exp_dir (str): The directory where the results of this run are stored.
            mission (Optional[ESAMission]): the ESA mission to use.
        """
        super().__init__(run_id, exp_dir, data_root)
        self.mission = mission

    @property
    def channels(self) -> List[str]:
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
        self, channel_id: str, mode: str = "train", overlapping_train: bool = True, **kwargs
    ) -> ESA:
        """Load the training or testing dataset for a given channel."""
        if self.mission is None:
            raise ValueError("Mission must be set for ESABenchmark")
            
        continual = kwargs.pop("continual", False)
            
        if mode == "train":
            return ESA(
                root=self.data_root,
                mission=self.mission,
                channel_id=channel_id,
                mode="prediction" if not continual else "continual",
                overlapping=overlapping_train,
                **kwargs
            )
        elif mode == "test":
            return ESA(
                root=self.data_root,
                mission=self.mission,
                channel_id=channel_id,
                mode="anomaly",
                overlapping=False,
                train=False,
                drop_last=False,
                **kwargs
            )
        else:
            # Flexible mode for challenge, continual, etc.
            return ESA(
                root=self.data_root,
                mission=self.mission,
                channel_id=channel_id,
                mode=mode,
                overlapping=overlapping_train if mode != "challenge" else False,
                train=False if mode == "challenge" else True,
                **kwargs
            )
