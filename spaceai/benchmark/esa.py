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
    from spaceai.models.detectors.anomaly_detector import AnomalyDetector
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
        save_metadata: bool = True,
        **kwargs
    ):
        """Initializes a new benchmark run.

        Args:
            run_id (str): A unique identifier for this run.
            exp_dir (str): The directory where the results of this run are stored.
            mission (Optional[ESAMission]): the ESA mission to use.
        """
        super().__init__(run_id, exp_dir, data_root, save_metadata)
        self.mission = mission
        self.date_overrides = kwargs

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


    def load_channel(self, channel_id: str, train: bool = True, challenge: bool = False, continual: bool = False, overlapping_train: bool = True, **kwargs) -> ESA:
        """Load the training or testing dataset for a given channel."""
        if self.mission is None: raise ValueError("Mission must be set for ESABenchmark")
            
        return ESA(
            root=self.data_root, mission=self.mission, channel_id=channel_id,
            challenge=challenge,
            continual=continual,
            overlapping=overlapping_train if train else False,
            train=train,
            drop_last=train,
            **{**self.date_overrides, **kwargs}
        )
