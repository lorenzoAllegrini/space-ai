"""OPS-SAT benchmark module for anomaly detection on OPS-SAT telemetry data."""

from __future__ import annotations
import pandas as pd
from typing import List, Tuple

from spaceai.data import OPSSAT

from .benchmark import Benchmark


class OPSSATBenchmark(Benchmark):
    """Benchmark for OPS-SAT telemetry anomaly detection dataset."""

    def __init__(self, split_percentage: float = 0.6, **kwargs):
        super().__init__(**kwargs)
        self.split_percentage = split_percentage

    @property
    def channels(self) -> List[str]:
        """Get the default list of channels for the benchmark."""
        return OPSSAT.channel_ids

    def get_global_temporal_params(self, channels: List[str]) -> Tuple[pd.Timestamp, float]:
        """Get global start time and period for event-level aggregation."""
        min_start_time = OPSSAT.train_test_split.tz_localize(None)
        min_period = OPSSAT.resampling_rule.total_seconds()
        return min_start_time, min_period

    def load_channel(
        self, channel_id: str, train: bool = True, overlapping_train: bool = True, **kwargs
    ) -> OPSSAT:
        """Load the training or testing dataset for a given channel."""
        return OPSSAT(
            root=self.data_root,
            channel_id=channel_id,
            overlapping=overlapping_train if train else False,
            train=train,
            drop_last=train,
            split_percentage=self.split_percentage,
            **kwargs
        )
