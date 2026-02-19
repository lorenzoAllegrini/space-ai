"""OPS-SAT benchmark module for anomaly detection on OPS-SAT telemetry data."""

from __future__ import annotations
import pandas as pd
from typing import List, Tuple

from spaceai.data import OPSSAT

from .benchmark import Benchmark


class OPSSATBenchmark(Benchmark):
    """Benchmark for OPS-SAT telemetry anomaly detection dataset."""

    def get_default_channels(self) -> List[str]:
        """Get the default list of channels for the benchmark."""
        return OPSSAT.channel_ids

    def get_global_temporal_params(self, channels: List[str]) -> Tuple[pd.Timestamp, float]:
        """Get global start time and period for event-level aggregation."""
        min_start_time = OPSSAT.train_test_split.tz_localize(None)
        min_period = OPSSAT.resampling_rule.total_seconds()
        return min_start_time, min_period

    def load_channel(
        self, channel_id: str, overlapping_train: bool = True
    ) -> Tuple[OPSSAT, OPSSAT]:
        """Load the training and testing datasets for a given channel.

        Args:
            channel_id (str): the ID of the channel to be used
            overlapping_train (bool): whether to use overlapping sequences for training

        Returns:
            Tuple[OPSSAT, OPSSAT]: training and testing datasets
        """
        train_channel = OPSSAT(
            root=self.data_root,
            channel_id=channel_id,
            mode="anomaly",
            overlapping=overlapping_train,
            seq_length=self.seq_length,
            n_predictions=self.n_predictions,
        )

        test_channel = OPSSAT(
            root=self.data_root,
            channel_id=channel_id,
            mode="anomaly",
            overlapping=False,
            seq_length=self.seq_length,
            train=False,
            drop_last=False,
            n_predictions=1,
        )

        return train_channel, test_channel
