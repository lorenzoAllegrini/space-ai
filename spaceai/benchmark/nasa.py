"""NASA benchmark module for anomaly detection on NASA telemetry data."""

from __future__ import annotations

from typing import Tuple, List, Optional
import pandas as pd

from spaceai.data import NASA

from .benchmark import Benchmark


class NASABenchmark(Benchmark):
    """Benchmark for NASA telemetry anomaly detection dataset."""

    def get_default_channels(self) -> List[str]:
        """Get the default list of channels for the benchmark."""
        return NASA.channel_ids

    def get_global_temporal_params(self, channels: List[str]) -> Tuple[Optional[pd.Timestamp], Optional[float]]:
        """Get global start time and period for event-level aggregation."""
        return None, None

    def load_channel(
        self, channel_id: str, overlapping_train: bool = True
    ) -> Tuple[NASA, NASA]:
        """Load the training and testing datasets for a given channel.

        Args:
            channel_id (str): the ID of the channel to be used
            overlapping_train (bool): whether to use overlapping sequences for training

        Returns:
            Tuple[NASA, NASA]: training and testing datasets
        """
        train_channel = NASA(
            root=self.data_root,
            channel_id=channel_id,
            mode="prediction",
            overlapping=overlapping_train,
            seq_length=self.seq_length,
            n_predictions=self.n_predictions,
        )

        test_channel = NASA(
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
