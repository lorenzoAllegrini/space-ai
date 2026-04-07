"""NASA benchmark module for anomaly detection on NASA telemetry data."""

from __future__ import annotations

from typing import Tuple, List, Optional
import pandas as pd

from spaceai.data import NASA

from .benchmark import Benchmark


class NASABenchmark(Benchmark):
    """Benchmark for NASA telemetry anomaly detection dataset."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def channels(self) -> List[str]:
        """Get the default list of channels for the benchmark."""
        return NASA.channel_ids

    def get_global_temporal_params(self, channels: List[str]) -> Tuple[Optional[pd.Timestamp], Optional[float]]:
        """Get global start time and period for event-level aggregation."""
        return None, None

    def load_channel(
        self, channel_id: str, train: bool = True, overlapping_train: bool = True, **kwargs
    ) -> NASA:
        """Load the training or testing dataset for a given channel."""
        return NASA(
            root=self.data_root,
            channel_id=channel_id,
            overlapping=overlapping_train if train else False,
            train=train,
            drop_last=train,
            **kwargs
        )
