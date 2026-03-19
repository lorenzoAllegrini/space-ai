"""Base anomaly dataset module."""

import os

from torch.utils.data import Dataset
from typing import List, Tuple, Optional

class AnomalyDataset(Dataset):
    """Base class for anomaly detection datasets."""

    def __init__(self, root: str):
        """Initialize the AnomalyDataset.

        Args:
            root (str): Root directory where the dataset is stored.
        """
        super().__init__()
        self.root = root

    @property
    def raw_folder(self) -> str:
        """Return the path to the raw data folder."""
        return os.path.join(self.root, self.__class__.__name__)

    @property
    def sampling_period(self) -> float:
        """Return the sampling period in seconds. Default is 1.0."""
        return 1.0

    def __getitem__(self, index):
        raise NotImplementedError


class AnomalyDatasetSubset(AnomalyDataset):
    """A continuous subset view of an AnomalyDataset."""

    def __init__(self, parent: AnomalyDataset, start_idx: int, end_idx: int):
        self.parent = parent
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.root = getattr(parent, "root", "")

    @property
    def raw_folder(self) -> str:
        return getattr(self.parent, "raw_folder", "")

    @property
    def sampling_period(self) -> float:
        return getattr(self.parent, "sampling_period", 1.0)

    @property
    def data(self):
        if hasattr(self.parent, "data"):
            return self.parent.data[self.start_idx : self.end_idx + 1]
        return None

    @property
    def timestamps(self):
        ts = getattr(self.parent, "timestamps", None)
        if ts is not None and len(ts) > 0:
            return ts[self.start_idx : self.end_idx + 1]
        return None

    @property
    def anomalies(self) -> Optional[List[Tuple[int, int]]]:
        parent_anom = getattr(self.parent, "anomalies", None)
        if parent_anom is not None:
            new_anom = []
            for s, e in parent_anom:
                ov_s = max(s, self.start_idx)
                ov_e = min(e, self.end_idx)
                if ov_s <= ov_e:
                    new_anom.append((ov_s - self.start_idx, ov_e - self.start_idx))
            return new_anom
        return None

    @property
    def block_intervals(self) -> List[Tuple[int, int]]:
        return [(0, len(self))]

    def __getitem__(self, idx):
        return self.parent[self.start_idx + idx]

    def __len__(self):
        return self.end_idx - self.start_idx + 1
