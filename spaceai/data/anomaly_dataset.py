"""Base anomaly dataset module."""

import os

from torch.utils.data import Dataset


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

    def __getitem__(self, index):
        raise NotImplementedError
