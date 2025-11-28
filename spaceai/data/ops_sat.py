"""OPS-SAT dataset module."""

import logging
import math
import os
import zipfile
from typing import (
    Literal,
    Optional,
    Tuple,
    Union,
)

import more_itertools as mit
import numpy as np
import pandas as pd  # type: ignore
import torch

from .anomaly_dataset import AnomalyDataset
from .utils import download_file


class OPSSAT(AnomalyDataset):
    """OPS-SAT benchmark dataset for anomaly detection."""

    resource = "https://zenodo.org/api/records/12588359/files-archive"

    channel_ids = [
        "CADC0872",
        "CADC0873",
        "CADC0874",
        "CADC0884",
        "CADC0886",
        "CADC0888",
        "CADC0890",
        "CADC0892",
        "CADC0894",
    ]

    def __init__(
        self,
        root: str,
        channel_id: str,
        mode: Literal["prediction", "anomaly"],
        overlapping: bool = False,
        seq_length: Optional[int] = 250,
        n_predictions: int = 1,
        train: bool = True,
        download: bool = True,
        drop_last: bool = True,
    ):
        """Initialize the dataset for a given channel.

        Args:
            channel_id (str): the ID of the channel to be used

            seq_length (int): the size of the sliding window
            train (bool): whether to use the training or test data
            download (bool): whether to download the dataset
            drop_last (bool): whether to drop the last incomplete sequence
        """
        super().__init__(root)
        if seq_length is None or seq_length < 1:
            raise ValueError(f"Invalid window size: {seq_length}")
        self.channel_id: str = channel_id
        self._mode: Literal["prediction", "anomaly"] = mode
        self.overlapping: bool = overlapping
        self.window_size: int = seq_length if seq_length else 250
        self.train: bool = train
        self.drop_last: bool = drop_last
        self.n_predictions: int = n_predictions

        if not channel_id in self.channel_ids:
            raise ValueError(f"Channel ID {channel_id} is not valid")

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError(
                "Dataset not found. You can use download=True to download it"
            )

        if self._mode == "anomaly" and self.overlapping:
            logging.warning(
                "Channel %s is in anomaly mode and overlapping is set to True."
                " Anomalies will be repeated in the dataset.",
                channel_id,
            )

        self.data, self.anomalies = self.load_and_preprocess()
        self.data = self.data.reshape(-1, 1)

    def __getitem__(self, index: int) -> Union[
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        """Return the data at the given index."""
        if index < 0 or index >= len(self):
            raise IndexError(f"Index {index} out of bounds")
        first_idx = (
            index
            if self.overlapping
            else index * (self.window_size + self.n_predictions - 1)
        )
        last_idx = first_idx + self.window_size
        if last_idx > len(self.data) - self.n_predictions:
            last_idx = len(self.data) - self.n_predictions
        x, y_true = (
            torch.tensor(self.data[first_idx:last_idx]),
            torch.from_numpy(
                np.stack(
                    [
                        self.data[first_idx + i + 1 : last_idx + i + 1, 0]
                        for i in range(self.n_predictions)
                    ]
                )
            ).T,
        )
        return x, y_true

    def __len__(self) -> int:
        if self.overlapping:
            length = self.data.shape[0] - self.window_size - self.n_predictions + 1
            return length
        length = int(self.data.shape[0] / (self.window_size + self.n_predictions))
        if self.drop_last:
            return math.floor(length)
        return math.ceil(length)

    def _check_exists(self) -> bool:
        """Check if the dataset exists on the local filesystem."""
        return os.path.exists(os.path.join(self.split_folder, self.channel_id + ".csv"))

    def download(self):
        """Download the OPS-SAT dataset and save filtered train data by channel."""

        if self._check_exists():
            return

        zip_filepath = "ops_sat.zip"
        download_file(self.resource, to=zip_filepath)

        os.makedirs(self.raw_folder, exist_ok=True)

        with zipfile.ZipFile(zip_filepath, "r") as zip_ref:
            zip_ref.extract("segments.csv", path=self.raw_folder)

        os.remove(zip_filepath)

        csv_path = os.path.join(self.raw_folder, "segments.csv")
        df = pd.read_csv(csv_path)

        train_df = df[df["train"] == 1]
        channel_df = train_df[train_df["channel"] == self.channel_id]
        channel_df = channel_df.drop(columns=["train", "channel"])
        channel_path = self.split_folder
        os.makedirs(channel_path, exist_ok=True)
        output_file = os.path.join(channel_path, f"{self.channel_id}.csv")
        channel_df.to_csv(output_file, index=False)

        os.remove(csv_path)

    def load_and_preprocess(self) -> Tuple[np.ndarray, list[list[int]] | None]:
        """Load and preprocess the dataset."""

        df = pd.read_csv(os.path.join(self.split_folder, f"{self.channel_id}.csv"))

        data = df["value"].astype(np.float32).values

        if self._mode == "prediction":
            return data, None

        anomaly_indices = df.index[df["anomaly"] == 1].tolist()
        groups = [list(group) for group in mit.consecutive_groups(anomaly_indices)]
        anomalies = [[group[0], group[-1]] for group in groups]

        return data, anomalies

    @property
    def split_folder(self) -> str:
        """Return the path to the folder containing the split data."""
        return os.path.join(self.raw_folder, "data", "train" if self.train else "test")

    @property
    def in_features_size(self) -> int:
        """Return the size of the input features."""
        return self.data.shape[-1]

    @property
    def mode(self) -> str:
        """Return the mode of the dataset."""
        return self._mode

    @mode.setter
    def mode(self, mode: Literal["prediction", "anomaly"]):
        """Set the mode of the dataset."""
        if mode not in ["prediction", "anomaly"]:
            raise ValueError(f"Invalid mode {mode}")
        self._mode = mode
        self.data, self.anomalies = self.load_and_preprocess()
