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
    
    train_test_split = pd.to_datetime("2022-06-02 03:00:00+00:00").tz_localize(None)
    resampling_rule=pd.Timedelta(seconds=1)

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
        max_gap_sigma: float = 3.0,
        split_percentage: Optional[float] = 0.6,
    ):
        """Initialize the dataset for a given channel.

        Args:
            root (str): The root directory of the dataset.
            channel_id (str): The channel ID to be used.
            mode (Literal["prediction", "anomaly"]): The mode of the dataset.
            overlapping (bool): The flag that indicates whether the dataset is overlapping.
            seq_length (int): the size of the sliding window
            train (bool): whether to use the training or test data
            download (bool): whether to download the dataset
            drop_last (bool): whether to drop the last incomplete sequence
            max_gap_sigma (float): max sigma for gap detection
        """
        super().__init__(root)
        if seq_length is None or seq_length < 1:
            raise ValueError(f"Invalid window size: {seq_length}")
        self._mode: Literal["prediction", "anomaly"] = mode
        self.overlapping: bool = overlapping
        self.window_size: int = seq_length if seq_length else 250
        self.train: bool = train
        self.drop_last: bool = drop_last
        self.n_predictions: int = n_predictions
        self.max_gap_sigma = max_gap_sigma
        self.split_percentage = split_percentage

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError(
                "Dataset not found. You can use download=True to download it"
            )

        self.load_channel(channel_id)

    @property
    def sampling_period(self) -> float:
        """Return the sampling period in seconds."""
        return self.resampling_rule.total_seconds()

    def load_channel(self, channel_id: str):
        """Load specific channel data into the dataset instance."""
        self.channel_id = channel_id
        if self._mode == "anomaly" and self.overlapping:
            logging.warning(
                "Channel %s is in anomaly mode and overlapping is set to True."
                " Anomalies will be repeated in the dataset.",
                channel_id,
            )
        
        # Unpack the returned values
        (
            self.data, 
            self.anomalies, 
            self.timestamps, 
            self.start_time, 
            self.end_time, 
            self.block_intervals,
        ) = self.load_and_preprocess(channel_id)
        
        if self.data.size == 0:
            logging.warning("Channel %s has no data after filtering.", channel_id)
            self.data = np.empty((0, 1), dtype=np.float32)
        else:
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
            length = max(0, self.data.shape[0] - self.window_size - self.n_predictions + 1)
            return length
        length = int(self.data.shape[0] / (self.window_size + self.n_predictions))
        if self.drop_last:
            return math.floor(length)
        return math.ceil(length)

    def _check_exists(self) -> bool:
        """Check if the dataset exists on the local filesystem."""
        return os.path.exists(os.path.join(self.raw_folder, "data"))

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
        
        # Process ALL channels and save to raw/data folder
        data_path = os.path.join(self.raw_folder, "data")
        os.makedirs(data_path, exist_ok=True)
        
        for channel_id in self.channel_ids:
            channel_df = df[df["channel"] == channel_id].copy()
            
            # Sort by timestamp
            if "timestamp" in channel_df.columns:
                channel_df["timestamp"] = pd.to_datetime(channel_df["timestamp"])
                channel_df = channel_df.sort_values(by="timestamp")

            target_df = channel_df.drop(columns=["train", "channel", "label", "sampling"])
            output_file = os.path.join(data_path, f"{channel_id}.csv")
            target_df.to_csv(output_file, index=False)
            
        os.remove(csv_path)

    def _apply_resampling_rule_(
        self, channel_df: pd.DataFrame, start_date: pd.Timestamp, end_date: pd.Timestamp
    ) -> pd.DataFrame:
        """Resample the dataframe using zero order hold.

        Args:
            channel_df (pd.DataFrame): The dataframe to resample.
            start_date (pd.Timestamp): The start date of the dataframe.
            end_date (pd.Timestamp): The end date of the dataframe.

        Returns:
            pd.DataFrame: The resampled dataframe.
        """
        # Determine split date
        split_date = self.train_test_split
        if self.split_percentage is not None:
            split_date = start_date + (end_date - start_date) * self.split_percentage

        # Adjust bounds based on train/test mode and split date
        if self.train:
            if end_date > split_date:
                end_date = split_date
        else:
            if start_date < split_date:
                start_date = split_date
        
        # Filter by adjusted bounds first
        channel_df = channel_df[(channel_df.index >= start_date) & (channel_df.index <= end_date)].copy()
        
        if len(channel_df) == 0:
            return pd.DataFrame(), []

        # Ensure index is unique before processing
        channel_df = channel_df[~channel_df.index.duplicated(keep='first')]
        timestamps = channel_df.index.values

        # Detect gaps
        if len(timestamps) > 1:
            diffs = np.diff(timestamps)
            diffs_sec = diffs.astype('timedelta64[ms]').astype(float) / 1000.0
            
            median_dt = np.median(diffs_sec)
            std_dt = np.std(diffs_sec)
            
            sigma_val = max(std_dt, median_dt * 0.1)
            gap_threshold_sec = median_dt + self.max_gap_sigma * sigma_val
            
            gap_threshold = np.timedelta64(int(gap_threshold_sec * 1000), 'ms')
            
            # boolean mask for gaps
            is_gap = diffs > gap_threshold
        else:
            is_gap = np.array([], dtype=bool)

        if len(is_gap) > 0:
            split_indices = np.where(is_gap)[0] + 1
            block_boundaries = np.concatenate(([0], split_indices, [len(timestamps)]))
        else:
            block_boundaries = np.array([0, len(timestamps)])
        
        resampled_blocks = []
        
        for i in range(len(block_boundaries) - 1):
            start_idx, end_idx = block_boundaries[i], block_boundaries[i+1]
            if start_idx >= end_idx: continue

            block_df = channel_df.iloc[start_idx:end_idx]
            
            # Resample block
            block_start_date = block_df.index[0]
            block_end_date = block_df.index[-1]

            first_index_resampled = pd.Timestamp(block_start_date).floor(
                freq=self.resampling_rule
            )
            last_index_resampled = pd.Timestamp(block_end_date).ceil(
                freq=self.resampling_rule
            )
            resampled_range = pd.date_range(
                first_index_resampled,
                last_index_resampled,
                freq=self.resampling_rule,
            )
            
            block_resampled = block_df.reindex(resampled_range, method="ffill")
            
            # Handle start of block potential NaN if grid starts before data
            if len(block_df) > 0 and pd.isna(block_resampled.iloc[0]["value"]):
                block_resampled.iloc[0] = block_df.iloc[0]
                block_resampled = block_resampled.ffill()

            resampled_blocks.append(block_resampled)
            
        if len(resampled_blocks) > 0:
            final_param_df = pd.concat(resampled_blocks)
            
            # Reconstruct valid block intervals
            block_intervals = []
            curr_idx = 0
            for block in resampled_blocks:
                block_intervals.append((curr_idx, curr_idx + len(block)))
                curr_idx += len(block)
        else:
            final_param_df = pd.DataFrame()
            block_intervals = []

        return final_param_df, block_intervals



    def load_and_preprocess(
        self,
        channel_id:str,
    ) -> Tuple[np.ndarray, list[list[int]] | None, np.ndarray | None, pd.Timestamp | None, pd.Timestamp | None, float | None]:
        """Load and preprocess the dataset.
        
        Returns:
            Tuple: (data, anomalies, timestamps, start_time, end_time, sampling_period)
        """

        if channel_id in self.channel_ids:
            channel_df = pd.read_csv(os.path.join(self.raw_folder, "data", f"{channel_id}.csv"))
        else:
            raise ValueError("channel_id not in available channels")

        # Convert timestamp before passing to resampling
        block_intervals = []
        if "timestamp" in channel_df.columns:
            channel_df["timestamp"] = pd.to_datetime(channel_df["timestamp"]).dt.tz_localize(None)
            channel_df = channel_df.set_index("timestamp").sort_index()
            
            # Use timestamps for resampling bounds if available
            channel_df, block_intervals = self._apply_resampling_rule_(
                channel_df,
                channel_df.index[0],
                channel_df.index[-1],
            )
        else:
            # If no timestamp column, assume continuous block
            block_intervals = [(0, len(channel_df))]
        
        if len(channel_df) == 0:
            return np.array([]), None, None, None, None, block_intervals

        channel_df = channel_df.ffill().bfill().astype(np.float32)

        timestamps = None
        start_time = None
        end_time = None


        if "timestamp" in channel_df.columns or isinstance(channel_df.index, pd.DatetimeIndex):
            if isinstance(channel_df.index, pd.DatetimeIndex):
                timestamps = channel_df.index.values
            else:
                timestamps = pd.to_datetime(channel_df["timestamp"]).dt.tz_localize(None).values

            if timestamps is not None and len(timestamps) > 0:
                start_time = timestamps[0]
                end_time = timestamps[-1]
    
        data = channel_df["value"].astype(np.float32).values
        anomalies = None

        if self._mode == "prediction":
            return data, None, timestamps, start_time, end_time, block_intervals

        if "anomaly" in channel_df.columns:
            anomalies_series = channel_df["anomaly"].fillna(0).astype(int)
            anomaly_indices = np.where(anomalies_series == 1)[0].tolist()
            groups = [list(group) for group in mit.consecutive_groups(anomaly_indices)]
            anomalies = [[group[0], group[-1]] for group in groups]
        
        return data, anomalies, timestamps, start_time, end_time, block_intervals

    @property
    def split_folder(self) -> str:
        """Return the path to the folder containing the split data."""
        return os.path.join(self.raw_folder, "data", "train" if self.train else "test")

    @property
    def in_features_size(self) -> int:
        """Return the size of the input features."""
        if self.data is None:
            return 1
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
        if self.channel_id:
            self.load_channel(self.channel_id)
