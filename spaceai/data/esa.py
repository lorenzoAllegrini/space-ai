import logging
import math
import os
from dataclasses import dataclass
from enum import Enum
from typing import (
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

import numpy as np
import pandas as pd  # type: ignore
import torch

from .anomaly_dataset import AnomalyDataset
from .utils import download_and_extract_zip


class AnnotationLabel(Enum):
    """Enuemeration of annotation labels for ESA dataset."""

    NOMINAL = 0
    ANOMALY = 1
    RARE_EVENT = 2
    GAP = 3
    INVALID = 4


@dataclass
class ESAMission:
    """ESA mission dataclass with metadata of a single mission."""

    index: int
    """The index of the mission."""
    url_source: str
    """The URL source of the mission data."""
    dirname: str
    """The directory name of the mission data."""
    train_test_split: pd.Timestamp
    """The split date between training and testing data."""
    start_date: pd.Timestamp
    """The start date of the mission."""
    end_date: pd.Timestamp
    """The end date of the mission."""
    resampling_rule: pd.Timedelta
    """The resampling rule for the data."""
    monotonic_channel_range: tuple[int, int]
    """The range of monotonic channels."""
    parameters: list[str]
    """The list of parameters."""
    telecommands: list[str]
    """The list of telecommands."""
    target_channels: list[str]
    """The list of target channels."""

    @property
    def inner_dirpath(self):
        return os.path.join(self.dirname, self.dirname)

    @property
    def all_channels(self):
        return self.parameters + self.telecommands


class ESAMissions(Enum):
    """ESA missions enumeration that contains metadata of mission1 and mission2."""

    MISSION_1: ESAMission = ESAMission(
        index=1,
        url_source="https://zenodo.org/records/12528696/files/ESA-Mission1.zip?download=1",
        dirname="ESA-Mission1",
        train_test_split=pd.to_datetime("2007-01-01"),
        start_date=pd.to_datetime("2000-01-01"),
        end_date=pd.to_datetime("2014-01-01"),
        resampling_rule=pd.Timedelta(seconds=30),
        monotonic_channel_range=(4, 11),
        parameters=[f"channel_{i + 1}" for i in range(76)],
        telecommands=[f"telecommand_{i + 1}" for i in range(698)],
        target_channels=[
            f"channel_{i}"
            for i in [*list(range(12, 53)), *list(range(57, 67)), *list(range(70, 77))]
        ],
    )
    MISSION_2: ESAMission = ESAMission(
        index=2,
        url_source="https://zenodo.org/records/12528696/files/ESA-Mission2.zip?download=1",
        dirname="ESA-Mission2",
        train_test_split=pd.to_datetime("2001-10-01"),
        start_date=pd.to_datetime("2000-01-01"),
        end_date=pd.to_datetime("2003-07-01"),
        resampling_rule=pd.Timedelta(seconds=18),
        monotonic_channel_range=(29, 46),
        parameters=[f"channel_{i + 1}" for i in range(100)],
        telecommands=[f"telecommand_{i + 1}" for i in range(123)],
        target_channels=[
            f"channel_{i}"
            for i in [
                *list(range(9, 29)),
                *list(range(58, 60)),
                *list(range(70, 92)),
                *list(range(96, 99)),
            ]
        ],
    )


class ESA(
    AnomalyDataset,
):
    """ESA benchmark dataset for anomaly detection.

    The dataset consists of multivariate time series data collected from ESA's
    spacecrafts telemetry data. The data is used to detect anomalies in the spacecrafts'
    telemetry data and evaluate the performance of anomaly detection algorithms.
    """

    def __init__(
        self,
        root: str,
        mission: ESAMission,
        channel_id: str,
        challenge: bool = False,
        continual: bool = False,
        overlapping: bool = False,
        seq_length: Optional[int] = 250,
        n_predictions: int = 1,
        train: bool = True,
        download: bool = True,
        uniform_start_end_date: bool = True,
        drop_last: bool = True,
        use_telecommands: bool = False,
        train_start_date: Optional[Union[str, pd.Timestamp]] = None,
        train_end_date: Optional[Union[str, pd.Timestamp]] = None,
        test_start_date: Optional[Union[str, pd.Timestamp]] = None,
        test_end_date: Optional[Union[str, pd.Timestamp]] = None,
    ):
        """ESABenchmark class that preprocesses and loads ESA dataset for training and
        testing.

        Args:
            root (str): The root directory of the dataset.
            mission (ESAMission): The mission type of the dataset.
            channel_id (str): The channel ID to be used.
            challenge (bool): The flag that indicates whether the dataset is for challenge.
            continual (bool): The flag that indicates whether the dataset is for continual learning.
            overlapping (bool): The flag that indicates whether the dataset is overlapping.
            seq_length (Optional[int]): The length of the sequence for each sample.
            n_predictions (int): The number of predictions.
            train (bool): The flag that indicates whether the dataset is for training or testing.
            download (bool): The flag that indicates whether the dataset should be downloaded.
            uniform_start_end_date (bool): The flag that indicates whether the dataset should be
                resampled to have uniform start and end date.
            drop_last (bool): The flag that indicates whether the last sample should be dropped.
            use_telecommands (bool): The flag that indicates whether to use telecommands.
        """
        super().__init__(root)
        if seq_length is None or seq_length < 1:
            raise ValueError(f"Invalid window size: {seq_length}")

        self.root = root
        self.mission = mission
        self.channel_id: str = channel_id
        self.challenge: bool = challenge
        self.continual: bool = continual
        self.overlapping: bool = overlapping
        self.window_size: int = seq_length if seq_length else 250
        self.train: bool = train
        self.uniform_start_end_date: bool = uniform_start_end_date
        self.drop_last: bool = drop_last
        self.n_predictions: int = n_predictions
        self.use_telecommands: bool = use_telecommands
        
        # Date overrides
        self.train_start_date = pd.to_datetime(train_start_date) if train_start_date else None
        self.train_end_date = pd.to_datetime(train_end_date) if train_end_date else None
        self.test_start_date = pd.to_datetime(test_start_date) if test_start_date else None
        self.test_end_date = pd.to_datetime(test_end_date) if test_end_date else None

        if not channel_id in self.mission.all_channels:
            raise ValueError(f"Channel ID {channel_id} is not valid")

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError(
                "Dataset not found. You can use download=True to download it"
            )

        if not self.train and not self.challenge and self.overlapping:
            logging.warning(
                "Channel %s is in anomaly mode and overlapping is set to True."
                " Anomalies will be repeated in the dataset.",
                channel_id,
            )

        self.timestamps = None
        self.data, self.anomalies, self.communication_gaps, self.block_intervals = self.load_and_preprocess(
            channel_id
        )

    def __len__(self) -> int:
        if self.overlapping:
            length = self.data.shape[0] - \
                self.window_size - self.n_predictions + 1
            return length
        length = int(self.data.shape[0] /
                     (self.window_size + self.n_predictions))
        if self.drop_last:
            return math.floor(length)
        return math.ceil(length)

    def __getitem__(self, index: Union[int, slice]) -> Union[
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

    def download(self):
        """Download the dataset from the given URL and extract it to the given
        directory."""
        if self._check_exists():
            return
        download_and_extract_zip(
            self.mission.url_source,
            os.path.join(self.root, self.mission.dirname),
            cleanup=True,
        )

    def _check_exists(self) -> bool:
        """Check if the dataset exists on the local filesystem."""
        return os.path.exists(os.path.join(self.root, self.mission.dirname))

    def _apply_resampling_rule_(
        self, 
        channel_df: pd.DataFrame, 
        start_date: pd.Timestamp, 
        end_date: pd.Timestamp,
        gap_intervals: List[Tuple[pd.Timestamp, pd.Timestamp]] = None
    ) -> Tuple[pd.DataFrame, List[Tuple[int, int]]]:
        """Resample the dataframe using zero order hold, respecting official gaps.
        """
        # Filter by bounds
        channel_df = channel_df[(channel_df.index >= start_date) & (channel_df.index <= end_date)].copy()
        
        if len(channel_df) == 0:
            return pd.DataFrame(), []

        channel_df = channel_df[~channel_df.index.duplicated(keep='first')]
        timestamps = channel_df.index.values

        # Detect gaps
        if len(timestamps) > 1:
            if gap_intervals:
                is_gap = np.zeros(len(timestamps) - 1, dtype=bool)
                for g_start, g_end in gap_intervals:
                    mask = (timestamps[:-1] < g_end) & (timestamps[1:] > g_start)
                    is_gap |= mask
            else:
                diffs = np.diff(timestamps)
                diffs_sec = diffs.astype('timedelta64[ms]').astype(float) / 1000.0
                median_dt = np.median(diffs_sec)
                resampling_seconds = pd.Timedelta(self.mission.resampling_rule).total_seconds()
                gap_threshold_sec = max(median_dt * 20.0, resampling_seconds * 20.0, 3000.0)
                gap_threshold = np.timedelta64(int(gap_threshold_sec * 1000), 'ms')
                is_gap = diffs > gap_threshold
        else:
            is_gap = np.array([], dtype=bool)

        if len(is_gap) > 0:
            # We only want to split when is_gap transitions, or correctly:
            # every True in is_gap represents a boundary where we must REFRESH the timeline.
            # However, consecutive True's mean consecutive individual points.
            # We want to find contiguous ranges of False (real blocks) and skip True.
            
            # Find indices where is_gap is True.
            gap_indices = np.where(is_gap)[0]
            
            # We split the data into blocks. A block is a range of consecutive False.
            # The indices in block_boundaries should represent the start/end of these False ranges.
            split_indices = np.where(is_gap)[0] + 1
            # Optimization: merge consecutive split indices to avoid many 1-sample blocks
            if len(split_indices) > 1:
                # Keep only indices that are NOT consecutive
                diffs = np.diff(split_indices)
                mask = np.concatenate(([True], diffs > 1))
                split_indices = split_indices[mask]

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
                freq=self.mission.resampling_rule
            )
            last_index_resampled = pd.Timestamp(block_end_date).ceil(
                freq=self.mission.resampling_rule
            )
            resampled_range = pd.date_range(
                first_index_resampled,
                last_index_resampled,
                freq=self.mission.resampling_rule,
            )
            
            block_resampled = block_df.reindex(resampled_range, method="ffill")
            
            if pd.isna(block_resampled.iloc[0, 0]):
                block_resampled.iloc[0] = block_df.iloc[0]
                block_resampled = block_resampled.ffill()

            resampled_blocks.append(block_resampled)
            
        if not resampled_blocks:
            return pd.DataFrame(), []

        
        final_df = pd.concat(resampled_blocks)
        block_intervals = []
        curr_idx = 0
        for block in resampled_blocks:
            block_intervals.append((curr_idx, curr_idx + len(block)))
            curr_idx += len(block)
            
        return final_df, block_intervals
        
        return pd.DataFrame(), []

    def load_and_preprocess(
        self,
        channel_id: str,
    ) -> Tuple[np.ndarray, List[Tuple[int, int]], List[Tuple[int, int]], List[Tuple[int, int]]]:
        """Preprocess the channel dataset."""
        source_folder = os.path.join(self.root, self.mission.inner_dirpath)
        if not self.train and self.challenge:
            return self.load_challenge_channel(channel_id)

        # 1. Load telemetry
        if channel_id in self.mission.parameters:
            channel_df = pd.read_pickle(os.path.join(source_folder, "channels", f"{channel_id}.zip"))
        elif channel_id in self.mission.telecommands:
            channel_df = pd.read_pickle(os.path.join(source_folder, "telecommands", f"{channel_id}.zip"))
        else:
            raise ValueError("channel_id not in available channels")

        # 2. Extract gap timestamps early from labels
        labels_df = pd.read_csv(os.path.join(source_folder, "labels.csv"))
        anomaly_types_df = pd.read_csv(os.path.join(source_folder, "anomaly_types.csv"))
        labels_df = pd.merge(labels_df, anomaly_types_df, how="inner", on="ID")
        for dcol in ["StartTime", "EndTime"]:
            labels_df[dcol] = pd.to_datetime(labels_df[dcol]).dt.tz_localize(None)
        
        chan_labels = labels_df.loc[labels_df["Channel"] == channel_id]
        gaps = chan_labels[chan_labels["Category"] == "Communication Gap"]
        gap_intervals = [(row["StartTime"], row["EndTime"]) for _, row in gaps.iterrows()]

        # 3. Determine Global Temporal Bounds
        if self.train:
            global_start = self.train_start_date or self.mission.start_date
            global_end = self.train_end_date or self.mission.train_test_split
        else:
            global_start = self.test_start_date or self.mission.train_test_split
            global_end = self.test_end_date or self.mission.end_date

        if not self.challenge and not self.continual:
            # Legacy handling if specifically requested, but overrides take precedence
            if self.train:
                if global_end > self.mission.train_test_split and self.train_end_date is None:
                    global_end = self.mission.train_test_split
            else:
                if global_start < self.mission.train_test_split and self.test_start_date is None:
                    global_start = self.mission.train_test_split
        
        # Override with data-specific bounds if not using uniform_start_end_date
        if not self.uniform_start_end_date:
            global_start = max(global_start, channel_df.index[0])
            global_end = min(global_end, channel_df.index[-1])

        # 4. Apply resampling (Single pass)
        channel_df, block_intervals = self._apply_resampling_rule_(
            channel_df, global_start, global_end, gap_intervals=gap_intervals
        )

        channel_df = channel_df.ffill().bfill().astype(np.float32)

        # 5. Telecommands
        if self.use_telecommands:
            telecommands_csv = pd.read_csv(os.path.join(source_folder, "telecommands.csv"))
            prioritized_tcs = telecommands_csv.loc[telecommands_csv["Priority"] >= 3, "Telecommand"].to_numpy().flatten()

            telecommand_dfs = []
            for i, tc in enumerate(prioritized_tcs):
                tc_file = os.path.join(source_folder, "telecommands", f"{tc}.zip")
                if os.path.exists(tc_file):
                    df_tc = pd.read_pickle(tc_file)
                    df_tc_bool = pd.Series(0, index=channel_df.index, name=tc, dtype=np.int8)
                    for ts in df_tc.index:
                        pos = channel_df.index.searchsorted(ts, side="left")
                        if pos < len(channel_df.index):
                            df_tc_bool.iloc[pos] = 1
                    telecommand_dfs.append(df_tc_bool.to_frame())

            if telecommand_dfs:
                tele_df = telecommand_dfs[0]
                for df in telecommand_dfs[1:]:
                    tele_df = tele_df.join(df, how="outer")
                channel_df = channel_df.join(tele_df.fillna(0), how="left")

        # 6. Map labels
        anomalies = []
        communication_gaps_idx = [] 

        for _, row in chan_labels.iterrows():
            start_t = row["StartTime"].floor(freq=self.mission.resampling_rule)
            end_t = row["EndTime"].ceil(freq=self.mission.resampling_rule)
            
            s_idx = channel_df.index.searchsorted(start_t, side="left")
            e_idx = channel_df.index.searchsorted(end_t, side="right") - 1
            
            if s_idx < len(channel_df) and e_idx >= 0 and s_idx <= e_idx:
                if row["Category"] in ["Anomaly", "Rare Event"]:
                    anomalies.append((int(s_idx), int(e_idx)))
                elif row["Category"] == "Communication Gap":
                    communication_gaps_idx.append((int(s_idx), int(e_idx)))

        self.timestamps = channel_df.index.values
        return channel_df.values.astype(np.float32), sorted(anomalies), sorted(communication_gaps_idx), block_intervals

    def load_challenge_channel(self, channel_id: str):
        """Load the challenge channel data."""

        import pyarrow.parquet as pq  # type: ignore

        source_folder = os.path.join(self.root, "ESA-Mission1/ESA-Mission1-challenge")
        table = pq.read_table(os.path.join(source_folder, "test.parquet"))
        df = table.to_pandas()

        telecommand_cols = [col for col in df.columns if col.startswith("telecommand_")]
        telecommand_cols = sorted(
            telecommand_cols, key=lambda col: int(col.split("_")[1])
        )

        selected_cols = [channel_id] + telecommand_cols
        channel = df[selected_cols]

        return channel.values.astype(np.float32), [], [], [(0, len(channel))]

    @property
    def sampling_period(self) -> float:
        """Return the sampling period in seconds."""
        return self.mission.resampling_rule.total_seconds()

    @property
    def in_features_size(self) -> int:
        """Return the size of the input features."""
        return self.data.shape[-1]
