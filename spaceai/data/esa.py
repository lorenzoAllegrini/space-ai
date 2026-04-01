#
import logging
import math
import os
from dataclasses import dataclass
from enum import Enum
from typing import (
    Literal,
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
        mode: Literal["prediction", "anomaly", "challenge"],
        overlapping: bool = False,
        seq_length: Optional[int] = 250,
        n_predictions: int = 1,
        train: bool = True,
        download: bool = True,
        uniform_start_end_date: bool = True,
        drop_last: bool = True,
        use_telecommands: bool = True,
    ):
        """ESABenchmark class that preprocesses and loads ESA dataset for training and
        testing.

        Args:
            root (str): The root directory of the dataset.
            mission (ESAMission): The mission type of the dataset.
            channel_id (str): The channel ID to be used.
            mode (Literal["prediction", "anomaly"]): The mode of the dataset.
            overlapping (bool): The flag that indicates whether the dataset is overlapping.
            seq_length (Optional[int]): The length of the sequence for each sample.
            train (bool): The flag that indicates whether the dataset is for training or testing.
            download (bool): The flag that indicates whether the dataset should be downloaded.
            uniform_start_end_date (bool): The flag that indicates whether the dataset should be
                resampled to have uniform start and end date.
            drop_last (bool): The flag that indicates whether the last sample should be dropped.
        """
        super().__init__(root)
        if seq_length is None or seq_length < 1:
            raise ValueError(f"Invalid window size: {seq_length}")

        if mode not in ["prediction", "anomaly", "challenge", "continual"]:
            raise ValueError(f"Invalid mode {mode}")

        self.root = root
        self.mission = mission
        self.channel_id: str = channel_id
        self._mode: Literal["prediction", "anomaly", "challenge", "continual"] = mode
        self.overlapping: bool = overlapping
        self.window_size: int = seq_length if seq_length else 250
        self.train: bool = train
        self.uniform_start_end_date: bool = uniform_start_end_date
        self.drop_last: bool = drop_last
        self.n_predictions: int = n_predictions
        self.use_telecommands: bool = use_telecommands

        if not channel_id in self.mission.all_channels:
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

        self.timestamps = None
        self.data, self.anomalies, self.communication_gaps, self.block_intervals = self.load_and_preprocess(
            channel_id
        )

    def __getitem__(self, index: int) -> Union[
        Tuple[torch.Tensor, torch.Tensor],
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
        length = self.data.shape[0] / (self.window_size + self.n_predictions)
        if self.drop_last:
            return math.floor(length)
        return math.ceil(length)

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
        # Resample using zero order hold
        if self._mode == "challenge" or self._mode == "continual":
            end_date = self.mission.end_date
            start_date = self.mission.start_date
        elif self.train:
            if end_date > self.mission.train_test_split:
                end_date = self.mission.train_test_split
        else:
            if start_date < self.mission.train_test_split:
                start_date = self.mission.train_test_split
        
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
    
            resampling_seconds = pd.Timedelta(self.mission.resampling_rule).total_seconds()
            gap_threshold_sec = max(median_dt * 20.0, resampling_seconds * 20.0, 3000.0)
            
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
            
            # Handle start of block potential NaN if grid starts before data
            if len(block_df) > 0 and pd.isna(block_resampled.iloc[0, 0]):
                block_resampled.iloc[0] = block_df.iloc[0]
                block_resampled = block_resampled.ffill()

            resampled_blocks.append(block_resampled)
            
        if len(resampled_blocks) > 0:
            final_param_df = pd.concat(resampled_blocks)

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
        channel_id: str,
    ) -> pd.DataFrame:
        """Preprocess the channel dataset by loading the raw channel dataset.

        Args:
            channel_id (str): The channel ID to preprocess.

        Returns:
            pd.DataFrame: The preprocessed channel dataset.
        """
        source_folder = os.path.join(self.root, self.mission.inner_dirpath)
        if not self.train and self._mode == "challenge":
            return self.load_challenge_channel(channel_id)

        if channel_id in self.mission.parameters:
            channel_df = pd.read_pickle(
                os.path.join(source_folder, "channels", f"{channel_id}.zip")
            )

        elif channel_id in self.mission.telecommands:
            channel_df = pd.read_pickle(
                os.path.join(source_folder, "telecommands", f"{channel_id}.zip"),
            )

        else:
            raise ValueError("channel_id not in available channels")

        channel_df, block_intervals = self._apply_resampling_rule_(
            channel_df,
            channel_df.index[0],
            channel_df.index[-1],
        )

        channel_df = channel_df.ffill().bfill().astype(np.float32)

        if self.use_telecommands:
            telecommands_csv = pd.read_csv(
                os.path.join(source_folder, "telecommands.csv")
            )
            prioritized_tcs = (
                telecommands_csv.loc[telecommands_csv["Priority"] >= 3, "Telecommand"]
                .to_numpy()
                .flatten()
            )

            telecommand_dfs = []

            for tc in prioritized_tcs:
                tc_file = os.path.join(source_folder, "telecommands", f"{tc}.zip")
                if os.path.exists(tc_file):
                    df_tc = pd.read_pickle(tc_file)
                    df_tc_bool = pd.Series(0, index=channel_df.index, name=tc)

                    channel_index = channel_df.index
                    for ts in df_tc.index:
                        pos = channel_index.searchsorted(ts, side="left")
                        if pos < len(channel_index):
                            df_tc_bool.iloc[pos] = 1
                    telecommand_dfs.append(df_tc_bool.to_frame())
                else:
                    logging.warning("Telecommand file %s not found.", tc_file)

            if telecommand_dfs:
                telecommands_df = telecommand_dfs[0]
                for df in telecommand_dfs[1:]:
                    telecommands_df = telecommands_df.join(df, how="outer")
                
                telecommands_df = telecommands_df.fillna(0)
                channel_df = channel_df.join(telecommands_df, how="left")

        if self.uniform_start_end_date:
            channel_df, block_intervals = self._apply_resampling_rule_(
                channel_df,
                self.mission.start_date,
                self.mission.end_date,
            )
        
        map_datetime_index = pd.DataFrame(
            list(range(0, len(channel_df))), index=channel_df.index, columns=["value"]
        )
        min_dt, max_dt = channel_df.index.min(), channel_df.index.max()
        labels_df = pd.read_csv(os.path.join(source_folder, "labels.csv"))

        anomaly_types_df = pd.read_csv(os.path.join(source_folder, "anomaly_types.csv"))
        labels_df = pd.merge(labels_df, anomaly_types_df, how="inner", on="ID")

        for dcol in ["StartTime", "EndTime"]:
            labels_df[dcol] = pd.to_datetime(labels_df[dcol]).dt.tz_localize(None)
        labels_df = labels_df.loc[labels_df["Channel"] == channel_id]

        anomalies = []
        communication_gaps = []

        for _, label_row in labels_df.iterrows():
            start_time = label_row["StartTime"].floor(freq=self.mission.resampling_rule)
            end_time = label_row["EndTime"].ceil(freq=self.mission.resampling_rule)
            if end_time < min_dt:
                continue
            if start_time > max_dt:
                continue

            map_datetime_index_range = map_datetime_index[
                np.logical_and(
                    start_time <= map_datetime_index.index,
                    map_datetime_index.index <= end_time,
                )
            ]
            if len(map_datetime_index_range) > 0:
                start_idx = map_datetime_index_range.iloc[0]["value"]
                end_idx = map_datetime_index_range.iloc[-1]["value"]
                if (
                    label_row["Category"] == "Anomaly"
                    or label_row["Category"] == "Rare Event"
                ):
                    anomalies.append((start_idx, end_idx))
                elif label_row["Category"] == "Communication Gap":
                    communication_gaps.append((start_idx, end_idx))
        
        if isinstance(channel_df.index, pd.DatetimeIndex):
            self.timestamps = channel_df.index.values
        else:
            self.timestamps = None

        channel = channel_df.values.astype(np.float32)
        anomalies = sorted(anomalies, key=lambda x: x[0])

        communication_gaps = sorted(communication_gaps, key=lambda x: x[0])

        return channel, anomalies, communication_gaps, block_intervals

    def load_challenge_channel(self, channel_id: str):
        """Load the challenge channel data."""

        import pyarrow.parquet as pq  # type: ignore

        source_folder = os.path.join(self.root, self.mission.dirname, "ESA-Mission1-challenge")
        table = pq.read_table(os.path.join(source_folder, "test.parquet"))
        df = table.to_pandas()

        # Seleziona le colonne che iniziano con "telecommand_" e ordinali per numero crescente
        telecommand_cols = [col for col in df.columns if col.startswith("telecommand_")]
        telecommand_cols = sorted(
            telecommand_cols, key=lambda col: int(col.split("_")[1])
        )

        selected_cols = [channel_id] + telecommand_cols
        channel = df[selected_cols]

        for id_col in ["UT", "id"]:
            if id_col in df.columns:
                self.timestamps = df[id_col].values
                self.id_column_name = id_col
                break

        return channel.values.astype(np.float32), [], [], [(0, len(channel))]

    @property
    def sampling_period(self) -> float:
        """Return the sampling period in seconds."""
        return self.mission.resampling_rule.total_seconds()

    @property
    def in_features_size(self) -> int:
        """Return the size of the input features."""
        return self.data.shape[-1]
