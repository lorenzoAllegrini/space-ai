import os 
import sys 
import pandas as pd
import numpy as np
import ast 
import math
import random 
import itertools
from sktime.transformations.panel.rocket import Rocket
from scipy.stats import kurtosis, skew

from spaceai.segmentators.functions import (
    spectral_energy,
    autoregressive_deviation,
    moving_average_prediction_error,
    stft_spectral_std,
    calculate_slope,
    spearman_correlation,
    diff_peaks,
    diff_var,
)
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Tuple,
)

import os 
import sys 
import pandas as pd
import numpy as np
import ast 
import math
import random 
import itertools
from sktime.transformations.panel.rocket import Rocket
from spaceai.data.esa import ESA
from scipy.stats import kurtosis, skew

from spaceai.segmentators.functions import *

from spaceai.segmentators.functions import (
    spectral_energy,
    autoregressive_deviation,
    moving_average_prediction_error,
    stft_spectral_std,
    calculate_slope,
    spearman_correlation,
    diff_peaks,
    diff_var,
)
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Tuple,
)
import more_itertools as mit 

class EsaDatasetSegmentator:

    available_transformations = {
        "se":  spectral_energy,
        "ar":  autoregressive_deviation,
        "ma":  moving_average_prediction_error,
        "stft": stft_spectral_std,
        "slope": calculate_slope,
        "sp_correlation": spearman_correlation,
        "mean": np.mean,
        "var":  np.var,
        "std":  np.std,
        "kurtosis": kurtosis,
        "skew":     skew,
        "diff_peaks": diff_peaks,
        "diff_var":  diff_var,
        "median": np.median,
        "n_peaks":            number_of_peaks_finding,
        "smooth10_n_peaks":   smooth10_n_peaks,
        "smooth20_n_peaks":   smooth20_n_peaks,
        "diff2_peaks":        diff2_peaks,
        "diff2_var":          diff2_var,
    }

    def __init__(
        self,
        transformations: List[str],
        segment_duration: int = 100,
        step_duration: int = 1,
        save_csv: bool = True,
        telecommands: bool = False,
        extract_features: bool = True,
        exp_dir: str = "experiments",
        segments_id:str = "esa_segments"
    ) -> None:
        
        for transformation in transformations:
            if transformation not in self.available_transformations:
                raise RuntimeError(f"Transformation {transformation} not available")
        self.transformations = transformations
        self.segment_duration = segment_duration
        self.step_duration = step_duration
        self.save_csv = save_csv
        self.telecommands = telecommands
        self.exp_dir = exp_dir
        self.extract_features = extract_features
        self.segments_id = segments_id


    def segment(self, esa_channel: ESA) :

        output_dir = os.path.join(self.exp_dir, self.segments_id)
        train_file_name = f"{esa_channel.channel_id}_segments_train_.csv"
        test_file_name = f"{esa_channel.channel_id}_segments_test_.csv"
        output_file = train_file_name if esa_channel.train else test_file_name
        csv_path = os.path.join(output_dir, output_file)

        if os.path.exists(csv_path) and self.extract_features:
            df = pd.read_csv(csv_path)
            segments = df.values.tolist()
            anomalies = self.get_event_intervals(segments=segments, label=1)

        else:
            segments = self.create_segments_from_channel(
                esa_channel.data, np.array(esa_channel.anomalies)
                )
            anomalies = self.get_event_intervals(segments=segments, label=1)

        if self.extract_features: 
            base_columns = self.transformations.copy()
            if self.telecommands:
                base_columns.extend([f"telecommand_{i}" for i in range(1, esa_channel.data.shape[1])])
            columns = ["event"] + base_columns
            df = pd.DataFrame(segments, columns=columns)
            if self.save_csv:
                os.makedirs(output_dir, exist_ok=True)
                df.to_csv(csv_path, index=False)
            df = df.drop(columns=df.filter(like="event").columns)
        else:
            segments = [segment[1:] for segment in segments]
            return segments, anomalies

        return df, anomalies

        
    def create_segments_from_channel(
        self,
        data: np.ndarray,
        anomaly_indices: np.ndarray
    ) -> List[List[float]]:
        segments = []             
        index = 0
        anomaly_index = 0
        while index + self.segment_duration < len(data):     
            seg_start = index
            seg_end = index + self.segment_duration
            while anomaly_index < anomaly_indices.shape[0] and seg_start > anomaly_indices[anomaly_index][1]:
                anomaly_index += 1

            # Check for intersection with anomaly
            intersects_anomaly = 0
            if anomaly_index < anomaly_indices.shape[0]:
                if max(seg_start, anomaly_indices[anomaly_index][0]) < min(seg_end, anomaly_indices[anomaly_index][1]):
                    intersects_anomaly = 1
            values = data[index:(index + self.segment_duration), 0]
            segment = [intersects_anomaly]
            if self.extract_features:
                segment.extend([
                    self.available_transformations[transformation](values)
                    for transformation in self.transformations
                ])
                if self.telecommands:
                    for telecommand_idx in range(1, data.shape[1]):
                        segment.append(float(np.sum(data[index:(index + self.segment_duration), telecommand_idx])))
            else:
                segment.extend(values)
            segments.append(segment)
            index += self.step_duration
        return segments

        
    def get_event_intervals(self, segments: list, label:int) -> list:
        labels = np.array([int(seg[0]) for seg in segments])
        indices = np.where(labels == label)[0]
        if indices.size == 0:
            return []
        groups = [list(group) for group in mit.consecutive_groups(indices)]

        intervals = [[group[0], group[-1]] for group in groups]
        return intervals