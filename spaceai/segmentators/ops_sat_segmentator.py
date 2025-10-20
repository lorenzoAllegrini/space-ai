import os 
import sys 
import pandas as pd
import numpy as np
import ast 
import math
import random 
import itertools
from scipy.stats import kurtosis, skew
from spaceai.data.ops_sat import OPSSAT
import more_itertools as mit 

from spaceai.segmentators.functions import *
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Tuple,
)


class OPSSATDatasetSegmentator:
    available_transformations = {
        # --- trasformazioni già presenti ---
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
        transformations: Optional[List[str]] = ["mean"],
        segment_duration: int = 100,
        step_duration: int = 1,
        telecommands: bool = True,
        extract_features: bool = True,
        run_id: str = "esa_segments",
        exp_dir: str = "experiments",
    ) -> None:
        
        for transformation in transformations:
            if transformation not in self.available_transformations:
                raise RuntimeError(f"Transformation {transformation} not available")
        self.transformations = transformations
        self.segment_duration = segment_duration
        self.step_duration = step_duration
        self.telecommands = telecommands
        self.run_id = run_id
        self.exp_dir = exp_dir
        self.extract_features = extract_features

    def segment(self, nasa_channel: OPSSAT) :
        if nasa_channel.anomalies is None:
            nasa_channel.anomalies = []
        segments = self.create_segments_from_channel(
            nasa_channel.data, np.array(nasa_channel.anomalies)
            )
        anomalies = self.get_event_intervals(segments=segments, label=1)

        segments = [s[1:] for s in segments]

        all_columns = self.transformations.copy()

        if self.extract_features:
            segments = pd.DataFrame(segments, columns=all_columns)

        return segments, anomalies
        
    def create_segments_from_channel(
        self,
        data: np.ndarray,
        anomalies
    ) -> List[List[float]]:
        if anomalies is None:
            anomalies = []
        segments = []             
        index = 0
        anomaly_index = 0
        while index + self.segment_duration < len(data):  
            while  len(anomalies) != 0 and anomaly_index < anomalies.shape[0] and index > anomalies[anomaly_index][1]:
                anomaly_index += 1 
            event = 0
            if len(anomalies) != 0 and anomaly_index < anomalies.shape[0]:
                if max(index, anomalies[anomaly_index][0]) < min(index + self.segment_duration, anomalies[anomaly_index][1]):
                    event=1
              

            values = data[index:(index + self.segment_duration), 0]
            segment = [event]
            if self.extract_features:
                segment.extend([
                    self.available_transformations[transformation](values)
                    for transformation in self.transformations
                ])
                if self.telecommands:
                    for telecommand_idx in range(1, data.shape[1]):
                        segment.append(float(np.sum(data[index:(index + self.segment_duration), telecommand_idx])))
            else:  
                segment = [event] + values.tolist()
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