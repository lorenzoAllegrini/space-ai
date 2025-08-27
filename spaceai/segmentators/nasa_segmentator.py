import os 
import sys 
import pandas as pd
import numpy as np
import ast 
import math
import random 
import itertools
from scipy.stats import kurtosis, skew
from spaceai.data.nasa import NASA
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


class NasaDatasetSegmentator:
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
    
    available_poolings = {
        "max": np.max,
        "min": np.min,
        "mean": np.mean,
        "std": np.std,
    }
    def __init__(
        self,
        transformations: Optional[List[str]] = ["mean"],
        segment_duration: int = 100,
        step_duration: int = 1,
        telecommands: bool = False,
        extract_features: bool = True,
        run_id: str = "esa_segments",
        exp_dir: str = "experiments",
        poolings: Optional[List[str]] = None,
        pooling_config: Optional[Dict[str, Dict[str, str]]] = None,
        pooling_segment_len: int = 2,
        pooling_segment_stride: int = 20,
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
        self.poolings = poolings or []        # [] ⇒ nessun pooling
        for p in self.poolings:
            if p not in self.available_poolings:
                raise RuntimeError(f"Pooling {p} not available")

        self.pooling_config = pooling_config or {}
        self.pooling_segment_len = pooling_segment_len
        self.pooling_segment_stride = pooling_segment_stride

    def segment(self, nasa_channel: NASA) :
        if nasa_channel.anomalies is None:
            nasa_channel.anomalies = []
        segments = self.create_segments_from_channel(
            nasa_channel.data, np.array(nasa_channel.anomalies)
            )
        anomalies = self.get_event_intervals(segments=segments, label=1)
        if self.extract_features:
            base_cols = ["event"]
            base_cols += self.transformations.copy()
            if self.telecommands:
                base_cols += [f"telecommand_{i}" for i in range(1, nasa_channel.data.shape[1])]

            # -- pooling -----------------------------------------------------
            if self.poolings:
                segments, pooled_cols = self.pooling_segmentation(segments, base_cols)
                columns = pooled_cols
            else:
                columns = base_cols

            df = pd.DataFrame(segments, columns=columns)
            df = df.drop(columns=df.filter(like="event").columns)
            return df, anomalies
        
        segments = [s[1:] for s in segments]
        # se non estraggo feature, segments è ancora list‑of‑values
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
                segment.extend(values)
            segments.append(segment)
            index += self.step_duration
        return segments
    
    def pooling_segmentation(
        self,
        segments: List[List[float]],
        columns: List[str],
    ) -> Tuple[List[List[float]], List[str]]:
        """
        Rolling‑window pooling identico al Segmentatore ESA.
        """
        data = np.array(segments)                  # shape (N, F)
        N, _ = data.shape
        w, s = self.pooling_segment_len, self.pooling_segment_stride

        # ---- header ---------------------------
        pooled_cols = []
        for feat in columns:
            if feat in self.pooling_config:
                for _, out_name in self.pooling_config[feat].items():
                    pooled_cols.append(out_name)
            else:
                for p in self.poolings:
                    pooled_cols.append(f"{p}_{feat}")

        # ---- sliding window pooling -----------
        pooled_rows = []
        for start in range(0, N - w + 1, s):
            window = data[start : start + w]       # (w, F)
            if window[:, 0].min() == -1:           # salta finestre con event == -1
                continue

            row = []
            for j, feat in enumerate(columns):
                if feat in self.pooling_config:
                    for p in self.pooling_config[feat]:
                        func = self.available_poolings[p]
                        row.append(func(window[:, j], axis=0))
                else:
                    for p in self.poolings:
                        func = self.available_poolings[p]
                        row.append(func(window[:, j], axis=0))
            pooled_rows.append(row)

        return pooled_rows, pooled_cols
    
    def get_event_intervals(self, segments: list, label:int) -> list:
        labels = np.array([int(seg[0]) for seg in segments])
        indices = np.where(labels == label)[0]
        if indices.size == 0:
            return []
        groups = [list(group) for group in mit.consecutive_groups(indices)]

        intervals = [[group[0], group[-1]] for group in groups]
        return intervals