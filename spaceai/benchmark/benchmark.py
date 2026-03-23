"""Base benchmark class for anomaly detection benchmarks."""

from __future__ import annotations

import bisect
import json
import logging
import os
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Tuple,
)

import more_itertools as mit
import numpy as np
import pandas as pd  # type: ignore
import zmq
import torch
from torch.utils.data import (
    DataLoader,
    Subset,
)
from tqdm import tqdm  # type: ignore

from spaceai.data.utils import seq_collate_fn

from .callbacks import CallbackHandler
from spaceai.preprocessing.ts_splitter import TimeSeriesSplitter

if TYPE_CHECKING:
    from spaceai.models.predictors import SequenceModel
    from spaceai.models.anomaly import AnomalyDetector
    from .callbacks import Callback
import zmq
import json
import time

import threading
import json
import torch
from spaceai.preprocessing import TimeSeriesSplitter
from spaceai.models.anomaly_classifier import AnomalyClassifier

class Benchmark:
    """Base class for benchmark runners."""

    def __init__(
        self,
        run_id: str,
        exp_dir: str,
        data_root: str = "datasets",
    ):
        """Initializes a new benchmark run.

        Args:
            run_id (str): A unique identifier for this run.
            exp_dir (str): The directory where the results of this run are stored.
            seq_length (int): The length of the sequences used for training and testing.
            data_root (str): The root directory of the dataset.
        """
        self.run_id = run_id
        self.exp_dir = exp_dir
        self.data_root: str = data_root
        self.all_results: List[Dict[str, Any]] = []
        
        self.trained_classifiers: Dict[str, Any] = {}
        self.channel_predictions: Dict[str, Any] = {}
        self.global_results: Dict[str, Any] = {"channel_id": "GLOBAL_EVENT_LEVEL"}
        self.event_labels_global: List[Any] = []
        self.predicted_events_global: List[Any] = []

    def set_classifier(self, channel_id: str, classifier: Any):
        """Manually inject a pre-trained classifier into the benchmark state.
        
        Args:
            channel_id (str): The ID of the channel.
            classifier (Any): The pre-trained model/classifier instance.
        """
        self.trained_classifiers[channel_id] = classifier

    @staticmethod
    def merge_intervals(
        intervals: List[Tuple[int, int]]
    ) -> List[Tuple[int, int]]:
        """Merge overlapping intervals."""
        if not intervals:
            return []

        events = []
        for interval in intervals: 
            events.extend([(interval[0], 1), (interval[1], -1)]) # +1 for start, -1 for end of an interval
        
        events.sort(key=lambda e: (e[0], -e[1]))
        current_depth = 0
        curr_start = 0

        res = []
        for event in events:
            if current_depth == 0 and event[1] == 1:
                curr_start = event[0]

            current_depth += event[1]

            if current_depth == 0:
                res.append((curr_start, event[0]))

        return res
        
    def compute_global_event_metrics(
        self,
        channels: Optional[List[str]] = None,
        time_aware: bool = True,
    ) -> Dict[str, Any]:
        """Compute aggregated event-level metrics from internally accumulated state.
        
        This should be called AFTER all channels have been trained and tested
        (via train_channel_* and test_channel_* methods), which populate
        self.event_labels_global and self.predicted_events_global.
        
        Args:
            channels: List of channel IDs (used for temporal normalization).
            time_aware: Whether to normalize intervals to a common time axis.
            
        Returns:
            Dict[str, Any]: Global event-level metrics.
        """
        if channels is None:
            channels = self.get_default_channels()

        for metric in [m for m in self.global_results.keys() if m.endswith("cpu")]:
            self.global_results[metric] /= max(len(channels), 1)

        event_labels = Benchmark.merge_intervals(self.event_labels_global)
        predicted_events = Benchmark.merge_intervals(self.predicted_events_global)
        
        adtqc_metrics = Benchmark.adtqc_score(event_labels, predicted_events)
        self.global_results.update(adtqc_metrics)
        
        min_start_time, min_period = self.get_global_temporal_params(channels)

        if time_aware and min_start_time is not None and min_period is not None:
            event_labels = [
                (
                    int((pd.Timestamp(s) - min_start_time).total_seconds() / min_period),
                    int((pd.Timestamp(e) - min_start_time).total_seconds() / min_period)
                ) for s, e in event_labels
            ]
            predicted_events = [
                (
                    int((pd.Timestamp(s) - min_start_time).total_seconds() / min_period),
                    int((pd.Timestamp(e) - min_start_time).total_seconds() / min_period)
                ) for s, e in predicted_events
            ]

        self.global_results.update(
            Benchmark.compute_metrics(event_labels, predicted_events)
        )
        
        logging.info("Global Event-Level Results: %s", self.global_results)

        self.all_results.append(self.global_results)
        pd.DataFrame.from_records(self.all_results).to_csv(
            os.path.join(self.run_dir, "results.csv"), index=False
        )

        return self.global_results

    @staticmethod
    def _evaluate_predictions(classifier: AnomalyClassifier, data: Any, true_intervals: List[Any], y_pred: np.ndarray) -> Dict[str, float]:
        """Process predictions into intervals and compute metrics vs ground truth."""
        pred_intervals = Benchmark.process_pred_anomalies(y_pred, 0)
        
        true_intervals_ts = classifier.map_to_timestamps(data, true_intervals)
        pred_intervals_ts = classifier.map_to_timestamps(data, pred_intervals)
        
        return Benchmark.compute_metrics(
            true_intervals, pred_intervals, total_length=len(y_pred),
            true_anomalies_ts=true_intervals_ts, pred_anomalies_ts=pred_intervals_ts
        )

    def fit_channel(
        self,
        channel_id: str,
        classifier: AnomalyClassifier,
    ) -> Dict[str, Any]:
        """Trains the anomaly classifier for a given channel and saves it to state."""
        
        train_channel = self.load_channel(channel_id, mode="train")
        logging.info("Fitting the anomaly classifier for channel %s...", channel_id)
        metrics = classifier.fit(train_channel)

        self.trained_classifiers[channel_id] = classifier

        os.makedirs(self.run_dir, exist_ok=True)
        classifier_path = os.path.join(self.run_dir, f"classifier-{channel_id}.pt")
        classifier.save(classifier_path)
    
        self._update_global_state(channel_id, metrics)

        return classifier, metrics

    def _finalize_channel_results(
        self,
        channel_id: str,
        classifier: AnomalyClassifier,
        test_dataset: Any,
        y_pred: np.ndarray,
        extra_metrics: Optional[Dict[str, Any]] = None,
        pred_buffer: int = 0,
    ) -> Tuple[Dict[str, Any], List[Any], List[Any]]:
        """Shared logic for computing metrics, saving results, and updating global state.

        Used by both ``test_channel`` and ``test_continual``.
        """
        results: Dict[str, Any] = {"channel_id": channel_id}
        if extra_metrics:
            results.update(extra_metrics)

        pred_anomalies = Benchmark.process_pred_anomalies(y_pred, pred_buffer)
        test_anomalies = classifier.prepare_labels(test_dataset)

        true_anomaly_intervals_ts = classifier.map_to_timestamps(test_dataset, test_anomalies)
        pred_intervals_ts = classifier.map_to_timestamps(test_dataset, pred_anomalies)

        all_metrics = Benchmark.compute_metrics(
            test_anomalies, pred_anomalies, total_length=len(y_pred),
            true_anomalies_ts=true_anomaly_intervals_ts, pred_anomalies_ts=pred_intervals_ts
        )
        results.update(all_metrics)

        logging.info("Results for channel %s: %s", channel_id, results)

        self.channel_predictions[channel_id] = {
            "y_pred": y_pred.tolist(),
            "pred_anomalies": pred_anomalies,
            "true_anomalies": test_anomalies,
        }

        self.all_results.append(results)
        os.makedirs(self.run_dir, exist_ok=True)
        pd.DataFrame.from_records(self.all_results).to_csv(
            os.path.join(self.run_dir, "results.csv"), index=False
        )

        try:
            y_true_pointwise = np.zeros(len(y_pred), dtype=int)
            for start, end in test_anomalies:
                y_true_pointwise[max(0, int(start)):min(len(y_pred), int(end) + 1)] = 1
            
            plot_path = os.path.join(self.run_dir, f"{channel_id}_degradation.png")
            window_size = max(min(len(y_pred) // 10, 2000), 100)
            Benchmark.save_degradation_plot(channel_id, y_true_pointwise, y_pred, plot_path, window_size)
        except Exception as e:
            logging.warning("Could not save degradation plot for %s: %s", channel_id, e)

        if true_anomaly_intervals_ts or pred_intervals_ts:
            self._update_global_state(channel_id, results, true_anomaly_intervals_ts, pred_intervals_ts)

        return results, true_anomaly_intervals_ts, pred_intervals_ts

    def test_channel(
        self,
        channel_id: str,
        pred_buffer: int = 0,
        classifier: Optional[AnomalyClassifier] = None,
    ) -> Tuple[Dict[str, Any], List[Any], List[Any]]:
        """Tests the fitted anomaly classifier for a given channel using internal state."""

        if classifier is None:
            if channel_id not in self.trained_classifiers:
                logging.warning("Classifier for channel %s not found in state.", channel_id)
                return {"channel_id": channel_id}, [], []
            classifier = self.trained_classifiers[channel_id]
        
        test_channel = self.load_channel(channel_id, mode="test")

        logging.info("Predicting the test data for channel %s...", channel_id)
        y_pred, metrics = classifier.predict(test_channel)

        return self._finalize_channel_results(
            channel_id, classifier, test_channel, y_pred,
            extra_metrics=metrics, pred_buffer=pred_buffer,
        )

    def test_continual(
        self,
        channel_id: str,
        classifier: Optional[AnomalyClassifier] = None,
        experience_size: Union[int, str, pd.Timedelta] = 500,
    ) -> Dict[str, Any]:
        """Simulate continual real-time streaming telemetry and compute
        the same metrics as ``test_channel`` for comparability.

        The AnomalyClassifier needs to be fitted on the training set before
        executing this method.
        """

        if classifier is None:
            if channel_id not in self.trained_classifiers:
                logging.warning("Classifier for channel %s not found in state.", channel_id)
                return {"channel_id": channel_id}
            classifier = self.trained_classifiers[channel_id]
            
        test_dataset = self.load_channel(channel_id, mode="test", overlapping_train=False)
        
        experience_splitter = TimeSeriesSplitter(window_size=experience_size, step_size=experience_size)
        splitted = experience_splitter.segment_dataset(test_dataset, mode="experience")

        logging.info("Streaming dataset for channel %s experience by experience...", channel_id)

        experience_log = {}
        all_predictions = []
        all_point_labels = []
        
        for i, (data, point_labels, (start_idx, end_idx)) in enumerate(zip(splitted.segments, splitted.labels, splitted.segment_indices)):
            logging.info("experience %d/%d", i+1, len(splitted.segments))
            
            if hasattr(classifier, "step"):
                experience_predictions, metrics = classifier.step(data, point_labels)
            else:
                experience_predictions, test_metrics = classifier.predict(data)
                train_metrics = classifier.fit(data)
                metrics = {**test_metrics, **train_metrics}

            all_predictions.extend(experience_predictions)
            all_point_labels.extend(point_labels)
            
            # Per-experience metrics
            true_forward_intervals = classifier.prepare_labels(data)
            exp_metrics = self._evaluate_predictions(classifier, data, true_forward_intervals, experience_predictions)
            exp_metrics.update({k: v for k, v in metrics.items() if k not in exp_metrics})
            experience_log[f"experience_{i}"] = exp_metrics
            
            # Incremental save
            os.makedirs(self.run_dir, exist_ok=True)
            with open(os.path.join(self.run_dir, f"{channel_id}_stream_history.json"), "w") as f:
                json.dump(experience_log, f, indent=2, default=str)
            
            np.savez_compressed(
                os.path.join(self.run_dir, f"{channel_id}_streaming_results.npz"),
                predictions=np.array(all_predictions),
                labels=np.array(all_point_labels),
            )

        logging.info("Streaming for channel %s completed.", channel_id)

        # Global metrics (reuses the same logic as test_channel)
        y_pred_all = np.array(all_predictions)
        results, _, _ = self._finalize_channel_results(
            channel_id, classifier, test_dataset, y_pred_all,
        )

        # Save experience log
        with open(os.path.join(self.run_dir, f"{channel_id}_stream_history.json"), "w") as f:
            json.dump(experience_log, f, indent=2, default=str)

        return results


    def _update_global_state(self, channel_id, metrics, true_intervals=None, pred_intervals=None):
        """Helper to accumulate global metrics and save interval logs."""
        for k, v in metrics.items():
            if k.endswith(("time", "cpu")):
                self.global_results[k] = self.global_results.get(k, 0) + v
                
        if true_intervals is not None and pred_intervals is not None:
            self.event_labels_global.extend(true_intervals)
            self.predicted_events_global.extend(pred_intervals)

            with open(os.path.join(self.run_dir, f"{channel_id}_intervals.json"), "w") as f:
                json.dump({
                    "pred_intervals": [[str(s), str(e)] for s, e in pred_intervals],
                    "true_intervals": [[str(s), str(e)] for s, e in true_intervals],
                }, f, indent=2)

    @staticmethod
    def save_degradation_plot(
        channel_id: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        save_path: str,
        window_size: int = 2000,
    ):
        """Generates and saves a degradation plot (cumulative errors and rolling FPR)."""
        import matplotlib.pyplot as plt
        
        false_positives = (y_pred == 1) & (y_true == 0)
        false_negatives = (y_pred == 0) & (y_true == 1)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        ax1.plot(np.cumsum(false_positives), label='Cumulative False Positives', color='red', linewidth=2)
        ax1.plot(np.cumsum(false_negatives), label='Cumulative False Negatives', color='orange', linewidth=2)
        ax1.set_title(f"[{channel_id}] Cumulative Errors over Time (Degradation Indicator)")
        ax1.set_ylabel("Total Error Count")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        fp_rolling_rate = pd.Series(false_positives).rolling(window=window_size).mean() * 100
        
        ax2.plot(fp_rolling_rate, label=f'Rolling FPR (window {window_size})', color='purple')
        ax2.set_title(f"[{channel_id}] Rolling False Positive Rate")
        ax2.set_xlabel("Time step")
        ax2.set_ylabel("FPR (%)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close(fig)

    @staticmethod
    def process_pred_anomalies(
        y_pred: np.ndarray, pred_buffer: int
    ) -> List[List[int]]:
        """Process predicted anomalies by grouping consecutive indices and applying buffer."""
        y_pred = np.atleast_1d(y_pred)
        pred_anomalies = np.where(y_pred == 1)[0]

        if len(pred_anomalies) > 0:

            groups = [list(group) for group in mit.consecutive_groups(pred_anomalies)]
            buffered_intervals = [
                [max(0, int(group[0] - pred_buffer)), min(len(y_pred) - 1, int(group[-1] + pred_buffer))]
                for group in groups
            ]

            merged_intervals: List[List[int]] = []
            for interval in sorted(buffered_intervals, key=lambda x: x[0]):
                if not merged_intervals or interval[0] > merged_intervals[-1][1]:
                    merged_intervals.append(interval)
                else:
                    merged_intervals[-1][1] = max(merged_intervals[-1][1], interval[1])

            return merged_intervals
        else:
            return []
            

    @staticmethod
    def compute_metrics(
        true_anomalies: List[Tuple[int, int]],
        pred_anomalies: List[Tuple[int, int]],
        total_length: Optional[int] = None,
        true_anomalies_ts: Optional[List[Tuple[Any, Any]]] = None,
        pred_anomalies_ts: Optional[List[Tuple[Any, Any]]] = None,
    ) -> Dict[str, Any]:
        """Compute all range-level classification metrics including corrected variants.

        Computes base metrics (TP, FP, FN, precision, recall, F1) and
        TNR-corrected metrics (corrected precision, corrected F0.5, corrected F1).

        Args:
            true_anomalies (List[Tuple[int, int]]): the true anomaly intervals.
            pred_anomalies (List[Tuple[int, int]]): the predicted anomaly intervals.
            total_length (Optional[int]): the total length of the sequence. If None,
                it is inferred from the maximum endpoint of the intervals.

        Returns:
            Dict[str, Any]: dictionary with all computed metrics.
        """
        results = {
            "n_anomalies": len(true_anomalies),
            "n_detected": len(pred_anomalies),
            "true_positives": 0,
            "false_positives": 0,
            "false_negatives": 0,
        }

        # --- Base classification metrics ---
        matched_true_seqs = []
        true_indices_grouped = [list(range(e[0], e[1] + 1)) for e in true_anomalies]
        true_indices_flat = set(i for group in true_indices_grouped for i in group)
        for e_seq in pred_anomalies:
            i_anom_predicted = set(range(e_seq[0], e_seq[1] + 1))

            matched_indices = list(i_anom_predicted & true_indices_flat)
            valid = len(matched_indices) > 0

            if valid:
                true_seq_index = [
                    i
                    for i in range(len(true_indices_grouped))
                    if len(
                        np.intersect1d(list(i_anom_predicted), true_indices_grouped[i])
                    )
                    > 0
                ]

                if true_seq_index[0] not in matched_true_seqs:
                    matched_true_seqs.append(true_seq_index[0])
                    results["true_positives"] += 1
            else:
                results["false_positives"] += 1

        results["false_negatives"] = len(
            np.delete(true_anomalies, matched_true_seqs, axis=0)
        )

        tpfp = results["true_positives"] + results["false_positives"]
        results["precision"] = results["true_positives"] / tpfp if tpfp > 0 else 1
        tpfn = results["true_positives"] + results["false_negatives"]
        results["recall"] = results["true_positives"] / tpfn if tpfn > 0 else 1
        results["f1"] = (
            (
                2
                * (results["precision"] * results["recall"])
                / (results["precision"] + results["recall"])
            )
            if results["precision"] + results["recall"] > 0
            else 0
        )

        # --- TNR-corrected metrics ---
        if total_length is None:
            total_length = 0
            if true_anomalies:
                total_length = max(total_length, true_anomalies[-1][1])
            if pred_anomalies:
                total_length = max(total_length, pred_anomalies[-1][1])

        indices_pred_grouped = [list(range(e[0], e[1] + 1)) for e in pred_anomalies]
        indices_pred_flat = set(i for group in indices_pred_grouped for i in group)
        indices_all_flat = true_indices_flat.union(indices_pred_flat)
        n_e = total_length - len(true_indices_flat)
        tn_e = total_length - len(indices_all_flat)

        results["tnr"] = tn_e / n_e if n_e > 0 else 1
        results["test_length"] = total_length
        results["test_negatives"] = n_e
        results["detected_negatives"] = tn_e
        results["precision_corrected"] = results["precision"] * results["tnr"]
        results["corrected_f0.5"] = (
            (
                (1 + 0.5**2)
                * (results["precision_corrected"] * results["recall"])
                / (0.5**2 * results["precision_corrected"] + results["recall"])
            )
            if results["precision_corrected"] + results["recall"] > 0
            else 0
        )
        results["corrected_f1"] = (
            (
                2
                * (results["precision_corrected"] * results["recall"])
                / (results["precision_corrected"] + results["recall"])
            )
            if results["precision_corrected"] + results["recall"] > 0
            else 0
        )
        
        if true_anomalies_ts is not None and pred_anomalies_ts is not None:
            results.update(Benchmark.adtqc_score(true_anomalies_ts, pred_anomalies_ts))
            
        return results

    @staticmethod
    def timing_curve(x, a, b, exponent):
        assert a >= pd.Timedelta(0)
        assert b >= pd.Timedelta(0)
        if (a == pd.Timedelta(0) or b == pd.Timedelta(0)) and x == pd.Timedelta(0):
            return 1
        if x <= -a or x >= b:
            return 0
        if -a < x <= pd.Timedelta(0):
            return ((x + a)/a)**exponent
        if pd.Timedelta(0) < x < b:
            denom_part = x/(b - x)
            return 1. / (1. + denom_part**exponent)
    
    @staticmethod
    def adtqc_score(
        label_intervals: List[Tuple],
        pred_intervals: List[Tuple],
        exponent: int = 2,
        segment_duration: pd.Timedelta = pd.Timedelta(0),
    ) -> dict:
        """Compute ADTQC (Anomaly Detection Time-Quality Curve) score.

        Args:
            label_intervals: Sorted, merged list of (start, end) ground truth intervals (timestamps).
            pred_intervals: Sorted, merged list of (start, end) predicted intervals (timestamps).
            exponent: Exponent for the timing curve function.

        Returns:
            Dictionary with adtqc_n_before, adtqc_n_after, adtqc_after_rate, and adtqc_score.
        """
        if not label_intervals:
            return {"adtqc_n_before": 0, "adtqc_n_after": 0,
                    "adtqc_after_rate": np.nan, "adtqc_score": np.nan}

        pred_starts = [pd.Timestamp(s) for s, _ in pred_intervals]

        before_tps = []
        after_tps = []
        curve_scores = []

        for i, (gt_start, gt_end) in enumerate(label_intervals):
            gt_start = pd.Timestamp(gt_start)
            gt_end = pd.Timestamp(gt_end)
            anomaly_length = gt_end - gt_start

            # Alpha: min(anomaly_length, distance to previous anomaly start)
            if i > 0:
                prev_start = pd.Timestamp(label_intervals[i - 1][0])
                alpha = min(anomaly_length, gt_start - prev_start)
            else:
                alpha = anomaly_length

            window_start = gt_start - alpha
            idx = bisect.bisect_left(pred_starts, window_start)

            first_detection = None
            search_start = max(0, idx - 1)
            for j in range(search_start, len(pred_intervals)):
                p_start = pd.Timestamp(pred_intervals[j][0])
                p_end = pd.Timestamp(pred_intervals[j][1])

                if p_start >= gt_end:
                    break

                if p_end >= window_start and p_start < gt_end:
                    first_detection = p_start
                    break

            if first_detection is None:
                continue

            latency = (first_detection + segment_duration) - gt_start
            metric_value = Benchmark.timing_curve(latency, alpha, anomaly_length, exponent)
            curve_scores.append(metric_value)

            if latency < pd.Timedelta(0):
                before_tps.append(metric_value)
            else:
                after_tps.append(metric_value)

        curve_scores = np.array(curve_scores)

        return {
            "adtqc_n_before": len(before_tps),
            "adtqc_n_after": len(after_tps),
            "adtqc_after_rate": len(after_tps) / len(curve_scores) if len(curve_scores) > 0 else np.nan,
            "adtqc_score": np.mean(curve_scores) if len(curve_scores) > 0 else np.nan,
        }


    def get_default_channels(self) -> List[str]:
        """Get the default list of channels for the benchmark.

        Returns:
            List[str]: list of channel IDs.
        """
        raise NotImplementedError("Subclasses must implement get_default_channels.")

    def get_global_temporal_params(self, channels: List[str]) -> Tuple[Optional[pd.Timestamp], Optional[float]]:
        """Get global start time and period for event-level aggregation.

        Args:
            channels (List[str]): List of channels.

        Returns:
            Tuple[Optional[pd.Timestamp], Optional[float]]: global start time and period.
        """
        raise NotImplementedError("Subclasses must implement get_global_temporal_params.")

    @property
    def run_dir(self) -> str:
        """Returns the directory where the results of this run are stored."""
        return os.path.join(self.exp_dir, self.run_id)
