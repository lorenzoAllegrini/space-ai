"""Base benchmark class for anomaly detection benchmarks."""
from __future__ import annotations

import bisect
import json
import logging
import os
import threading
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import more_itertools as mit
import numpy as np
import pandas as pd  # type: ignore
import torch
import zmq
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm  # type: ignore

from spaceai.data.utils import seq_collate_fn
from spaceai.models.anomaly_classifier import AnomalyClassifier
from spaceai.preprocessing import TimeSeriesSplitter
from .callbacks import CallbackHandler

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from spaceai.models.anomaly import AnomalyDetector
    from spaceai.models.predictors import SequenceModel
    from .callbacks import Callback


class Benchmark:
    """Base class for benchmark runners."""

    def __init__(
        self,
        run_id: str,
        exp_dir: str,
        data_root: str = "datasets",
        challenge: bool = False,
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
        self.challenge: bool = challenge
        self.all_results: List[Dict[str, Any]] = []
        self.processed_channels: set[str] = set()
        
        self.trained_classifiers: Dict[str, Any] = {}
        self.global_results: Dict[str, Any] = {"channel_id": "GLOBAL_EVENT_LEVEL"}
        self.event_labels_global: List[Any] = []
        self.predicted_events_global: List[Any] = []
        
        self.global_results_val: Dict[str, Any] = {"channel_id": "GLOBAL_EVENT_LEVEL_VAL"}
        self.event_labels_global_val: List[Any] = []
        self.predicted_events_global_val: List[Any] = []
        
        self.all_metadata: List[Dict[str, Any]] = []
        self.channel_val_metrics: Dict[str, Dict[str, Any]] = {}
        self.channel_fit_metrics: Dict[str, Dict[str, Any]] = {}
        
        # Challenge mode state
        self.challenge_pointwise_predictions: Dict[str, np.ndarray] = {}
        self.challenge_timestamps: Optional[np.ndarray] = None

    def set_classifier(self, channel_id: str, classifier: Any):
        """Manually inject a pre-trained classifier into the benchmark state.
        
        Args:
            channel_id (str): The ID of the channel.
            classifier (Any): The pre-trained model/classifier instance.
        """
        self.trained_classifiers[channel_id] = classifier

    def recover_global_state(self):
        """Recover global state from experiment directory."""
        if os.path.exists(self.run_dir):
            for channel_id in self.get_default_channels():
                json_path = os.path.join(self.run_dir, f"{channel_id}_intervals.json")
                if not os.path.exists(json_path):
                    continue
                with open(json_path) as f:
                    intervals = json.load(f)
                
                self.event_labels_global.extend([(pd.Timestamp(s), pd.Timestamp(e)) for s, e in intervals["true_intervals"]])
                self.predicted_events_global.extend([(pd.Timestamp(s), pd.Timestamp(e)) for s, e in intervals["pred_intervals"]])
                self.processed_channels.add(channel_id)
        else:
            raise ValueError(f"No global state found in {self.run_dir}. Please train all channels first.")
            
    def load_incremental_state(self):
        """Load already processed models and results from a previous crashed run."""
        results_csv = os.path.join(self.run_dir, "results.csv")
        metadata_csv = os.path.join(self.run_dir, "metadata.csv")
        
        if os.path.exists(results_csv):
            try:
                df = pd.read_csv(results_csv)
                df = df[~df["channel_id"].str.contains("GLOBAL_EVENT_LEVEL", na=False)]
                self.all_results = df.to_dict('records')
                self.processed_channels.update(df["channel_id"].tolist())
                # Restore channel_val_metrics from results.csv
                for record in self.all_results:
                    chid = record.get("channel_id")
                    if chid:
                        val_m = {k[4:]: v for k, v in record.items() if k.startswith("val_")}
                        if val_m:
                            self.channel_val_metrics[chid] = val_m
                logging.info(f"Loaded {len(self.processed_channels)} processed channels from results.csv")
            except Exception as e:
                logging.warning(f"Could not load incremental results.csv: {e}")
                
        if os.path.exists(metadata_csv):
            try:
                meta_df = pd.read_csv(metadata_csv)
                self.all_metadata = meta_df.to_dict('records')
                # Restore channel_fit_metrics from metadata.csv
                for record in self.all_metadata:
                    chid = record.get("channel_id")
                    if chid:
                        self.channel_fit_metrics[chid] = record
            except Exception as e:
                pass
                
        try:
            self.recover_global_state()
        except Exception:
            pass
        
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
        recover_state: bool = True,
        **extra_metrics,
    ) -> Dict[str, Any]:
        """Compute aggregated event-level metrics from internally accumulated state."""
        if channels is None:
            channels = self.get_default_channels()

        if recover_state and len(self.processed_channels) < len(channels):
            self.recover_global_state()
            
        # 1. Compute Test Metrics
        self.global_results = self._compute_metrics_from_global_events(
            self.event_labels_global, self.predicted_events_global, channels, time_aware=time_aware
        )
        self.global_results["channel_id"] = "GLOBAL_EVENT_LEVEL"
        
        # 2. Compute Validation Metrics
        if self.event_labels_global_val or self.predicted_events_global_val:
            val_metrics = self._compute_metrics_from_global_events(
                self.event_labels_global_val, self.predicted_events_global_val, channels, time_aware=time_aware
            )
            # Add prefixed keys to the main results for the row
            for k, v in val_metrics.items():
                self.global_results[f"val_{k}"] = v
        
        # 3. Add any extra metrics provided
        if extra_metrics:
            self.global_results.update(extra_metrics)
        
        logging.info("Global Event-Level Results: %s", self.global_results)

        self.all_results.append(self.global_results)
        pd.DataFrame.from_records(self.all_results).to_csv(
            os.path.join(self.run_dir, "results.csv"), index=False
        )
        
        if self.all_metadata:
             pd.DataFrame.from_records(self.all_metadata).to_csv(
                os.path.join(self.run_dir, "metadata.csv"), index=False
            )

        # --- Aggregate Challenge Predictions ---
        if self.challenge and self.challenge_pointwise_predictions:
            logging.info("Generating aggregate challenge prediction files...")
            
            # 1. Prediction for each channel as columns
            agg_df = pd.DataFrame(self.challenge_pointwise_predictions)
            
            # Add timestamps if available
            if self.challenge_timestamps is not None:
                # We try to use 'UT' or whatever is common
                agg_df.insert(0, "UT", self.challenge_timestamps)
            
            agg_channels_path = os.path.join(self.run_dir, "challenge_channels_predictions.csv")
            agg_df.to_csv(agg_channels_path, index=False)
            logging.info("Aggregate channel predictions saved to %s", agg_channels_path)
            
            # 2. Unified submission (Logical OR across channels)
            # We skip the timestamp column for OR operation
            channel_cols = [c for c in agg_df.columns if c != "UT"]
            global_prediction = (agg_df[channel_cols].sum(axis=1) > 0).astype(int)
            
            submission_df = pd.DataFrame({
                "prediction": global_prediction
            })
            if self.challenge_timestamps is not None:
                submission_df.insert(0, "UT", self.challenge_timestamps)
            
            submission_path = os.path.join(self.run_dir, "challenge_submission.csv")
            submission_df.to_csv(submission_path, index=False)
            logging.info("Unified challenge submission saved to %s", submission_path)

        return self.global_results

    def _compute_metrics_from_global_events(self, global_labels, global_preds, channels, time_aware=True):
        """Helper to compute metrics from raw global events."""
        event_labels = Benchmark.merge_intervals(global_labels)
        predicted_events = Benchmark.merge_intervals(global_preds)
        
        metrics = Benchmark.adtqc_score(event_labels, predicted_events)
        
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

        metrics.update(Benchmark.compute_metrics(event_labels, predicted_events))
        return metrics

    @staticmethod
    def _evaluate_predictions(classifier: AnomalyClassifier, data: Any, true_intervals: List[Any], y_pred: np.ndarray, pred_buffer:int=1) -> Dict[str, float]:
        """Process predictions into intervals and compute metrics vs ground truth."""
        pred_intervals = Benchmark.process_pred_anomalies(y_pred, pred_buffer)
        
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
        train: bool = True,
        data: Any = None
    ) -> Dict[str, Any]:
        """Trains the anomaly classifier for a given channel and saves it to state."""
        
        train_channel = data if data is not None else self.load_channel(channel_id, train=train)
        logging.info("Fitting the anomaly classifier for channel %s...", channel_id)
        
        chan_results_dir = os.path.join(self.run_dir, channel_id)
        os.makedirs(chan_results_dir, exist_ok=True)
        metrics = classifier.fit(train_channel, results_dir=chan_results_dir)
        
        if getattr(classifier, "val_results_", None) is not None:
            self._process_validation_metrics(channel_id, classifier, train_channel, classifier.val_results_)

        # Collect metadata for this channel
        meta = {"channel_id": channel_id}
        
        # Add scalar metrics (exclude complex types)
        meta.update({k: v for k, v in metrics.items() if not isinstance(v, (np.ndarray, list, dict))})
        
        # Flatten feature scores if available (one column per feature score)
        feature_scores = metrics.get("feature_scores")
        if isinstance(feature_scores, dict):
             meta.update({f"fscore_{k}": v for k, v in feature_scores.items()})

        # Add selected features as comma-separated string for readability
        selected_features = metrics.get("selected_features")
        if isinstance(selected_features, list):
            meta["selected_features"] = ",".join(map(str, selected_features))

        self.all_metadata.append(meta)
        self.channel_fit_metrics[channel_id] = meta

        self.trained_classifiers[channel_id] = classifier

        os.makedirs(self.run_dir, exist_ok=True)
        classifier_path = os.path.join(chan_results_dir, f"classifier-{channel_id}.pt")
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
        pred_buffer: int = 1,
    ) -> Tuple[Dict[str, Any], List[Any], List[Any]]:
        """Shared logic for computing metrics, saving results, and updating global state.

        Used by both ``test_channel`` and ``test_continual``.
        """
        results: Dict[str, Any] = {"channel_id": channel_id}
        
        # Include fitting metrics (callback times, etc.), excluding selection metadata
        if channel_id in self.channel_fit_metrics:
            fit_meta = self.channel_fit_metrics[channel_id]
            # Solo tempi e metriche tecniche, no feature selection e score intermedi
            results.update({
                k: v for k, v in fit_meta.items() 
                if k not in ["selected_features"] and not k.startswith("fscore_")
            })
        pred_anomalies = Benchmark.process_pred_anomalies(y_pred, pred_buffer)
        test_anomalies = classifier.prepare_labels(test_dataset)

        true_anomaly_intervals_ts = classifier.map_to_timestamps(test_dataset, test_anomalies)
        pred_intervals_ts = classifier.map_to_timestamps(test_dataset, pred_anomalies)

        # --- Handle Challenge Mode (Blind Inference) ---
        if self.challenge:
            logging.info("Challenge mode active: Generating pointwise predictions for submission...")
            original_indices = extra_metrics.get("original_indices") if extra_metrics else None
            
            if original_indices is not None:
                y_pointwise = np.zeros(len(test_dataset.data), dtype=int)
                for i, pred in enumerate(y_pred):
                    if pred == 1:
                        start, end = original_indices[i]
                        y_pointwise[int(start):int(end)] = 1
                
                submission_df = pd.DataFrame({
                    "prediction": y_pointwise
                })
                
                if hasattr(test_dataset, "timestamps") and test_dataset.timestamps is not None:
                    id_name = getattr(test_dataset, "id_column_name", "UT")
                    submission_df.insert(0, id_name, test_dataset.timestamps)
                
                chan_results_dir = os.path.join(self.run_dir, channel_id)
                csv_path = os.path.join(chan_results_dir, f"challenge_predictions_{channel_id}.csv")
                submission_df.to_csv(csv_path, index=False)
                logging.info("Challenge predictions saved to %s", csv_path)
                
                # Store for global aggregation
                self.challenge_pointwise_predictions[channel_id] = y_pointwise
                if self.challenge_timestamps is None:
                    self.challenge_timestamps = test_dataset.timestamps
            
            results["n_detected"] = len(pred_anomalies)
        else:
            all_metrics = Benchmark.compute_metrics(
                test_anomalies, pred_anomalies, total_length=len(y_pred),
                true_anomalies_ts=true_anomaly_intervals_ts, pred_anomalies_ts=pred_intervals_ts
            )
            results.update(all_metrics)

        # Include validation metrics AFTER test metrics for requested CSV order
        if channel_id in self.channel_val_metrics:
            for k, v in self.channel_val_metrics[channel_id].items():
                results[f"val_{k}"] = v

        self.processed_channels.add(channel_id)

        logging.info("Results for channel %s: %s", channel_id, results)

        self.all_results.append(results)
        os.makedirs(self.run_dir, exist_ok=True)
        pd.DataFrame.from_records(self.all_results).to_csv(
            os.path.join(self.run_dir, "results.csv"), index=False
        )

        if self.challenge:
            return results, [], pred_intervals_ts

        try:
            y_true_pointwise = np.zeros(len(y_pred), dtype=int)
            for start, end in test_anomalies:
                y_true_pointwise[max(0, int(start)):min(len(y_pred), int(end) + 1)] = 1
            
            chan_results_dir = os.path.join(self.run_dir, channel_id)
            plot_path = os.path.join(chan_results_dir, f"{channel_id}_degradation.png")
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
        classifier: Optional[AnomalyClassifier] = None,
        train: bool = False,
        data: Any = None,
        pred_buffer: int = 2,
    ) -> Tuple[Dict[str, Any], List[Any], List[Any]]:
        """Tests the fitted anomaly classifier for a given channel using internal state."""

        if classifier is None:
            if channel_id not in self.trained_classifiers:
                logging.warning("Classifier for channel %s not found in state.", channel_id)
                return {"channel_id": channel_id}, [], []
            classifier = self.trained_classifiers[channel_id]
        
        test_channel = data if data is not None else self.load_channel(channel_id, train=train)

        logging.info("Predicting the test data for channel %s...", channel_id)
        
        chan_results_dir = os.path.join(self.run_dir, channel_id)
        os.makedirs(chan_results_dir, exist_ok=True)
        y_pred, metrics = classifier.predict(test_channel, results_dir=chan_results_dir)

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
            
            true_forward_intervals = classifier.prepare_labels(data)
            exp_metrics = self._evaluate_predictions(classifier, data, true_forward_intervals, experience_predictions)
            exp_metrics.update({k: v for k, v in metrics.items() if k not in exp_metrics})
            experience_log[f"experience_{i}"] = exp_metrics
            
            os.makedirs(self.run_dir, exist_ok=True)
            with open(os.path.join(self.run_dir, f"{channel_id}_stream_history.json"), "w") as f:
                json.dump(experience_log, f, indent=2, default=str)
            
            chan_results_dir = os.path.join(self.run_dir, channel_id)
            np.savez_compressed(
                os.path.join(chan_results_dir, f"{channel_id}_streaming_results.npz"),
                predictions=np.array(all_predictions),
                labels=np.array(all_point_labels),
            )

        logging.info("Streaming for channel %s completed.", channel_id)

        y_pred_all = np.array(all_predictions)
        results, _, _ = self._finalize_channel_results(
            channel_id, classifier, test_dataset, y_pred_all,
        )

        chan_results_dir = os.path.join(self.run_dir, channel_id)
        with open(os.path.join(chan_results_dir, f"{channel_id}_stream_history.json"), "w") as f:
            json.dump(experience_log, f, indent=2, default=str)

        return results


    def _update_global_state(self, channel_id, metrics, true_intervals=None, pred_intervals=None, validation=False):
        """Helper to accumulate global metrics and save interval logs."""
        for k, v in metrics.items():
            if k.endswith(("time", "cpu")):
                self.global_results[k] = self.global_results.get(k, 0) + v
                
        if true_intervals is not None or pred_intervals is not None:
            true_ints = true_intervals if true_intervals is not None else []
            pred_ints = pred_intervals if pred_intervals is not None else []
            if validation:
                self.event_labels_global_val.extend(true_ints)
                self.predicted_events_global_val.extend(pred_ints)
            else:
                self.event_labels_global.extend(true_ints)
                self.predicted_events_global.extend(pred_ints)

            chan_results_dir = os.path.join(self.run_dir, channel_id)
            with open(os.path.join(chan_results_dir, f"{channel_id}_{'val_' if validation else ''}intervals.json"), "w") as f:
                json.dump({
                    "pred_intervals": [[str(s), str(e)] for s, e in pred_intervals],
                    "true_intervals": [[str(s), str(e)] for s, e in true_intervals],
                }, f, indent=2)

    def _process_validation_metrics(self, channel_id, classifier, train_channel, val_results):
        """Extract and aggregate validation metrics from the fitness process."""
        y_pred = val_results["y_pred"]
        original_indices = val_results["original_indices"]
        true_intervals_rel = val_results.get("true_intervals", [])

        # 1. Convert segment predictions to absolute intervals, then merge
        raw_intervals = [
            (int(original_indices[i][0]), int(original_indices[i][1]))
            for i, pred in enumerate(y_pred)
            if pred == 1 and original_indices is not None
        ]
        pred_intervals_abs = Benchmark.merge_intervals(raw_intervals)

        # 2. Convert relative true_intervals to absolute
        if original_indices is not None and original_indices.ndim == 2:
            # Windowed case: true_intervals_rel are window indices
            true_intervals_abs = Benchmark.merge_intervals([
                (int(original_indices[int(s)][0]), int(original_indices[int(min(e, len(original_indices)-1))][1]))
                for s, e in true_intervals_rel
                if int(s) < len(original_indices)
            ])
        else:
            # Pointwise case: true_intervals_rel are already point-relative offsets
            start_idx = original_indices[0] if original_indices is not None else 0
            if isinstance(start_idx, (list, np.ndarray, tuple)):
                 start_idx = start_idx[0]
            true_intervals_abs = Benchmark.merge_intervals(
                [(int(s + start_idx), int(e + start_idx)) for s, e in true_intervals_rel]
            )

        # 3. Map to timestamps directly from dataset indices
        timestamps = getattr(train_channel, "timestamps", None)
        offset = getattr(train_channel, "start_idx", 0)
        
        if timestamps is not None:
            true_intervals_ts = [(timestamps[s], timestamps[min(e, len(timestamps)-1)]) for s, e in true_intervals_abs]
            pred_intervals_ts = [(timestamps[s], timestamps[min(e, len(timestamps)-1)]) for s, e in pred_intervals_abs]
        else:
            true_intervals_ts = [(s + offset, e + offset) for s, e in true_intervals_abs]
            pred_intervals_ts = [(s + offset, e + offset) for s, e in pred_intervals_abs]

        val_metrics = Benchmark.compute_metrics(
            true_intervals_abs, pred_intervals_abs, total_length=len(train_channel.data),
            true_anomalies_ts=true_intervals_ts, pred_anomalies_ts=pred_intervals_ts
        )

        # 4. Update global state for validation
        self._update_global_state(channel_id, val_metrics, true_intervals_ts, pred_intervals_ts, validation=True)
        self.channel_val_metrics[channel_id] = val_metrics
        logging.info("Validation metrics for channel %s aggregated into global state.", channel_id)

    @staticmethod
    def save_degradation_plot(
        channel_id: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        save_path: str,
        window_size: int = 2000,
    ):
        """Generates and saves a degradation plot (cumulative errors and rolling FPR)."""
        
        # Take the first 50,000 points as requested to keep the plot manageable but high resolution
        max_plot_len = 50000
        y_true = y_true[:max_plot_len]
        y_pred = y_pred[:max_plot_len]
        
        false_positives = (y_pred == 1) & (y_true == 0)
        false_negatives = (y_pred == 0) & (y_true == 1)
        
        # Calculate full series
        cum_fp = np.cumsum(false_positives)
        cum_fn = np.cumsum(false_negatives)
        fp_rolling_rate = pd.Series(false_positives).rolling(window=window_size).mean() * 100
        
        n_points = len(y_pred)
        x_axis = np.arange(n_points)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        ax1.plot(x_axis, cum_fp, label='Cumulative False Positives', color='red', linewidth=2)
        ax1.plot(x_axis, cum_fn, label='Cumulative False Negatives', color='orange', linewidth=2)
        ax1.set_title(f"[{channel_id}] Cumulative Errors over Time (First {max_plot_len} points)")
        ax1.set_ylabel("Total Error Count")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(x_axis, fp_rolling_rate, label=f'Rolling FPR (window {window_size})', color='purple')
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
        y_pred: np.ndarray, pred_buffer: int = 1
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
        true_indices_grouped = [list(range(int(e[0]), int(e[1]) + 1)) for e in true_anomalies]
        true_indices_flat = set(i for group in true_indices_grouped for i in group)
        
        correct_predictions = 0
        for e_seq in pred_anomalies:
            i_anom_predicted = set(range(int(e_seq[0]), int(e_seq[1]) + 1))

            matched_indices = list(i_anom_predicted & true_indices_flat)
            if len(matched_indices) > 0:
                correct_predictions += 1
                for i, gt_indices in enumerate(true_indices_grouped):
                    if any(idx in i_anom_predicted for idx in gt_indices):
                        if i not in matched_true_seqs:
                            matched_true_seqs.append(i)

        results["true_positives"] = len(matched_true_seqs)
        results["false_positives"] = len(pred_anomalies) - correct_predictions
        results["false_negatives"] = len(true_anomalies) - results["true_positives"]

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
        if hasattr(self, "channels"):
            return self.channels
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
