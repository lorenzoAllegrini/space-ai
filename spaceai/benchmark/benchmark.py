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
from torch.utils.data import (
    DataLoader,
    Subset,
)
from tqdm import tqdm  # type: ignore

from spaceai.data.utils import seq_collate_fn

from .callbacks import CallbackHandler
from .utils import merge_intervals

if TYPE_CHECKING:
    from spaceai.models.predictors import SequenceModel
    from spaceai.models.anomaly import AnomalyDetector
    from .callbacks import Callback
import zmq
import json
import time

import threading
import json

class Benchmark:
    """Base class for benchmark runners."""

    def __init__(
        self,
        run_id: str,
        exp_dir: str,
        segmentator: Any,
        feature_extractor: Optional[Any] = None,
        seq_length: int = 250,
        n_predictions: int = 1,
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
        self.seq_length: int = seq_length
        self.n_predictions: int = n_predictions
        self.all_results: List[Dict[str, Any]] = []
        self.segmentator = segmentator
        self.feature_extractor = feature_extractor
        
        # Internal state for decoupled execution and SML
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

        event_labels = merge_intervals(self.event_labels_global)
        predicted_events = merge_intervals(self.predicted_events_global)
        
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

    def anomaly_listener(self, channel_id, server_ip, pub_port):
        """Listen passively for anomaly alerts from the server."""
        ctx = zmq.Context.instance()
        sub_socket = ctx.socket(zmq.SUB)
        sub_socket.connect(f"tcp://{server_ip}:{pub_port}")
        
        sub_socket.setsockopt_string(zmq.SUBSCRIBE, channel_id)
        
        predictions = []
        while True:
            try:
                frames = sub_socket.recv_multipart(flags=zmq.NOBLOCK)
                if len(frames) == 2:
                    alert = json.loads(frames[1].decode("utf-8"))
                    if alert.get("channel_id") == channel_id:
                        predictions.append(alert.get("timestep"))
            except zmq.Again:
                time.sleep(0.1)
            except Exception as e:
                logging.error("Listener thread error: %s", e)
                break
                
        self.global_results[channel_id] = Benchmark.process_pred_anomalies(np.array(predictions) if predictions else np.zeros((0,)), 0)
        
    def simulate_stream_to_server(
        self,
        channel_id: str,
        server_ip: str,
        port: int,
        sample_rate_ms: int = 1000,
        max_duration_s: Optional[int] = 500,
    ) -> Dict[str, List[int]]:
        """Simulate real-time streaming telemetry to the SML server.

        Args:
            server_ip: IP address of the remote SML server.
            port: Port the SML server is listening on.
            sample_rate_ms: Delay in milliseconds between sending each window.
            max_duration_s: Maximum duration in seconds to stream.

        Returns:
            Dict[str, List[int]]: Empty dict (PUSH architecture does not receive predictions).
        """

        pub_port = port + 1

        listener = threading.Thread(
            target=self.anomaly_listener,
            args=(channel_id, server_ip, pub_port),
            daemon=True
        )
        listener.start()

        _, test_dataset = self.load_channel(channel_id, overlapping_train=False)
        if test_dataset is None:
            return {}

        data = test_dataset.data[:, 0]
        logging.info("Streaming dataset for channel %s to %s:%d...", channel_id, server_ip, port)

        channel_bytes = channel_id.encode("utf-8")
        channel_start_time = time.perf_counter()
        
        with zmq.Context.instance() as context:
            with context.socket(zmq.PUSH) as socket:
                socket.connect(f"tcp://{server_ip}:{port}")

                for timestep in range(len(data)):
                    if max_duration_s is not None and (time.perf_counter() - channel_start_time) > max_duration_s:
                        logging.info("Max duration reached. Halting channel %s.", channel_id)
                        break

                    payload = {
                        "timestep": timestep,
                        "value": float(data[timestep])
                    }
                    
                    socket.send_multipart([
                        channel_bytes, 
                        json.dumps(payload).encode("utf-8")
                    ])

                    if sample_rate_ms > 0:
                        time.sleep(sample_rate_ms / 1000.0)

        return {}

    def train_channel_telemanom(
        self,
        channel_id: str,
        predictor: SequenceModel,
        fit_predictor_args: Optional[Dict[str, Any]] = None,
        perc_eval: Optional[float] = 0.2,
        restore_predictor: bool = False,
        overlapping_train: bool = True,
        callbacks: Optional[List[Callback]] = None,
        call_every_ms: int = 100,
    ) -> Dict[str, Any]:
        """Trains the telemanom predictor for a given channel and saves it to state."""
        callback_handler = CallbackHandler(
            callbacks=callbacks if callbacks is not None else [],
            call_every_ms=call_every_ms,
        )
        train_channel, _ = self.load_channel(
            channel_id, overlapping_train=overlapping_train
        )
        os.makedirs(self.run_dir, exist_ok=True)

        results: Dict[str, Any] = {"channel_id": channel_id}
        train_history = None
        
        if (
            os.path.exists(os.path.join(self.run_dir, f"predictor-{channel_id}.pt"))
            and restore_predictor
        ):
            logging.info("Restoring predictor for channel %s...", channel_id)
            predictor.load(os.path.join(self.run_dir, f"predictor-{channel_id}.pt"))

        elif fit_predictor_args is not None:
            logging.info("Fitting the predictor for channel %s...", channel_id)
            batch_size = fit_predictor_args.pop("batch_size", 64)
            eval_channel = None
            try:
                if perc_eval is not None:
                    indices = np.arange(len(train_channel))
                    np.random.shuffle(indices)
                    eval_size = int(len(train_channel) * perc_eval)
                    eval_channel = Subset(train_channel, indices[:eval_size].tolist())
                    train_channel = Subset(train_channel, indices[eval_size:].tolist())  # type: ignore[assignment]
                train_loader = DataLoader(
                    train_channel,
                    batch_size=batch_size,
                    shuffle=True,
                    collate_fn=seq_collate_fn(n_inputs=2, mode="batch"),
                )
                eval_loader = (
                    DataLoader(
                        eval_channel,
                        batch_size=batch_size,
                        shuffle=False,
                        collate_fn=seq_collate_fn(n_inputs=2, mode="batch"),
                    )
                    if eval_channel is not None
                    else None
                )
            except Exception as e:
                logging.error("Failed to prepare data loaders: %s", e)
                return results

            callback_handler.start()
            predictor.stateful = False
            train_history = predictor.fit(
                train_loader=train_loader,
                valid_loader=eval_loader,
                **fit_predictor_args,
            )
            callback_handler.stop()
            results.update(
                {
                    f"train_{k}": v
                    for k, v in callback_handler.collect(reset=True).items()
                }
            )
            logging.info(
                "Training time on channel %s: %s", channel_id, results.get('train_time', 0)
            )
            if train_history:
                pd.DataFrame.from_records(train_history).to_csv(
                    os.path.join(self.run_dir, f"train_history-{channel_id}.csv"),
                    index=False,
                )
            predictor_path = os.path.join(self.run_dir, f"predictor-{channel_id}.pt")
            predictor.save(predictor_path)
            if os.path.exists(predictor_path):
                results["disk_usage"] = os.path.getsize(predictor_path)
                
            if train_history is not None and len(train_history) > 0:
                results["train_loss"] = train_history[-1].get("loss_train")
                if eval_loader is not None:
                    results["eval_loss"] = train_history[-1].get("loss_eval")

        self.trained_classifiers[channel_id] = predictor
        
        for metric in [m for m in results.keys() if m.endswith("time") or m.endswith("cpu")]:
            if metric not in self.global_results:
                self.global_results[metric] = results[metric]
            else:
                self.global_results[metric] += results[metric]
                
        return results

    def test_channel_telemanom(
        self,
        channel_id: str,
        detector: AnomalyDetector,
        callbacks: Optional[List[Callback]] = None,
        call_every_ms: int = 100,
    ) -> Tuple[Dict[str, Any], List[Any], List[Any]]:
        """Tests the trained predictor and detector for a given channel."""
        if channel_id not in self.trained_classifiers:
            logging.warning("Predictor for channel %s not found in state. Call train_channel_telemanom first.", channel_id)
            return {"channel_id": channel_id}, [], []

        predictor = self.trained_classifiers[channel_id]

        callback_handler = CallbackHandler(
            callbacks=callbacks if callbacks is not None else [],
            call_every_ms=call_every_ms,
        )
        _, test_channel = self.load_channel(
            channel_id, overlapping_train=False
        )
        
        results: Dict[str, Any] = {"channel_id": channel_id}

        if getattr(predictor, "model", None) is not None:
            predictor.model.eval()
            
        logging.info("Predicting the test data for channel %s...", channel_id)
        test_loader = DataLoader(
            test_channel,
            batch_size=1,
            shuffle=False,
            collate_fn=seq_collate_fn(n_inputs=2, mode="time"),
        )
        callback_handler.start()
        predictor.stateful = True
        y_pred, y_trg = zip(
            *[
                (
                    predictor(x.to(predictor.device)).detach().cpu().squeeze().numpy(),
                    y.detach().cpu().squeeze().numpy(),
                )
                for x, y in tqdm(test_loader, desc="Predicting")
            ]
        )
        y_pred, y_trg = [
            np.concatenate(seq)[test_channel.window_size - 1 :]
            for seq in [y_pred, y_trg]
        ]
        callback_handler.stop()
        results.update(
            {f"predict_{k}": v for k, v in callback_handler.collect(reset=True).items()}
        )
        results["test_loss"] = np.mean(((y_pred - y_trg) ** 2)) 
        logging.info("Test loss for channel %s: %s", channel_id, results['test_loss'])
        logging.info(
            "Prediction time for channel %s: %s", channel_id, results.get('predict_time', 0)
        )

        # Testing the detector
        logging.info("Detecting anomalies for channel %s", channel_id)
        callback_handler.start()
        if len(y_trg) < 2500:
            detector.ignore_first_n_factor = 1
        if len(y_trg) < 1800:
            detector.ignore_first_n_factor = 0
            
        pred_anomalies = detector.detect_anomalies(np.array(y_pred), np.array(y_trg))
        pred_anomalies += detector.flush_detector()
        callback_handler.stop()
        
        results.update(
            {f"detect_{k}": v for k, v in callback_handler.collect(reset=True).items()}
        )
        logging.info(
            "Detection time for channel %s: %s", channel_id, results.get('detect_time', 0)
        )

        true_anomalies = test_channel.anomalies
        all_metrics = Benchmark.compute_metrics(
            true_anomalies, pred_anomalies, total_length=len(y_pred)
        )
        results.update(all_metrics)

        logging.info("Results for channel %s", channel_id)
        
        self.channel_predictions[channel_id] = {
            "y_pred": y_pred.tolist(),
            "pred_anomalies": pred_anomalies,
            "true_anomalies": true_anomalies
        }

        self.all_results.append(results)

        pd.DataFrame.from_records(self.all_results).to_csv(
            os.path.join(self.run_dir, "results.csv"), index=False
        )

        offset = test_channel.window_size - 1
        pred_anomalies_global = [
            (int(s + offset), int(e + offset)) for s, e in pred_anomalies
        ]

        if hasattr(test_channel, "timestamps") and test_channel.timestamps is not None:
            pred_intervals = [
                (test_channel.timestamps[s], test_channel.timestamps[e]) 
                for s, e in pred_anomalies_global
            ]
            label_intervals = [
                (test_channel.timestamps[s], test_channel.timestamps[e]) 
                for s, e in true_anomalies
            ]
        else:
            pred_intervals = pred_anomalies_global
            label_intervals = true_anomalies

        for metric in [m for m in results.keys() if m.endswith("time") or m.endswith("cpu")]:
            if metric not in self.global_results:
                self.global_results[metric] = results[metric]
            else:
                self.global_results[metric] += results[metric]
                
        self.event_labels_global.extend(label_intervals)
        self.predicted_events_global.extend(pred_intervals)

        with open(os.path.join(self.run_dir, f"{channel_id}_intervals.json"), "w") as f:
            json.dump({
                "pred_intervals": [[str(s), str(e)] for s, e in pred_intervals],
                "true_intervals": [[str(s), str(e)] for s, e in label_intervals],
            }, f, indent=2)

        return results, label_intervals, pred_intervals

    def train_channel_rolling_stats(
        self,
        channel_id: str,
        classifier,
        overlapping_train: Optional[bool] = True,
        callbacks: Optional[List[Callback]] = None,
        call_every_ms: int = 100,
        supervised: bool = True,
    ) -> Dict[str, Any]:
        """Trains the anomaly classifier for a given channel and saves it to state."""
        callback_handler = CallbackHandler(
            callbacks=callbacks if callbacks is not None else [],
            call_every_ms=call_every_ms,
        )
        
        train_channel, _ = self.load_channel(
            channel_id, overlapping_train=overlapping_train if overlapping_train is not None else True
        )
        os.makedirs(self.run_dir, exist_ok=True)
        results: Dict[str, Any] = {"channel_id": channel_id}

        if self.segmentator is not None:
            callback_handler.start()
            seg_result = self.segmentator.segment(train_channel)
            callback_handler.stop()
            results.update({f"train_set_segmentation_{k}": v for k, v in callback_handler.collect(reset=True).items()})
            train_channel = seg_result["segments"]
            train_labels = seg_result["labels"]
        else:
            train_anomalies = train_channel.anomalies
            num_segments = len(train_channel)
            train_labels = np.zeros(num_segments, dtype=int)
            if train_anomalies is not None:
                for start, end in train_anomalies:
                    start = max(0, start)
                    end = min(num_segments - 1, end)
                    train_labels[start : end + 1] = 1
            
        if len(train_channel) == 0:
            logging.warning("No training data for channel %s. Skipping...", channel_id)
            return results

        if self.feature_extractor is not None:
            callback_handler.start()
            train_channel = self.feature_extractor.fit_transform(train_channel)
            callback_handler.stop()
            results.update({f"train_set_feature_extraction_{k}": v for k, v in callback_handler.collect(reset=True).items()})
            
        logging.info("Fitting the classifier for channel %s...", channel_id)

        callback_handler.start()
        if supervised:
            classifier.fit(X=train_channel, y=train_labels)
        else:
            classifier.fit(X=train_channel, y=train_labels)
        callback_handler.stop()
        results.update({f"train_{k}": v for k, v in callback_handler.collect(reset=True).items()})

        self.trained_classifiers[channel_id] = classifier

        classifier_path = os.path.join(self.run_dir, f"classifier-{channel_id}.pt")
        if hasattr(classifier, 'save'):
            classifier.save(classifier_path)
        else:
            import torch
            torch.save(classifier, classifier_path)
    
        for metric in [m for m in results.keys() if m.endswith("time") or m.endswith("cpu")]:
            if metric not in self.global_results:
                self.global_results[metric] = results[metric]
            else:
                self.global_results[metric] += results[metric]

        return results

    def test_channel_rolling_stats(
        self,
        channel_id: str,
        pred_buffer: int = 0,
        callbacks: Optional[List[Callback]] = None,
        call_every_ms: int = 100,
    ) -> Tuple[Dict[str, Any], List[Any], List[Any]]:
        """Tests the fitted anomaly classifier for a given channel using internal state."""
        if channel_id not in self.trained_classifiers:
            logging.warning("Classifier for channel %s not found in state. Call train_channel_rolling_stats first.", channel_id)
            return {"channel_id": channel_id}, [], []
            
        classifier = self.trained_classifiers[channel_id]
        
        callback_handler = CallbackHandler(
            callbacks=callbacks if callbacks is not None else [],
            call_every_ms=call_every_ms,
        )
        _, test_channel = self.load_channel(
            channel_id, overlapping_train=False # Doesn't matter for test loading
        )
        results: Dict[str, Any] = {"channel_id": channel_id}
        original_test_channel = test_channel

        logging.info("Predicting the test data for channel %s...", channel_id)

        if self.segmentator is not None:
            callback_handler.start()
            seg_result = self.segmentator.segment(test_channel)
            callback_handler.stop()
            results.update({f"test_set_segmentation_{k}": v for k, v in callback_handler.collect(reset=True).items()})
            test_channel = seg_result["segments"]
            test_anomalies = seg_result["intervals"]
            segment_indices = seg_result["segment_indices"]
        else:
            test_anomalies = test_channel.anomalies
            segment_indices = None 

        if len(test_channel) == 0:
            logging.warning("No test data for channel %s. Skipping evaluation...", channel_id)
            return results, [], []

        if self.feature_extractor is not None:
            callback_handler.start()
            # MUST BE .transform, not .fit_transform
            test_channel = self.feature_extractor.transform(test_channel) 
            callback_handler.stop()
            results.update({f"test_set_feature_extraction_{k}": v for k, v in callback_handler.collect(reset=True).items()})
            
        callback_handler.start()
        y_pred = classifier.predict(X=test_channel)
        callback_handler.stop()

        pred_anomalies = Benchmark.process_pred_anomalies(y_pred, pred_buffer)

        results.update({f"predict_{k}": v for k, v in callback_handler.collect(reset=True).items()})
        
        combined_anomalies = test_anomalies
        combined_anomalies.sort()
        all_metrics = Benchmark.compute_metrics(
            combined_anomalies, pred_anomalies, total_length=len(y_pred)
        )
        results.update(all_metrics)
        
        test_anomalies_mask = np.zeros(len(y_pred), dtype=int)
        for start, end in test_anomalies:
            test_anomalies_mask[int(start) : int(end) + 1] = 1

        results.update({
            "test_length": len(test_channel),
            "test_negatives": len(test_channel) - test_anomalies_mask.sum(),
            "detected_negatives": int(((y_pred == 0) & (test_anomalies_mask == 0)).sum()),
        })
        
        logging.info("Results for channel %s: %s", channel_id, results)

        self.channel_predictions[channel_id] = {
            "y_pred": y_pred.tolist(),
            "pred_anomalies": pred_anomalies,
            "true_anomalies": combined_anomalies
        }
        
        self.all_results.append(results)
        pd.DataFrame.from_records(self.all_results).to_csv(
            os.path.join(self.run_dir, "results.csv"), index=False
        )

        if segment_indices is None or len(segment_indices) == 0:
            return results, [], []

        true_preds_intervals = [(segment_indices[s][0], segment_indices[e][1]) for s, e in pred_anomalies]
        true_anomaly_intervals = [(segment_indices[s][0], segment_indices[e][1]) for s, e in combined_anomalies]

        timestamps = getattr(original_test_channel, "timestamps", None)
        if timestamps is not None and len(timestamps) > 0:
            limit = len(timestamps)
            true_preds_intervals = [(timestamps[s], timestamps[e]) for s, e in true_preds_intervals if e < limit]
            true_anomaly_intervals = [(timestamps[s], timestamps[e]) for s, e in true_anomaly_intervals if e < limit]
        else:
            return results, [], []

        # Accumulate metrics globally for the final summary
        for metric in [m for m in results.keys() if m.endswith("time") or m.endswith("cpu")]:
            if metric not in self.global_results:
                self.global_results[metric] = results[metric]
            else:
                self.global_results[metric] += results[metric]
                
        self.event_labels_global.extend(true_anomaly_intervals)
        self.predicted_events_global.extend(true_preds_intervals)

        with open(os.path.join(self.run_dir, f"{channel_id}_intervals.json"), "w") as f:
            json.dump({
                "pred_intervals": [[str(s), str(e)] for s, e in true_preds_intervals],
                "true_intervals": [[str(s), str(e)] for s, e in true_anomaly_intervals],
            }, f, indent=2)

        return results, true_anomaly_intervals, true_preds_intervals


    @staticmethod
    def process_pred_anomalies(
        y_pred: np.ndarray, pred_buffer: int
    ) -> List[List[int]]:
        """Process predicted anomalies by grouping consecutive indices and applying buffer."""
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
    def aggregate_results_event_level(
        min_start_time: pd.Timestamp,
        min_period: pd.Timedelta,
        event_labels: List[pd.Timestamp, pd.Timestamp],
        predicted_events: List[pd.Timestamp, pd.Timestamp],
        time_aware: bool,
    ) -> Dict[str, Any]:

        """Aggregate event-level results.

        Args:
            event_labels (List[Tuple[int, int]]): List of true event intervals.
            predicted_events (List[Tuple[int, int]]): List of predicted event intervals.
        """
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

        event_labels = merge_intervals(event_labels)
        predicted_events = merge_intervals(predicted_events)
        
        return predicted_events, event_labels

    @staticmethod
    def compute_metrics(
        true_anomalies: List[Tuple[int, int]],
        pred_anomalies: List[Tuple[int, int]],
        total_length: Optional[int] = None,
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
