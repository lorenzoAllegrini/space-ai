"""Base benchmark class for anomaly detection benchmarks."""

from __future__ import annotations

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

    def run_event_level(
        self,
        channels: Optional[List[str]] = None,
        overlapping_train: bool = True,
        callbacks: Optional[List[Callback]] = None,
        call_every_ms: int = 100,
        predictor: Optional[SequenceModel] = None,
        detector: Optional[AnomalyDetector] = None,
        fit_predictor_args: Optional[Dict[str, Any]] = None,
        perc_eval: float = 0.2,
        restore_predictor: bool = False,
        pred_buffer: int = 0,
        supervised: bool = True,
        time_aware: bool = True,
    ):
        """Run benchmark on a set of channels."""

        if channels is None:
            channels = self.get_default_channels()

        event_labels = []
        predicted_events = []
        for channel_id in channels:
            
            if self.segmentator is not None:
                _, label_intervals, pred_intervals = self.run_channel_rolling_stats(
                    channel_id=channel_id,
                    classifier=predictor,
                    pred_buffer=pred_buffer,
                    overlapping_train=overlapping_train,
                    callbacks=callbacks,
                    call_every_ms=call_every_ms,
                    supervised=supervised,
                )
                event_labels.extend(label_intervals)
                predicted_events.extend(pred_intervals)

            elif predictor is not None and detector is not None:
                _, label_intervals, pred_intervals = self.run_channel_telemanom(
                    channel_id=channel_id,
                    predictor=predictor,
                    detector=detector,
                    fit_predictor_args=fit_predictor_args,
                    perc_eval=perc_eval,
                    restore_predictor=restore_predictor,
                    overlapping_train=overlapping_train,
                    callbacks=callbacks,
                    call_every_ms=call_every_ms,
                )
                event_labels.extend(label_intervals)
                predicted_events.extend(pred_intervals)
            else:
                raise ValueError(
                    "Either 'segmentator' or ('predictor' and 'detector') must be provided."
                )

        min_start_time, min_period = self.get_global_temporal_params(channels)
        if time_aware and min_start_time is not None and min_period is not None:
            event_labels_idx = [
                (
                    int((pd.Timestamp(s) - min_start_time).total_seconds() / min_period),
                    int((pd.Timestamp(e) - min_start_time).total_seconds() / min_period)
                ) for s, e in event_labels
            ]
            predicted_events_idx = [
                (
                    int((pd.Timestamp(s) - min_start_time).total_seconds() / min_period),
                    int((pd.Timestamp(e) - min_start_time).total_seconds() / min_period)
                ) for s, e in predicted_events
            ]
        else:
            event_labels_idx = event_labels
            predicted_events_idx = predicted_events

        return self._aggregate_and_save_results(
            event_labels_idx, 
            predicted_events_idx
        )



    def run_channel_telemanom(
        self,
        channel_id: str,
        predictor: SequenceModel,
        detector: AnomalyDetector,
        fit_predictor_args: Optional[Dict[str, Any]] = None,
        perc_eval: Optional[float] = 0.2,
        restore_predictor: bool = False,
        overlapping_train: bool = True,
        callbacks: Optional[List[Callback]] = None,
        call_every_ms: int = 100,
    ) -> Tuple[Dict[str, Any], List[Any], List[Any]]:
        """Runs the benchmark for a given channel.

        Args:
            channel_id (str): the ID of the channel to be used
            predictor (SequenceModel): the sequence model to be trained
            detector (AnomalyDetector): the anomaly detector to be used
            fit_predictor_args (Optional[Dict[str, Any]]): additional arguments for the predictor's fit method
            perc_eval (Optional[float]): the percentage of the training data to be used for evaluation
            restore_predictor (bool): whether to restore the predictor from a previous run
            overlapping_train (bool): whether to use overlapping sequences for training
            callbacks (Optional[List[Callback]]): a list of callbacks to be used during benchmark
            call_every_ms (int): the interval at which the callbacks are called
        """
        callback_handler = CallbackHandler(
            callbacks=callbacks if callbacks is not None else [],
            call_every_ms=call_every_ms,
        )
        train_channel, test_channel = self.load_channel(
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
                return

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
                "Training time on channel %s: %s", channel_id, results['train_time']
            )
            train_history = pd.DataFrame.from_records(train_history).to_csv(
                os.path.join(self.run_dir, f"train_history-{channel_id}.csv"),
                index=False,
            )
            predictor_path = os.path.join(self.run_dir, f"predictor-{channel_id}.pt")
            predictor.save(predictor_path)
            results["disk_usage"] = os.path.getsize(predictor_path)

        if predictor.model is not None:
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
        results["test_loss"] = np.mean(((y_pred - y_trg) ** 2))  # type: ignore[operator]
        logging.info("Test loss for channel %s: %s", channel_id, results['test_loss'])
        logging.info(
            "Prediction time for channel %s: %s", channel_id, results['predict_time']
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
            "Detection time for channel %s: %s", channel_id, results['detect_time']
        )

        true_anomalies = test_channel.anomalies
        classification_results = self.compute_classification_metrics(
            true_anomalies, pred_anomalies
        )
        corrected_results = self.compute_corrected_range_classification_metrics(
            classification_results,
            true_anomalies,
            pred_anomalies,  # type: ignore[arg-type]
            total_length=len(y_pred),
        )
        classification_results.update(corrected_results)
        results.update(classification_results)
        if train_history is not None:
            results["train_loss"] = train_history[-1]["loss_train"]
            if eval_loader is not None:
                results["eval_loss"] = train_history[-1]["loss_eval"]

        logging.info("Results for channel %s", channel_id)

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

        return results, label_intervals, pred_intervals

    def run_channel_rolling_stats(
        self,
        channel_id: str,
        classifier,
        pred_buffer: int = 0,
        overlapping_train: Optional[bool] = True,
        callbacks: Optional[List[Callback]] = None,
        call_every_ms: int = 100,
        supervised: bool = True,
    ) -> Dict[str, Any]:
        """
        Runs the anomaly classifier benchmark for a given channel.

        Args:
            channel_id (str): The channel ID to process.
            classifier (AnomalyClassifier): The supervised anomaly classifier to train and test.
            pred_buffer (int): A buffer to add to the predicted anomaly indices.
            overlapping_train (bool): Whether to use overlapping sequences for training.
            callbacks (Optional[List[Callback]]): Optional list of callbacks for monitoring.
            call_every_ms (int): Interval (in milliseconds) for calling callbacks.

        Returns:
            Dict[str, Any]: A dictionary containing the benchmark results.
        """
        callback_handler = CallbackHandler(
            callbacks=callbacks if callbacks is not None else [],
            call_every_ms=call_every_ms,
        )
        train_channel, test_channel = self.load_channel(
            channel_id, overlapping_train=overlapping_train if overlapping_train is not None else True
        )
        os.makedirs(self.run_dir, exist_ok=True)
        results: Dict[str, Any] = {"channel_id": channel_id}

        if self.segmentator is not None:
            seg_result = self.segmentator.segment(train_channel)
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
            train_channel = self.feature_extractor.fit_transform(train_channel)
        logging.info("Fitting the classifier for channel %s...", channel_id)

        callback_handler.start()
        if supervised:
            classifier.fit(X=train_channel, y=train_labels)
        else:
            classifier.fit(X=train_channel, y=train_labels)
        callback_handler.stop()
        results.update(
            {f"train_{k}": v for k, v in callback_handler.collect(reset=True).items()}
        )
        # Evaluate the classifier on test data
        logging.info("Predicting the test data for channel %s...", channel_id)
        # Keep reference to original test channel for timestamp access
        original_test_channel = test_channel

        if self.segmentator is not None:
            seg_result = self.segmentator.segment(test_channel)
            test_channel = seg_result["segments"]
            test_anomalies = seg_result["intervals"]
            segment_indices = seg_result["segment_indices"]
            
        else:
            test_anomalies = test_channel.anomalies
            segment_indices = None # Handle this case if needed, or assume it won't happen here

    
        if len(test_channel) == 0:
            logging.warning("No test data for channel %s. Skipping evaluation...", channel_id)
            return results, [], []

        if self.feature_extractor is not None:
            test_channel = self.feature_extractor.transform(test_channel)
        
        callback_handler.start()
        y_pred = classifier.predict(X=test_channel)
        pred_anomalies = self.process_pred_anomalies(y_pred, pred_buffer)
        callback_handler.stop()

        results.update(
            {f"predict_{k}": v for k, v in callback_handler.collect(reset=True).items()}
        )
        combined_anomalies = test_anomalies
        combined_anomalies.sort()
        classification_results = self.compute_classification_metrics(
            combined_anomalies, pred_anomalies
        )
    
        corrected_results = self.compute_corrected_range_classification_metrics(
            classification_results,
            combined_anomalies,
            pred_anomalies,  # type: ignore[arg-type]
            total_length=len(y_pred),
        )
        classification_results.update(corrected_results)
        results.update(classification_results)
        
        test_anomalies_mask = np.zeros(len(y_pred), dtype=int)
        for start, end in test_anomalies:
            test_anomalies_mask[int(start) : int(end) + 1] = 1

        results.update(
            {
                "test_length": len(test_channel),
                "test_negatives": len(test_channel) - test_anomalies_mask.sum(),
                "detected_negatives": int(
                    ((y_pred == 0) & (test_anomalies_mask == 0)).sum()
                ),
            }
        )
        logging.info("Results for channel %s: %s", channel_id, results)

        self.all_results.append(results)
        pd.DataFrame.from_records(self.all_results).to_csv(
            os.path.join(self.run_dir, "results.csv"), index=False
        )
        
        if segment_indices is not None:
            if len(segment_indices) == 0:
                return results, [], []


            true_preds_intervals = [(segment_indices[s][0], segment_indices[e][1]) for s, e in pred_anomalies]
            true_anomaly_intervals = [(segment_indices[s][0], segment_indices[e][1]) for s, e in combined_anomalies]

            timestamps = getattr(original_test_channel, "timestamps", None)
            if timestamps is not None and len(timestamps) > 0:
                limit = len(timestamps)
                true_preds_intervals = [(timestamps[s], timestamps[e]) for s, e in true_preds_intervals if e < limit]
                true_anomaly_intervals = [(timestamps[s], timestamps[e]) for s, e in true_anomaly_intervals if e < limit]
            else:
                true_preds_intervals = []
                true_anomaly_intervals = []

            return results, true_anomaly_intervals, true_preds_intervals
        else:
            # Fallback if no segmentation mapping is available (e.g. raw point anomalies)
            return results, test_channel.anomalies, pred_anomalies_time


    def process_pred_anomalies(
        self, y_pred: np.ndarray, pred_buffer: int
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

    def compute_classification_metrics(self, true_anomalies, pred_anomalies):
        """Compute range-level classification metrics comparing true and predicted anomalies."""
        results = {
            "n_anomalies": len(true_anomalies),
            "n_detected": len(pred_anomalies),
            "true_positives": 0,
            "false_positives": 0,
            "false_negatives": 0,
        }

        matched_true_seqs = []
        true_indices_grouped = [list(range(e[0], e[1] + 1)) for e in true_anomalies]
        true_indices_flat = set([i for group in true_indices_grouped for i in group])
        for e_seq in pred_anomalies:
            i_anom_predicted = set(range(e_seq[0], e_seq[1] + 1))

            matched_indices = list(i_anom_predicted & true_indices_flat)
            valid = True if len(matched_indices) > 0 else False

            if valid:
                true_seq_index = [
                    i
                    for i in range(len(true_indices_grouped))
                    if len(
                        np.intersect1d(list(i_anom_predicted), true_indices_grouped[i])
                    )
                    > 0
                ]

                if not true_seq_index[0] in matched_true_seqs:
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
        return results

    def compute_corrected_range_classification_metrics(
        self,
        results: Dict[str, Any],
        true_anomalies: List[Tuple[int, int]],
        pred_anomalies: List[Tuple[int, int]],
        total_length: int,
    ) -> Dict[str, Any]:
        """Compute corrected range-level classification metrics.

        Adds TNR-corrected precision and corrected F-scores to the base
        classification results.

        Args:
            results (Dict[str, Any]): the base classification results
            true_anomalies (List[Tuple[int, int]]): the true anomaly intervals
            pred_anomalies (List[Tuple[int, int]]): the predicted anomaly intervals
            total_length (int): the total length of the sequence

        Returns:
            Dict[str, Any]: the corrected metrics results
        """
        corrected_results = {}
        indices_true_grouped = [list(range(e[0], e[1] + 1)) for e in true_anomalies]
        indices_true_flat = set([i for group in indices_true_grouped for i in group])
        indices_pred_grouped = [list(range(e[0], e[1] + 1)) for e in pred_anomalies]
        indices_pred_flat = set([i for group in indices_pred_grouped for i in group])
        indices_all_flat = indices_true_flat.union(indices_pred_flat)
        n_e = total_length - len(indices_true_flat)
        tn_e = total_length - len(indices_all_flat)
        for k, v in results.items():
            corrected_results[k] = v
        corrected_results["tnr"] = tn_e / n_e if n_e > 0 else 1
        corrected_results["precision_corrected"] = results["precision"] * corrected_results["tnr"]
        corrected_results["corrected_f0.5"] = (
            (
                (1 + 0.5**2)
                * (corrected_results["precision_corrected"] * results["recall"])
                / (0.5**2 * corrected_results["precision_corrected"] + results["recall"])
            )
            if corrected_results["precision_corrected"] + results["recall"] > 0
            else 0
        )
        corrected_results["corrected_f1"] = (
            (
                2
                * (corrected_results["precision_corrected"] * results["recall"])
                / (corrected_results["precision_corrected"] + results["recall"])
            )
            if corrected_results["precision_corrected"] + results["recall"] > 0
            else 0
        )
        return corrected_results

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

    def _aggregate_and_save_results(
        self,
        event_labels: List[Tuple[int, int]],
        predicted_events: List[Tuple[int, int]],
    ) -> Dict[str, Any]:
        """Aggregate event-level results and save them to disk.

        Args:
            event_labels (List[Tuple[int, int]]): List of true event intervals.
            predicted_events (List[Tuple[int, int]]): List of predicted event intervals.
        """

        event_labels = merge_intervals(event_labels)
        predicted_events = merge_intervals(predicted_events)
        
        classification_results = self.compute_classification_metrics(
            event_labels, predicted_events
        )

        total_length = 0
        if event_labels:
            total_length = max(total_length, event_labels[-1][1])
        if predicted_events:
            total_length = max(total_length, predicted_events[-1][1])
            
        corrected_results = self.compute_corrected_range_classification_metrics(
            classification_results,
            event_labels, 
            predicted_events,
            total_length=total_length + 1000 # Add buffer
        )
        
        global_results = {"channel_id": "GLOBAL_EVENT_LEVEL"}
        global_results.update(classification_results)
        global_results.update(corrected_results)
        
        logging.info("Global Event-Level Results: %s", global_results)

        self.all_results.append(global_results)
        pd.DataFrame.from_records(self.all_results).to_csv(
            os.path.join(self.run_dir, "results.csv"), index=False
        )

        return global_results    


    @property
    def run_dir(self) -> str:
        """Returns the directory where the results of this run are stored."""
        return os.path.join(self.exp_dir, self.run_id)
