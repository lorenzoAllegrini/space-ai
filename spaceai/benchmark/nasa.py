from __future__ import annotations

import os
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Tuple,
)

import numpy as np
import pandas as pd
from torch.utils.data import (
    DataLoader,
    Subset,
)
from tqdm import tqdm

from spaceai.data import NASA
from spaceai.data.utils import seq_collate_fn

from .callbacks import CallbackHandler

if TYPE_CHECKING:
    from spaceai.models.predictors import SequenceModel
    from spaceai.models.anomaly import AnomalyDetector
    from .callbacks import Callback

import logging

import more_itertools as mit
from .benchmark import Benchmark
from spaceai.segmentators.nasa_segmentator import NasaDatasetSegmentator


class NASABenchmark(Benchmark):

    def __init__(
        self,
        run_id: str,
        exp_dir: str,
        segmentator: Optional[NasaDatasetSegmentator] = None,
        seq_length: int = 250,
        n_predictions: int = 1,
        data_root: str = "data/nasa",
    ):
        """Initializes a new NASA benchmark run.

        Args:
            run_id (str): A unique identifier for this run.
            exp_dir (str): The directory where the results of this run are stored.
            seq_length (int): The length of the sequences used for training and testing.
            data_root (str): The root directory of the NASA dataset.
        """
        super().__init__(run_id, exp_dir)
        self.data_root: str = data_root
        self.seq_length: int = seq_length
        self.n_predictions: int = n_predictions
        self.all_results: List[Dict[str, Any]] = []
        self.segmentator: NasaDatasetSegmentator = segmentator 

    def run(
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
    ):
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
            logging.info(f"Restoring predictor for channel {channel_id}...")
            predictor.load(os.path.join(self.run_dir, f"predictor-{channel_id}.pt"))

        elif fit_predictor_args is not None:
            logging.info(f"Fitting the predictor for channel {channel_id}...")
            # Training the predictor
            batch_size = fit_predictor_args.pop("batch_size", 64)
            eval_channel = None
            if perc_eval is not None:
                # Split the training data into training and evaluation sets
                indices = np.arange(len(train_channel))
                np.random.shuffle(indices)
                eval_size = int(len(train_channel) * perc_eval)
                eval_channel = Subset(train_channel, indices[:eval_size])
                train_channel = Subset(train_channel, indices[eval_size:])
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
                f"Training time on channel {channel_id}: {results['train_time']}"
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
        logging.info(f"Predicting the test data for channel {channel_id}...")
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
        logging.info(f"Test loss for channel {channel_id}: {results['test_loss']}")
        logging.info(
            f"Prediction time for channel {channel_id}: {results['predict_time']}"
        )

        # Testing the detector
        logging.info(f"Detecting anomalies for channel {channel_id}")
        callback_handler.start()
        if len(y_trg) < 2500:
            detector.ignore_first_n_factor = 1
        if len(y_trg) < 1800:
            detector.ignore_first_n_factor = 0
        pred_anomalies = detector.detect_anomalies(y_pred, y_trg)
        pred_anomalies += detector.flush_detector()
        callback_handler.stop()
        results.update(
            {f"detect_{k}": v for k, v in callback_handler.collect(reset=True).items()}
        )
        logging.info(
            f"Detection time for channel {channel_id}: {results['detect_time']}"
        )

        true_anomalies = test_channel.anomalies
        classification_results = self.compute_classification_metrics(
            true_anomalies, pred_anomalies
        )
        results.update(classification_results)
        if train_history is not None:
            results["train_loss"] = train_history[-1]["loss_train"]
            if eval_loader is not None:
                results["eval_loss"] = train_history[-1]["loss_eval"]

        logging.info(f"Results for channel {channel_id}")

        self.all_results.append(results)

        pd.DataFrame.from_records(self.all_results).to_csv(
            os.path.join(self.run_dir, "results.csv"), index=False
        )

    def run_classifier(
        self,
        channel_id: str,
        classifier,
        overlapping_train: Optional[bool] = True,
        callbacks: Optional[List[Callback]] = None,
        call_every_ms: Optional[int] = 100,
    ):
        """Esegue il benchmark diretto per classificazione di anomalie su segmenti.

        Args:
            channel_id (str): ID del canale da testare
            classifier (AnomalyClassifier): classificatore supervisionato
            fit_classifier_args (dict): argomenti addizionali per il fit del classificatore
            perc_eval (float): percentuale dei segmenti da usare per la validazione
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

        callback_handler.start()
        if self.segmentator is not None:
            train_channel, _ = self.segmentator.segment(train_channel)
        

        logging.info(f"Fitting the classifier for chaggdnnel {channel_id}...")

        classifier.fit(train_channel) 
        callback_handler.stop()
        results.update(
            {
                f"train_{k}": v
                for k, v in callback_handler.collect(reset=True).items()
            }
        )
        callback_handler.start()
        if self.segmentator is not None:
            test_channel, test_anomalies = self.segmentator.segment(test_channel)

        y_pred = classifier.predict(test_channel)
        pred_anomalies = np.where(y_pred == 1)[0]

        if len(pred_anomalies) > 0:
            groups = [list(group) for group in mit.consecutive_groups(pred_anomalies)]
            pred_anomalies = [
                [int(group[0]), int(group[-1])]
                for group in groups
            ]
        else:
            pred_anomalies = []

        callback_handler.stop()
        results.update(
            {f"predict_{k}": v for k, v in callback_handler.collect(reset=True).items()}
        )

        classification_results = self.compute_classification_metrics(
            test_anomalies, pred_anomalies
        )
        classification_results = self.compute_corrected_classification_metrics(
            classification_results, test_anomalies, pred_anomalies, total_length=len(y_pred)
        )
        print(f"pred_anomalies: {pred_anomalies}")
        print(f"true_anomalies: {test_anomalies}")
        print(f"precision: {classification_results['precision_corrected']}")
        print(f"recall: {classification_results['recall']}")

        print("---------------------------------------------")
        results.update(classification_results)

        logging.info(f"Results for channel {channel_id}")

        self.all_results.append(results)

        pd.DataFrame.from_records(self.all_results).to_csv(
            os.path.join(self.run_dir, "results.csv"), index=False
        )

    def load_channel(
        self, channel_id: str, overlapping_train: bool = True
    ) -> Tuple[NASA, NASA]:
        """Load the training and testing datasets for a given channel.

        Args:
            channel_id (str): the ID of the channel to be used
            overlapping_train (bool): whether to use overlapping sequences for training

        Returns:
            Tuple[NASA, NASA]: training and testing datasets
        """
        train_channel = NASA(
            root=self.data_root,
            channel_id=channel_id,
            mode="prediction",
            overlapping=overlapping_train,
            seq_length=self.seq_length,
            n_predictions=self.n_predictions,
        )

        test_channel = NASA(
            root=self.data_root,
            channel_id=channel_id,
            mode="anomaly",
            overlapping=False,
            seq_length=self.seq_length,
            train=False,
            drop_last=False,
            n_predictions=1,
        )

        return train_channel, test_channel

    def compute_classification_metrics(self, true_anomalies, pred_anomalies):
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

                (results["precision"] * results["recall"])
                / (results["precision"] + results["recall"])
            )
            if results["precision"] + results["recall"] > 0
            else 0
        )
        return results
    
    def compute_corrected_classification_metrics(
        self,
        results: Dict[str, Any],
        true_anomalies: List[Tuple[int, int]],
        pred_anomalies: List[Tuple[int, int]],
        total_length: int,
    ) -> Dict[str, Any]:
        """Compute ESA classification metrics.

        Args:
            results (Dict[str, Any]): the classification results
            true_anomalies (List[Tuple[int, int]]): the true anomalies
            pred_anomalies (List[Tuple[int, int]]): the predicted anomalies
            total_length (int): the total length of the sequence

        Returns:
            Dict[str, Any]: the ESA metrics results
        """
        esa_results = {}
        indices_true_grouped = [list(range(e[0], e[1] + 1)) for e in true_anomalies]
        indices_true_flat = set([i for group in indices_true_grouped for i in group])
        indices_pred_grouped = [list(range(e[0], e[1] + 1)) for e in pred_anomalies]
        indices_pred_flat = set([i for group in indices_pred_grouped for i in group])
        indices_all_flat = indices_true_flat.union(indices_pred_flat)
        n_e = total_length - len(indices_true_flat)
        tn_e = total_length - len(indices_all_flat)
        for k,v in results.items():
            esa_results[k] = v
        esa_results["tnr"] = tn_e / n_e if n_e > 0 else 1
        esa_results["precision_corrected"] = results["precision"] * esa_results["tnr"]
        esa_results["corrected_f0.5"] = (
            (
                (1 + 0.5**2)
                * (esa_results["precision_corrected"] * results["recall"])
                / (0.5**2 * esa_results["precision_corrected"] + results["recall"])
            )
            if esa_results["precision_corrected"] + results["recall"] > 0
            else 0
        )
        esa_results["corrected_f1"] = (
            (
              
                (esa_results["precision_corrected"] * results["recall"])
                / (esa_results["precision_corrected"] + results["recall"])
            )
            if esa_results["precision_corrected"] + results["recall"] > 0
            else 0
        )
        
        return esa_results
    
    
