from __future__ import annotations

"""Abstract base class for anomaly classifiers."""

from abc import abstractmethod
from typing import TYPE_CHECKING, Optional, Any, List, Union, Tuple, Dict, Callable
from contextlib import contextmanager

if TYPE_CHECKING:
    from spaceai.data import AnomalyDataset

import numpy as np
import pandas as pd
import torch
from spaceai.data import AnomalyDataset

from spaceai.models.classifiers.base import BaseClassifier
from spaceai.models.predictors.seq_model import SequenceModel
from spaceai.benchmark.callbacks import CallbackHandler
from spaceai.models.detectors import AnomalyDetector

from torch.utils.data import TensorDataset, DataLoader, Subset
from spaceai.data.utils import seq_collate_fn

class SequenceModelClassifier(BaseClassifier):
    """
    Abstract base for time-series wrappers: defines common interface and input preparation.
    """

    def __init__(
        self,
        predictor: SequenceModel,
        callback_handler: Optional[CallbackHandler] = None,
        detector: Optional[AnomalyDetector] = None,
        fit_predictor_args: Dict[str, Any] = {},
        scale_data: bool = True,
        n_predictions: int = 1
    ):
        super().__init__(callback_handler=callback_handler)
        self.predictor = predictor
        self.detector = detector
        self.fit_predictor_args = fit_predictor_args
        self.n_predictions = n_predictions
        self.scale_data = scale_data
        if self.scale_data:
            from sklearn.preprocessing import MinMaxScaler
            self.scaler = MinMaxScaler()

    def _scale_input(self, channel_data: Union[np.ndarray, List[np.ndarray], AnomalyDataset], fit: bool = False) -> Union[np.ndarray, List[np.ndarray], AnomalyDataset]:
        if not self.scale_data or isinstance(channel_data, AnomalyDataset):
            return channel_data
        
        if isinstance(channel_data, list):
            # Assumes list of 2D windows
            X_arr = np.array(channel_data)
            orig_shape = X_arr.shape
            X_2d = X_arr.reshape(-1, orig_shape[-1])
            if fit:
                X_2d = self.scaler.fit_transform(X_2d)
            else:
                X_2d = self.scaler.transform(X_2d)
            scaled_arr = X_2d.reshape(orig_shape)
            return [torch.from_numpy(x).float() for x in scaled_arr]
        elif isinstance(channel_data, np.ndarray):
            orig_shape = channel_data.shape
            X_2d = channel_data.reshape(-1, orig_shape[-1])
            if fit:
                X_2d = self.scaler.fit_transform(X_2d)
            else:
                X_2d = self.scaler.transform(X_2d)
            scaled_arr = X_2d.reshape(orig_shape)
            return torch.from_numpy(scaled_arr).float()
        return channel_data

    def fit(
        self,
        channel_data: Union[np.ndarray, List[np.ndarray], AnomalyDataset],
        channel_labels: Optional[np.ndarray] = None,
        results: Optional[Dict[str, Any]] = None,
        results_dir: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Fit the model on time-series data X, optionally with labels y.
        """
        results = results if results is not None else {}
        with self._callback_context("classifier_fit", results):
            scaled_data = self._scale_input(channel_data, fit=True)
            channel_loader, fit_predictor_args = self._prepare_fit_input(
                scaled_data, self.fit_predictor_args)

            with self._callback_context("train_", results):
                self.predictor.fit(
                    train_loader=channel_loader,
                    **fit_predictor_args,
                )
            self.is_fitted_ = True
            return results

    def predict(
        self,
        channel_data: Union[np.ndarray, List[np.ndarray], AnomalyDataset],
        results: Optional[Dict[str, Any]] = None,
        results_dir: Optional[str] = None,
        **test_predictor_args
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        results = results if results is not None else {}

        with self._callback_context("classifier_predict", results):
            scaled_data = self._scale_input(channel_data, fit=False)
            test_loader, test_predictor_args = self._prepare_predict_input(
                scaled_data, test_predictor_args)

            # Try to get window_size from the dataset, then from the predictor, finally default to 250
            window_size = 250
            if hasattr(channel_data, "window_size"):
                window_size = channel_data.window_size
            elif hasattr(self.predictor, "window_size"):
                window_size = self.predictor.window_size

            with self._callback_context("predict_", results):
                self.predictor.model.eval()
                self.predictor.stateful = True

                all_y_pred = []
                all_y_trg = []

                with torch.no_grad():
                    for x, y in test_loader:
                        x = x.to(self.predictor.device)
                        pred = self.predictor(
                            x).detach().cpu().squeeze().numpy()
                        trg = y.detach().cpu().squeeze().numpy()

                        all_y_pred.append(pred)
                        all_y_trg.append(trg)

                y_pred_arr = np.concatenate(all_y_pred)[window_size - 1:]
                y_trg_arr = np.concatenate(all_y_trg)[window_size - 1:]
                # If targets have multiple dimensions, take the first one for detection (the last dimension is used for repeating future datapoints like [[1, 2, 3], [2, 3, 4], ...])
                if y_trg_arr.ndim > 1:
                    y_trg_arr = y_trg_arr[:, 0]

            with self._callback_context("detect_", results):
                if len(y_trg_arr) < 2500:
                    self.detector.ignore_first_n_factor = 1
                if len(y_trg_arr) < 1800:
                    self.detector.ignore_first_n_factor = 0

                pred_anomalies_intervals = self.detector.detect_anomalies(
                    y_pred_arr, y_trg_arr)
                pred_anomalies_intervals += self.detector.flush_detector()
                self.detector.reset_state()

            anomaly_mask = np.zeros(len(y_pred_arr), dtype=int)
            for start, end in pred_anomalies_intervals:
                anomaly_mask[int(start): int(end) + 1] = 1

        return anomaly_mask, results

    def map_to_timestamps(self, channel_data: Union[np.ndarray, List[np.ndarray], AnomalyDataset], anomalies: List[Tuple[int, int]]) -> List[Tuple[Any, Any]]:
        has_timestamps = hasattr(channel_data, "timestamps") and channel_data.timestamps is not None and len(
            channel_data.timestamps) > 0

        limit = float('inf')
        if has_timestamps:
            limit = len(channel_data.timestamps)
        elif hasattr(channel_data, "data") and channel_data.data is not None:
            limit = len(channel_data.data)
        elif isinstance(channel_data, np.ndarray):
            limit = len(channel_data)

        offset = getattr(channel_data, "start_idx", 0)

        time_intervals = []
        for s, e in anomalies:
            if s >= limit:
                continue
            if e >= limit:
                e = limit - 1
            if has_timestamps:
                time_intervals.append(
                    (channel_data.timestamps[s], channel_data.timestamps[e]))
            else:
                time_intervals.append((s + offset, e + offset))

        return time_intervals

    def prepare_labels(self, channel_labels: Any) -> List[Tuple[int, int]]:
        """Prepare ground truth labels as interval tuples."""
        if hasattr(channel_labels, 'anomaly_sequences'):
            return channel_labels.anomaly_sequences
        return []

    def save(self, path: str) -> None:
        """Save the classifier to disk."""
        torch.save(self, path)

    @staticmethod
    def load(path: str) -> "SequenceModelClassifier":
        """Load a classifier from disk."""
        return torch.load(path, weights_only=False)

    def _prepare_fit_input(
        self,
        channel_data: Union[np.ndarray, List[np.ndarray], AnomalyDataset],
        fit_predictor_args: Optional[Dict[str, Any]] = None
    ) -> Tuple[DataLoader, Optional[Dict[str, Any]]]:  # pylint: disable=invalid-name

        if fit_predictor_args is None:
            fit_predictor_args = {}

        batch_size = fit_predictor_args.pop("batch_size", 32)
        perc_eval = fit_predictor_args.pop("perc_eval", None)

        # If data is not a dataset, we assume it's a collection of windows (N, W, F) or (N, W)
        # We need to split them into (X, Y) where Y is the last n_predictions points
        if not isinstance(channel_data, AnomalyDataset):
            if isinstance(channel_data, list):
                # Handle list of tensors from _scale_input
                X_tensor = torch.stack(channel_data)
            elif isinstance(channel_data, torch.Tensor):
                X_tensor = channel_data
            else:
                X_tensor = torch.from_numpy(np.array(channel_data)).float()
            
            if X_tensor.ndim == 2:
                X_tensor = X_tensor.unsqueeze(-1) # Add feature dim
            
            # Split windows into input and target
            X = X_tensor[:, :-self.n_predictions, :]
            Y = X_tensor[:, -self.n_predictions:, :]
            channel_data = TensorDataset(X, Y)

        if perc_eval is not None:
            eval_size = int(len(channel_data) * perc_eval)
            eval_indices = np.arange(eval_size)
            train_indices = np.arange(eval_size, len(channel_data))
            eval_channel = Subset(channel_data, eval_indices.tolist())
            channel_data = Subset(channel_data, train_indices.tolist())
            eval_loader = DataLoader(
                eval_channel,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=seq_collate_fn(n_inputs=2, mode="batch"),
            )
        else:
            eval_loader = None

        channel_loader = DataLoader(
            channel_data,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=seq_collate_fn(n_inputs=2, mode="batch"),
        )

        fit_predictor_args["valid_loader"] = eval_loader

        return channel_loader, fit_predictor_args

    def _prepare_predict_input(
        self,
        channel_data: Union[np.ndarray, List[np.ndarray], AnomalyDataset],
        test_predictor_args: Optional[Dict[str, Any]] = None
    ) -> Tuple[DataLoader, Optional[Dict[str, Any]]]:  # pylint: disable=invalid-name

        if not isinstance(channel_data, AnomalyDataset):
            if isinstance(channel_data, list):
                X_tensor = torch.stack(channel_data)
            elif isinstance(channel_data, torch.Tensor):
                X_tensor = channel_data
            else:
                X_tensor = torch.from_numpy(np.array(channel_data)).float()
            
            if X_tensor.ndim == 2:
                X_tensor = X_tensor.unsqueeze(-1)
            
            X = X_tensor[:, :-self.n_predictions, :]
            Y = X_tensor[:, -self.n_predictions:, :]
            channel_data = TensorDataset(X, Y)

        test_loader = DataLoader(
            channel_data,
            batch_size=1,
            shuffle=False,
            collate_fn=seq_collate_fn(n_inputs=2, mode="time"),
        )
        return test_loader, {}
