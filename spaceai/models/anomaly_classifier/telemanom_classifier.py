from __future__ import annotations

"""Abstract base class for anomaly classifiers."""

from abc import abstractmethod
from typing import TYPE_CHECKING, Optional, Any, List, Union, Dict

if TYPE_CHECKING:
    from spaceai.data import AnomalyDataset

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from spaceai.models.anomaly_classifier.anomaly_classifier import AnomalyClassifier
from spaceai.models.sequence_model import SequenceModel
from spaceai.benchmark.callbacks import CallbackHandler
from spaceai.models.anomaly import AnomalyDetector
from spaceai.data.utils import seq_collate_fn


class NumpyWindowDataset(Dataset):
    """Dataset for sliding window generation from numpy arrays."""
    def __init__(self, data: np.ndarray, window_size: int, n_predictions: int = 1):
        self.data = np.asarray(data, dtype=np.float32)
        self.window_size = window_size
        self.n_predictions = n_predictions

    def __len__(self):
        return max(0, len(self.data) - self.window_size - self.n_predictions + 1)

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.window_size]
        # Target y: the next n_predictions steps
        y = np.stack([
            self.data[idx + i + 1 : idx + self.window_size + i + 1, 0]
            for i in range(self.n_predictions)
        ], axis=1)
        return torch.tensor(x), torch.tensor(y).T


class TelemanomClassifier(AnomalyClassifier):
    """
    Abstract base for time-series wrappers: defines common interface and input preparation.
    """
    def __init__(self,
                predictor: SequenceModel,
                window_size: int = 250,
                n_predictions: int = 1,
                callback_handler: Optional[CallbackHandler] = None,
                detector: Optional[AnomalyDetector] = None,
                ):
        super().__init__(callback_handler=callback_handler)
        self.predictor = predictor
        self.window_size = window_size
        self.n_predictions = n_predictions
        self.detector = detector

    def fit( 
        self,
        channel_data: Union[np.ndarray, List[np.ndarray], AnomalyDataset],
        channel_labels: Optional[np.ndarray] = None,
        fit_predictor_args: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Fit the model on time-series data X, optionally with labels y.
        """
        results = {}
        batch_size = fit_predictor_args.pop("batch_size", 64) if fit_predictor_args else 64

        from spaceai.data import AnomalyDataset
        if isinstance(channel_data, AnomalyDataset):
            data_loader = DataLoader(
                channel_data,
                batch_size=batch_size,
                shuffle=True,
                collate_fn=seq_collate_fn(n_inputs=2, mode="batch"),
            )
        elif isinstance(channel_data, np.ndarray):
            dataset = NumpyWindowDataset(
                channel_data, 
                window_size=self.window_size, 
                n_predictions=self.n_predictions
            )
            data_loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                collate_fn=seq_collate_fn(n_inputs=2, mode="batch"),
            )
        else:
            raise ValueError(f"Unsupported data type: {type(channel_data)}")

        with self._callback_context("fitting", results):
            args_to_use = fit_predictor_args or {}
            self.predictor.fit(train_loader=data_loader, **args_to_use)

        return results

    @abstractmethod
    def predict(self, channel_data: Union[np.ndarray, List[np.ndarray], AnomalyDataset], batch_size: int = 64) -> np.ndarray:
        """
        Predict on time-series data X, returning a numpy array of outputs.
        """
        results = {}

        from spaceai.data import AnomalyDataset
        if isinstance(channel_data, AnomalyDataset):
            data_loader = DataLoader(
                channel_data,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=seq_collate_fn(n_inputs=2, mode="time"),
            )
            test_length = len(channel_data.data) # type: ignore
        elif isinstance(channel_data, np.ndarray):
            dataset = NumpyWindowDataset(
                channel_data, 
                window_size=self.window_size, 
                n_predictions=self.n_predictions
            )
            data_loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=seq_collate_fn(n_inputs=2, mode="time"),
            )
            test_length = len(channel_data)
        else:
            raise ValueError(f"Unsupported data type: {type(channel_data)}")

        with self._callback_context("prediction", results):
            self.predictor.model.eval()
            self.predictor.stateful = True
            
            y_preds = []
            for x, _ in data_loader:
                pred = self.predictor(x.to(self.predictor.device)).detach().cpu().squeeze().numpy()
                y_preds.append(pred)
                
            residuals = np.concatenate(y_preds)

        if self.detector is not None:
            with self._callback_context("detection", results):
                # Typically Telemanom detector requires both y_pred and y_true
                # For a wrapper, we might only pass residuals or we need to adjust detect()
                # Assuming detect handles the residuals produced.
                y_pred_anomaly = self.detector.detect(residuals)
            return y_pred_anomaly

        return residuals

    def save(self, path: str) -> None:
        """Save the classifier to disk."""
        torch.save(self, path)

    @staticmethod
    def load(path: str) -> "AnomalyClassifier":
        """Load a classifier from disk."""
        return torch.load(path, weights_only=False)

    @staticmethod
    def _prepare_input(X: np.ndarray) -> np.ndarray:  # pylint: disable=invalid-name
        """
        Ensure X is 3D with shape (n_samples, n_channels=1, n_timestamps).
        """
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("Input X must be 2D (n_samples, n_timestamps)")
        return X.reshape(X.shape[0], 1, X.shape[1])
