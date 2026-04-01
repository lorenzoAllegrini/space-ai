"""Rocket feature extractor module."""

from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
from sktime.transformations.panel.rocket import Rocket  # type: ignore

from .feature_extractor import FeatureExtractor


class RocketFeatureExtractor(FeatureExtractor):
    """
    Wrapper for Rocket to handle 2D input (n_samples, window_size)
    and convert it to 3D (n_samples, 1, window_size) for sktime.
    """

    def __init__(self, window_size: int, stride: int, num_kernels: int = 100, **kwargs):
        super().__init__(window_size, stride, **kwargs)
        self.num_kernels = num_kernels
        self.rocket = Rocket(num_kernels=num_kernels, n_jobs=1)

    @property
    def output_dim(self) -> int:
        return 2 * self.num_kernels

    def fit(
        self, *messages: "PipelineMessage"
    ) -> "RocketFeatureExtractor":
        """Fit the Rocket transformer using the first message."""
        if not messages:
            return self
        message = messages[0]
        results = message.results
        X = message.data
        with self._callback_context("feature_extraction_fit", results):
            X_prep = self._prepare_input(X)
            self.rocket.fit(X_prep)
        return self

    def transform(
        self, 
        message: "PipelineMessage"
    ) -> "PipelineMessage":
        """Transform the data in the message."""
        results = message.results
        save_dir = message.save_dir
        suffix = message.split_label
        X_segments = message.data

        with self._callback_context("feature_extraction", results):
            X_prep = self._prepare_input(X_segments)
            X_transformed = self.rocket.transform(X_prep)
            
            # sktime returns a pandas DataFrame for Rocket
            columns = [f"rocket_{i}" for i in range(X_transformed.shape[1])]
            df = pd.DataFrame(X_transformed, columns=columns)
            
            if save_dir:
                import os
                os.makedirs(save_dir, exist_ok=True)
                filename = "rocket_features"
                if suffix:
                    filename += f"_{suffix}"
                save_path = os.path.join(save_dir, f"{filename}.csv")
                df.to_csv(save_path, index=False)

        message.data = df
        return message

    def _prepare_input(  # pylint: disable=invalid-name
        self, X: np.ndarray
    ) -> np.ndarray:
        """Ensure X is 3D with shape (n_samples, n_channels=1, n_timestamps)."""
        X_arr = np.asarray(X)
        if X_arr.ndim != 2:
            # Se è già 3D o altro, proviamo a reshaperlo se ha senso
            if X_arr.ndim == 3 and X_arr.shape[1] == 1:
                return X_arr
            raise ValueError(f"Input X must be 2D (n_samples, n_timestamps), got shape {X_arr.shape}")
        return X_arr.reshape(X_arr.shape[0], 1, X_arr.shape[1])
