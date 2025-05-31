import numpy as np
from typing import Optional
from pyod.models.base import BaseDetector
from sktime.transformations.panel.rocket import Rocket
from .anomaly_classifier import AnomalyClassifier

class RocketClassifier(AnomalyClassifier):
    """
    A wrapper that applies ROCKET transformation to time series data
    and fits a given classifier.
    """

    def __init__(self, base_model: BaseDetector, num_kernels: int = 100):

        self.base_model = base_model
        self.num_kernels = num_kernels
        self.rocket: Optional[Rocket] = None

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:

        X = self._prepare_input(X)
        self.rocket = Rocket(num_kernels=self.num_kernels, n_jobs=-1)
        features_train = self.rocket.fit_transform(X)

        if y is None:
            self.base_model.fit(features_train.values)
        else:
            self.base_model.fit(features_train.values, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.rocket is None:
            raise RuntimeError("You must call `fit` before `predict`.")

        X = self._prepare_input(X)
        features_test = self.rocket.transform(X)
        return self.base_model.predict(features_test.values)
