"""ROCKAD anomaly classifier module."""

from typing import (
    Any,
    Optional,
)

import numpy as np
import pandas as pd  # type: ignore
from sklearn.metrics.pairwise import distance_metrics  # type: ignore
from sklearn.neighbors import NearestNeighbors  # type: ignore
from sklearn.preprocessing import (  # type: ignore
    PowerTransformer,
    StandardScaler,
)
from sklearn.utils import resample  # type: ignore
from sktime.transformations.panel.rocket import Rocket  # type: ignore

from .anomaly_classifier import AnomalyClassifier


class NearestNeighborOCC:
    """Nearest Neighbor One-Class Classifier."""

    def __init__(self, dist="euclidean"):
        self.scores_train = None
        self.dist = None

        metrics = distance_metrics()

        if isinstance(dist, str) and dist in metrics:
            self.dist = metrics[dist]
        elif dist in metrics.values():
            self.dist = dist
        else:
            raise ValueError("Distance metric not supported.")

    def fit(self, scores_train):
        """Fit the model."""
        _scores_train = scores_train

        if not isinstance(_scores_train, np.ndarray):
            _scores_train = np.array(scores_train.copy())

        if len(_scores_train.shape) == 1:
            _scores_train = _scores_train.reshape(-1, 1)

        self.scores_train = _scores_train

        return self

    def predict(self, scores_test):
        """
        Per definition (see [1]): 0 indicates an anomaly, 1 indicates normal.
        Here : -1 indicates an anomaly, 1 indicates normal.
        """
        predictions = []
        for score in scores_test:
            predictions.append(self.predict_score(score))
        return np.array(predictions)

    def predict_score(self, anomaly_score):
        """Predict the anomaly score."""
        prediction = None

        anomaly_score_arr = np.array(
            [anomaly_score for i in range(len(self.scores_train))]
        )

        _scores_train = self.scores_train.copy().reshape(-1, 1)
        anomaly_score_arr = anomaly_score_arr.reshape(-1, 1)
        nearest_neighbor_idx = np.argmin(self.dist(anomaly_score_arr, _scores_train))

        _scores_train = np.delete(_scores_train, nearest_neighbor_idx).reshape(-1, 1)

        nearest_neighbor_score = self.scores_train[nearest_neighbor_idx]
        neares_neighbot_score_arr = np.array(
            [nearest_neighbor_score for i in range(len(_scores_train))]
        )
        nearest_neighbor_score_arr = neares_neighbot_score_arr.reshape(-1, 1)
        nearest_nearest_neighbor_idx = np.argmin(
            self.dist(nearest_neighbor_score_arr, _scores_train)
        )
        nearest_nearest_neighbor_score = _scores_train[nearest_nearest_neighbor_idx]

        prediction = self.indicator_function(
            anomaly_score, nearest_neighbor_score, nearest_nearest_neighbor_score
        )

        return prediction

    def indicator_function(self, z_score, nearest_score, nearest_of_nearest_score):
        """Indicator function for anomaly detection."""

        # make it an array and reshape it to calculate the distance
        z_score_arr = np.array(z_score).reshape(1, -1)
        nearest_score_arr = np.array(nearest_score).reshape(1, -1)
        nearest_of_nearest_score_arr = np.array(nearest_of_nearest_score).reshape(1, -1)

        numerator = self.dist(z_score_arr, nearest_score_arr)
        denominator = self.dist(nearest_score_arr, nearest_of_nearest_score_arr)

        # error handling for corner cases
        if numerator == 0:
            return 1
        if denominator == 0:
            return -1
        return 1 if (numerator / denominator) <= 1 else -1


class NNEstimator:  # Renamed to avoid conflict with NN import
    """Nearest Neighbor Estimator."""

    def __init__(
        self,
        n_neighbors=5,
        n_jobs=1,
        dist="euclidean",
        random_state=42,
    ) -> None:

        self.n_neighbors = n_neighbors
        self.n_jobs = n_jobs
        self.dist = dist
        self.random_state = random_state
        self.nn = None

    def fit(self, x):
        """Fit the estimator."""
        self.nn = NearestNeighbors(
            n_neighbors=self.n_neighbors,
            n_jobs=self.n_jobs,
            metric=self.dist,
            algorithm="ball_tree",
        )

        self.nn.fit(x)

    def predict_proba(self, x, _y=None):
        """Predict class probabilities."""
        scores = self.nn.kneighbors(x)
        scores = scores[0].mean(axis=1).reshape(-1, 1)

        return scores


class ROCKAD:
    """ROCKAD (Rocket-based Anomaly Detection) classifier."""

    def __init__(
        self,
        n_estimators=10,
        n_kernels=100,
        n_neighbors=4,
        n_jobs=1,
        power_transform=True,
        random_state=42,
    ) -> None:
        self.random_state = random_state
        self.power_transform = power_transform

        self.n_estimators = n_estimators
        self.n_kernels = n_kernels
        self.n_neighbors = n_neighbors
        self.n_jobs = n_jobs
        self.n_inf_cols: list[int] = []
        self.x_transformed_power = None
        self.list_baggers: list[Any] = []

        self.estimator = NNEstimator
        self.rocket_transformer = Rocket(
            num_kernels=self.n_kernels,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
        )
        self.scaler = StandardScaler()
        self.power_transformer = PowerTransformer(standardize=False)

    def init(self, x):
        """Initialize the model with data."""

        # Fit Rocket & Transform into rocket feature space
        x_transformed = self.rocket_transformer.fit_transform(x)

        self.x_transformed_power = (
            None  # X: values, t: (rocket) transformed, p: power transformed
        )

        if self.power_transform is True:

            x_transformed_power = self.power_transformer.fit_transform(x_transformed)

            self.x_transformed_power = pd.DataFrame(x_transformed_power)

        else:
            self.x_transformed_power = pd.DataFrame(x_transformed)

    def fit_estimators(self):
        """Fit the ensemble of estimators."""

        x_transformed_power_scaled = None

        if self.power_transform is True:
            # Check for infinite columns and get indices
            self._check_inf_values(self.x_transformed_power)

            # Remove infinite columns
            self.x_transformed_power = self.x_transformed_power[
                self.x_transformed_power.columns[
                    ~self.x_transformed_power.columns.isin(self.n_inf_cols)
                ]
            ]

            # Fit Scaler
            x_transformed_power_scaled = self.scaler.fit_transform(
                self.x_transformed_power
            )

            x_transformed_power_scaled = pd.DataFrame(
                x_transformed_power_scaled, columns=self.x_transformed_power.columns
            )

            self._check_inf_values(x_transformed_power_scaled)

            x_transformed_power_scaled = x_transformed_power_scaled.astype(
                np.float32
            ).to_numpy()

        else:
            x_transformed_power_scaled = self.x_transformed_power.astype(
                np.float32
            ).to_numpy()

        self.list_baggers = []

        for idx_estimator in range(self.n_estimators):
            # Initialize estimator
            estimator = self.estimator(
                n_neighbors=self.n_neighbors,
                n_jobs=self.n_jobs,
            )

            # Bootstrap Aggregation
            x_transformed_power_scaled_sample = resample(
                x_transformed_power_scaled,
                replace=True,
                n_samples=None,
                random_state=self.random_state + idx_estimator,
                stratify=None,
            )
            # Fit estimator and append to estimator list
            estimator.fit(x_transformed_power_scaled_sample)
            self.list_baggers.append(estimator)

    def fit(self, x):
        """Fit the model."""
        self.init(x)
        self.fit_estimators()

        return self

    def predict_proba(self, x):
        """Predict class probabilities."""
        y_scores = np.zeros((len(x), self.n_estimators))

        # Transform into rocket feature space
        x_transformed = self.rocket_transformer.transform(x)

        x_transformed_power_scaled = None

        if self.power_transform:
            # Power Transform using yeo-johnson
            x_transformed_power = self.power_transformer.transform(x_transformed)
            x_transformed_power = pd.DataFrame(x_transformed_power)

            # Check for infinite columns and remove them
            self._check_inf_values(x_transformed_power)
            x_transformed_power = x_transformed_power[
                x_transformed_power.columns[
                    ~x_transformed_power.columns.isin(self.n_inf_cols)
                ]
            ]
            x_transformed_power_temp = x_transformed_power.copy()

            # Scale the data
            x_transformed_power_scaled = self.scaler.transform(x_transformed_power_temp)
            x_transformed_power_scaled = pd.DataFrame(
                x_transformed_power_scaled, columns=x_transformed_power_temp.columns
            )

            # Check for infinite columns and remove them
            self._check_inf_values(x_transformed_power_scaled)
            x_transformed_power_scaled = x_transformed_power_scaled[
                x_transformed_power_scaled.columns[
                    ~x_transformed_power_scaled.columns.isin(self.n_inf_cols)
                ]
            ]
            x_transformed_power_scaled = x_transformed_power_scaled.astype(
                np.float32
            ).to_numpy()

        else:
            x_transformed_power_scaled = x_transformed.astype(np.float32)

        for idx, bagger in enumerate(self.list_baggers):
            # Get scores from each estimator
            scores = bagger.predict_proba(x_transformed_power_scaled).squeeze()

            y_scores[:, idx] = scores

        # Average the scores to get the final score for each time series
        y_scores = y_scores.mean(axis=1)

        return y_scores

    def _check_inf_values(self, x):
        """Check for infinite values in the data."""
        if np.isinf(x[x.columns[~x.columns.isin(self.n_inf_cols)]]).any(axis=0).any():
            self.n_inf_cols.extend(x.columns.to_series()[np.isinf(x).any()])
            self.fit_estimators()
            return True
        return False


class RockadClassifier(AnomalyClassifier):
    """
    A fully unsupervised wrapper: costruisce un ensemble ROCKAD su X,
    poi allena un OCC sui punteggi di anomalia.
    """

    def __init__(
        self,
        base_model: Any = NearestNeighborOCC,
        num_kernels: int = 10000,
        n_estimators: int = 100,
    ):
        self.base_model = base_model
        self.num_kernels = num_kernels
        self.n_estimators = n_estimators
        self.rockad: Optional[ROCKAD] = None
        self.oc_model: Optional[Any] = None

    def fit(self, X: np.ndarray, y=None) -> None:  # pylint: disable=invalid-name
        """
        1) Applica ROCKAD su X a scapito di y.
        2) Prende i punteggi di anomalia e allena il one‐class model.
        """
        x_proc = self._prepare_input(X)

        # 1) Fit del solo ensemble ROCKAD (unsupervised)
        self.rockad = ROCKAD(
            n_estimators=self.n_estimators,
            n_kernels=self.num_kernels,
            n_jobs=1,
            power_transform=False,
        )
        # rockad.fit si aspetta solo X
        self.rockad.fit(x_proc)

    def predict(self, X: np.ndarray) -> np.ndarray:  # pylint: disable=invalid-name
        """
        Restituisce 1=normale, 0=anomalia, basandosi sul modello one‐class.
        """
        if self.rockad is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        x_proc = self._prepare_input(X)
        raw_scores = self.rockad.predict_proba(x_proc)

        base_model = self.base_model().fit(raw_scores)
        pred = base_model.predict(raw_scores)
        return pred
