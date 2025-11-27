from typing import (
    Any,
    Optional,
)

import numpy as np
import pandas as pd  # type: ignore
from sklearn.metrics.pairwise import (  # type: ignore
    distance_metrics,
    euclidean_distances,
)
from sklearn.neighbors import NearestNeighbors as NN  # type: ignore
from sklearn.preprocessing import (  # type: ignore
    PowerTransformer,
    StandardScaler,
)
from sklearn.utils import resample  # type: ignore
from sktime.transformations.panel.rocket import Rocket  # type: ignore

from .anomaly_classifier import AnomalyClassifier


class NearestNeighborOCC:

    def __init__(self, dist="euclidean"):
        self.scores_train = None
        self.dist = None

        metrics = distance_metrics()

        if type(dist) is str and dist in metrics.keys():
            self.dist = metrics[dist]
        elif dist in metrics.values():
            self.dist = dist
        elif False:
            # TODO: allow time series distance measures such as DTW or Matrix Profile
            pass
        else:
            raise Exception("Distance metric not supported.")

    def fit(self, scores_train):
        _scores_train = scores_train

        if type(_scores_train) is not np.array:
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

        # make it an array and reshape it to calculate the distance
        z_score_arr = np.array(z_score).reshape(1, -1)
        nearest_score_arr = np.array(nearest_score).reshape(1, -1)
        nearest_of_nearest_score_arr = np.array(nearest_of_nearest_score).reshape(1, -1)

        numerator = self.dist(z_score_arr, nearest_score_arr)
        denominator = self.dist(nearest_score_arr, nearest_of_nearest_score_arr)

        # error handling for corner cases
        if numerator == 0:
            return 1
        elif denominator == 0:
            return -1
        else:
            return 1 if (numerator / denominator) <= 1 else -1


class NNEstimator:  # Renamed to avoid conflict with NN import

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

    def fit(self, X):
        self.nn = NearestNeighbors(
            n_neighbors=self.n_neighbors,
            n_jobs=self.n_jobs,
            metric=self.dist,
            algorithm="ball_tree",
        )

        self.nn.fit(X)

    def predict_proba(self, X, y=None):
        scores = self.nn.kneighbors(X)
        scores = scores[0].mean(axis=1).reshape(-1, 1)

        return scores


class ROCKAD:

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

        self.estimator = NNEstimator
        self.rocket_transformer = Rocket(
            num_kernels=self.n_kernels,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
        )
        self.scaler = StandardScaler()
        self.power_transformer = PowerTransformer(standardize=False)

    def init(self, X):

        # Fit Rocket & Transform into rocket feature space
        Xt = self.rocket_transformer.fit_transform(X)

        self.Xtp = None  # X: values, t: (rocket) transformed, p: power transformed

        if self.power_transform is True:

            Xtp = self.power_transformer.fit_transform(Xt)

            self.Xtp = pd.DataFrame(Xtp)

        else:
            self.Xtp = pd.DataFrame(Xt)

    def fit_estimators(self):

        Xtp_scaled = None

        if self.power_transform is True:
            # Check for infinite columns and get indices
            self._check_inf_values(self.Xtp)

            # Remove infinite columns
            self.Xtp = self.Xtp[
                self.Xtp.columns[~self.Xtp.columns.isin(self.n_inf_cols)]
            ]

            # Fit Scaler
            Xtp_scaled = self.scaler.fit_transform(self.Xtp)

            Xtp_scaled = pd.DataFrame(Xtp_scaled, columns=self.Xtp.columns)

            self._check_inf_values(Xtp_scaled)

            Xtp_scaled = Xtp_scaled.astype(np.float32).to_numpy()

        else:
            Xtp_scaled = self.Xtp.astype(np.float32).to_numpy()

        self.list_baggers = []

        for idx_estimator in range(self.n_estimators):
            # Initialize estimator
            estimator = self.estimator(
                n_neighbors=self.n_neighbors,
                n_jobs=self.n_jobs,
            )

            # Bootstrap Aggregation
            Xtp_scaled_sample = resample(
                Xtp_scaled,
                replace=True,
                n_samples=None,
                random_state=self.random_state + idx_estimator,
                stratify=None,
            )
            # Fit estimator and append to estimator list
            estimator.fit(Xtp_scaled_sample)
            self.list_baggers.append(estimator)

    def fit(self, X):
        self.init(X)
        self.fit_estimators()

        return self

    def predict_proba(self, X):
        y_scores = np.zeros((len(X), self.n_estimators))

        # Transform into rocket feature space
        Xt = self.rocket_transformer.transform(X)

        Xtp_scaled = None

        if self.power_transform == True:
            # Power Transform using yeo-johnson
            Xtp = self.power_transformer.transform(Xt)
            Xtp = pd.DataFrame(Xtp)

            # Check for infinite columns and remove them
            self._check_inf_values(Xtp)
            Xtp = Xtp[Xtp.columns[~Xtp.columns.isin(self.n_inf_cols)]]
            Xtp_temp = Xtp.copy()

            # Scale the data
            Xtp_scaled = self.scaler.transform(Xtp_temp)
            Xtp_scaled = pd.DataFrame(Xtp_scaled, columns=Xtp_temp.columns)

            # Check for infinite columns and remove them
            self._check_inf_values(Xtp_scaled)
            Xtp_scaled = Xtp_scaled[
                Xtp_scaled.columns[~Xtp_scaled.columns.isin(self.n_inf_cols)]
            ]
            Xtp_scaled = Xtp_scaled.astype(np.float32).to_numpy()

        else:
            Xtp_scaled = Xt.astype(np.float32)

        for idx, bagger in enumerate(self.list_baggers):
            # Get scores from each estimator
            scores = bagger.predict_proba(Xtp_scaled).squeeze()

            y_scores[:, idx] = scores

        # Average the scores to get the final score for each time series
        y_scores = y_scores.mean(axis=1)

        return y_scores

    def _check_inf_values(self, X):
        if np.isinf(X[X.columns[~X.columns.isin(self.n_inf_cols)]]).any(axis=0).any():
            self.n_inf_cols.extend(X.columns.to_series()[np.isinf(X).any()])
            self.fit_estimators()
            return True


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

    def fit(self, X: np.ndarray, y=None) -> None:
        """
        1) Applica ROCKAD su X a scapito di y.
        2) Prende i punteggi di anomalia e allena il one‐class model.
        """
        X_proc = self._prepare_input(X)

        # 1) Fit del solo ensemble ROCKAD (unsupervised)
        self.rockad = ROCKAD(
            n_estimators=self.n_estimators,
            n_kernels=self.num_kernels,
            n_jobs=-1,
            power_transform=False,
        )
        # rockad.fit si aspetta solo X
        self.rockad.fit(X_proc)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Restituisce 1=normale, 0=anomalia, basandosi sul modello one‐class.
        """

        X_proc = self._prepare_input(X)
        raw_scores = self.rockad.predict_proba(X_proc)

        base_model = self.base_model().fit(raw_scores)
        pred = base_model.predict(raw_scores)
        return pred
