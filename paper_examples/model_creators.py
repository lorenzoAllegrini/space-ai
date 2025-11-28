"""Model creators module."""

from sklearn.dummy import DummyClassifier  # type: ignore
from sklearn.linear_model import RidgeClassifier  # type: ignore
from sklearn.pipeline import Pipeline  # type: ignore
from sklearn.preprocessing import RobustScaler  # type: ignore
from sklearn.svm import OneClassSVM  # type: ignore
from xgboost import XGBClassifier  # type: ignore

# from spaceai.models.anomaly_classifier import RockadClassifier
from spaceai.models.anomaly_classifier.dpmm_detector import (
    DPMMDetector,
    get_dpmm_argparser,
)

from .config import XGBOOST_N_THREAD


def get_ocsvm_classifier():
    """Get OneClassSVM classifier."""
    return OneClassSVM(), False


def get_rockad_classifier(num_kernels):
    """Get ROCKAD classifier."""
    return (
        DummyClassifier(strategy="constant", constant=0),
        False,
    )  # RockadClassifier(num_kernels=num_kernels), False


def get_xgboost_classifier():
    """Get XGBoost classifier."""
    return (
        XGBClassifier(eval_metric="logloss", base_score=0.5, nthread=XGBOOST_N_THREAD),
        True,
    )


def get_dpmm_classifier(model_type, mode, other_dpmm_args):
    """Get DPMM classifier."""
    parser = get_dpmm_argparser()
    config = parser.parse_args(other_dpmm_args)
    config_dict = vars(config)
    pipeline = Pipeline(
        [
            ("scaler", RobustScaler(with_centering=False)),
            ("dpmm", DPMMDetector(mode=mode, model_type=model_type, **config_dict)),
        ]
    )
    return pipeline, mode != "likelihood_threshold"


def get_ridge_regression_classifier():
    """Get Ridge Regression classifier."""
    return RidgeClassifier(), True
