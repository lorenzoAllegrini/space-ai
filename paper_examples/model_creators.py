from config import (
    XGBOOST_N_THREAD,
)
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import OneClassSVM
from xgboost import XGBClassifier

from spaceai.models.anomaly_classifier import RockadClassifier
from spaceai.models.anomaly_classifier.dpmm_detector import DPMMDetector


def get_ocsvm_classifier():
    return OneClassSVM(), False


def get_rockad_classifier(num_kernels):
    return (
        DummyClassifier(strategy="constant", constant=0),
        False,
    )  # RockadClassifier(num_kernels=num_kernels), False


def get_xgboost_classifier():
    return (
        XGBClassifier(eval_metric="logloss", base_score=0.5, nthread=XGBOOST_N_THREAD),
        True,
    )


def get_dpmm_classifier(model_type, mode):
    pipeline = Pipeline(
        [
            ("scaler", RobustScaler(with_centering=False)),
            (
                "dpmm",
                DPMMDetector(
                    mode=mode,
                    model_type=model_type,
                    K=100,
                    num_iterations=50,
                    lr=0.1,
                ),
            ),
        ]
    )
    return pipeline, False


def get_ridge_regression_classifier():
    return RidgeClassifier(), True
