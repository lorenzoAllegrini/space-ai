from .config import (
    XGBOOST_N_THREAD,
)
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import OneClassSVM
from xgboost import XGBClassifier

from spaceai.models.anomaly_classifier import RockadClassifier
from spaceai.models.anomaly_classifier.dpmm_detector import DPMMDetector, get_dpmm_argparser

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


def get_dpmm_classifier(model_type, mode, other_dpmm_args):
    parser = get_dpmm_argparser()
    config = parser.parse_args(other_dpmm_args)
    config_dict = vars(config)
    print(config_dict)
    pipeline = Pipeline([
        ("scaler", RobustScaler(with_centering=False)),
        ("dpmm", DPMMDetector(
            mode=mode,
            model_type=model_type,
            **config_dict
        ))
    ])
    return pipeline, mode != 'likelihood_threshold'

def get_ridge_regression_classifier():
    return RidgeClassifier(), True
