from spaceai.models.anomaly_classifier import RockadClassifier
from sklearn.svm import OneClassSVM
from xgboost import XGBClassifier
from sklearn.linear_model import RidgeClassifier
from spaceai.models.anomaly_classifier.dpmm_detector import DPMMWrapperDetector
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from config import DPMM_ENV_PATH


def get_ocsvm_classifier():
    return OneClassSVM(), False

def get_rockad_classifier(num_kernels):
    return RockadClassifier(num_kernels=num_kernels), False

def get_xgboost_classifier():
    return XGBClassifier(eval_metric="logloss", base_score=0.5, n_jobs=1), True

def get_dpmm_classifier(model_type, mode):
    pipeline = Pipeline([
        ("scaler", RobustScaler(with_centering=False)),
        ("dpmm", DPMMWrapperDetector(
            mode=mode,
            model_type=model_type,
            python_executable=DPMM_ENV_PATH,
        ))
    ])
    return pipeline, mode != 'likelihood_threshold'

def get_ridge_regression_classifier():
    return RidgeClassifier(), True

