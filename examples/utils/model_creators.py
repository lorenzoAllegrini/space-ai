"""Model creators module."""
import torch

from spaceai.models.anomaly import Telemanom
from spaceai.models.predictors import (
    ESN,
    LSTM,
)
from sklearn.dummy import DummyClassifier  # type: ignore
from sklearn.linear_model import RidgeClassifier  # type: ignore
from sklearn.pipeline import Pipeline  # type: ignore
from sklearn.preprocessing import RobustScaler  # type: ignore
from sklearn.svm import OneClassSVM  # type: ignore
from xgboost import XGBClassifier  # type: ignore
import numpy as np
import pandas as pd
from scipy.stats import iqr as scipy_iqr
from sklearn.base import BaseEstimator, TransformerMixin

# PyOD models
from pyod.models.iforest import IForest  # type: ignore
from pyod.models.pca import PCA as PyOD_PCA  # type: ignore
from pyod.models.knn import KNN  # type: ignore
from pyod.models.lof import LOF  # type: ignore
from pyod.models.ocsvm import OCSVM  # type: ignore
from pyod.models.ecod import ECOD  # type: ignore
from pyod.models.copod import COPOD  # type: ignore
from pyod.models.cblof import CBLOF  # type: ignore
from pyod.models.hbos import HBOS  # type: ignore

# from spaceai.models.anomaly_classifier import RockadClassifier
from spaceai.models.anomaly.dpmm_detector import (
    DPMM,
    get_dpmm_argparser,
)
from spaceai.models.anomaly.ndpm_detector import NDPMDetector
from spaceai.models.anomaly.ndpm_internal import Config as NdpmConfig
from spaceai.models.anomaly import (
    BaseClassifier, 
    SklearnClassifier,
    ThresholdDetector, 
    MoLooKDEDetector, 
    QuantileThresholdDetector,
    DPMMNativeDetector
)
import os
import logging
import torch

from .config import Config


class RollingRobustScalerWithPrior(BaseEstimator, TransformerMixin):
    """
    Rolling robust scaler that uses a historical IQR prior to stabilize 
    scaling in flat/low-noise regions.
    """

    def __init__(self, window: int = 10):
        self.window = window
        self.prior_iqr_ = None

    def fit(self, X, y=None):
        """Calculate global IQR on training data as a prior."""
        self.prior_iqr_ = scipy_iqr(X, axis=0) + 1e-12
        return self

    def transform(self, X):
        """Scale X using rolling median/IQR with a historical prior constraint."""
        if self.prior_iqr_ is None:
            raise ValueError("Scaler must be fitted before transform.")

        df = pd.DataFrame(X)
        roll = df.rolling(window=self.window, min_periods=1)

        rolling_median = roll.median()

        rolling_q75 = roll.quantile(0.75)
        rolling_q25 = roll.quantile(0.25)
        rolling_iqr = rolling_q75 - rolling_q25

        denominator = np.maximum(rolling_iqr.values, self.prior_iqr_)

        X_scaled = (df - rolling_median) / denominator
        res = X_scaled.values
        return res


def get_rockad_classifier(_num_kernels):
    """Get ROCKAD classifier."""
    return (
        DummyClassifier(strategy="constant", constant=0),
        False,
    )  # RockadClassifier(num_kernels=num_kernels), False


def get_xgboost_classifier(base_params=None):
    """Get XGBoost classifier."""
    base_params = base_params.copy() if base_params else {}
    dynamic_scaling = base_params.pop("dynamic_scaling", False)
    scaler_window = base_params.pop("scaler_window", 10)
    scaler = (
        RollingRobustScalerWithPrior(window=scaler_window)
        if dynamic_scaling
        else RobustScaler(with_centering=False)
    )

    params = base_params if base_params else {"eval_metric": "logloss", "base_score": 0.5}
    pipeline = Pipeline(
        [
            ("scaler", scaler),
            ("xgb", XGBClassifier(**params)),
        ]
    )
    return pipeline, True


def get_dpmm_classifier(model_type, mode, other_dpmm_args, base_params=None, callback_handler=None):
    """Get DPMM classifier."""
    parser = get_dpmm_argparser()
    config, _ = parser.parse_known_args(other_dpmm_args)
    config_dict = vars(config)

    # Merge with YAML params if provided
    base_params = base_params.copy() if base_params else {}
    dynamic_scaling = base_params.pop("dynamic_scaling", False)
    scaler_window = base_params.pop("scaler_window", 10)
    scaler = (
        RollingRobustScalerWithPrior(window=scaler_window)
        if dynamic_scaling
        else RobustScaler(with_centering=True)
    )

    if base_params:
        config_dict.update(base_params)

    pipeline = Pipeline(
        [
            ("scaler", scaler),
            ("dpmm", DPMM(mode=mode, model_type=model_type, callback_handler=callback_handler, **config_dict)),
        ]
    )
    return pipeline, mode != "likelihood_threshold"


def get_ndpm_classifier(args, device="cpu", input_dim=None):
    """Get NDPM classifier factory."""

    def factory():
        config_path = getattr(args, "ndpm_config", None)
        if not config_path or not os.path.exists(config_path):
            raise ValueError(
                f"Valid NDPM configuration required. Please check --ndpm_config (path: {config_path})"
            )

        logging.info("Loading NDPM config from %s", config_path)
        config = NdpmConfig.from_yaml_file(config_path)

        # Deducing x_w
        if input_dim is not None:
            x_w = input_dim
        else:
            # Fallback deduction from args if input_dim is not provided
            fe_type = getattr(args, "feature_extractor", "none")
            if fe_type == "base_statistics":
                from spaceai.preprocessing.functions import FEATURE_MAP
                x_w = len(FEATURE_MAP)
            elif fe_type == "rocket":
                n_kernel = getattr(args, "n_kernel", 100)
                x_w = 2 * n_kernel
            else:
                x_w = getattr(args, "window_size", 100)

        channel_id = getattr(args, "channel", "default") or "default"
        run_dir = getattr(args, "run_dir", ".") or "."
        config["log_dir"] = os.path.join(run_dir, "logs", channel_id)
        config["x_w"] = x_w

        detector = NDPMDetector(config, device=device)
        logging.info(
            "Initialized NDPM detector (x_w=%d) for channel: %s", x_w, channel_id
        )
        return detector

    return factory, False


def get_ridge_regression_classifier():
    """Get Ridge Regression classifier."""
    return RidgeClassifier, True


def get_iforest_classifier(base_params=None):
    """Get Isolation Forest (IForest) classifier."""
    base_params = base_params if base_params else {}
    
    # Extract meta-params for scaling
    dynamic_scaling = base_params.pop("dynamic_scaling", False)
    scaler_window = base_params.pop("scaler_window", 100)
    
    scaler = RollingRobustScalerWithPrior(window=scaler_window) if dynamic_scaling else RobustScaler(with_centering=False)
    
    pipeline = Pipeline([
        ("scaler", scaler),
        ("iforest", IForest(**base_params))
    ])
    return pipeline, False


def get_pca_classifier(base_params=None):
    """Get PCA anomaly detector classifier."""
    base_params = base_params if base_params else {}
    
    # Extract meta-params for scaling
    dynamic_scaling = base_params.pop("dynamic_scaling", False)
    scaler_window = base_params.pop("scaler_window", 100)
    
    scaler = RollingRobustScalerWithPrior(window=scaler_window) if dynamic_scaling else RobustScaler(with_centering=False)
    
    pipeline = Pipeline([
        ("scaler", scaler),
        ("pca", PyOD_PCA(**base_params))
    ])
    return pipeline, False


def get_knn_classifier(base_params=None):
    """Get K-Nearest Neighbors (KNN) classifier."""
    base_params = base_params if base_params else {}
    
    # Extract meta-params for scaling
    dynamic_scaling = base_params.pop("dynamic_scaling", False)
    scaler_window = base_params.pop("scaler_window", 100)
    
    scaler = RollingRobustScalerWithPrior(window=scaler_window) if dynamic_scaling else RobustScaler(with_centering=False)
    
    pipeline = Pipeline([
        ("scaler", scaler),
        ("knn", KNN(**base_params))
    ])
    return pipeline, False


def get_lof_classifier(base_params=None):
    """Get Local Outlier Factor (LOF) classifier."""
    base_params = base_params if base_params else {}
    
    # Extract meta-params for scaling
    dynamic_scaling = base_params.pop("dynamic_scaling", False)
    scaler_window = base_params.pop("scaler_window", 100)
    
    scaler = RollingRobustScalerWithPrior(window=scaler_window) if dynamic_scaling else RobustScaler(with_centering=False)
    
    pipeline = Pipeline([
        ("scaler", scaler),
        ("lof", LOF(**base_params))
    ])
    return pipeline, False


def get_pyod_ocsvm_classifier(base_params=None):
    """Get One-Class SVM (OCSVM) classifier from PyOD."""
    base_params = base_params if base_params else {}
    
    # Extract meta-params for scaling
    dynamic_scaling = base_params.pop("dynamic_scaling", False)
    scaler_window = base_params.pop("scaler_window", 100)
    
    scaler = RollingRobustScalerWithPrior(window=scaler_window) if dynamic_scaling else RobustScaler(with_centering=False)
    
    pipeline = Pipeline([
        ("scaler", scaler),
        ("ocsvm", OCSVM(**base_params))
    ])
    return pipeline, False


def get_ecod_classifier(base_params=None):
    """Get Empirical Cumulative Distribution (ECOD) classifier."""
    base_params = base_params if base_params else {}
    
    # Extract meta-params for scaling
    dynamic_scaling = base_params.pop("dynamic_scaling", False)
    scaler_window = base_params.pop("scaler_window", 100)
    
    scaler = RollingRobustScalerWithPrior(window=scaler_window) if dynamic_scaling else RobustScaler(with_centering=False)
    
    pipeline = Pipeline([
        ("scaler", scaler),
        ("ecod", ECOD(**base_params))
    ])
    return pipeline, False


def get_copod_classifier(base_params=None):
    """Get Copula-Based Outlier Detection (COPOD) classifier."""
    base_params = base_params if base_params else {}
    
    # Extract meta-params for scaling
    dynamic_scaling = base_params.pop("dynamic_scaling", False)
    scaler_window = base_params.pop("scaler_window", 100)
    
    scaler = RollingRobustScalerWithPrior(window=scaler_window) if dynamic_scaling else RobustScaler(with_centering=False)
    
    pipeline = Pipeline([
        ("scaler", scaler),
        ("copod", COPOD(**base_params))
    ])
    return pipeline, False


def get_cblof_classifier(base_params=None):
    """Get Cluster-based Local Outlier Factor (CBLOF) classifier."""
    base_params = base_params if base_params else {}
    
    # Extract meta-params for scaling
    dynamic_scaling = base_params.pop("dynamic_scaling", False)
    scaler_window = base_params.pop("scaler_window", 100)
    
    scaler = RollingRobustScalerWithPrior(window=scaler_window) if dynamic_scaling else RobustScaler(with_centering=False)
    
    pipeline = Pipeline([
        ("scaler", scaler),
        ("cblof", CBLOF(**base_params))
    ])
    return pipeline, False


def get_hbos_classifier(base_params=None):
    """Get Histogram-based Outlier Score (HBOS) classifier."""
    base_params = base_params if base_params else {}
    
    # Extract meta-params for scaling
    dynamic_scaling = base_params.pop("dynamic_scaling", False)
    scaler_window = base_params.pop("scaler_window", 100)
    
    scaler = RollingRobustScalerWithPrior(window=scaler_window) if dynamic_scaling else RobustScaler(with_centering=False)
    
    pipeline = Pipeline([
        ("scaler", scaler),
        ("hbos", HBOS(**base_params))
    ])
    return pipeline, False


def format_str(s):
    """Format string to CamelCase."""
    if "_" not in s:
        return s.lower()

    parts = s.split("_")
    return "".join([parts[0].lower()] + [x.capitalize() for x in parts[1:]])


def get_ocsvm_classifier(base_params=None):
    """Get One-Class SVM (OCSVM) classifier from sklearn."""
    base_params = base_params if base_params else {}
    
    # Extract meta-params for scaling
    dynamic_scaling = base_params.pop("dynamic_scaling", False)
    scaler_window = base_params.pop("scaler_window", 100)
    
    scaler = RollingRobustScalerWithPrior(window=scaler_window) if dynamic_scaling else RobustScaler(with_centering=False)
    
    pipeline = Pipeline([
        ("scaler", scaler),
        ("ocsvm", OneClassSVM(**base_params))
    ])
    return pipeline, False


def create_classifier(args, other_args, callback_handler=None):
    """Create the classifier factory based on arguments."""
    model_id = format_str(args.model)
    # Extract extra parameters from YAML config if available
    base_params = getattr(args, "base_classifier_params", {})

    if model_id == "dpmm":
        if getattr(args, "detector", None) == "dpmm_native":
            base_params["return_likelihood"] = True
            
        classifier, supervised = get_dpmm_classifier(
            args.dpmm_type, args.dpmm_mode, other_args, base_params=base_params, callback_handler=callback_handler
        )
    elif model_id == "ocsvm":
        classifier, supervised = get_ocsvm_classifier(base_params=base_params)
    elif model_id == "rockad":
        classifier, supervised = get_rockad_classifier(args.n_kernel)
    elif model_id == "xgboost":
        classifier, supervised = get_xgboost_classifier(base_params=base_params)
    elif model_id == "ridge_regression":
        classifier, supervised = get_ridge_regression_classifier()
    elif model_id == "iforest":
        classifier, supervised = get_iforest_classifier(base_params=base_params)
    elif model_id == "pca":
        classifier, supervised = get_pca_classifier(base_params=base_params)
    elif model_id == "knn":
        classifier, supervised = get_knn_classifier(base_params=base_params)
    elif model_id == "lof":
        classifier, supervised = get_lof_classifier(base_params=base_params)
    elif model_id == "pyod_ocsvm":
        classifier, supervised = get_pyod_ocsvm_classifier(base_params=base_params)
    elif model_id == "ecod":
        classifier, supervised = get_ecod_classifier(base_params=base_params)
    elif model_id == "copod":
        classifier, supervised = get_copod_classifier(base_params=base_params)
    elif model_id == "cblof":
        classifier, supervised = get_cblof_classifier(base_params=base_params)
    elif model_id == "hbos":
        classifier, supervised = get_hbos_classifier(base_params=base_params)
    elif model_id == "ndpm":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        classifier, supervised = get_ndpm_classifier(args, device)
    else:
        raise ValueError(f"Modello {args.model} non supportato!")

    # Wrap in SklearnClassifier if it's a raw model (missing legacy role or message compatibility)
    if not hasattr(classifier, "role"):
        classifier = SklearnClassifier(
            model=classifier,
            supervised=supervised,
            callback_handler=callback_handler,
            return_labels=getattr(args, "detector", "none") in [None, "none", "no_detector"]
        )
    
    return classifier, supervised


def create_detector(args, callback_handler=None):
    """Create the anomaly detector based on arguments."""
    detector_params = getattr(args, 'detector_params', {})
    
    if args.detector == "threshold":
        return ThresholdDetector(**{**dict(threshold=0.9, callback_handler=callback_handler), **detector_params})
    elif args.detector == "quantile":
        return QuantileThresholdDetector(**{**dict(quantile=0.95, callback_handler=callback_handler), **detector_params})
    elif args.detector == "molookde":
        return MoLooKDEDetector(**{**dict(alpha=0.001, callback_handler=callback_handler), **detector_params})
    elif args.detector == "dpmm_native":
        return DPMMNativeDetector(**{**dict(callback_handler=callback_handler), **detector_params})
    elif args.detector == "none":
        return None
    else:
        raise ValueError(f"Detector {args.detector} non supportato!")


def get_esn_predictor(config: Config):
    """Get ESN predictor."""
    return ESN(
        input_size=1,
        layers=config.layers,
        output_size=config.n_predictions,
        reduce_out="mean",
        gradient_based=True,
        washout=200,
        activation=config.activation,
        leakage=config.leakage,
        input_scaling=config.input_scaling,
        rho=config.rho,
        kernel_initializer=config.kernel_initializer,
        recurrent_initializer=config.recurrent_initializer,
        net_gain_and_bias=config.net_gain_and_bias,
        bias=config.bias,
    )


def get_lstm_predictor(config: Config):
    """Get LSTM predictor."""
    return lambda input_size: LSTM(
        input_size=input_size,
        hidden_sizes=config.layers,
        output_size=config.n_predictions,
        reduce_out="first",
        dropout=config.dropout,
    )


def get_telemanom_detector(config: Config):
    """Get Telemanom detector."""
    return lambda: Telemanom(pruning_factor=config.p)


def create_predictor(model_name, config: Config):
    """Create predictor based on model name."""
    model_id = format_str(model_name)
    if model_id == "esn":
        return get_esn_predictor(config)
    elif model_id == "lstm":
        return get_lstm_predictor(config)
    else:
        raise ValueError(f"Predictor {model_name} not supported!")
