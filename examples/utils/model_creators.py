"""Model creators module."""
import torch

from spaceai.models.detectors.telemanom import Telemanom
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

# Sequence models
from spaceai.models.predictors import LSTM
from spaceai.models.detectors import Telemanom
from spaceai.models.anomaly_pipeline import SequenceModelClassifier

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
from spaceai.models.anomaly_classifier.dpmm_detector import (
    DPMM,
    get_dpmm_argparser,
)
from spaceai.models.anomaly_classifier.ndpm_detector import NDPMDetector
from spaceai.models.anomaly_classifier.ndpm_internal import Config as NdpmConfig
import os
import logging
import torch

from .config import Config


from spaceai.models.anomaly_classifier.base import SklearnClassifier
from spaceai.models.utils.scalers import RollingRobustScalerWithPrior
    


def get_rockad_classifier(_num_kernels):
    """Get ROCKAD classifier."""
    from spaceai.models.anomaly_classifier.rockad import RockadClassifier
    return RockadClassifier(num_kernels=_num_kernels), False


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
            ("xgb", SklearnClassifier(XGBClassifier(**params), supervised=True)),
        ]
    )
    return pipeline, True


def get_dpmm_classifier(model_type, mode, other_dpmm_args, base_params=None):
    """Get DPMM classifier."""
    # Ensure defaults if None
    model_type = model_type if model_type is not None else "full"
    mode = mode if mode is not None else "likelihood_threshold"
    
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
        else RobustScaler(with_centering=False)
    )

    if base_params:
        config_dict.update(base_params)

    config_dict["return_likelihood"] = True

    pipeline = Pipeline(
        [
            ("scaler", scaler),
            ("dpmm", DPMM(mode=mode, model_type=model_type, **config_dict)),
        ]
    )
    supervised = base_params.pop("supervised", mode != "likelihood_threshold")
    return SklearnClassifier(pipeline, supervised=supervised), supervised


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


def get_ridge_regression_classifier(base_params=None):
    """Get Ridge Regression classifier."""
    base_params = base_params if base_params else {}
    return SklearnClassifier(RidgeClassifier(**base_params), supervised=True), True


def get_iforest_classifier(base_params=None):
    """Get Isolation Forest (IForest) classifier."""
    base_params = base_params.copy() if base_params else {}
    dynamic_scaling = base_params.pop("dynamic_scaling", False)
    scaler_window = base_params.pop("scaler_window", 10)
    scaler = (
        RollingRobustScalerWithPrior(window=scaler_window)
        if dynamic_scaling
        else RobustScaler(with_centering=False)
    )

    supervised = base_params.pop("supervised", False)
    pipeline = Pipeline(
        [
            ("scaler", scaler),
            ("iforest", SklearnClassifier(IForest(**base_params), supervised=supervised)),
        ]
    )
    return pipeline, supervised


def get_pca_classifier(base_params=None):
    """Get PCA anomaly detector classifier."""
    base_params = base_params if base_params else {}
    return SklearnClassifier(PyOD_PCA(**base_params)), False


def get_knn_classifier(base_params=None):
    """Get K-Nearest Neighbors (KNN) classifier."""
    base_params = base_params if base_params else {}
    return SklearnClassifier(KNN(**base_params)), False


def get_lof_classifier(base_params=None):
    """Get Local Outlier Factor (LOF) classifier."""
    base_params = base_params if base_params else {}
    return SklearnClassifier(LOF(**base_params)), False


def get_pyod_ocsvm_classifier(base_params=None):
    """Get One-Class SVM (OCSVM) classifier from PyOD."""
    base_params = base_params.copy() if base_params else {}
    dynamic_scaling = base_params.pop("dynamic_scaling", False)
    scaler_window = base_params.pop("scaler_window", 10)
    base_params.pop("random_state", None)  # OCSVM doesn't support random_state
    scaler = (
        RollingRobustScalerWithPrior(window=scaler_window)
        if dynamic_scaling
        else RobustScaler(with_centering=False)
    )

    pipeline = Pipeline(
        [
            ("scaler", scaler),
            ("ocsvm", SklearnClassifier(OCSVM(**base_params))),
        ]
    )
    return pipeline, False


def get_ecod_classifier(base_params=None):
    """Get Empirical Cumulative Distribution (ECOD) classifier."""
    base_params = base_params if base_params else {}
    return SklearnClassifier(ECOD(**base_params)), False


def get_copod_classifier(base_params=None):
    """Get Copula-Based Outlier Detection (COPOD) classifier."""
    base_params = base_params if base_params else {}
    return SklearnClassifier(COPOD(**base_params)), False


def get_cblof_classifier(base_params=None):
    """Get Cluster-based Local Outlier Factor (CBLOF) classifier."""
    base_params = base_params if base_params else {}
    return SklearnClassifier(CBLOF(**base_params)), False


def get_hbos_classifier(base_params=None):
    """Get Histogram-based Outlier Score (HBOS) classifier."""
    base_params = base_params if base_params else {}
    return SklearnClassifier(HBOS(**base_params)), False


def format_str(s):
    """Format string to CamelCase."""
    if "_" not in s:
        return s.lower()

    parts = s.split("_")
    return "".join([parts[0].lower()] + [x.capitalize() for x in parts[1:]])


def get_ocsvm_classifier(base_params=None):
    """Get One-Class SVM (OCSVM) classifier from sklearn."""
    base_params = base_params if base_params else {}
    return SklearnClassifier(OneClassSVM(**base_params)), False


def create_classifier(args, other_args):
    """Create the classifier factory based on arguments."""
    model_id = format_str(args.model)
    # Extract extra parameters from YAML config if available
    base_params = getattr(args, "base_classifier_params", {})

    if model_id == "dpmm":
        return get_dpmm_classifier(
            args.dpmm_type, args.dpmm_mode, other_args, base_params=base_params
        )
    elif model_id == "ocsvm":
        return get_ocsvm_classifier(base_params=base_params)
    elif model_id == "rockad":
        return get_rockad_classifier(args.n_kernel)
    elif model_id == "xgboost":
        return get_xgboost_classifier(base_params=base_params)
    elif model_id == "ridgeRegression":
        return get_ridge_regression_classifier(base_params=base_params)
    elif model_id == "iforest":
        return get_iforest_classifier(base_params=base_params)
    elif model_id == "pca":
        return get_pca_classifier(base_params=base_params)
    elif model_id == "knn":
        return get_knn_classifier(base_params=base_params)
    elif model_id == "lof":
        return get_lof_classifier(base_params=base_params)
    elif model_id == "pyodOcsvm":
        return get_pyod_ocsvm_classifier(base_params=base_params)
    elif model_id == "ecod":
        return get_ecod_classifier(base_params=base_params)
    elif model_id == "copod":
        return get_copod_classifier(base_params=base_params)
    elif model_id == "cblof":
        return get_cblof_classifier(base_params=base_params)
    elif model_id == "hbos":
        return get_hbos_classifier(base_params=base_params)
    elif model_id == "ndpm":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return get_ndpm_classifier(args, device, input_dim=None)
    elif model_id == "telemanom":
        return get_telemanom_sequence_classifier(base_params=base_params)
    else:
        raise ValueError(f"Modello {args.model} non supportato!")


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


def get_telemanom_sequence_classifier(base_params=None):
    """Get SequenceModelClassifier with LSTM predictor and Telemanom detector."""
    base_params = base_params.copy() if base_params else {}
    
    # LSTM params
    input_size = base_params.pop("input_size", 1)
    hidden_sizes = base_params.pop("hidden_sizes", [80, 80])
    output_size = base_params.pop("output_size", 1)
    dropout = base_params.pop("dropout", 0.3)
    n_predictions = base_params.pop("n_predictions", 1)
    
    # Training params (stored for server-side use)
    epochs = base_params.pop("epochs", 35)
    lr = base_params.pop("lr", 0.001)
    patience = base_params.pop("patience", 10)
    min_delta = base_params.pop("min_delta", 0.0003)
    batch_size = base_params.pop("batch_size", 32)
    perc_eval = base_params.pop("perc_eval", 0.15)
    
    # Telemanom detector params
    # Map 'error_offset' from both 'error_offset' and 'error_buffer' (v32 legacy)
    error_offset = base_params.pop("error_offset", base_params.pop("error_buffer", 100))
    smoothing_perc = base_params.pop("smoothing_perc", 0.05)
    pruning_factor = base_params.pop("pruning_factor", base_params.pop("p", 0.12))
    n_eval_per_window = base_params.pop("n_eval_per_window", 70)
    tele_window_size = base_params.pop("telemanom_window_size", 2100)
    
    predictor = LSTM(
        input_size=input_size,
        hidden_sizes=hidden_sizes,
        output_size=n_predictions,
        reduce_out="first",
        dropout=dropout,
    )
    predictor.build()
    
    detector = Telemanom(
        pruning_factor=pruning_factor,
        error_offset=error_offset,
        smoothing_perc=smoothing_perc,
        n_eval_per_window=n_eval_per_window,
        window_size=tele_window_size,
        pred_buffer=base_params.get("l_s", 250) # Fallback if present
    )
    
    import torch.nn as nn
    classifier = SequenceModelClassifier(
        predictor=predictor,
        detector=detector,
    )
    # Store training hyperparams on the classifier for server-side access
    classifier._fit_args = {
        "criterion": nn.MSELoss(),
        "optimizer_class": torch.optim.Adam,
        "lr": lr,
        "epochs": epochs,
        "patience_before_stopping": patience,
        "min_delta": min_delta,
        "batch_size": batch_size,
        "perc_eval": perc_eval,
    }
    return classifier, False


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
