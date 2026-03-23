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
from spaceai.models.anomaly_classifier import NDPMDetector
from spaceai.models.anomaly_classifier.ndpm_internal import Config as NdpmConfig
import os
import logging
import torch

from .config import Config


def get_ocsvm_classifier():
    """Get OneClassSVM classifier."""
    return OneClassSVM, False


def get_rockad_classifier(_num_kernels):
    """Get ROCKAD classifier."""
    return (
        DummyClassifier(strategy="constant", constant=0),
        False,
    )  # RockadClassifier(num_kernels=num_kernels), False


def get_xgboost_classifier():
    """Get XGBoost classifier."""
    return (
        XGBClassifier(eval_metric="logloss", base_score=0.5),
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
            ("dpmm", DPMM(mode=mode, model_type=model_type, **config_dict)),
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


def get_iforest_classifier():
    """Get Isolation Forest (IForest) classifier."""
    return IForest, False


def get_pca_classifier():
    """Get PCA anomaly detector classifier."""
    return PyOD_PCA, False


def get_knn_classifier():
    """Get K-Nearest Neighbors (KNN) classifier."""
    return KNN, False


def get_lof_classifier():
    """Get Local Outlier Factor (LOF) classifier."""
    return LOF, False


def get_pyod_ocsvm_classifier():
    """Get One-Class SVM (OCSVM) classifier from PyOD."""
    return OCSVM, False


def get_ecod_classifier():
    """Get Empirical Cumulative Distribution (ECOD) classifier."""
    return ECOD, False


def get_copod_classifier():
    """Get Copula-Based Outlier Detection (COPOD) classifier."""
    return COPOD, False


def get_cblof_classifier():
    """Get Cluster-based Local Outlier Factor (CBLOF) classifier."""
    return CBLOF, False


def get_hbos_classifier():
    """Get Histogram-based Outlier Score (HBOS) classifier."""
    return HBOS, False


def format_str(s):
    """Format string to CamelCase."""
    if "_" not in s:
        return s.lower()

    parts = s.split("_")
    return "".join([parts[0].lower()] + [x.capitalize() for x in parts[1:]])


def create_classifier(args, other_args):
    """Create the classifier factory based on arguments."""
    model_id = format_str(args.model)
    if model_id == "dpmm":
        return get_dpmm_classifier(args.dpmm_type, args.dpmm_mode, other_args)
    elif model_id == "ocsvm":
        return get_ocsvm_classifier()
    elif model_id == "rockad":
        return get_rockad_classifier(args.n_kernel)
    elif model_id == "xgboost":
        return get_xgboost_classifier()
    elif model_id == "ridge_regression":
        return get_ridge_regression_classifier()
    elif model_id == "iforest":
        return get_iforest_classifier()
    elif model_id == "pca":
        return get_pca_classifier()
    elif model_id == "knn":
        return get_knn_classifier()
    elif model_id == "lof":
        return get_lof_classifier()
    elif model_id == "pyod_ocsvm":
        return get_pyod_ocsvm_classifier()
    elif model_id == "ecod":
        return get_ecod_classifier()
    elif model_id == "copod":
        return get_copod_classifier()
    elif model_id == "cblof":
        return get_cblof_classifier()
    elif model_id == "hbos":
        return get_hbos_classifier()
    elif model_id == "ndpm":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return get_ndpm_classifier(args, device, input_dim=input_dim)
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
