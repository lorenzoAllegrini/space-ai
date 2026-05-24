"""Model creators module."""
import torch

from spaceai.models.detectors.telemanom import Telemanom
from spaceai.models.predictors import ESN, LSTM
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

from spaceai.models.predictors.sequence_model_classifier import SequenceModelClassifier

from pyod.models.iforest import IForest  # type: ignore
from pyod.models.pca import PCA as PyOD_PCA  # type: ignore
from pyod.models.knn import KNN  # type: ignore
from pyod.models.lof import LOF  # type: ignore
from pyod.models.ocsvm import OCSVM  # type: ignore
from pyod.models.ecod import ECOD  # type: ignore
from pyod.models.copod import COPOD  # type: ignore
from pyod.models.cblof import CBLOF  # type: ignore
from pyod.models.hbos import HBOS  # type: ignore

from spaceai.models.classifiers.dpmm_detector import (
    DPMM,
    get_dpmm_argparser,
)
import os
import torch.nn as nn

from .config import Config


from spaceai.models.classifiers import SklearnClassifier
from spaceai.models.utils.scalers import RollingRobustScalerWithPrior


def _pop_wrapper_params(base_params):
    """Pop wrapper/metadata parameters from base_params and return them."""
    params = base_params.copy() if base_params else {}
    wrapper_params = {
        "dynamic_scaling": params.pop("dynamic_scaling", False),
        "scaler_window": params.pop("scaler_window", 10),
        "supervised": params.pop("supervised", False),
        "return_proba": params.pop("return_proba", False),
        "return_labels": params.pop("return_labels", False),
    }
    return params, wrapper_params


def _get_scaler(wrapper_params):
    """Helper to create the appropriate scaler."""
    return (
        RollingRobustScalerWithPrior(window=wrapper_params["scaler_window"])
        if wrapper_params["dynamic_scaling"]
        else RobustScaler(with_centering=False)
    )

def get_rockad_classifier(_num_kernels):
    """Get ROCKAD classifier."""
    from spaceai.models.legacy.rockad import RockadClassifier
    return RockadClassifier(num_kernels=_num_kernels), False


def get_xgboost_classifier(base_params=None):
    """Get XGBoost classifier."""
    params, wp = _pop_wrapper_params(base_params)
    scaler = _get_scaler(wp)

    pipeline = Pipeline(
        [
            ("scaler", scaler),
            ("xgboost", SklearnClassifier(
                XGBClassifier(**params),
                supervised=True,
                return_labels=wp["return_labels"],
                return_proba=wp["return_proba"]
            )),
        ]
    )
    return SklearnClassifier(pipeline, supervised=True), True


def get_dpmm_classifier(model_type, mode, other_dpmm_args, base_params=None):
    """Get DPMM classifier."""
    model_type = model_type if model_type is not None else "full"
    mode = mode if mode is not None else "likelihood_threshold"

    parser = get_dpmm_argparser()
    config, _ = parser.parse_known_args(other_dpmm_args)
    config_dict = vars(config)

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


def get_ridge_regression_classifier(base_params=None):
    """Get Ridge Regression classifier."""
    base_params = base_params if base_params else {}
    return SklearnClassifier(RidgeClassifier(**base_params), supervised=True), True


def get_iforest_classifier(base_params=None):
    """Get Isolation Forest (IForest) classifier."""
    params, wp = _pop_wrapper_params(base_params)
    scaler = _get_scaler(wp)

    pipeline = Pipeline(
        [
            ("scaler", scaler),
            ("iforest", SklearnClassifier(
                IForest(**params),
                supervised=wp["supervised"],
                return_proba=wp["return_proba"],
                return_labels=wp["return_labels"]
            )),
        ]
    )
    return SklearnClassifier(pipeline, supervised=wp["supervised"]), wp["supervised"]


def get_pca_classifier(base_params=None):
    """Get PCA anomaly detector classifier."""
    params, wp = _pop_wrapper_params(base_params)
    return SklearnClassifier(
        PyOD_PCA(**params),
        supervised=wp["supervised"],
        return_labels=wp["return_labels"],
        return_proba=wp["return_proba"]
    ), wp["supervised"]


def get_knn_classifier(base_params=None):
    """Get K-Nearest Neighbors (KNN) classifier."""
    params, wp = _pop_wrapper_params(base_params)
    return SklearnClassifier(
        KNN(**params),
        supervised=wp["supervised"],
        return_labels=wp["return_labels"],
        return_proba=wp["return_proba"]
    ), wp["supervised"]


def get_lof_classifier(base_params=None):
    """Get Local Outlier Factor (LOF) classifier."""
    params, wp = _pop_wrapper_params(base_params)
    scaler = _get_scaler(wp)
    pipeline = Pipeline(
        [
            ("scaler", scaler),
            ("lof", SklearnClassifier(
                LOF(**params),
                supervised=wp["supervised"],
                return_proba=wp["return_proba"],
                return_labels=wp["return_labels"]
            )),
        ]
    )
    return SklearnClassifier(pipeline, supervised=wp["supervised"]), wp["supervised"]


def get_pyod_ocsvm_classifier(base_params=None):
    """Get One-Class SVM (OCSVM) classifier from PyOD."""
    params, wp = _pop_wrapper_params(base_params)
    params.pop("random_state", None)  # OCSVM doesn't support random_state
    scaler = _get_scaler(wp)

    pipeline = Pipeline(
        [
            ("scaler", scaler),
            ("ocsvm", SklearnClassifier(
                OCSVM(**params),
                supervised=wp["supervised"],
                return_proba=wp["return_proba"],
                return_labels=wp["return_labels"]
            )),
        ]
    )
    return SklearnClassifier(pipeline, supervised=wp["supervised"]), wp["supervised"]


def get_ecod_classifier(base_params=None):
    """Get Empirical Cumulative Distribution (ECOD) classifier."""
    params, wp = _pop_wrapper_params(base_params)
    return SklearnClassifier(
        ECOD(**params),
        supervised=wp["supervised"],
        return_labels=wp["return_labels"],
        return_proba=wp["return_proba"]
    ), wp["supervised"]


def get_copod_classifier(base_params=None):
    """Get Copula-Based Outlier Detection (COPOD) classifier."""
    params, wp = _pop_wrapper_params(base_params)
    return SklearnClassifier(
        COPOD(**params),
        supervised=wp["supervised"],
        return_labels=wp["return_labels"],
        return_proba=wp["return_proba"]
    ), wp["supervised"]


def get_cblof_classifier(base_params=None):
    """Get Cluster-based Local Outlier Factor (CBLOF) classifier."""
    params, wp = _pop_wrapper_params(base_params)
    return SklearnClassifier(
        CBLOF(**params),
        supervised=wp["supervised"],
        return_labels=wp["return_labels"],
        return_proba=wp["return_proba"]
    ), wp["supervised"]


def get_hbos_classifier(base_params=None):
    """Get Histogram-based Outlier Score (HBOS) classifier."""
    params, wp = _pop_wrapper_params(base_params)
    return SklearnClassifier(
        HBOS(**params),
        supervised=wp["supervised"],
        return_labels=wp["return_labels"],
        return_proba=wp["return_proba"]
    ), wp["supervised"]


def format_str(s):
    """Format string to CamelCase."""
    if "_" not in s:
        return s.lower()

    parts = s.split("_")
    return "".join([parts[0].lower()] + [x.capitalize() for x in parts[1:]])


def get_ocsvm_classifier(base_params=None):
    """Get One-Class SVM (OCSVM) classifier from sklearn."""
    params, wp = _pop_wrapper_params(base_params)
    return SklearnClassifier(
        OneClassSVM(**params),
        supervised=wp["supervised"],
        return_labels=wp["return_labels"],
        return_proba=wp["return_proba"]
    ), wp["supervised"]


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
    elif model_id == "telemanom":
        return get_telemanom_sequence_classifier(base_params=base_params)
    elif model_id == "dcvae":
        return get_dcvae_classifier(base_params=base_params)
    else:
        raise ValueError(f"Model {args.model} not supported!")


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
        reduce_out=config.reduce_out,
        dropout=config.dropout,
    )


def get_telemanom_sequence_classifier(base_params=None):
    """Get SequenceModelClassifier with LSTM predictor and Telemanom detector."""
    base_params = base_params.copy() if base_params else {}

    input_size = base_params.pop("input_size", 1)
    hidden_sizes = base_params.pop("hidden_sizes", [80, 80])
    dropout = base_params.pop("dropout", 0.3)
    n_predictions = base_params.pop("n_predictions", 1)
    reduce_out = base_params.pop("reduce_out", "first")

    device = "cpu"
    if base_params.get("device", None):
        device = torch.device(base_params["device"])

    epochs = base_params.pop("epochs", 35)
    lr = base_params.pop("lr", 0.001)
    patience = base_params.pop("patience", 10)
    min_delta = base_params.pop("min_delta", 0.0003)
    batch_size = base_params.pop("batch_size", 32)
    perc_eval = base_params.pop("perc_eval", 0.15)

    error_offset = base_params.pop(
        "error_offset", base_params.pop("error_buffer", 100))
    smoothing_perc = base_params.pop("smoothing_perc", 0.05)
    pruning_factor = base_params.pop(
        "pruning_factor", base_params.pop("p", 0.12))
    n_eval_per_window = base_params.pop("n_eval_per_window", 70)
    tele_window_size = base_params.pop("telemanom_window_size", 2100)

    optimizer_builder = getattr(
        torch.optim, base_params.pop("optimizer_class", "Adam"))
    criterion = getattr(torch.nn, base_params.pop("criterion", "MSELoss"))()

    def optimizer_builder_func(model):
        return optimizer_builder(model.parameters(), lr=lr)

    fit_predictor_args = {
        "criterion": criterion,
        "optimizer_builder": optimizer_builder_func,
        "epochs": epochs,
        "patience_before_stopping": patience,
        "min_delta": min_delta,
        "perc_eval": perc_eval,
        "batch_size": batch_size,
    }

    predictor = LSTM(
        input_size=input_size,
        hidden_sizes=hidden_sizes,
        output_size=n_predictions,
        reduce_out=reduce_out,
        dropout=dropout,
        device=device,
    )
    predictor.build()

    detector = Telemanom(
        pruning_factor=pruning_factor,
        error_offset=error_offset,
        smoothing_perc=smoothing_perc,
        n_eval_per_window=n_eval_per_window,
        window_size=tele_window_size,
        pred_buffer=base_params.get("l_s", 250)  # Fallback if present
    )

    from spaceai.models.predictors.sequence_model_classifier import SequenceModelClassifier
    classifier = SequenceModelClassifier(
        predictor=predictor,
        detector=detector,
        fit_predictor_args=fit_predictor_args,
        n_predictions=n_predictions
    )
    return classifier, False


def get_dcvae_classifier(base_params=None):
    """Get DCVAE classifier."""
    from spaceai.models.predictors.dcvae import DCVAEClassifier
    return DCVAEClassifier(**(base_params if base_params else {})), False


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
