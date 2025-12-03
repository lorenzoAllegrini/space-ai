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
    return lambda: OneClassSVM(), False


def get_rockad_classifier(num_kernels):
    """Get ROCKAD classifier."""
    return (
        lambda: DummyClassifier(strategy="constant", constant=0),
        False,
    )  # RockadClassifier(num_kernels=num_kernels), False


def get_xgboost_classifier():
    """Get XGBoost classifier."""
    return (
        lambda: XGBClassifier(eval_metric="logloss", base_score=0.5, nthread=XGBOOST_N_THREAD),
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
    return lambda: pipeline, mode != "likelihood_threshold"


def get_ridge_regression_classifier():
    """Get Ridge Regression classifier."""
    return lambda: RidgeClassifier(), True


def format_str(s):
    """Format string to CamelCase."""
    if "_" not in s:
        return s.lower()

    l = s.split("_")
    return "".join([l[0].lower()] + [x.capitalize() for x in l[1:]])


def create_classifier(args, other_args):
    """Create the classifier factory based on arguments."""
    model_id = format_str(args.model)
    match model_id:
        case "dpmm":
            return get_dpmm_classifier(args.dpmm_type, args.dpmm_mode, other_args)
        case "ocsvm":
            return get_ocsvm_classifier()
        case "rockad":
            return get_rockad_classifier(args.n_kernel)
        case "xgboost":
            return get_xgboost_classifier()
        case "ridge_regression":
            return get_ridge_regression_classifier()
        case _:
            raise ValueError(f"Modello {args.model} non supportato!")
            raise ValueError(f"Modello {args.model} non supportato!")


from spaceai.models.predictors import ESN, LSTM
from spaceai.models.anomaly import Telemanom
from .config import Config


def get_esn_predictor(config: Config):
    """Get ESN predictor."""
    return lambda input_size: ESN(
        input_size=input_size,
        hidden_size=config.layers,
        output_size=config.n_predictions,
        reduce_out="mean",  # TODO: make configurable?
        gradient_based=True, # TODO: make configurable?
        washout=200, # TODO: make configurable?
        activation=config.activation,
        leakage=config.leakage,
        input_scaling=config.input_scaling,
        rho=config.rho,
        kernel_initializer=config.kernel_initializer,
        recurrent_initializer=config.recurrent_initializer,
        net_gain_and_bias=config.net_gain_and_bias,
        bias=config.bias,
        l2=config.l2[0] if isinstance(config.l2, list) else config.l2,
    )


def get_lstm_predictor(config: Config):
    """Get LSTM predictor."""
    return lambda input_size: LSTM(
        input_size=input_size,
        hidden_sizes=config.layers,
        output_size=config.n_predictions,
        dropout=config.dropout,
    )


def get_telemanom_detector(config: Config):
    """Get Telemanom detector."""
    return lambda: Telemanom(pruning_factor=config.p)


def create_predictor(model_name, config: Config):
    """Create predictor based on model name."""
    model_id = format_str(model_name)
    match model_id:
        case "esn":
            return get_esn_predictor(config)
        case "lstm":
            return get_lstm_predictor(config)
        case _:
            raise ValueError(f"Predictor {model_name} not supported!")
