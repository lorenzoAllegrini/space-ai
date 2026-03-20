"""Run continual experiment module."""

import argparse
import warnings
import logging

from spaceai.preprocessing import (
    TimeSeriesSplitter,
    get_feature_extractor,
)

from utils.dataset_exp import (
    get_dataset_benchmark,
)
from utils.model_creators import (
    create_classifier,
)
from spaceai.benchmark.callbacks import SystemMonitorCallback, CallbackHandler
from spaceai.models.anomaly_classifier.rolling_window_classifier import RollingWindowClassifier
from spaceai.benchmark import ESABenchmark

warnings.simplefilter("ignore", FutureWarning)

DATASET_LIST = ["ops", "nasa", "esa"]
MODEL_LIST = [
    "ocsvm",
    "xgboost",
    "ridge_regression",
    "dpmm",
    "iforest",
    "pca",
    "knn",
    "lof",
    "pyod_ocsvm",
    "ecod",
    "copod",
    "cblof",
    "hbos",
    "ndpm"
]
DPMM_MODEL_TYPE = ["full", "diagonal", "single", "unit"]
DPMM_MODE = ["likelihood_threshold", "cluster_labels"]
FEATURE_EXTRACTOR_LIST = ["none", "base_statistics", "rocket"]


def parse_exp_args(str_args=None):
    """Parse experiment arguments."""
    parser = argparse.ArgumentParser(description="continual learning experiments execution")
    parser.add_argument("--base_dir", required=True)
    parser.add_argument("--exp-dir", default="experiments")
    parser.add_argument("--dataset", choices=DATASET_LIST, required=True)
    parser.add_argument("--mission-id", type=int, default=1)
    parser.add_argument("--model", choices=MODEL_LIST, required=True)
    parser.add_argument("--segmentator", action="store_true")
    parser.add_argument(
        "--feature-extractor", choices=FEATURE_EXTRACTOR_LIST, default="none"
    )
    parser.add_argument("--channels", type=str, nargs="+")
    parser.add_argument("--n-kernel", type=int)
    parser.add_argument("--dpmm-type", choices=DPMM_MODEL_TYPE)
    parser.add_argument("--dpmm-mode", choices=DPMM_MODE)
    parser.add_argument("--window-size", type=int, default=50)
    parser.add_argument("--step-size", type=int, default=50)
    parser.add_argument("--experience-size", type=str, default="30D", help="Size of each experience (int or time duration like '30D')")
    parser.add_argument("--ndpm_config", type=str, default=None, help="Path to NDPM config")
    return parser.parse_known_args(str_args)


def run_exp(args, other_args=None):
    """Run experiment."""

    ts_splitter = TimeSeriesSplitter(
        window_size=args.window_size,
        step_size=args.step_size,
    )
        
    feature_extractor = get_feature_extractor(
        args.feature_extractor,
        window_size=args.window_size,
        stride=args.step_size,
        n_kernel=args.n_kernel,
    )

    classifier, is_supervised = create_classifier(args, other_args)
    
    handler = CallbackHandler([SystemMonitorCallback()], call_every_ms=100)

    run_id = f"continual_{args.dataset}_{args.model}"
    if args.model == "dpmm":
        run_id += f"_{args.dpmm_type}_{args.dpmm_mode}"
    
    rolling_window_classifier = RollingWindowClassifier(
        base_classifier=classifier,
        supervised_classifier=is_supervised,
        ts_splitter=ts_splitter,
        feature_extractor=feature_extractor,
        callback_handler=handler,
    )

    benchmark = get_dataset_benchmark(
        dataset_name=args.dataset,
        data_path=args.base_dir,
        exp_dir=args.exp_dir,
        run_id=run_id,
        mission_id=args.mission_id,
    )

    channels = benchmark.channels if args.channels is None else args.channels
     
    for channel_name in channels:
        logging.info("Starting continual test for channel %s...", channel_name)

        fitted_classifier, fitting_metrics = benchmark.fit_channel(
            channel_id=channel_name,
            classifier=rolling_window_classifier,
        )

        print(fitting_metrics)

        experience_log = benchmark.test_continual(
            channel_id=channel_name,
            classifier=fitted_classifier,
            experience_size=args.experience_size,
        )


def main():
    """Main function."""
    args, other_args = parse_exp_args()
    logging.basicConfig(level=logging.INFO)
    run_exp(args, other_args)


if __name__ == "__main__":
    main()
