"""Run pipeline experiment module.

Uses AnomalyDetectionPipeline instead of RollingWindowClassifier.
Same CLI/YAML config as run_segment_extraction_exp.py.
"""

import argparse
import warnings
import logging
import yaml
import numpy as np

from spaceai.preprocessing import (
    TimeSeriesSplitter,
    get_feature_extractor,
)

from utils.dataset_exp import get_dataset_benchmark
from utils.model_creators import create_classifier
from utils.reproducibility import set_seed

from spaceai.benchmark.callbacks import SystemMonitorCallback, CallbackHandler
from spaceai.models.anomaly_classifier import (
    AnomalyDetectionPipeline,
)
from spaceai.models.anomaly.base import SklearnClassifier
from spaceai.benchmark import Benchmark, ESABenchmark
from spaceai.models.anomaly import ThresholdDetector, MoLooKDEDetector, QuantileThresholdDetector
from utils.model_creators import create_detector

warnings.simplefilter("ignore", FutureWarning)

DATASET_LIST = ["ops", "nasa", "esa"]
MODEL_LIST = [
    "ocsvm", "xgboost", "ridge_regression", "dpmm",
    "iforest", "pca", "knn", "lof", "pyod_ocsvm",
    "ecod", "copod", "cblof", "hbos", "ndpm"
]
FEATURE_EXTRACTOR_LIST = ["none", "base_statistics", "rocket"]


def parse_exp_args(str_args=None):
    """Parse experiment arguments with YAML config support."""
    conf_parser = argparse.ArgumentParser(add_help=False)
    conf_parser.add_argument("--config", type=str, default=None,
                             help="Path to the YAML configuration file")
    args, remaining_argv = conf_parser.parse_known_args(str_args)

    defaults = {
        "exp_dir": "experiments",
        "mission_id": 1,
        "feature_extractor": "none",
        "window_size": 50,
        "step_size": 50,
        "seed": 42,
    }

    if args.config:
        try:
            with open(args.config, "r") as f:
                yaml_config = yaml.safe_load(f)
                if yaml_config:
                    defaults.update(yaml_config)
        except FileNotFoundError:
            logging.warning(f"Config file '{args.config}' not found.")

    parser = argparse.ArgumentParser(
        description="Pipeline-based anomaly detection experiments",
        parents=[conf_parser]
    )
    parser.add_argument("--base_dir", help="Base directory for the dataset")
    parser.add_argument("--exp-dir", help="Experiments output directory")
    parser.add_argument("--dataset", choices=DATASET_LIST)
    parser.add_argument("--mission-id", type=int)
    parser.add_argument("--model", choices=MODEL_LIST)
    parser.add_argument("--segmentator", action="store_true")
    parser.add_argument("--feature-extractor", choices=FEATURE_EXTRACTOR_LIST)
    parser.add_argument("--channels", type=str, nargs="+")
    parser.add_argument("--n-kernel", type=int)
    parser.add_argument("--dpmm-type", type=str)
    parser.add_argument("--dpmm-mode", type=str)
    parser.add_argument("--window-size", type=int)
    parser.add_argument("--step-size", type=int)
    parser.add_argument("--min-window", type=int)
    parser.add_argument("--max-window", type=int)
    parser.add_argument("--perc-step-size", type=float)
    parser.add_argument("--ndpm_config", type=str)
    parser.add_argument("--detector", choices=["threshold", "quantile", "molookde", "none"], default="threshold")
    parser.add_argument("--seed", type=int)

    parser.set_defaults(**defaults)
    parsed_args, extra_argv = parser.parse_known_args(remaining_argv)

    if not parsed_args.base_dir:
        parser.error("--base_dir is required (in config.yaml or via CLI)")
    if not parsed_args.dataset:
        parser.error("--dataset is required (in config.yaml or via CLI)")
    if not parsed_args.model:
        parser.error("--model is required (in config.yaml or via CLI)")

    return parsed_args, extra_argv


def run_exp(args, other_args=None):
    """Run experiment using the modular AnomalyDetectionPipeline."""
    set_seed(getattr(args, 'seed', 42))

    detector = create_detector(args)
    if detector is None:
        args.detector = "no_detector"

    handler = CallbackHandler([SystemMonitorCallback()], call_every_ms=100)

    run_id = f"pipeline_{args.feature_extractor}_{args.dataset}_{args.model}_{args.detector}"

    wrapper_params = getattr(args, 'wrapper_params', {})
    eval_perc = getattr(args, 'eval_perc', wrapper_params.get('eval_perc', None))

    benchmark = get_dataset_benchmark(
        dataset_name=args.dataset,
        data_path=args.base_dir,
        exp_dir=args.exp_dir,
        run_id=run_id,
        mission_id=args.mission_id,
    )
    channels = benchmark.channels if args.channels is None else args.channels

    for channel_name in channels:

        timeseries_splitter = TimeSeriesSplitter(
            window_size=args.window_size,
            step_size=args.step_size,
            min_window=getattr(args, 'min_window', None) or 10,
            max_window=getattr(args, 'max_window', None) or 300,
            perc_step_size=getattr(args, 'perc_step_size', None) or 1.0,
        )

        feature_extractor = get_feature_extractor(
            args.feature_extractor,
            window_size=args.window_size,
            stride=args.step_size,
            n_kernel=getattr(args, 'n_kernel', None),
            **getattr(args, 'feature_extraction_params', getattr(args, 'fe_params', {}))
        )

        classifier, is_supervised = create_classifier(args, other_args)


        pipeline = AnomalyDetectionPipeline(
            [
                ("timeseries_splitter", timeseries_splitter),
                ("feature_extractor", feature_extractor),
                ("classifier", classifier),
                ("detector", detector),
            ],
            callback_handler=handler,
            eval_perc=eval_perc,
        )

        fitted_pipeline, fitting_metrics = benchmark.fit_channel(
            channel_id=channel_name,
            classifier=pipeline,
        )
        print(fitting_metrics)

        results = benchmark.test_channel(
            channel_id=channel_name,
            classifier=fitted_pipeline,
        )

    if isinstance(benchmark, ESABenchmark):
        results = benchmark.compute_global_event_metrics(channels=channels)
        print(results)


def main():
    """Main function."""
    args, other_args = parse_exp_args()
    run_exp(args, other_args)


if __name__ == "__main__":
    main()
