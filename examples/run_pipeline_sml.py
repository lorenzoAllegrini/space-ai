"""Run pipeline experiment module.

Uses AnomalyDetectionPipeline instead of RollingWindowClassifier.
Same CLI/YAML config as run_segment_extraction_exp.py.
"""

import argparse
import warnings
import logging
import yaml
import numpy as np
from datetime import datetime

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
    SMLClientPipeline
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
        "window_size": None,
        "step_size": None,
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
    parser.add_argument("--detector", choices=["threshold", "quantile", "molookde", "dpmm_native", "none"], default="threshold")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--run-id", type=str, help="Override automatically generated run_id")
    parser.add_argument("--challenge", action="store_true", help="Active blind inference for challenge")
    parser.add_argument("--skip-existent", action="store_true", help="Skip existing channels if results exist")
    parser.add_argument("--server-ip", type=str, default="127.0.0.1", help="SML server IP")
    parser.add_argument("--port", type=int, default=5557, help="SML server port")

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

    handler = CallbackHandler([SystemMonitorCallback()], call_every_ms=100)
    detector = create_detector(args, callback_handler=handler)

    wrapper_params = getattr(args, 'wrapper_params', {})
    eval_perc = getattr(args, 'eval_perc', getattr(args, 'perc_eval', wrapper_params.get('eval_perc', 0.0)))
    
    # Extract dynamic_scaling for run_id
    base_params = getattr(args, 'base_classifier_params', {})
    ds = "T" if base_params.get('dynamic_scaling', False) else "F"
    ch = "T" if getattr(args, 'challenge', False) else "F"
    
    # Extract sweep hyperparameters for identifier
    lr = base_params.get('lr', 0.1)
    n_cl = base_params.get('n_clusters', 100)
    adp = base_params.get('alpha_dp', 3.0)
    vp = base_params.get('var_prior', 3.0)
    vps = base_params.get('var_prior_strength', 1.0)
    mps = base_params.get('mu_prior_strength', 0.001)
    q = base_params.get('quantile', 0.001)
    
    dp = getattr(args, 'detector_params', {})
    alpha = dp.get('alpha', 0.05)
    pot = dp.get('pot_percentile', 0.0)
    p = dp.get('p', 0.0)
    
    if getattr(args, 'run_id', None):
        run_id = args.run_id
        print(f"Using provided run_id: {run_id}")
    else:
        run_id = f"pipeline_{args.feature_extractor}_{args.dataset}_{args.model}_{args.detector}_lr{lr}_nc{n_cl}_adp{adp}_vp{vp}_vps{vps}_mps{mps}_q{q}_al{alpha}_po{pot}_p{p}_ds{ds}_ch{ch}_ep{eval_perc}"
        print(f"Generated run_id: {run_id}")


    benchmark = get_dataset_benchmark(
        dataset_name=args.dataset,
        data_path=args.base_dir,
        exp_dir=args.exp_dir,
        run_id=run_id,
        mission_id=args.mission_id,
        challenge=args.challenge,
    )
    
    if getattr(args, 'skip_existent', False):
        benchmark.load_incremental_state()
    
    channels = benchmark.channels if args.channels is None else args.channels
    if isinstance(channels, str):
        channels = [channels]

    start_time = datetime.now()
    start_str = start_time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"Starting validation on {len(channels)} channels at {start_str}")

    for channel_name in channels:
        if getattr(args, 'skip_existent', False) and channel_name in benchmark.processed_channels:
            print(f"Skipping already processed channel: {channel_name}")
            continue

        timeseries_splitter = TimeSeriesSplitter(
            window_size=args.window_size,
            step_size=args.step_size,
            min_window=getattr(args, 'min_window', None) or 10,
            max_window=getattr(args, 'max_window', None) or 300,
            perc_step_size=getattr(args, 'perc_step_size', None) or 1.0,
            callback_handler=handler,
        )

        feature_extractor = get_feature_extractor(
            args.feature_extractor,
            window_size=args.window_size,
            stride=args.step_size,
            callback_handler=handler,
            n_kernel=getattr(args, 'n_kernel', None),
            **getattr(args, 'feature_extraction_params', getattr(args, 'fe_params', {}))
        )

        classifier, is_supervised = create_classifier(args, other_args, callback_handler=handler)

        pipeline = SMLClientPipeline(
            local_pipeline=AnomalyDetectionPipeline(
                [
                    ("timeseries_splitter", timeseries_splitter),
                    ("feature_extractor", feature_extractor),
                    ("classifier", classifier),
                    ("detector", detector),
                ],
                callback_handler=handler,
                eval_perc=eval_perc,
            ),
            server_ip=args.server_ip,
            port=args.port,
            channel_id=channel_name,
            callback_handler=handler,
            eval_perc=eval_perc,
        )

        fitted_pipeline, fitting_metrics = benchmark.fit_channel(
            channel_id=channel_name,
            classifier=pipeline,
        )

        if getattr(fitted_pipeline, "kill_switch_active", False):
            logging.warning("Kill switch activated for channel %s. Skipping predictions to avoid polluting results.", channel_name)
            continue

        results = benchmark.test_channel(
            channel_id=channel_name,
            classifier=fitted_pipeline,
        )

    end_time = datetime.now()
    end_str = end_time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"Validation completed at {end_str}")

    if isinstance(benchmark, ESABenchmark):
        results = benchmark.compute_global_event_metrics(
            channels=channels,
            start_validation=start_str,
            end_validation=end_str
        )
        print(results)
    else:
        # For other benchmarks, we still want to save these times
        if hasattr(benchmark, "compute_global_event_metrics"):
             benchmark.compute_global_event_metrics(
                 channels=channels,
                 start_validation=start_str,
                 end_validation=end_str
             )


def main():
    """Main function."""
    args, other_args = parse_exp_args()
    run_exp(args, other_args)


if __name__ == "__main__":
    main()
