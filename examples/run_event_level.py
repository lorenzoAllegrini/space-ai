"""Event-level benchmark execution module."""

import argparse
import os
import warnings

import torch

from spaceai.benchmark import (
    ESABenchmark,
    OPSSATBenchmark,
)
from spaceai.data import ESAMissions, OPSSAT
from spaceai.preprocessing import (
    TimeSeriesSplitter,
    get_feature_extractor,
)
from spaceai.benchmark.callbacks import SystemMonitorCallback
from utils.model_creators import (
    create_classifier,
)
from xgboost import XGBClassifier
warnings.simplefilter("ignore", FutureWarning)

DATASET_LIST = ["ops", "esa"]
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
]
DPMM_MODEL_TYPE = ["full", "diagonal", "single", "unit"]
DPMM_MODE = ["likelihood_threshold", "cluster_labels"]
FEATURE_EXTRACTOR_LIST = ["none", "base_statistics", "rocket"]


def parse_args():
    """Parse benchmark arguments."""
    parser = argparse.ArgumentParser(description="Event-level benchmark execution")
    parser.add_argument("--base-dir", required=True, help="Path to dataset root")
    parser.add_argument("--exp-dir", default="experiments", help="Directory for results")
    parser.add_argument("--dataset", choices=DATASET_LIST, required=True, help="Dataset to benchmark")
    parser.add_argument("--mission", default=1, help="Mission of ESA dataset to benchmark")
    parser.add_argument("--model", choices=MODEL_LIST, required=True, help="Anomaly detection model")
    
    
    # Preprocessing
    parser.add_argument("--segmentator", action="store_true", help="Use sliding window segmentation")
    parser.add_argument("--window-size", type=int, default=100, help="Window size for segmentation")
    parser.add_argument("--step-size", type=int, default=50, help="Step size for segmentation")
    parser.add_argument(
        "--feature-extractor", 
        choices=FEATURE_EXTRACTOR_LIST, 
        default="none",
        help="Feature extraction method"
    )
    
    # Model specific
    parser.add_argument("--n-kernel", type=int, default=10000, help="Number of kernels for ROCKET")
    parser.add_argument("--dpmm-type", choices=DPMM_MODEL_TYPE, default="full")
    parser.add_argument("--dpmm-mode", choices=DPMM_MODE, default="likelihood_threshold")
    
    return parser.parse_known_args()


def run_benchmark(args, other_args=None):
    """Run the benchmark."""

    classifier_factory, is_supervised = create_classifier(args, other_args)

    run_id = f"{args.dataset}_{args.model}"
    if args.model == "dpmm":
        run_id += f"_{args.dpmm_type}_{args.dpmm_mode}"
    if args.segmentator:
        run_id += f"_win{args.window_size}_step{args.step_size}"

    if args.dataset == "esa":
        mission = ESAMissions.MISSION_1.value if args.mission == 1 else ESAMissions.MISSION_2.value
        benchmark = ESABenchmark(
            data_root=args.base_dir,
            mission=mission,
            run_id=run_id,
            exp_dir=args.exp_dir,
        )
        
        target_channels = [f'channel_{n}' for n in range(9,12)]

        # Train all channels
        for channel_id in target_channels:
            benchmark.fit_channel(
                channel_id=channel_id,
                classifier=classifier_factory(),
            )

        for channel_id in target_channels:
            benchmark.test_channel(channel_id=channel_id)

        # aggregate global event-level metrics
        results = benchmark.compute_global_event_metrics(channels=target_channels)
        print(results)

    elif args.dataset == "ops":
        benchmark = OPSSATBenchmark(
            data_root=args.base_dir,
            run_id=run_id,
            exp_dir=args.exp_dir,
            split_percentage=None,
        )

        channels = benchmark.get_default_channels()

        # Train all channels
        for channel_id in channels:
            benchmark.fit_channel(
                channel_id=channel_id,
                classifier=classifier_factory(),
            )

        # Test all channels
        for channel_id in channels:
            benchmark.test_channel(
                channel_id=channel_id,
            )

        # Aggregate global event-level metrics
        results = benchmark.compute_global_event_metrics(channels=channels)
        print(results)


def main():
    """Main function."""
    args, other_args = parse_args()
    run_benchmark(args, other_args)


if __name__ == "__main__":
    main()
