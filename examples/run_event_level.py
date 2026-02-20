"""Event-level benchmark execution module."""

import argparse
import warnings

from spaceai.benchmark import (
    ESABenchmark,
    OPSSATBenchmark,
)
from spaceai.data import ESAMissions, OPSSAT
from spaceai.preprocessing import (
    SpaceAISegmentator,
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

    segmentator = None
    if args.segmentator:
        segmentator = SpaceAISegmentator(
            window_size=args.window_size,
            step_size=args.step_size,
        )

    feature_extractor = get_feature_extractor(
        args.feature_extractor, n_kernel=args.n_kernel
    )
    callbacks = [SystemMonitorCallback()]

    run_id = f"{args.dataset}_{args.model}"
    if args.model == "dpmm":
        run_id += f"_{args.dpmm_type}_{args.dpmm_mode}"
    if args.segmentator:
        run_id += f"_win{args.window_size}_step{args.step_size}"

    if args.dataset == "esa":
        mission = ESAMissions.MISSION_1.value if args.mission == 1 else ESAMissions.MISSION_2.value
        benchmark = ESABenchmark(
            data_root=args.base_dir,
            segmentator=segmentator,
            mission=mission,
            feature_extractor=feature_extractor,
            run_id=run_id,
            exp_dir=args.exp_dir,
        )
        
        target_channels = [f'channel_{n}' for n in range(9,12)]

        
        classifier = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            scale_pos_weight=0.02
        )
        
        benchmark.run_event_level(
            channels=target_channels,
            predictor=classifier, 
            callbacks=callbacks,
            supervised=is_supervised
        )

    elif args.dataset == "ops":
        benchmark = OPSSATBenchmark(
            data_root=args.base_dir,
            segmentator=segmentator,
            feature_extractor=feature_extractor,
            run_id=run_id,
            exp_dir=args.exp_dir,
        )
        results = benchmark.run_event_level(
            channels=None, 
            predictor=classifier_factory(),
            supervised=is_supervised
        )

        print(results)


def main():
    """Main function."""
    args, other_args = parse_args()
    run_benchmark(args, other_args)


if __name__ == "__main__":
    main()
