"""Run experiment module."""

import argparse
import warnings

from spaceai.preprocessing import (
    SpaceAISegmentator,
    get_feature_extractor,
)

from .utils.dataset_exp import (
    get_dataset_benchmark,
    run_dataset_experiment,
)
from .utils.model_creators import (
    create_classifier,
)

warnings.simplefilter("ignore", FutureWarning)

DATASET_LIST = ["ops", "nasa", "esa"]
MODEL_LIST = ["ocsvm", "xgboost", "ridge_regression", "dpmm"]
DPMM_MODEL_TYPE = ["full", "diagonal", "single", "unit"]
DPMM_MODE = ["likelihood_threshold", "cluster_labels"]
FEATURE_EXTRACTOR_LIST = ["none", "base_statistics", "rocket"]


def parse_exp_args(str_args=None):
    """Parse experiment arguments."""
    parser = argparse.ArgumentParser(description="paper experiments execution")
    parser.add_argument("--base_dir", required=True)
    parser.add_argument("--exp-dir", default="experiments")
    parser.add_argument("--dataset", choices=DATASET_LIST, required=True)
    parser.add_argument("--model", choices=MODEL_LIST, required=True)
    parser.add_argument("--segmentator", action="store_true")
    parser.add_argument(
        "--feature-extractor", choices=FEATURE_EXTRACTOR_LIST, default="none"
    )
    parser.add_argument("--n-kernel", type=int)
    parser.add_argument("--dpmm-type", choices=DPMM_MODEL_TYPE)
    parser.add_argument("--dpmm-mode", choices=DPMM_MODE)
    parser.add_argument("--window-size", type=int, default=50)
    parser.add_argument("--step-size", type=int, default=50)
    return parser.parse_known_args(str_args)


def run_exp(args, other_args=None, _suppress_output=False):
    """Run experiment."""
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

    benchmark = get_dataset_benchmark(
        dataset_name=args.dataset,
        data_path=args.base_dir,
        exp_dir=args.exp_dir,
        segmentator=segmentator,
        feature_extractor=feature_extractor,
    )

    run_dataset_experiment(
        benchmark=benchmark,
        classifier_factory=classifier_factory,
        is_supervised=is_supervised,
        model_id=args.model,
    )


def main():
    """Main function."""
    args, other_args = parse_exp_args()
    run_exp(args, other_args)


if __name__ == "__main__":
    main()
