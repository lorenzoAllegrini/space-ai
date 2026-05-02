"""Run experiment module."""

import argparse
import logging
import warnings

from spaceai.benchmark import ESABenchmark, NASABenchmark, OPSSATBenchmark
from spaceai.data import ESAMissions
from spaceai.preprocessing import TimeSeriesSplitter, get_feature_extractor

warnings.simplefilter("ignore", FutureWarning)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [CLIENT] %(message)s")

DATASET_LIST = ["ops-sat", "nasa", "esa_m1", "esa_m2"]
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
    "ndpm",
]
DPMM_MODEL_TYPE = ["full", "diagonal", "single", "unit"]
DPMM_MODE = ["likelihood_threshold", "cluster_labels"]
FEATURE_EXTRACTOR_LIST = ["none", "base_statistics", "rocket"]


def parse_exp_args(str_args=None):
    """Parse experiment arguments."""
    parser = argparse.ArgumentParser(description="SML Inference Client")
    parser.add_argument("--server_ip", required=True)
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--base_dir", default="datasets")
    parser.add_argument("--exp-dir", default="experiments")
    parser.add_argument("--dataset", choices=DATASET_LIST, required=True)
    parser.add_argument("--model", choices=MODEL_LIST, required=True)
    parser.add_argument("--segmentator", action="store_true")
    parser.add_argument(
        "--feature-extractor", choices=FEATURE_EXTRACTOR_LIST, default="none"
    )
    parser.add_argument("--channel", type=str, default="all")
    parser.add_argument("--n-kernel", type=int)
    parser.add_argument("--dpmm-type", choices=DPMM_MODEL_TYPE)
    parser.add_argument("--dpmm-mode", choices=DPMM_MODE)
    parser.add_argument("--window-size", type=int, default=100)
    parser.add_argument("--step-size", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sample-rate-ms", type=int, default=0)
    parser.add_argument("--max-duration-s", type=int, default=None)
    return parser.parse_known_args(str_args)


def create_benchmark(args):
    """Create a Benchmark instance matching the test configuration."""
    segmentator = None
    if args.segmentator:
        segmentator = TimeSeriesSplitter(
            window_size=args.window_size,
            step_size=args.step_size,
        )

    feature_extractor = get_feature_extractor(
        args.feature_extractor,
        window_size=args.window_size,
        stride=args.step_size,
        n_kernel=args.n_kernel,
    )

    run_id = f"{args.dataset}_{args.model}" if args.model != "dpmm" else f"{args.dataset}_{args.model}_{args.dpmm_type}_{args.dpmm_mode}"

    if args.dataset == "ops-sat":
        return OPSSATBenchmark(
            data_root=args.base_dir,
            segmentator=segmentator,
            feature_extractor=feature_extractor,
            run_id=run_id,
            exp_dir=args.exp_dir,
            split_percentage=None,
        )
    elif args.dataset == "nasa":
        return NASABenchmark(
            data_root=args.base_dir,
            segmentator=segmentator,
            feature_extractor=feature_extractor,
            run_id=run_id,
            exp_dir=args.exp_dir,
        )
    elif args.dataset.startswith("esa"):
        mission = ESAMissions.MISSION_1.value if args.dataset.endswith("1") else ESAMissions.MISSION_2.value
        return ESABenchmark(
            data_root=args.base_dir,
            segmentator=segmentator,
            feature_extractor=feature_extractor,
            mission=mission,
            run_id=run_id,
            exp_dir=args.exp_dir,
        )


def main():
    args, _other_args = parse_exp_args()

    benchmark = create_benchmark(args)

    channels = [args.channel] if args.channel != "all" else getattr(benchmark, 'get_default_channels', lambda: [])()
    if not channels and hasattr(benchmark, 'target_channels'):
        channels = benchmark.target_channels

    logging.info("Starting stream to %s:%d...", args.server_ip, args.port)
    for channel_id in channels:
        from spaceai.models.legacy.sml_client_classifier import SMLClientClassifier
        classifier = SMLClientClassifier(server_ip=args.server_ip, port=args.port, channel_id=channel_id)
        benchmark.test_continual(
            channel_id=channel_id,
            classifier=classifier,
            experience_size="10D", # Default or configurable
        )
    
    results = benchmark.compute_global_event_metrics(channels=channels)
    print(results)


if __name__ == "__main__":
    main()
