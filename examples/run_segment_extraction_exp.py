"""Run experiment module."""

import argparse
import warnings
import logging
import yaml

from spaceai.preprocessing import (
    TimeSeriesSplitter,
    get_feature_extractor,
)

from utils.dataset_exp import (
    get_dataset_benchmark,
    run_dataset_experiment,
)
from utils.model_creators import (
    create_classifier,
)
from utils.reproducibility import set_seed
from spaceai.benchmark.callbacks import SystemMonitorCallback, CallbackHandler
from spaceai.preprocessing.ts_splitter import TimeSeriesSplitter
from spaceai.models.anomaly_classifier.rolling_window_classifier import RollingWindowClassifier
from spaceai.benchmark import Benchmark, ESABenchmark
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
    """Parse experiment arguments with YAML config support."""
    
    # --- FASE 1: Parser preliminare per catturare SOLO il file di config ---
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
            logging.warning(f"Config file '{args.config}' non trovato. Uso i default hardcoded.")

    parser = argparse.ArgumentParser(
        description="Segment extraction experiments execution",
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
    parser.add_argument("--dpmm-type", choices=DPMM_MODEL_TYPE)
    parser.add_argument("--dpmm-mode", choices=DPMM_MODE)
    parser.add_argument("--window-size", type=int)
    parser.add_argument("--step-size", type=int)
    parser.add_argument("--min-window", type=int)
    parser.add_argument("--max-window", type=int)
    parser.add_argument("--perc-step-size", type=float)
    parser.add_argument("--ndpm_config", type=str, help="Path to NDPM config")
    parser.add_argument("--detector", choices=["threshold", "molookde", "none"], default="threshold")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")

    parser.set_defaults(**defaults)
    parsed_args, extra_argv = parser.parse_known_args(remaining_argv)

    if not parsed_args.base_dir:
        parser.error("Il parametro --base_dir è obbligatorio (nel config.yaml o via CLI)")
    if not parsed_args.dataset:
        parser.error("Il parametro --dataset è obbligatorio (nel config.yaml o via CLI)")
    if not parsed_args.model:
        parser.error("Il parametro --model è obbligatorio (nel config.yaml o via CLI)")

    return parsed_args, extra_argv


def run_exp(args, other_args=None, _suppress_output=False):
    """Run experiment."""
    set_seed(getattr(args, 'seed', 40))

    from spaceai.models.anomaly import ThresholdDetector, MoLooKDEDetector
    detector_params = getattr(args, 'detector_params', {})
    detector = None
    if args.detector == "threshold":
        detector = ThresholdDetector(**{**dict(threshold=0.9), **detector_params})
    elif args.detector == "molookde":
        detector = MoLooKDEDetector(**{**dict(alpha=0.001), **detector_params})
    else:  # "none"
        detector = None
        args.detector = "no_detector"
    
    handler = CallbackHandler([SystemMonitorCallback()], call_every_ms=100)

    run_id = f"{args.feature_extractor}_{args.dataset}_{args.model}_{args.detector}"
    if args.model == "dpmm":
        run_id += f"{args.dpmm_type}_{args.dpmm_mode}"
    
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
        ts_splitter = TimeSeriesSplitter(
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
            n_kernel=args.n_kernel,
            **getattr(args, 'feature_extraction_params', getattr(args, 'fe_params', {}))
        )

        classifier, is_supervised = create_classifier(args, other_args)

        rolling_window_classifier = RollingWindowClassifier(
            base_classifier=classifier,
            supervised_classifier=is_supervised,
            ts_splitter=ts_splitter,
            feature_extractor=feature_extractor,
            callback_handler=handler,
            detector=detector,
            eval_perc=eval_perc,
        )

        fitted_classifier, fitting_metrics = benchmark.fit_channel(
            channel_id=channel_name,
            classifier=rolling_window_classifier,
        )
        print(fitting_metrics)
        
        results = benchmark.test_channel(
            channel_id=channel_name,
            classifier=fitted_classifier,
        )
        print(results)

    if isinstance(benchmark, ESABenchmark):
        results = benchmark.compute_global_event_metrics(channels=channels)
        print(results)

def main():
    """Main function."""
    args, other_args = parse_exp_args()
    run_exp(args, other_args)


if __name__ == "__main__":
    main()
