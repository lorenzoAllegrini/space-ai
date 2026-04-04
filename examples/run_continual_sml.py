"""Run SML Continual Experiment module."""

import argparse
import warnings
import logging
import yaml
import numpy as np

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
from utils.reproducibility import set_seed
from spaceai.benchmark.callbacks import SystemMonitorCallback, CallbackHandler
from spaceai.models.anomaly_classifier.rolling_window_classifier import RollingWindowClassifier
from spaceai.models.anomaly_classifier.sml_client_classifier import SMLClientClassifier
from spaceai.benchmark import ESABenchmark
warnings.simplefilter("ignore", FutureWarning)

DATASET_LIST = ["ops", "nasa", "esa"]
MODEL_LIST = ["ocsvm", "xgboost", "ridge_regression", "dpmm", "iforest", "pca", "knn", "lof", "pyod_ocsvm", "ecod", "copod", "cblof", "hbos", "ndpm"]
FEATURE_EXTRACTOR_LIST = ["none", "base_statistics", "rocket"]

def parse_sml_args(str_args=None):
    """Parse args for SML experiment."""
    conf_parser = argparse.ArgumentParser(add_help=False)
    conf_parser.add_argument("--config", type=str, default=None)
    args, remaining_argv = conf_parser.parse_known_args(str_args)

    defaults = {
        "exp_dir": "experiments_sml_continual",
        "mission_id": 1,
        "feature_extractor": "none",
        "window_size": 50,
        "step_size": 50,
        "experience_size": "30D",
        "drift_detector": True,
        "seed": 42,
        "server_ip": "localhost",
        "server_port": 5555
    }

    if args.config:
        with open(args.config, "r") as f:
            yaml_config = yaml.safe_load(f)
            if yaml_config: defaults.update(yaml_config)

    parser = argparse.ArgumentParser(description="SML Bridge Continual Experiment", parents=[conf_parser])
    parser.add_argument("--base_dir", help="Base directory for the dataset")
    parser.add_argument("--exp-dir", help="Experiments output directory")
    parser.add_argument("--dataset", choices=DATASET_LIST)
    parser.add_argument("--mission-id", type=int)
    parser.add_argument("--model", choices=MODEL_LIST)
    parser.add_argument("--feature-extractor", choices=FEATURE_EXTRACTOR_LIST)
    parser.add_argument("--channels", type=str, nargs="+")
    parser.add_argument("--window-size", type=int)
    parser.add_argument("--step-size", type=int)
    parser.add_argument("--experience-size", type=str, help="Size of each experience (e.g. '30D')")
    parser.add_argument("--drift-detector", action="store_true", help="Enable drift detection & adaptation")
    parser.add_argument("--detector", choices=["threshold", "molookde", "none"], default="threshold")
    parser.add_argument("--server-ip", type=str)
    parser.add_argument("--server-port", "--port", type=int)
    parser.add_argument("--seed", type=int)

    parser.set_defaults(**defaults)
    return parser.parse_known_args(remaining_argv)

def run_sml_continual():
    args, other_args = parse_sml_args()
    set_seed(args.seed)

    from spaceai.models.anomaly import ThresholdDetector, MoLooKDEDetector
    detector_params = getattr(args, 'detector_params', {})
    if args.detector == "threshold":
        detector = ThresholdDetector(**{**dict(threshold=0.9), **detector_params})
    elif args.detector == "molookde":
        detector = MoLooKDEDetector(**{**dict(alpha=0.001), **detector_params})
    else:
        detector = None
    
    handler = CallbackHandler([SystemMonitorCallback()], call_every_ms=100)

    # Generate run_id
    run_id = f"SML_continual_{args.model}"
    for attr in ['type', 'mode']:
        val = getattr(args, f"{args.model}_{attr}", getattr(args, attr, None))
        if val: run_id += f"_{val}"
    run_id += f"_{args.detector}"
    
    benchmark = get_dataset_benchmark(
        dataset_name=args.dataset,
        data_path=args.base_dir,
        exp_dir=args.exp_dir,
        run_id=run_id,
        mission_id=args.mission_id,
    )
    
    channels = benchmark.channels if args.channels is None else args.channels
    wrapper_params = getattr(args, 'wrapper_params', {})
    eval_perc = getattr(args, 'eval_perc', wrapper_params.get('eval_perc', None))

    for channel_name in channels:
        logging.info("--- Starting SML Continual Pipeline for Channel %s ---", channel_name)
        
        ts_splitter = TimeSeriesSplitter(
            window_size=args.window_size,
            step_size=args.step_size,
            min_window=getattr(args, 'min_window', 10),
            max_window=getattr(args, 'max_window', 300),
            perc_step_size=getattr(args, 'perc_step_size', 1.0),
        )
        
        feature_extractor = get_feature_extractor(
            args.feature_extractor,
            window_size=args.window_size,
            stride=args.step_size,
            n_kernel=getattr(args, 'n_kernel', None),
            **getattr(args, 'fe_params', {})
        )

        base_classifier, is_supervised = create_classifier(args, other_args)

        # In SML mode, the server initializes the Adaptive/Rolling wrapper 
        # based on the recipe we send.
        sml_client = SMLClientClassifier(
            server_ip=args.server_ip,
            port=args.server_port,
            channel_id=channel_name,
            base_classifier=base_classifier, # Just for local reference
            args=(args, other_args)
        )

        # Dataset-specific parameters
        dataset_kwargs = {}
        if args.dataset == "esa":
            dataset_kwargs["use_telecommands"] = getattr(args, "use_telecommands", False)

        logging.info("[CLIENT] Initial remote FIT for channel %s...", channel_name)
        fitted_client, fitting_metrics = benchmark.fit_channel(
            channel_id=channel_name,
            classifier=sml_client,
            **dataset_kwargs
        )
        
        logging.info("[CLIENT] Starting CONTINUAL STREAMING for channel %s...", channel_name)
        experience_log = benchmark.test_continual(
            channel_id=channel_name,
            classifier=fitted_client,
            experience_size=args.experience_size,
            **dataset_kwargs
        )
        
        # Log summary
        logging.info("Continual test completed for %s. Experiences processed: %d", 
                     channel_name, len(experience_log))

    if isinstance(benchmark, ESABenchmark):
        results = benchmark.compute_global_event_metrics(channels=channels)
        print(f"Global ESA Metrics: {results}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [SML-CONTINUAL] %(message)s")
    run_sml_continual()
