"""Run continual experiment module."""

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
)
from utils.model_creators import (
    create_classifier,
)
from utils.reproducibility import set_seed
from spaceai.benchmark.callbacks import SystemMonitorCallback, CallbackHandler
from spaceai.models.anomaly_classifier.adaptive_rolling_window_classifier import AdaptiveRollingWindowClassifier
from spaceai.models.drift_detectors.adwin_detector import ADWINDetector
from spaceai.models.drift_detectors.utils.replay_buffers import TimeDecayReplayBuffer
from spaceai.models.drift_detectors.utils.filters import SafeRampUpFilter
from spaceai.benchmark import ESABenchmark
from spaceai.models.anomaly import ThresholdDetector, MoLooKDEDetector

warnings.simplefilter("ignore", FutureWarning)
warnings.simplefilter("ignore", RuntimeWarning)

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
    conf_parser.add_argument("--config", type=str, default="dpmm_unit_config.yaml", 
                             help="Path to the YAML configuration file")
    
    # ParseKnownArgs estrae --config e lascia il resto intatto
    args, remaining_argv = conf_parser.parse_known_args(str_args)

    defaults = {
        "exp_dir": "experiments",
        "mission_id": 1,
        "feature_extractor": "none",
        "drift_detector": False,
        "window_size": 100,
        "step_size": 100,
        "experience_size": "30D",
        "seed": 42,
    }

    if args.config:
        try:
            with open(args.config, "r") as f:
                yaml_config = yaml.safe_load(f)
                if yaml_config:
                    defaults.update(yaml_config)
        except FileNotFoundError:
            logging.warning(f"Config file '{args.config}' non trovato. Uso i default hardcoded e la CLI.")

    parser = argparse.ArgumentParser(
        description="Continual learning experiments execution",
        parents=[conf_parser]
    )
    
    parser.add_argument("--base_dir", help="Base directory for the dataset")
    parser.add_argument("--exp-dir", help="Experiments output directory")
    parser.add_argument("--dataset", choices=DATASET_LIST)
    parser.add_argument("--mission-id", type=int)
    parser.add_argument("--model", choices=MODEL_LIST)
    parser.add_argument("--segmentator", action="store_true")
    parser.add_argument("--feature-extractor", choices=FEATURE_EXTRACTOR_LIST)
    parser.add_argument("--drift-detector", type=bool) 
    parser.add_argument("--channels", type=str, nargs="+")
    parser.add_argument("--n-kernel", type=int)
    parser.add_argument("--dpmm-type", choices=DPMM_MODEL_TYPE)
    parser.add_argument("--dpmm-mode", choices=DPMM_MODE)
    parser.add_argument("--window-size", type=int)
    parser.add_argument("--step-size", type=int)
    parser.add_argument("--experience-size", type=str, help="Size of each experience (e.g. '30D')")
    parser.add_argument("--detector", choices=["threshold", "molookde", "none"], default="threshold")
    parser.add_argument("--min-window", type=int)
    parser.add_argument("--max-window", type=int)
    parser.add_argument("--perc-step-size", type=float)
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")

    parser.set_defaults(**defaults)
    parsed_args, other_args = parser.parse_known_args(remaining_argv)

    if not parsed_args.base_dir:
        parser.error("Il parametro --base_dir è obbligatorio (nel config.yaml o via CLI)")
    if not parsed_args.dataset:
        parser.error("Il parametro --dataset è obbligatorio (nel config.yaml o via CLI)")
    if not parsed_args.model:
        parser.error("Il parametro --model è obbligatorio (nel config.yaml o via CLI)")

    return parsed_args, other_args


def run_exp(args, other_args=None):
    """Run experiment."""
    set_seed(getattr(args, 'seed', 42))

    handler = CallbackHandler([SystemMonitorCallback()], call_every_ms=100)

    ts_splitter = TimeSeriesSplitter(
        window_size=args.window_size,
        step_size=args.step_size,
        min_window=getattr(args, 'min_window', None) or 10,
        max_window=getattr(args, 'max_window', None) or 300,
        perc_step_size=getattr(args, 'perc_step_size', None) or 1.0,
        callback_handler=handler,
    )
    print(f"[DEBUG] TimeSeriesSplitter: window_size={ts_splitter.window_size_raw}, min_window={ts_splitter.min_window}, max_window={ts_splitter.max_window}, perc_step_size={ts_splitter.perc_step_size}")
        
    feature_extractor = get_feature_extractor(
        args.feature_extractor, 
        window_size=args.window_size, 
        stride=args.step_size, 
        callback_handler=handler,
        n_kernel=args.n_kernel,
        **getattr(args, 'fe_params', {})  # Inietta automaticamente 'selected_features'!
    )

    classifier, is_supervised = create_classifier(args, other_args, callback_handler=handler)

    run_id = f"continual_{args.dataset}_{args.model}"
    if args.model == "dpmm":
        run_id += f"_{args.dpmm_type}_{args.dpmm_mode}"
    
    drift_detector = None
    if args.drift_detector:
        drift_detector = ADWINDetector(
            delta=0.2,  
            filters=[SafeRampUpFilter(lookahead_steps=5, max_safe_score=0.4)],
        )
    # 1. Configurazione Replay Buffer
    replay_params = getattr(args, 'replay_params', {})
    # Merge con i default
    default_replay = dict(max_size=100000, half_life_segments="180D", min_prob=1e-6)
    replay_buffer = TimeDecayReplayBuffer(**{**default_replay, **replay_params})

    # 2. Configurazione Detector
    detector_params = getattr(args, 'detector_params', {})
    detector = None
    if args.detector == "threshold":
        detector = ThresholdDetector(**{**dict(threshold=0.9, callback_handler=handler), **detector_params})
    elif args.detector == "molookde":
        detector = MoLooKDEDetector(**{**dict(alpha=0.05, callback_handler=handler), **detector_params})
    else:  # "none"
        detector = None

    # 3. Configurazione Adaptive Wrapper
    wrapper_params = getattr(args, 'wrapper_params', {})
    rolling_window_classifier = AdaptiveRollingWindowClassifier(
        drift_detector=drift_detector,
        replay_buffer=replay_buffer,
        base_classifier=classifier,
        supervised_classifier=is_supervised,
        ts_splitter=ts_splitter,
        feature_extractor=feature_extractor,
        callback_handler=handler,
        detector=detector,
        **wrapper_params
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
