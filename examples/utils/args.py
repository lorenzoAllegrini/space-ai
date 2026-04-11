import argparse
import logging
import yaml

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
            logging.warning(f"Config file '{args.config}' not found. Using hardcoded defaults.")

    parser = argparse.ArgumentParser(
        description="Experiments execution",
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
    parser.add_argument("--replay", choices=["time-decay", "none"], default="time-decay")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    parser.add_argument("--challenge", action="store_true", help="Enable challenge mode")
    parser.add_argument("--run-id", type=str, help="Experiment run ID")
    
    # Custom dates
    parser.add_argument("--train-start-date", type=str, help="Custom start date for training")
    parser.add_argument("--train-end-date", type=str, help="Custom end date for training")
    parser.add_argument("--test-start-date", type=str, help="Custom start date for testing")
    parser.add_argument("--test-end-date", type=str, help="Custom end date for testing")

    parser.set_defaults(**defaults)
    parsed_args, extra_argv = parser.parse_known_args(remaining_argv)

    if not parsed_args.base_dir:
        parser.error("The --base_dir parameter is required (in config.yaml or via CLI)")
    if not parsed_args.dataset:
        parser.error("The --dataset parameter is required (in config.yaml or via CLI)")
    if not parsed_args.model:
        parser.error("The --model parameter is required (in config.yaml or via CLI)")

    return parsed_args, extra_argv
