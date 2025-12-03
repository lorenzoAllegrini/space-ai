"""Run prediction experiment module."""

import argparse
import os
import sys
import warnings

from spaceai.benchmark.callbacks import SystemMonitorCallback

from .utils.config import Config, OMP_NUM_THREADS
from .utils.dataset_exp import get_dataset_benchmark, run_prediction_experiment
from .utils.model_creators import (
    create_predictor,
    get_telemanom_detector,
    format_str,
)

os.environ["OMP_NUM_THREADS"] = f"{OMP_NUM_THREADS}"

warnings.simplefilter("ignore", FutureWarning)

DATASET_LIST = ["ops", "nasa", "esa"]
MODEL_LIST = ["esn", "lstm"]


def parse_exp_args(str_args=None):
    """Parse experiment arguments."""
    parser = argparse.ArgumentParser(description="prediction experiments execution")
    parser.add_argument("--base_dir", required=True)
    parser.add_argument("--exp-dir", default="experiments")
    parser.add_argument("--dataset", choices=DATASET_LIST, required=True)
    parser.add_argument("--model", choices=MODEL_LIST, required=True)
    
    # Config overrides
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--window-size", type=int) # l_s / seq_length
    parser.add_argument("--prediction-steps", type=int) # n_predictions
    
    return parser.parse_known_args(str_args)


def run_exp(args, other_args=None):
    """Run experiment."""
    config = Config()
    
    # Override config with args
    if args.epochs: config.epochs = args.epochs
    if args.batch_size: config.batch_size = args.batch_size
    if args.learning_rate: config.learning_rate = args.learning_rate
    if args.window_size: config.l_s = args.window_size
    if args.prediction_steps: config.n_predictions = args.prediction_steps

    predictor_factory = create_predictor(args.model, config)
    detector_factory = get_telemanom_detector(config)

    benchmark = get_dataset_benchmark(
        dataset_name=args.dataset,
        data_path=args.base_dir,
        exp_dir=args.exp_dir,
        run_id=f"{args.dataset}_{args.model}_pred",
        n_predictions=config.n_predictions,
    )

    callbacks = [SystemMonitorCallback()]

    run_prediction_experiment(
        benchmark=benchmark,
        predictor_factory=predictor_factory,
        detector_factory=detector_factory,
        config=config,
        callbacks=callbacks,
    )


def main():
    """Main function."""
    args, other_args = parse_exp_args()
    run_exp(args, other_args)


if __name__ == "__main__":
    main()
