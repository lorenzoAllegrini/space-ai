"""Run experiment module."""

import argparse
import os
import sys
import warnings

from spaceai.models.anomaly_classifier import RocketClassifier

from .config import OMP_NUM_THREADS
from .dataset_exp import run_esa_exp, run_nasa_exp, run_ops_exp
from .model_creators import (
    get_dpmm_classifier,
    get_ocsvm_classifier,
    get_ridge_regression_classifier,
    get_rockad_classifier,
    get_xgboost_classifier,
)

os.environ["OMP_NUM_THREADS"] = f"{OMP_NUM_THREADS}"

warnings.simplefilter("ignore", FutureWarning)

DATASET_LIST = ["ops", "nasa", "esa"]
MODEL_LIST = ["ocsvm", "xgboost", "ridge_regression", "dpmm"]  #'rockad'
DPMM_MODEL_TYPE = ["full", "diagonal", "single", "unit"]
DPMM_MODE = ["likelihood_threshold", "cluster_labels"]
SEGMENTATOR_LIST = ["base_stats", "rocket"]


def format_str(s):
    """Format string to CamelCase."""
    if "_" not in s:
        return s.lower()

    l = s.split("_")
    return "".join([l[0].lower()] + [x.capitalize() for x in l[1:]])


def parse_exp_args(str_args=None):
    """Parse experiment arguments."""
    parser = argparse.ArgumentParser(description="paper experiments execution")
    parser.add_argument("--base_dir", required=True)
    parser.add_argument("--exp-dir", default="experiments")
    parser.add_argument("--dataset", choices=DATASET_LIST, required=True)
    parser.add_argument("--model", choices=MODEL_LIST, required=True)
    parser.add_argument("--segmentator", choices=SEGMENTATOR_LIST, required=True)
    parser.add_argument("--n-kernel", type=int)
    parser.add_argument("--dpmm-type", choices=DPMM_MODEL_TYPE)
    parser.add_argument("--dpmm-mode", choices=DPMM_MODE)
    return parser.parse_known_args(str_args)


def create_classifier(args, other_args):
    """Create the classifier based on arguments."""
    model_id = format_str(args.model)
    if args.model == "dpmm":
        classifier, is_supervised = get_dpmm_classifier(
            args.dpmm_type, args.dpmm_mode, other_args
        )
        model_id += f"_{format_str(args.dpmm_type)}_{format_str(args.dpmm_mode)}"
    elif args.model == "ocsvm":
        classifier, is_supervised = get_ocsvm_classifier()
    elif args.model == "rockad":
        classifier, is_supervised = get_rockad_classifier(args.n_kernel)
    elif args.model == "xgboost":
        classifier, is_supervised = get_xgboost_classifier()
    elif args.model == "ridge_regression":
        classifier, is_supervised = get_ridge_regression_classifier()
    else:
        raise ValueError(f"Modello {args.model} non supportato!")
    return classifier, is_supervised, model_id


def run_exp(args, other_args=None, suppress_output=False):
    """Run experiment."""
    # create the classifier
    classifier, is_supervised, model_id = create_classifier(args, other_args)

    # add kernels if needed
    extract_features = args.model != "rockad"

    if args.segmentator == "rocket" and args.model != "rockad":
        classifier = RocketClassifier(base_model=classifier, num_kernels=args.n_kernel)
        extract_features = False

    # create the exp
    if suppress_output:
        sys.stdout = open(os.devnull, "w", encoding="utf-8")
        sys.stderr = open(os.devnull, "w", encoding="utf-8")

    exp_dir = args.exp_dir
    base_dir = f"{format_str(args.dataset)}_{format_str(args.segmentator)}_{model_id}"
    result_path = os.path.join(exp_dir, base_dir)
    if os.path.exists(result_path):
        raise FileExistsError(
            "Exp folder already exists! Remove the folder to execute the experiment again!"
        )

    if args.dataset == "esa":
        run_esa_exp(classifier, is_supervised, extract_features, exp_dir, base_dir)
    elif args.dataset == "nasa":
        if args.model in ["xgboost", "ridge_regression"]:
            raise ValueError("NASA supports only unsupervised methods!")
        run_nasa_exp(classifier, extract_features, exp_dir, base_dir)
    elif args.dataset == "ops":
        run_ops_exp(classifier, is_supervised, extract_features, exp_dir, base_dir)
    else:
        raise ValueError(f"Dataset {args.dataset} not supported!")


if __name__ == "__main__":
    parsed_args, other_parsed_args = parse_exp_args()
    run_exp(parsed_args, other_parsed_args)
