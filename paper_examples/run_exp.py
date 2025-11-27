from config import OMP_NUM_THREADS
import os
os.environ['OMP_NUM_THREADS'] = f'{OMP_NUM_THREADS}'

import argparse
import sys

from model_creators import *
from dataset_exp import *
from spaceai.models.anomaly_classifier import RocketClassifier

DATASET_LIST = ["ops", "nasa", "esa"]
MODEL_LIST = ['ocsvm', 'xgboost', 'ridge_regression', 'dpmm'] #'rockad'
DPMM_MODEL_TYPE = ['full', 'diagonal', 'single', 'unit']
DPMM_MODE = ['likelihood_threshold', 'cluster_labels']
SEGMENTATOR_LIST = ["base_stats", "rocket"]


def format(s):
    if '_' not in s:
        return s.lower()
    else:
        l = s.split('_')
        return ''.join([l[0].lower()]+[x.capitalize() for x in l[1:]])


def parse_exp_args(str_args=None):
    parser = argparse.ArgumentParser(description="Esecuzione esperimenti paper")
    #parser.add_argument("--base_dir", required=True)
    parser.add_argument("--exp-dir", default='experiments')
    parser.add_argument("--dataset", choices=DATASET_LIST, required=True)
    parser.add_argument("--model", choices=MODEL_LIST, required=True)
    parser.add_argument("--segmentator", choices=SEGMENTATOR_LIST, required=True)
    parser.add_argument("--n-kernel", type=int)
    parser.add_argument("--dpmm-type", choices=DPMM_MODEL_TYPE)
    parser.add_argument("--dpmm-mode", choices=DPMM_MODE)
    return parser.parse_known_args(str_args)

def run_exp(args, other_args=None, suppress_output=False):
    # create the classifier
    model_id = format(args.model)
    if args.model == "dpmm":
        classifier, is_supervised = get_dpmm_classifier(args.dpmm_type, args.dpmm_mode, other_args)
        model_id += f"_{format(args.dpmm_type)}_{format(args.dpmm_mode)}"
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

    # add kernels if needed
    extract_features = True if args.model != 'rockad' else False

    if args.segmentator == "rocket" and args.model != 'rockad':
        classifier = RocketClassifier(base_model=classifier, num_kernels=args.n_kernel)
        extract_features = False

    # create the exp
    if suppress_output:
        sys.stdout = open(os.devnull, "w")
        sys.stderr = open(os.devnull, "w")

    exp_dir = args.exp_dir
    base_dir = f'{format(args.dataset)}_{format(args.segmentator)}_{model_id}'
    result_path = os.path.join(exp_dir, base_dir)
    if os.path.exists(result_path):
        raise FileExistsError(f'Exp folder already exists! Remove the folder to execute the experiment again!')

    if args.dataset == "esa":
        run_esa_exp(classifier, is_supervised, extract_features, exp_dir, base_dir)
    elif args.dataset == "nasa":
        if args.model in ['xgboost', 'ridge_regression']:
            raise ValueError("NASA dataset supporta solo metodi unsupervised!")
        run_nasa_exp(classifier, extract_features, exp_dir, base_dir)
    elif args.dataset == "ops":
        run_ops_exp(classifier, is_supervised, extract_features, exp_dir, base_dir)
    else:
        raise ValueError(f"Dataset {args.dataset} non supportato!")

if __name__ == "__main__":
    args, other_args = parse_exp_args()
    run_exp(args, other_args)


