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
from spaceai.models.legacy import RollingWindowClassifier
from spaceai.models.detectors import ThresholdDetector, MoLooKDEDetector
from spaceai.benchmark import Benchmark, ESABenchmark
warnings.simplefilter("ignore", FutureWarning)
import torch


from utils.args import (
    parse_exp_args, 
    DATASET_LIST, 
    MODEL_LIST, 
    DPMM_MODEL_TYPE, 
    DPMM_MODE, 
    FEATURE_EXTRACTOR_LIST
)


# parse_exp_args moved to utils.args


def run_exp(args, other_args=None, _suppress_output=False):
    """Run experiment."""
    set_seed(getattr(args, 'seed', 40))

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

    # -------------------------------------------------------------------------
    # RUN ID GENERATION
    # -------------------------------------------------------------------------
    wrapper_params = getattr(args, 'wrapper_params', {})
    eval_perc = getattr(args, 'eval_perc', wrapper_params.get('eval_perc', None))
    filter_valid = getattr(args, 'filter_valid', wrapper_params.get('filter_valid', wrapper_params.get('filter_valid_for_detector', False)))

    if getattr(args, 'run_id', None) is not None:
        run_id = args.run_id
    else:
        run_id = f"{args.feature_extractor}_{args.dataset}_{args.model}_{args.detector}"
        if args.model == "dpmm":
            run_id += f"_{args.dpmm_type}_{args.dpmm_mode}"
        
        # Add eval_perc if present
        if eval_perc is not None:
            run_id += f"_ep{eval_perc}"
        
        # Add dynamic scaling indicator
        base_params = getattr(args, 'base_classifier_params', {})
        ds = base_params.get('dynamic_scaling', False)
        run_id += f"_ds{'T' if ds else 'F'}"
        
        # Add filter_valid indicator
        if filter_valid:
            run_id += "_fv"
    # -------------------------------------------------------------------------

    benchmark = get_dataset_benchmark(
        dataset_name=args.dataset,
        data_path=args.base_dir,
        exp_dir=args.exp_dir,
        run_id=run_id,
        mission_id=args.mission_id,
        save_metadata=getattr(args, 'save_metadata', True),
    )
    channels = benchmark.channels if args.channels is None else args.channels
     
    for channel_name in channels:
        _ws = args.window_size
        _ss = args.step_size
        _minw = getattr(args, 'min_window', None) or 10
        _maxw = getattr(args, 'max_window', None) or 300
        _pss = getattr(args, 'perc_step_size', None) or 1.0
        _fe_params = getattr(args, 'feature_extraction_params', getattr(args, 'fe_params', {}))
        

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
            filter_valid_for_detector=filter_valid,
        )

        dataset_kwargs = {}
        if hasattr(args, 'challenge'):
            dataset_kwargs["challenge"] = getattr(args, 'challenge')
        fitted_classifier, fitting_metrics = benchmark.fit_channel(
            channel_id=channel_name,
            classifier=rolling_window_classifier,
            **dataset_kwargs
        )
        
        results = benchmark.test_channel(
            channel_id=channel_name,
            classifier=fitted_classifier,
            **dataset_kwargs
        )

    if isinstance(benchmark, ESABenchmark):
        results = benchmark.compute_global_event_metrics(channels=channels)

def main():
    """Main function."""
    args, other_args = parse_exp_args()
    run_exp(args, other_args)


if __name__ == "__main__":
    main()
