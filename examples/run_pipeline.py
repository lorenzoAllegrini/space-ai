"""Run decoupled pipeline experiment module."""

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
from spaceai.benchmark import Benchmark, ESABenchmark
from spaceai.models.anomaly_pipeline.anomaly_classifier import AnomalyDetectionPipeline, PipelineStep, PhaseConfig
import time
from spaceai.models.detectors import ThresholdDetector, MoLooKDEDetector
from spaceai.models.classifiers import DPMMDetector
warnings.simplefilter("ignore", FutureWarning)

from utils.args import parse_exp_args

def run_exp(args, other_args=None):
    """Run decoupled pipeline experiment."""
    set_seed(getattr(args, 'seed', 40))

    wrapper_params = getattr(args, 'wrapper_params', {})
    eval_perc = getattr(args, 'eval_perc', wrapper_params.get('eval_perc', None))
    filter_valid = getattr(args, 'filter_valid', wrapper_params.get('filter_valid', wrapper_params.get('filter_valid_for_detector', False)))

    detector_params = getattr(args, 'detector_params', {})
    detector = None
    if args.detector == "threshold":
        detector = ThresholdDetector(**{**dict(threshold=None, filter_valid=filter_valid), **detector_params})
    elif args.detector == "molookde":
        detector = MoLooKDEDetector(**{**dict(alpha=0.001, filter_valid=filter_valid), **detector_params})
    else:
        detector = None
        args.detector = "no_detector"
    
    handler = CallbackHandler([SystemMonitorCallback()], call_every_ms=100)


    if getattr(args, 'run_id', None) is not None:
        run_id = args.run_id
    else:
        run_id = f"{args.feature_extractor}_{args.dataset}_{args.model}_{args.detector}"
        if args.model == "dpmm":
            run_id += f"_{args.dpmm_type}_{args.dpmm_mode}"
        
        if eval_perc is not None:
            run_id += f"_eval_perc{eval_perc}"
        
        base_params = getattr(args, 'base_classifier_params', {})
        ds = base_params.get('dynamic_scaling', False)
        run_id += f"_dynamic_scaling{'T' if ds else 'F'}"
        
        if filter_valid:
            run_id += "_filtered_detector"

    date_overrides = {}
    for date_key in ["train_start_date", "train_end_date", "test_start_date", "test_end_date"]:
        if hasattr(args, date_key) and getattr(args, date_key):
            date_overrides[date_key] = getattr(args, date_key)

    benchmark = get_dataset_benchmark(
        dataset_name=args.dataset,
        data_path=args.base_dir,
        exp_dir=args.exp_dir,
        run_id=run_id,
        mission_id=args.mission_id,
        save_metadata=getattr(args, 'save_metadata', True),
        **date_overrides
    )
    channels = benchmark.channels if args.channels is None else args.channels
     
    for channel_name in channels:
        ts_splitter = TimeSeriesSplitter(
            window_size=args.window_size,
            step_size=args.step_size,
            min_window=getattr(args, 'min_window', None) or 100,
            max_window=getattr(args, 'max_window', None) or 500,
            perc_step_size=getattr(args, 'perc_step_size', None) or 1.0,
            callback_handler=handler
        )
        
        feature_extractor = get_feature_extractor(
            args.feature_extractor,
            window_size=args.window_size,
            stride=args.step_size,
            n_kernel=args.n_kernel,
            **getattr(args, 'feature_extraction_params', getattr(args, 'fe_params', {})),
            callback_handler=handler
        )

        classifier, is_supervised = create_classifier(args, other_args)

        pipeline = AnomalyDetectionPipeline(
            steps=[
                PipelineStep(
                    name="ts_splitter", 
                    processor=ts_splitter,
                    phases={"train": None, "val": None, "predict": None}
                ),
                PipelineStep(
                    name="feature_extractor", 
                    processor=feature_extractor,
                    phases={
                        "train": PhaseConfig(method="fit_transform", supervised=True), 
                        "val": PhaseConfig(method="transform", supervised=True),
                        "predict": "transform"
                    }
                ),
                PipelineStep(
                    name="data_filter", 
                    processor=detector,
                    phases={
                        "train": PhaseConfig(method="filter", supervised=True),
                        "val": PhaseConfig(method="filter", supervised=True),
                    }  
                ),
                PipelineStep(
                    name="base_classifier", 
                    processor=classifier,
                    phases={
                        "train": PhaseConfig(method="fit", supervised=True), 
                        "val": PhaseConfig(method="predict", supervised=True),
                        "predict": "predict"
                    }
                ),
                PipelineStep(
                    name="detector", 
                    processor=detector,
                    phases={
                        "val": PhaseConfig(method="fit", supervised=True), 
                        "predict": "detect"
                    }  
                ),
            ],
            eval_perc=eval_perc,
            phase_map={"fit": ["train", "val"], "predict": ["predict"]}
        )

        dataset_kwargs = {}
        if hasattr(args, 'challenge'):
            dataset_kwargs["challenge"] = getattr(args, 'challenge')

        fitted_classifier, fitting_metrics = benchmark.fit_channel(
            channel_id=channel_name,
            classifier=pipeline,
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
    args, other_args = parse_exp_args()
    run_exp(args, other_args)

if __name__ == "__main__":
    main()
