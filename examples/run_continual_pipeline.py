"""Run decoupled pipeline experiment with continual learning (replay buffer)."""

import argparse
import warnings
import logging
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
from spaceai.benchmark import Benchmark, ESABenchmark
from spaceai.models.anomaly_pipeline.anomaly_classifier import AnomalyDetectionPipeline, PipelineStep
from spaceai.models.replay import TimeDecayReplayBuffer
from spaceai.models.replay.buffer_handler import BufferHandler
from spaceai.models.detectors import ThresholdDetector, MoLooKDEDetector
from spaceai.model_selection.validation_splitter import ValidationSplitter
warnings.simplefilter("ignore", FutureWarning)

from utils.args import parse_exp_args

def generate_run_id(args, replay_type=None, eval_perc=None, filter_valid=False):
    if getattr(args, 'run_id', None):
        return args.run_id
        
    parts = ["continual", args.feature_extractor, args.dataset, args.model, args.detector]
    if args.model == "dpmm":
        parts.extend([args.dpmm_type, args.dpmm_mode])
    
    if replay_type and replay_type != 'none':
        parts.append(f"replay_{replay_type}")
    if eval_perc is not None:
        parts.append(f"eval_perc{eval_perc}")
        
    ds = getattr(args, 'base_classifier_params', {}).get('dynamic_scaling', False)
    parts.append(f"dynamic_scaling{'T' if ds else 'F'}")
    
    if filter_valid:
        parts.append("filtered_detector")
        
    return "_".join(map(str, parts))

def run_exp(args, other_args=None):
    set_seed(getattr(args, 'seed', 40))

    w_params = getattr(args, 'wrapper_params', {})
    eval_perc = getattr(args, 'eval_perc', w_params.get('eval_perc', None))
    filter_valid = getattr(args, 'filter_valid', w_params.get('filter_valid', w_params.get('filter_valid_for_detector', False)))
    replay_type = getattr(args, 'replay', 'none')

    detector_params = getattr(args, 'detector_params', {})
    if args.detector == "threshold":
        detector = ThresholdDetector(threshold=None, filter_valid=filter_valid, **detector_params)
    elif args.detector == "molookde":
        detector = MoLooKDEDetector(alpha=0.001, filter_valid=filter_valid, **detector_params)
    else:
        detector, args.detector = None, "no_detector"

    run_id = generate_run_id(args, replay_type=replay_type, eval_perc=eval_perc, filter_valid=filter_valid)
    handler = CallbackHandler([SystemMonitorCallback()], call_every_ms=100)

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
    
    channels = args.channels
    if channels is None:
        channels = benchmark.channels
    elif isinstance(channels, str):
        channels = [c.strip() for c in channels.split(",")]
    elif isinstance(channels, list) and len(channels) == 1 and "," in channels[0]:
        channels = [c.strip() for c in channels[0].split(",")]

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

        replay_buffer = TimeDecayReplayBuffer(
            max_size=getattr(args, 'buffer_size', 5000), 
            half_life_segments=getattr(args, 'half_life', 2000),
            callback_handler=handler
            ) if getattr(args, 'replay', 'none') == 'time-decay' else None

        replay_detector_params = getattr(args, 'replay_detector_params', {})
        replay_detector = ThresholdDetector(**replay_detector_params)
        
        buffer_handler = BufferHandler(buffer=replay_buffer, replay_detector=replay_detector)
        
        classifier, is_supervised = create_classifier(args, other_args)

        from spaceai.models.anomaly_pipeline.anomaly_classifier import PhaseConfig
        
        validation_splitter = ValidationSplitter(eval_perc=eval_perc)

        pipeline = AnomalyDetectionPipeline(
            steps = [
                PipelineStep(
                    name="ts_splitter", 
                    processor=ts_splitter,
                    phases={
                        "train": None,  "predict": None, 
                        "train_continual": None, 
                    }
                ),
                PipelineStep(
                    name="feature_extractor", 
                    processor=feature_extractor,
                    phases={
                        "train": PhaseConfig(method="fit_transform", supervised=True), 
                        #"val": PhaseConfig(method="transform", supervised=True),
                        "predict": "transform", 
                        "train_continual": PhaseConfig(method="fit_transform", supervised=False),
                        #"val_continual": PhaseConfig(method="transform", supervised=False)
                    }
                ),
                PipelineStep(
                    name="validation_splitter", 
                    processor=validation_splitter,
                    phases={
                        "train": PhaseConfig(method="split", supervised=True), 
                        "val": PhaseConfig(method="inject_val", supervised=True),
                        "train_continual": PhaseConfig(method="split", supervised=False), 
                        "val_continual": PhaseConfig(method="inject_val", supervised=False),
                    }
                ),
                PipelineStep(
                    name="buffer_collector",
                    processor=buffer_handler,
                    phases={
                        "val": PhaseConfig(method="collect", supervised=False),
                        "val_continual": PhaseConfig(method="collect", supervised=False)
                    }
                ),
                PipelineStep(
                    name="buffer_sampler",
                    processor=buffer_handler,
                    phases={
                        "train": PhaseConfig(method="sample", supervised=True),
                        "train_continual": PhaseConfig(method="sample", supervised=False)
                    }
                ),
                PipelineStep(
                    name="base_classifier", 
                    processor=classifier,
                    phases={
                        "train": PhaseConfig(method="fit", supervised=True), 
                        "val": PhaseConfig(method="predict", supervised=True),
                        "predict": "predict",
                        "train_continual": PhaseConfig(method="fit", supervised=False),
                        "val_continual": PhaseConfig(method="predict", supervised=False)
                    }
                ),
                PipelineStep(
                    name="filter_trainer", 
                    processor=replay_detector,
                    phases={
                        "val": PhaseConfig(method="fit", supervised=True),
                        "val_continual": PhaseConfig(method="fit", supervised=False)
                    }  
                ),
                PipelineStep(
                    name="buffer_thresholder",
                    processor=buffer_handler,
                    phases={
                        "val": PhaseConfig(method="threshold", supervised=False),
                        "val_continual": PhaseConfig(method="threshold", supervised=False)
                    }
                ),
                PipelineStep(
                    name="filter_trainer", 
                    processor=replay_detector,
                    phases={
                        "val": PhaseConfig(method="filter", supervised=True),
                        "val_continual": PhaseConfig(method="filter", supervised=False)
                    }  
                ),
                PipelineStep(
                    name="detector", 
                    processor=detector,
                    phases={
                        "val": PhaseConfig(method="fit", supervised=False), 
                        "val_continual": PhaseConfig(method="fit", supervised=False),
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
        pipeline.phase_map["fit"] = ["train_continual", "val_continual"]

        results = benchmark.test_continual(
            channel_id=channel_name,
            classifier=fitted_classifier,
            experience_size=getattr(args, 'experience_size', 500)
        )

    if isinstance(benchmark, ESABenchmark):
        results = benchmark.compute_global_event_metrics(channels=channels)

def main():
    """Main function."""
    args, other_args = parse_exp_args()
    run_exp(args, other_args)

if __name__ == "__main__":
    main()
