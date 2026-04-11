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
from spaceai.models.legacy import AdaptiveRollingWindowClassifier
from spaceai.models.drift_detectors.adwin_detector import ADWINDetector
from spaceai.models.drift_detectors.utils.replay_buffers import TimeDecayReplayBuffer
from spaceai.models.drift_detectors.utils.filters import SafeRampUpFilter
from spaceai.benchmark import ESABenchmark
from spaceai.models.detectors import ThresholdDetector, MoLooKDEDetector

warnings.simplefilter("ignore", FutureWarning)
warnings.simplefilter("ignore", RuntimeWarning)

from utils.args import (
    parse_exp_args, 
    DATASET_LIST, 
    MODEL_LIST, 
    DPMM_MODEL_TYPE, 
    DPMM_MODE, 
    FEATURE_EXTRACTOR_LIST
)

# parse_exp_args moved to utils.args


def run_exp(args, other_args=None):
    """Run experiment."""
    set_seed(getattr(args, 'seed', 42))

    ts_splitter = TimeSeriesSplitter(
        window_size=args.window_size,
        step_size=args.step_size,
        min_window=getattr(args, 'min_window', None) or 10,
        max_window=getattr(args, 'max_window', None) or 300,
        perc_step_size=getattr(args, 'perc_step_size', None) or 1.0,
    )
    print(f"[DEBUG] TimeSeriesSplitter: window_size={ts_splitter.window_size_raw}, min_window={ts_splitter.min_window}, max_window={ts_splitter.max_window}, perc_step_size={ts_splitter.perc_step_size}")
        
    feature_extractor = get_feature_extractor(
        args.feature_extractor, 
        window_size=args.window_size, 
        stride=args.step_size, 
        n_kernel=args.n_kernel,
        **getattr(args, 'fe_params', {})  # Inietta automaticamente 'selected_features'!
    )

    classifier, is_supervised = create_classifier(args, other_args)
    
    handler = CallbackHandler([SystemMonitorCallback()], call_every_ms=100)

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
        detector = ThresholdDetector(**{**dict(threshold=0.9), **detector_params})
    elif args.detector == "molookde":
        detector = MoLooKDEDetector(**{**dict(alpha=0.05), **detector_params})
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
