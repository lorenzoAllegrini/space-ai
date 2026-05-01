"""Run decoupled pipeline experiment module."""

from utils.args import parse_exp_args
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
from spaceai.model_selection import PhaseSplitter
warnings.simplefilter("ignore", FutureWarning)


def run_exp(args, other_args=None):
    """Run decoupled pipeline experiment."""
    set_seed(getattr(args, 'seed', 40))

    wrapper_params = getattr(args, 'wrapper_params', {})
    calibration_perc = getattr(args, 'calibration_perc',
                               wrapper_params.get('calibration_perc', None))
    validation_perc = getattr(args, 'validation_perc',
                              wrapper_params.get('validation_perc', None))
    eval_perc = getattr(args, 'eval_perc',
                        wrapper_params.get('eval_perc', None))
    if calibration_perc is None and eval_perc is not None:
        calibration_perc = eval_perc
    filter_valid = getattr(args, 'filter_valid', wrapper_params.get(
        'filter_valid', wrapper_params.get('filter_valid_for_detector', False)))

    detector_params = getattr(args, 'detector_params', {})
    detector = None
    if args.detector == "threshold":
        detector = ThresholdDetector(
            **{**dict(threshold=None, filter_valid=filter_valid), **detector_params})
    elif args.detector == "molookde":
        detector = MoLooKDEDetector(
            **{**dict(alpha=0.001, filter_valid=filter_valid), **detector_params})
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

        if calibration_perc is not None or validation_perc is not None:
            run_id += f"_cal{calibration_perc}_val{validation_perc}"

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

    temporal_validation = getattr(args, 'temporal_validation', False)
    val_train_end = getattr(args, 'val_train_end_date', "2005-01-01")
    val_test_start = getattr(args, 'val_test_start_date', "2005-01-01")
    val_test_end = getattr(args, 'val_test_end_date', "2007-01-01")

    def _build_pipeline(channel_name):
        """Build a fresh AnomalyDetectionPipeline for a given channel."""
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

        clf, _ = create_classifier(args, other_args)

        fit_phases = ["train"]
        phase_percs = {}
        if calibration_perc and calibration_perc > 0:
            fit_phases.append("calibration")
            phase_percs["calibration"] = calibration_perc

        phase_splitter = PhaseSplitter(phases=phase_percs) if phase_percs else None

        return AnomalyDetectionPipeline(
            steps=[
                ps for ps in [
                    PipelineStep(
                        name="ts_splitter",
                        processor=ts_splitter,
                        phases={"train": None, "predict": None}
                    ) if args.segmentator else None,
                    PipelineStep(
                        name="feature_extractor",
                        processor=feature_extractor,
                        phases={
                            "train": PhaseConfig(method="fit_transform", supervised=True),
                            "predict": "transform"
                        }
                    ) if feature_extractor is not None else None,
                    PipelineStep(
                        name="phase_splitter",
                        processor=phase_splitter,
                        phases={
                            "train": PhaseConfig(method="split", supervised=True),
                            "calibration": PhaseConfig(method="switch_phase", supervised=True),
                        }
                    ) if phase_splitter is not None else None,
                    PipelineStep(
                        name="data_filter",
                        processor=detector,
                        phases={
                            "train": PhaseConfig(method="filter", supervised=True),
                            "calibration": PhaseConfig(method="filter", supervised=True),
                        }
                    ) if detector is not None else None,
                    PipelineStep(
                        name="base_classifier",
                        processor=clf,
                        phases={
                            "train": PhaseConfig(method="fit", supervised=True),
                            "calibration": PhaseConfig(method="predict", supervised=True),
                            "predict": "predict"
                        }
                    ),
                    PipelineStep(
                        name="detector",
                        processor=detector,
                        phases={
                            "calibration": PhaseConfig(method="fit", supervised=True),
                            "predict": "detect"
                        }
                    ) if detector is not None else None,
                ] if ps is not None
            ],
            eval_perc=None,
            phase_map={"fit": fit_phases, "predict": ["predict"]}
        )

    dataset_kwargs = {}
    if hasattr(args, 'challenge'):
        dataset_kwargs["challenge"] = getattr(args, 'challenge')
        dataset_kwargs["n_predictions"] = getattr(args, 'n_predictions', 1)

    # ── Pass 1: Temporal Validation (optional) ──
    if temporal_validation:
        logging.debug("=== Temporal Validation Pass ===")
        benchmark.results_filename = "validation_results.csv"
        benchmark.date_overrides["train_end_date"] = val_train_end
        benchmark.date_overrides["test_start_date"] = val_test_start
        benchmark.date_overrides["test_end_date"] = val_test_end

        for channel_name in channels:
            pipeline = _build_pipeline(channel_name)
            fitted_classifier, _ = benchmark.fit_channel(
                channel_id=channel_name,
                classifier=pipeline,
                **dataset_kwargs
            )
            benchmark.test_channel(
                channel_id=channel_name,
                classifier=fitted_classifier,
                **dataset_kwargs
            )

        if isinstance(benchmark, ESABenchmark):
            benchmark.compute_global_event_metrics(channels=channels)

        benchmark.reset()

    # ── Pass 2: Full Test ──
    logging.debug("=== Full Test Pass ===")
    # Restore default date overrides (from original config or ESAMission defaults)
    benchmark.date_overrides = date_overrides.copy()

    for channel_name in channels:
        pipeline = _build_pipeline(channel_name)
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
