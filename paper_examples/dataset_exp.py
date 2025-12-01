"""Dataset experiment execution module."""

from typing import Any, Callable, Optional

from spaceai.benchmark import (
    ESABenchmark,
    NASABenchmark,
    OPSSATBenchmark,
)
from spaceai.benchmark.callbacks import SystemMonitorCallback
from spaceai.data import (
    NASA,
    ESAMissions,
)
from spaceai.data.ops_sat import OPSSAT
from spaceai.preprocessing import (
    FEATURE_MAP,
    SpaceAISegmentator,
)


def get_dataset_benchmark(
    dataset_name: str,
    data_path: str,
    segmentator: Optional[SpaceAISegmentator] = None,
    feature_extractor: Optional[Any] = None,
    run_id: str = "exp",
    exp_dir: str = "experiments",
):
    """Get the benchmark object for the dataset."""
    if dataset_name == "esa":
        return ESABenchmark(
            data_root=data_path,
            segmentator=segmentator,
            feature_extractor=feature_extractor,
            run_id=run_id,
            exp_dir=exp_dir,
        )
    elif dataset_name == "nasa":
        return NASABenchmark(
            data_root=data_path,
            segmentator=segmentator,
            feature_extractor=feature_extractor,
            run_id=run_id,
            exp_dir=exp_dir,
        )
    elif dataset_name == "ops":
        return OPSSATBenchmark(
            data_root=data_path,
            segmentator=segmentator,
            feature_extractor=feature_extractor,
            run_id=run_id,
            exp_dir=exp_dir,
        )
    else:
        raise ValueError(f"Dataset {dataset_name} not supported.")


def run_dataset_experiment(
    benchmark: Any,
    classifier_factory: Callable[[], Any],
    is_supervised: bool,
    model_id: str,
    exp_dir: str = "experiments",
):
    """
    Run experiment for a specific dataset using the provided benchmark.

    Args:
        benchmark (Any): The benchmark instance.
        classifier_factory (Callable): Function that returns a new classifier instance.
        is_supervised (bool): Whether the model is supervised.
        model_id (str): ID of the model.
        exp_dir (str): Experiment directory.
    """
    if isinstance(benchmark, ESABenchmark):
        run_esa_experiment(
            benchmark, classifier_factory, is_supervised, model_id, exp_dir
        )
    elif isinstance(benchmark, NASABenchmark):
        run_nasa_experiment(
            benchmark, classifier_factory, is_supervised, model_id, exp_dir
        )
    elif isinstance(benchmark, OPSSATBenchmark):
        run_ops_sat_experiment(
            benchmark, classifier_factory, is_supervised, model_id, exp_dir
        )
    else:
        raise ValueError(f"Benchmark type {type(benchmark)} not supported.")


def run_esa_experiment(
    benchmark: ESABenchmark,
    classifier_factory: Callable[[], Any],
    is_supervised: bool,
    model_id: str,
    exp_dir: str,
):
    """Run ESA experiment."""
    for mission_wrapper in ESAMissions:
        mission = mission_wrapper.value
        if mission.index != 1:
            continue
        for channel_id in mission.target_channels:
            if (
                int(channel_id.split("_")[1]) < 41
                or int(channel_id.split("_")[1]) > 46
            ):
                continue
            print(f"Running classifier on channel {channel_id}")

            classifier = classifier_factory()
            benchmark.run_classifier(
                mission=mission,
                channel_id=channel_id,
                classifier=classifier,
                supervised=supervised,
                model_id=model_id,
                exp_dir=exp_dir,
            )


def run_nasa_experiment(
    benchmark: NASABenchmark,
    classifier_factory: Callable[[], Any],
    is_supervised: bool,
    model_id: str,
    exp_dir: str,
):
    """Run NASA experiment."""
    channels = NASA.channel_ids
    for channel_id in channels:
        print(f"Running classifier on channel {channel_id}")
        classifier = classifier_factory()
        benchmark.run_classifier(
            channel_id=channel_id,
            classifier=classifier,
            supervised=supervised,
            model_id=model_id,
            exp_dir=exp_dir,
        )


def run_ops_sat_experiment(
    benchmark: OPSSATBenchmark,
    classifier_factory: Callable[[], Any],
    is_supervised: bool,
    model_id: str,
    exp_dir: str,
):
    """Run OPS-SAT experiment."""
    channels = OPSSAT.channel_ids
    for channel_id in channels:
        print(f"Running classifier on channel {channel_id}")
        classifier = classifier_factory()
        benchmark.run_classifier(
            channel_id=channel_id,
            classifier=classifier,
            supervised=supervised,
            model_id=model_id,
            exp_dir=exp_dir,
        )

