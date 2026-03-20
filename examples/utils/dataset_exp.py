"""Dataset experiment execution module."""

from typing import (
    Any,
    Callable,
    Optional,
)

import os

import torch

from spaceai.benchmark import (
    ESABenchmark,
    NASABenchmark,
    OPSSATBenchmark,
)
from spaceai.data import (
    NASA,
    ESAMissions,
)
from spaceai.data.ops_sat import OPSSAT
from spaceai.preprocessing import TimeSeriesSplitter


def get_dataset_benchmark(
    dataset_name: str,
    data_path: str,
    run_id: str = "exp",
    exp_dir: str = "experiments",
    mission_id: int = 1,
):
    """Get the benchmark object for the dataset."""
    if dataset_name == "esa":
        mission = ESAMissions.MISSION_1.value if mission_id == 1 else ESAMissions.MISSION_2.value

        return ESABenchmark(
            data_root=data_path,
            run_id=run_id,
            exp_dir=exp_dir,
            mission=mission,
        )
    elif dataset_name == "nasa":
        return NASABenchmark(
            data_root=data_path,
            run_id=run_id,
            exp_dir=exp_dir,
        )
    elif dataset_name == "ops":
        return OPSSATBenchmark(
            data_root=data_path,
            run_id=run_id,
            exp_dir=exp_dir,
        )
    else:
        raise ValueError(f"Dataset {dataset_name} not supported.")


def run_dataset_experiment(
    benchmark: Any,
    classifier_factory: Callable[[], Any],
    model_id: str,
    exp_dir: str = "experiments",
):
    """
    Run experiment for a specific dataset using the provided benchmark.

    Args:
        benchmark (Any): The benchmark instance.
        classifier_factory (Callable): Function that returns a new classifier instance.
        model_id (str): ID of the model.
        exp_dir (str): Experiment directory.
    """
    if isinstance(benchmark, ESABenchmark):
        run_esa_experiment(
            benchmark, classifier_factory, model_id, exp_dir,
        )
    elif isinstance(benchmark, NASABenchmark):
        run_nasa_experiment(
            benchmark, classifier_factory, model_id, exp_dir,
        )
    elif isinstance(benchmark, OPSSATBenchmark):
        run_ops_sat_experiment(
            benchmark, classifier_factory, model_id, exp_dir,
        )
    else:
        raise ValueError(f"Benchmark type {type(benchmark)} not supported.")


def run_esa_experiment(
    benchmark: ESABenchmark,
    classifier_factory: Callable[[], Any],
    _model_id: str,
    _exp_dir: str,
):
    """Run ESA experiment."""
    for mission_wrapper in ESAMissions:
        mission = mission_wrapper.value
        if mission.index != 1:
            continue
        for channel_id in mission.target_channels:
            if int(channel_id.split("_")[1]) < 41 or int(channel_id.split("_")[1]) > 46:
                continue

            classifier = classifier_factory()
            benchmark.mission = mission
            benchmark.fit_channel(
                channel_id=channel_id,
                classifier=classifier,
            )
            benchmark.test_channel(
                channel_id=channel_id,
            )


def run_nasa_experiment(
    benchmark: NASABenchmark,
    classifier_factory: Callable[[], Any],
    _model_id: str,
    _exp_dir: str,
):
    """Run NASA experiment."""
    channels = NASA.channel_ids
    for channel_id in channels:

        classifier = classifier_factory()
        benchmark.fit_channel(
            channel_id=channel_id,
            classifier=classifier,
        )
        benchmark.test_channel(
            channel_id=channel_id,
        )


def run_ops_sat_experiment(
    benchmark: OPSSATBenchmark,
    classifier_factory: Callable[[], Any],
    _model_id: str,
    _exp_dir: str,
):
    """Run OPS-SAT experiment."""
    channels = OPSSAT.channel_ids
    for channel_id in channels:

        classifier = classifier_factory()
        benchmark.fit_channel(
            channel_id=channel_id,
            classifier=classifier,
        )
        benchmark.test_channel(
            channel_id=channel_id,
        )
