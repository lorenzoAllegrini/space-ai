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
from spaceai.preprocessing import TSSplitter


def get_dataset_benchmark(
    dataset_name: str,
    data_path: str,
    segmentator: Optional[TSSplitter] = None,
    feature_extractor: Optional[Any] = None,
    run_id: str = "exp",
    exp_dir: str = "experiments",
    n_predictions: int = 10,
):
    """Get the benchmark object for the dataset."""
    if dataset_name == "esa":
        return ESABenchmark(
            data_root=data_path,
            segmentator=segmentator,
            feature_extractor=feature_extractor,
            run_id=run_id,
            exp_dir=exp_dir,
            n_predictions=n_predictions,
        )
    elif dataset_name == "nasa":
        return NASABenchmark(
            data_root=data_path,
            segmentator=segmentator,
            feature_extractor=feature_extractor,
            run_id=run_id,
            exp_dir=exp_dir,
            n_predictions=n_predictions,
        )
    elif dataset_name == "ops":
        return OPSSATBenchmark(
            data_root=data_path,
            segmentator=segmentator,
            feature_extractor=feature_extractor,
            run_id=run_id,
            exp_dir=exp_dir,
            n_predictions=n_predictions,
        )
    else:
        raise ValueError(f"Dataset {dataset_name} not supported.")


def run_dataset_experiment(
    benchmark: Any,
    classifier_factory: Callable[[], Any],
    is_supervised: bool,
    model_id: str,
    exp_dir: str = "experiments",
    callbacks: Optional[list] = None,
):
    """
    Run experiment for a specific dataset using the provided benchmark.

    Args:
        benchmark (Any): The benchmark instance.
        classifier_factory (Callable): Function that returns a new classifier instance.
        is_supervised (bool): Whether the model is supervised.
        model_id (str): ID of the model.
        exp_dir (str): Experiment directory.
        callbacks (list): List of callbacks to use.
    """
    if isinstance(benchmark, ESABenchmark):
        run_esa_experiment(
            benchmark, classifier_factory, is_supervised, model_id, exp_dir, callbacks
        )
    elif isinstance(benchmark, NASABenchmark):
        run_nasa_experiment(
            benchmark, classifier_factory, is_supervised, model_id, exp_dir, callbacks
        )
    elif isinstance(benchmark, OPSSATBenchmark):
        run_ops_sat_experiment(
            benchmark, classifier_factory, is_supervised, model_id, exp_dir, callbacks
        )
    else:
        raise ValueError(f"Benchmark type {type(benchmark)} not supported.")

    if benchmark.feature_extractor is not None:
        torch.save(
            benchmark.feature_extractor,
            os.path.join(benchmark.run_dir, "feature_extractor.pt"),
        )


def run_sml_dataset_experiment(
    benchmark: Any,
    classifier_factory: Callable[[], Any],
    is_supervised: bool,
    model_id: str,
    server_ip: str,
    port: int,
    exp_dir: str = "experiments",
    callbacks: Optional[list] = None,
):
    """
    Run SML experiment for a specific dataset using the provided benchmark.
    """
    if isinstance(benchmark, ESABenchmark):
        run_esa_sml_experiment(
            benchmark, classifier_factory, is_supervised, model_id, server_ip, port, exp_dir, callbacks
        )
    elif isinstance(benchmark, NASABenchmark):
        run_nasa_sml_experiment(
            benchmark, classifier_factory, is_supervised, model_id, server_ip, port, exp_dir, callbacks
        )
    elif isinstance(benchmark, OPSSATBenchmark):
        run_ops_sat_sml_experiment(
            benchmark, classifier_factory, is_supervised, model_id, server_ip, port, exp_dir, callbacks
        )
    else:
        raise ValueError(f"Benchmark type {type(benchmark)} not supported.")

    if benchmark.feature_extractor is not None:
        torch.save(
            benchmark.feature_extractor,
            os.path.join(benchmark.run_dir, "feature_extractor.pt"),
        )


def run_esa_experiment(
    benchmark: ESABenchmark,
    classifier_factory: Callable[[], Any],
    is_supervised: bool,
    _model_id: str,
    _exp_dir: str,
    callbacks: Optional[list] = None,
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
            benchmark.train_channel_rolling_stats(
                channel_id=channel_id,
                classifier=classifier,
                supervised=is_supervised,
                callbacks=callbacks,
            )
            benchmark.test_channel_rolling_stats(
                channel_id=channel_id,
            )
        
        # Aggregate global event-level metrics for ESA
        target_channels = [c for c in mission.target_channels if 41 <= int(c.split("_")[1]) <= 46]
        benchmark.compute_global_event_metrics(channels=target_channels)

    if benchmark.feature_extractor is not None:
        torch.save(benchmark.feature_extractor, os.path.join(benchmark.run_dir, "feature_extractor.pt"))


def run_esa_sml_experiment(
    benchmark: ESABenchmark,
    classifier_factory: Callable[[], Any],
    is_supervised: bool,
    _model_id: str,
    server_ip: str,
    port: int,
    _exp_dir: str,
    callbacks: Optional[list] = None,
):
    """Run ESA experiment with SML offloading."""
    for mission_wrapper in ESAMissions:
        mission = mission_wrapper.value
        if mission.index != 1:
            continue
        for channel_id in mission.target_channels:
            if int(channel_id.split("_")[1]) < 41 or int(channel_id.split("_")[1]) > 46:
                continue

            classifier = classifier_factory()
            benchmark.mission = mission
            benchmark.channel_rolling_stats_sml(
                channel_id=channel_id,
                classifier=classifier,
                server_ip=server_ip,
                port=port,
                supervised=is_supervised,
                callbacks=callbacks,
            )

        # Aggregate global event-level metrics for ESA (SML)
        target_channels = [c for c in mission.target_channels if 41 <= int(c.split("_")[1]) <= 46]
        benchmark.compute_global_event_metrics(channels=target_channels)


def run_nasa_experiment(
    benchmark: NASABenchmark,
    classifier_factory: Callable[[], Any],
    is_supervised: bool,
    _model_id: str,
    _exp_dir: str,
    callbacks: Optional[list] = None,
):
    """Run NASA experiment."""
    channels = NASA.channel_ids
    for channel_id in channels:

        classifier = classifier_factory()
        benchmark.train_channel_rolling_stats(
            channel_id=channel_id,
            classifier=classifier,
            supervised=is_supervised,
            callbacks=callbacks,
        )
        benchmark.test_channel_rolling_stats(
            channel_id=channel_id,
        )

    if benchmark.feature_extractor is not None:
        torch.save(benchmark.feature_extractor, os.path.join(benchmark.run_dir, "feature_extractor.pt"))


def run_nasa_sml_experiment(
    benchmark: NASABenchmark,
    classifier_factory: Callable[[], Any],
    is_supervised: bool,
    _model_id: str,
    server_ip: str,
    port: int,
    _exp_dir: str,
    callbacks: Optional[list] = None,
):
    """Run NASA experiment with SML offloading."""
    channels = NASA.channel_ids
    for channel_id in channels:
        classifier = classifier_factory()
        benchmark.channel_rolling_stats_sml(
            channel_id=channel_id,
            classifier=classifier,
            server_ip=server_ip,
            port=port,
            supervised=is_supervised,
            callbacks=callbacks,
        )


def run_ops_sat_experiment(
    benchmark: OPSSATBenchmark,
    classifier_factory: Callable[[], Any],
    is_supervised: bool,
    _model_id: str,
    _exp_dir: str,
    callbacks: Optional[list] = None,
):
    """Run OPS-SAT experiment."""
    channels = OPSSAT.channel_ids
    for channel_id in channels:

        classifier = classifier_factory()
        benchmark.train_channel_rolling_stats(
            channel_id=channel_id,
            classifier=classifier,
            supervised=is_supervised,
            callbacks=callbacks,
        )
        benchmark.test_channel_rolling_stats(
            channel_id=channel_id,
        )

    if benchmark.feature_extractor is not None:
        torch.save(benchmark.feature_extractor, os.path.join(benchmark.run_dir, "feature_extractor.pt"))


def run_ops_sat_sml_experiment(
    benchmark: OPSSATBenchmark,
    classifier_factory: Callable[[], Any],
    is_supervised: bool,
    _model_id: str,
    server_ip: str,
    port: int,
    _exp_dir: str,
    callbacks: Optional[list] = None,
):
    """Run OPS-SAT experiment with SML offloading."""
    channels = OPSSAT.channel_ids
    for channel_id in channels:
        classifier = classifier_factory()
        benchmark.channel_rolling_stats_sml(
            channel_id=channel_id,
            classifier=classifier,
            server_ip=server_ip,
            port=port,
            supervised=is_supervised,
            callbacks=callbacks,
        )


def run_prediction_experiment(
    benchmark: Any,
    predictor_factory: Callable[[int], Any],
    detector_factory: Callable[[], Any],
    config: Any,
    callbacks: Optional[list] = None,
):
    """Run prediction experiment."""
    if isinstance(benchmark, ESABenchmark):
        run_esa_prediction_experiment(
            benchmark, predictor_factory, detector_factory, config, callbacks
        )
    elif isinstance(benchmark, NASABenchmark):
        run_nasa_prediction_experiment(
            benchmark, predictor_factory, detector_factory, config, callbacks
        )
    elif isinstance(benchmark, OPSSATBenchmark):
        run_ops_sat_prediction_experiment(
            benchmark, predictor_factory, detector_factory, config, callbacks
        )
    else:
        raise ValueError(f"Benchmark type {type(benchmark)} not supported.")


def run_esa_prediction_experiment(
    benchmark: ESABenchmark,
    predictor_factory: Callable[[int], Any],
    detector_factory: Callable[[], Any],
    config: Any,
    callbacks: Optional[list] = None,
):
    """Run ESA prediction experiment."""
    from torch import (
        nn,
        optim,
    )

    from spaceai.data import ESA

    for mission_wrapper in ESAMissions:
        mission = mission_wrapper.value

        for channel_id in mission.target_channels:
            esa_channel = ESA(
                benchmark.data_root, mission, channel_id, mode="anomaly", train=False
            )

            detector = detector_factory()
            predictor = predictor_factory(esa_channel.in_features_size)
            predictor.build()

            benchmark.mission = mission
            benchmark.train_channel_telemanom(
                channel_id,
                predictor,
                fit_predictor_args=dict(
                    criterion=nn.MSELoss(),
                    optimizer=optim.Adam(
                        predictor.model.parameters(), lr=config.learning_rate
                    ),
                    epochs=config.epochs,
                    patience_before_stopping=config.patience,
                    min_delta=config.min_delta,
                    batch_size=config.batch_size,
                    restore_best=False,
                ),
                overlapping_train=True,
                restore_predictor=not config.train,
                callbacks=callbacks,
            )
            benchmark.test_channel_telemanom(
                channel_id,
                detector,
                callbacks=callbacks,
            )

    if benchmark.feature_extractor is not None:
        torch.save(
            benchmark.feature_extractor,
            os.path.join(benchmark.run_dir, "feature_extractor.pt"),
        )

    if benchmark.feature_extractor is not None:
        torch.save(
            benchmark.feature_extractor,
            os.path.join(benchmark.run_dir, "feature_extractor.pt"),
        )


def run_nasa_prediction_experiment(
    benchmark: NASABenchmark,
    predictor_factory: Callable[[int], Any],
    detector_factory: Callable[[], Any],
    config: Any,
    callbacks: Optional[list] = None,
):
    """Run NASA prediction experiment."""
    from torch import (
        nn,
        optim,
    )

    from spaceai.data import NASA

    channels = NASA.channel_ids
    for channel_id in channels:
        nasa_channel = NASA(
            benchmark.data_root, channel_id, mode="anomaly", train=False
        )

        detector = detector_factory()
        predictor = predictor_factory(nasa_channel.in_features_size)
        predictor.build()

        benchmark.train_channel_telemanom(
            channel_id,
            predictor,
            fit_predictor_args=dict(
                criterion=nn.MSELoss(),
                optimizer=optim.Adam(
                    predictor.model.parameters(), lr=config.learning_rate
                ),
                epochs=config.epochs,
                patience_before_stopping=config.patience,
                min_delta=config.min_delta,
                batch_size=config.batch_size,
                restore_best=False,
            ),
            overlapping_train=True,
            restore_predictor=not config.train,
            callbacks=callbacks,
        )
        benchmark.test_channel_telemanom(
            channel_id,
            detector,
            callbacks=callbacks,
        )


def run_ops_sat_prediction_experiment(
    benchmark: OPSSATBenchmark,
    predictor_factory: Callable[[int], Any],
    detector_factory: Callable[[], Any],
    config: Any,
    callbacks: Optional[list] = None,
):
    """Run OPS-SAT prediction experiment."""
    from torch import (
        nn,
        optim,
    )

    from spaceai.data.ops_sat import OPSSAT

    channels = OPSSAT.channel_ids
    for channel_id in channels:
        ops_channel = OPSSAT(
            benchmark.data_root, channel_id, mode="anomaly", train=False
        )

        detector = detector_factory()
        predictor = predictor_factory(ops_channel.in_features_size)
        predictor.build()

        benchmark.train_channel_telemanom(
            channel_id,
            predictor,
            fit_predictor_args=dict(
                criterion=nn.MSELoss(),
                optimizer=optim.Adam(
                    predictor.model.parameters(), lr=config.learning_rate
                ),
                epochs=config.epochs,
                patience_before_stopping=config.patience,
                min_delta=config.min_delta,
                batch_size=config.batch_size,
                restore_best=False,
            ),
            overlapping_train=True,
            restore_predictor=not config.train,
            callbacks=callbacks,
        )
        benchmark.test_channel_telemanom(
            channel_id,
            detector,
            callbacks=callbacks,
        )
