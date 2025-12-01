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


            classifier = classifier_factory()
            benchmark.run_classifier(
                mission=mission,
                channel_id=channel_id,
                classifier=classifier,
                supervised=is_supervised,
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

        classifier = classifier_factory()
        benchmark.run_classifier(
            channel_id=channel_id,
            classifier=classifier,
            supervised=is_supervised,
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

        classifier = classifier_factory()
        benchmark.run_classifier(
            channel_id=channel_id,
            classifier=classifier,
            supervised=is_supervised,
        )

        classifier = classifier_factory()
        benchmark.run_classifier(
            channel_id=channel_id,
            classifier=classifier,
            supervised=is_supervised,
        )


def run_prediction_experiment(
    benchmark: Any,
    predictor_factory: Callable[[int], Any],
    detector_factory: Callable[[], Any],
    config: Any,
    callbacks: list = None,
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
    callbacks: list = None,
):
    """Run ESA prediction experiment."""
    from spaceai.data import ESA
    from torch import nn, optim

    for mission_wrapper in ESAMissions:
        mission = mission_wrapper.value
        # Filter missions/channels if needed (logic from original script)
        # if mission.index != 1: continue
        
        for channel_id in mission.target_channels:
            # Filter channels if needed
            # if int(channel_id.split("_")[1]) < 41 or int(channel_id.split("_")[1]) > 46: continue

            esa_channel = ESA(
                benchmark.data_root, mission, channel_id, mode="anomaly", train=False
            )

            detector = detector_factory()
            predictor = predictor_factory(esa_channel.in_features_size)
            predictor.build()

            benchmark.run(
                mission,
                channel_id,
                predictor,
                detector,
                fit_predictor_args=dict(
                    criterion=nn.MSELoss(), # TODO: make configurable
                    optimizer=optim.Adam(predictor.model.parameters(), lr=config.learning_rate),
                    epochs=config.epochs,
                    patience_before_stopping=config.patience,
                    min_delta=config.min_delta,
                    batch_size=config.batch_size, # or esn_batch_number/lstm_batch_size
                    restore_best=False,
                ),
                overlapping_train=True,
                restore_predictor=not config.train,
                callbacks=callbacks,
            )


def run_nasa_prediction_experiment(
    benchmark: NASABenchmark,
    predictor_factory: Callable[[int], Any],
    detector_factory: Callable[[], Any],
    config: Any,
    callbacks: list = None,
):
    """Run NASA prediction experiment."""
    from spaceai.data import NASA
    from torch import nn, optim

    channels = NASA.channel_ids
    for channel_id in channels:
        nasa_channel = NASA(
            benchmark.data_root, channel_id, mode="anomaly", train=False
        )
        
        detector = detector_factory()
        predictor = predictor_factory(nasa_channel.in_features_size)
        predictor.build()

        benchmark.run(
            channel_id,
            predictor,
            detector,
            fit_predictor_args=dict(
                criterion=nn.MSELoss(),
                optimizer=optim.Adam(predictor.model.parameters(), lr=config.learning_rate),
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


def run_ops_sat_prediction_experiment(
    benchmark: OPSSATBenchmark,
    predictor_factory: Callable[[int], Any],
    detector_factory: Callable[[], Any],
    config: Any,
    callbacks: list = None,
):
    """Run OPS-SAT prediction experiment."""
    from spaceai.data.ops_sat import OPSSAT
    from torch import nn, optim

    channels = OPSSAT.channel_ids
    for channel_id in channels:
        ops_channel = OPSSAT(
            benchmark.data_root, channel_id, mode="anomaly", train=False
        )

        detector = detector_factory()
        predictor = predictor_factory(ops_channel.in_features_size)
        predictor.build()

        benchmark.run(
            channel_id,
            predictor,
            detector,
            fit_predictor_args=dict(
                criterion=nn.MSELoss(),
                optimizer=optim.Adam(predictor.model.parameters(), lr=config.learning_rate),
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
