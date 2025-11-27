from spaceai.benchmark import (
    ESABenchmark,
    NASABenchmark,
    OPSSATBenchmark,
)
from spaceai.benchmark.callbacks import SystemMonitorCallback
from spaceai.data import (
    ESA,
    NASA,
    ESAMissions,
)
from spaceai.data.ops_sat import OPSSAT
from spaceai.segmentators import (
    FEATURE_MAP,
    SpaceAISegmentator,
)


def run_esa_exp(classifier, is_supervised, extract_features, exp_dir, run_id):

    segmentator = SpaceAISegmentator(
        window_size=50,
        step_size=50,
        extract_features=extract_features,
        transformations=FEATURE_MAP,
        run_id="channel_segments",
        exp_dir=exp_dir,
    )

    benchmark = ESABenchmark(
        run_id=run_id,
        exp_dir=exp_dir,
        data_root="datasets",
        segmentator=segmentator,
    )
    callbacks = [SystemMonitorCallback()]

    for mission_wrapper in ESAMissions:
        mission = mission_wrapper.value
        if mission.index != 1:
            continue
        for channel_id in mission.target_channels:
            if int(channel_id.split("_")[1]) < 41 or int(channel_id.split("_")[1]) > 46:
                continue

            benchmark.run_classifier(
                mission=mission,
                channel_id=channel_id,
                classifier=classifier,
                callbacks=callbacks,
                supervised=is_supervised,
            )


def run_nasa_exp(classifier, extract_features, exp_dir, run_id):

    segmentator = SpaceAISegmentator(
        window_size=50,
        step_size=10,
        extract_features=extract_features,
        transformations=FEATURE_MAP,
        exp_dir=exp_dir,
        run_id=run_id,
    )
    benchmark = NASABenchmark(
        run_id=run_id,
        exp_dir=exp_dir,
        data_root="datasets",
        segmentator=segmentator,
    )
    callbacks = [SystemMonitorCallback()]

    channels = NASA.channel_ids
    for i, channel_id in enumerate(channels):

        benchmark.run_classifier(
            channel_id,
            classifier=classifier,
            callbacks=callbacks,
        )


def run_ops_exp(classifier, is_supervised, extract_features, exp_dir, run_id):

    segmentator = SpaceAISegmentator(
        window_size=50,
        step_size=50,
        extract_features=extract_features,
        transformations=FEATURE_MAP,
        telecommands=True,
        exp_dir=exp_dir,
        run_id=run_id,
    )

    benchmark = OPSSATBenchmark(
        run_id=run_id,
        exp_dir=exp_dir,
        data_root="datasets",
        segmentator=segmentator,
    )

    callbacks = [SystemMonitorCallback()]

    channels = OPSSAT.channel_ids
    for i, channel_id in enumerate(channels):

        benchmark.run_classifier(
            channel_id,
            classifier=classifier,
            callbacks=callbacks,
            supervised=is_supervised,
        )
