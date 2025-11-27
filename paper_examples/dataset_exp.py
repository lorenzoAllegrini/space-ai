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


<<<<<<< HEAD
=======
BASE_STATISTICS_LIST=[
            "mean",
            "var",
            "std",
            "n_peaks",
            "smooth10_n_peaks",
            "smooth20_n_peaks",
            "diff_peaks",
            "diff2_peaks",
            "diff_var",
            "diff2_var"
        ]

>>>>>>> dpmm_pull
def run_esa_exp(classifier, is_supervised, extract_features, exp_dir, run_id):

    segmentator = SpaceAISegmentator(
        window_size=50,
        step_size=50,
        extract_features=extract_features,
        transformations=FEATURE_MAP,
        run_id="channel_segments",
        exp_dir=exp_dir,
    )
    print('Loading data...')

    benchmark = ESABenchmark(
        run_id=run_id,
        exp_dir=exp_dir,
        data_root="datasets",
        segmentator=segmentator,
    )
    callbacks = [SystemMonitorCallback()]

    print('Data loaded!')

    for mission_wrapper in ESAMissions:
        mission = mission_wrapper.value
        if mission.index != 1:
            continue
        for channel_id in mission.target_channels:
            if int(channel_id.split("_")[1]) < 41 or int(channel_id.split("_")[1]) > 46:
                continue
            print(f'Running classifier on channel {channel_id}')

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
    print('Loading data...')

    benchmark = NASABenchmark(
        run_id=run_id,
        exp_dir=exp_dir,
        data_root="datasets",
        segmentator=segmentator,
    )
    callbacks = [SystemMonitorCallback()]

    print('Data loaded!')

    channels = NASA.channel_ids
    for i, channel_id in enumerate(channels):
<<<<<<< HEAD
=======
        print(f'Running classifier on channel {channel_id}')
>>>>>>> dpmm_pull

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
<<<<<<< HEAD
        transformations=FEATURE_MAP,
        telecommands=True,
        exp_dir=exp_dir,
        run_id=run_id,
=======
        transformations=BASE_STATISTICS_LIST + ["kurtosis", "skew"]
>>>>>>> dpmm_pull
    )

    print('Loading data...')

    benchmark = OPSSATBenchmark(
        run_id=run_id,
        exp_dir=exp_dir,
        data_root="datasets",
        segmentator=segmentator,
    )
    callbacks = [SystemMonitorCallback()]

    print('Data loaded!')

    channels = OPSSAT.channel_ids
    for i, channel_id in enumerate(channels):
        print(f'Running classifier on channel {channel_id}')
        benchmark.run_classifier(
            channel_id,
            classifier=classifier,
            callbacks=callbacks,
            supervised=is_supervised,
        )
