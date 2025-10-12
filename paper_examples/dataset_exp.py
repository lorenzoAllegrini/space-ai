from spaceai.data import ESA, ESAMissions
from spaceai.benchmark import ESABenchmark
from spaceai.segmentators.esa_segmentator import EsaDatasetSegmentator

from spaceai.data import NASA
from spaceai.benchmark import NASABenchmark
from spaceai.segmentators.nasa_segmentator import NasaDatasetSegmentator

from spaceai.data.ops_sat import OPSSAT
from spaceai.benchmark import OPSSATBenchmark
from spaceai.segmentators.ops_sat_segmentator import OPSSATDatasetSegmentator

from spaceai.benchmark.callbacks import SystemMonitorCallback


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
            "diff2_var",
            "kurtosis",
            "skew",
        ]

def run_esa_exp(classifier, is_supervised, extract_features, exp_dir, run_id):

    segmentator = EsaDatasetSegmentator(
        segment_duration=50,
        step_duration=50,
        extract_features=extract_features,
        transformations=BASE_STATISTICS_LIST,
        segments_id="channel_segments",
        save_csv=False
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

    segmentator = NasaDatasetSegmentator(
        segment_duration=50,
        step_duration=10,
        extract_features=extract_features,
        transformations=BASE_STATISTICS_LIST,
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
        print(f"{i+1}/{len(channels)}: {channel_id}")

        benchmark.run_classifier(
            channel_id,
            classifier=classifier,
            callbacks=callbacks,
        )


def run_ops_exp(classifier, is_supervised, extract_features, exp_dir, run_id):

    segmentator = OPSSATDatasetSegmentator(
        segment_duration=50,
        step_duration=50,
        extract_features=extract_features,
        transformations=BASE_STATISTICS_LIST,
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