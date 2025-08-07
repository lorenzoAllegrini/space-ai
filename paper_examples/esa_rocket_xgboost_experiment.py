import os
import pandas as pd

from spaceai.data import ESA, ESAMissions
from spaceai.benchmark import ESABenchmark
from spaceai.benchmark.callbacks import SystemMonitorCallback
from spaceai.segmentators.esa_segmentator import EsaDatasetSegmentator
from spaceai.models.anomaly_classifier import RocketClassifier
from xgboost import XGBClassifier
from utils import compute_and_print_metrics


def main():
    run_id = "esa_rocket_xgboost_experiment"
    nasa_segmentator = EsaDatasetSegmentator(
        segment_duration=50,
        step_duration=50,
        extract_features=False,
        segments_id="channel_segments",
        save_csv=False,
    )
    benchmark = ESABenchmark(
        run_id=run_id,
        exp_dir="experiments",
        data_root="datasets",
        segmentator=nasa_segmentator,
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
                classifier=RocketClassifier(
                    base_model=XGBClassifier(),
                    num_kernels=10,
                ),
                callbacks=callbacks,
                supervised=True,
            )

    results_df = pd.read_csv(os.path.join(benchmark.run_dir, "results.csv"))
    compute_and_print_metrics(results_df, "esa")


if __name__ == "__main__":
    main()
