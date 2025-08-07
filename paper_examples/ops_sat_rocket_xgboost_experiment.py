import os
import pandas as pd

from spaceai.data.ops_sat import OPSSAT
from spaceai.benchmark import OPSSATBenchmark
from spaceai.benchmark.callbacks import SystemMonitorCallback
from spaceai.segmentators.ops_sat_segmentator import OPSSATDatasetSegmentator
from spaceai.models.anomaly_classifier import RocketClassifier
from xgboost import XGBClassifier
from utils import compute_and_print_metrics


def main():
    run_id = "ops_sat_rocket_xgboost"
    nasa_segmentator = OPSSATDatasetSegmentator(
        segment_duration=50,
        step_duration=50,
        extract_features=False,
    )
    benchmark = OPSSATBenchmark(
        run_id=run_id,
        exp_dir="experiments",
        data_root="datasets",
        segmentator=nasa_segmentator,
    )
    callbacks = [SystemMonitorCallback()]

    channels = OPSSAT.channel_ids
    for i, channel_id in enumerate(channels):
        print(f"{i+1}/{len(channels)}: {channel_id}")

        base_classifier = XGBClassifier()
        benchmark.run_classifier(
            channel_id,
            classifier=RocketClassifier(
                base_model=base_classifier,
                num_kernels=1000,
            ),
            callbacks=callbacks,
            supervised=True,
        )

    results_df = pd.read_csv(os.path.join(benchmark.run_dir, "results.csv"))
    compute_and_print_metrics(results_df, "ops")


if __name__ == "__main__":
    main()
