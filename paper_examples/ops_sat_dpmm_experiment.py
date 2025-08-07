import os
import pandas as pd

from spaceai.data.ops_sat import OPSSAT
from spaceai.benchmark import OPSSATBenchmark
from spaceai.benchmark.callbacks import SystemMonitorCallback
from spaceai.segmentators.ops_sat_segmentator import OPSSATDatasetSegmentator
from spaceai.models.anomaly_classifier.dpmm_detector import DPMMWrapperDetector
from utils import compute_and_print_metrics


def main():
    model_types = ["Full", "Diagonal", "Single", "Unit"]

    for model_type in model_types:
        run_id = f"ops_sat_dpmm_likelihood_{model_type.lower()}"
        nasa_segmentator = OPSSATDatasetSegmentator(
            segment_duration=50,
            step_duration=50,
            extract_features=True,
            transformations=[
                "mean", "var", "std", "n_peaks",
                "smooth10_n_peaks", "smooth20_n_peaks",
                "diff_peaks", "diff2_peaks", "diff_var", "diff2_var", "kurtosis", "skew",
            ],
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

            classifier = DPMMWrapperDetector(
                mode="likelihood",
                model_type=model_type,
                K=100,
                num_iterations=50,
                lr=0.8,
                python_executable="/opt/homebrew/Caskroom/miniconda/base/envs/dpmm_env/bin/python",
            )
            benchmark.run_classifier(
                channel_id,
                classifier=classifier,
                callbacks=callbacks,
                supervised=False,
            )

        results_df = pd.read_csv(os.path.join(benchmark.run_dir, "results.csv"))
        compute_and_print_metrics(results_df, "ops")


if __name__ == "__main__":
    main()
