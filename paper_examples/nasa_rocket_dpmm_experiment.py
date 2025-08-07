import os
import pandas as pd

from spaceai.data import NASA
from spaceai.benchmark import NASABenchmark
from spaceai.benchmark.callbacks import SystemMonitorCallback
from spaceai.segmentators.nasa_segmentator import NasaDatasetSegmentator
from spaceai.models.anomaly_classifier import RocketClassifier
from spaceai.models.anomaly_classifier.dpmm_detector import DPMMWrapperDetector
from utils import compute_and_print_metrics


def main():
    model_types = ["Full", "Diagonal", "Single", "Unit"]

    for model_type in model_types:
        run_id = f"nasa_rocket_dpmm_{model_type.lower()}"
        nasa_segmentator = NasaDatasetSegmentator(
            segment_duration=50,
            step_duration=50,
            extract_features=False,
        )
        benchmark = NASABenchmark(
            run_id=run_id,
            exp_dir="experiments",
            data_root="datasets",
            segmentator=nasa_segmentator,
        )
        callbacks = [SystemMonitorCallback()]
        channels = NASA.channel_ids
        for i, channel_id in enumerate(channels):
            print(f"{i+1}/{len(channels)}: {channel_id}")

            base_classifier = DPMMWrapperDetector(
                mode="likelihood",
                model_type=model_type,
                K=100,
                num_iterations=50,
                lr=0.8,
                python_executable="/opt/homebrew/Caskroom/miniconda/base/envs/dpmm_env/bin/python",
            )
            benchmark.run_classifier(
                channel_id,
                classifier=RocketClassifier(
                    base_model=base_classifier,
                    num_kernels=50,
                ),
                callbacks=callbacks,
            )

        results_df = pd.read_csv(os.path.join(benchmark.run_dir, "results.csv"))
        compute_and_print_metrics(results_df, "nasa")


if __name__ == "__main__":
    main()
