import os
import pandas as pd

from spaceai.data import NASA
from spaceai.benchmark import NASABenchmark
from spaceai.benchmark.callbacks import SystemMonitorCallback
from spaceai.segmentators.nasa_segmentator import NasaDatasetSegmentator
from spaceai.models.anomaly_classifier import RocketClassifier
from spaceai.models.anomaly_classifier.dpmm_detector import DPMMWrapperDetector


def main():
    model_types = ["Full", "Diagonal", "Single", "Unit"]
    for model_type in model_types:
        run_id = f"nasa_rocket_dpmm_{model_type.lower()}"
        segmentator = NasaDatasetSegmentator(
            segment_duration=50,
            step_duration=50,
            extract_features=False,
        )
        benchmark = NASABenchmark(
            run_id=run_id,
            exp_dir="experiments",
            data_root="datasets",
            segmentator=segmentator,
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
                lr=0.1,
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
        tp = results_df["true_positives"].sum()
        fp = results_df["false_positives"].sum()
        fn = results_df["false_negatives"].sum()

        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        f0_5 = (
            (1 + 0.5**2) * precision * recall / (0.5**2 * precision + recall)
            if (0.5**2 * precision + recall) > 0
            else 0.0
        )

        print(f"Results for {model_type}:")
        print("True Positives:", tp)
        print("False Positives:", fp)
        print("False Negatives:", fn)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1:", f1)
        print("F0.5:", f0_5)


if __name__ == "__main__":
    main()

