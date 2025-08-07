import os
import pandas as pd

from spaceai.data import ESA, ESAMissions
from spaceai.benchmark import ESABenchmark
from spaceai.benchmark.callbacks import SystemMonitorCallback
from spaceai.models.anomaly_classifier.dpmm_detector import DPMMWrapperDetector
from spaceai.segmentators.esa_segmentator import EsaDatasetSegmentator
from utils import compute_and_print_metrics


def main():
    model_types = ["Full", "Diagonal", "Single", "Unit"]

    for model_type in model_types:
        run_id = f"esa_dpmm_new_cluster_{model_type.lower()}"
        nasa_segmentator = EsaDatasetSegmentator(
            segment_duration=50,
            step_duration=50,
            extract_features=True,
            transformations=[
                "mean", "var", "std", "n_peaks",
                "smooth10_n_peaks", "smooth20_n_peaks",
                "diff_peaks", "diff2_peaks", "diff_var", "diff2_var",
            ],
            segments_id="channel_segments",
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
                detector = DPMMWrapperDetector(
                    mode="new_cluster",
                    model_type=model_type,
                    K=100,
                    num_iterations=50,
                    lr=0.8,
                    python_executable="/opt/homebrew/Caskroom/miniconda/base/envs/dpmm_env/bin/python",
                )

                benchmark.run_classifier(
                    mission=mission,
                    channel_id=channel_id,
                    classifier=detector,
                    callbacks=callbacks,
                    supervised=False,
                )

        results_df = pd.read_csv(os.path.join(benchmark.run_dir, "results.csv"))
        compute_and_print_metrics(results_df, "esa")


if __name__ == "__main__":
    main()
