import os

import pandas as pd
from spaceai.data import ESA, ESAMissions
from spaceai.benchmark import ESABenchmark

from spaceai.benchmark.callbacks import SystemMonitorCallback
from spaceai.models.anomaly_classifier.dpmm_detector import DPMMWrapperDetector
from spaceai.segmentators.esa_segmentator import EsaDatasetSegmentator

def main():
    run_id = "esa_dpmm_likelihood_experiment"
    nasa_segmentator = EsaDatasetSegmentator(
        segment_duration=50,
        step_duration=50,
        extract_features=True,
        transformations=[
            "mean", "var", "std", "n_peaks",
            "smooth10_n_peaks", "smooth20_n_peaks", 
            "diff_peaks", "diff2_peaks", "diff_var", "diff2_var",
        ],
        segments_id="channel_segments"
    )
    benchmark = ESABenchmark(
        run_id=run_id, 
        exp_dir="experiments", 
        data_root="datasets",
        segmentator = nasa_segmentator,
    )
    callbacks = [SystemMonitorCallback()]

    for mission_wrapper in ESAMissions:
        mission = mission_wrapper.value
        if mission.index != 1:
            continue
        for channel_id in mission.target_channels:
            if int(channel_id.split("_")[1])<41 or int(channel_id.split("_")[1])>46:
                continue
            detector = DPMMWrapperDetector(
                mode="likelihood",      # oppure "new_cluster"
                model_type="Full",
                K=100,
                num_iterations=50,
                lr=0.8,
                python_executable="/opt/homebrew/Caskroom/miniconda/base/envs/dpmm_env/bin/python"  # Inserisci il percorso corretto del tuo ambiente Python
            )

            benchmark.run_classifier(
                mission=mission,
                channel_id=channel_id,
                classifier=detector,
                callbacks=callbacks,
                supervised=False
            )
            
        results_df = pd.read_csv(os.path.join(benchmark.run_dir, "results.csv"))
        tp = results_df['true_positives'].sum()
        fp = results_df['false_positives'].sum()
        fn = results_df['false_negatives'].sum()

        total_precision = tp / (tp + fp)
        total_recall = tp / (tp + fn)
        total_f1 = 2 * (total_precision * total_recall) / \
            (total_precision + total_recall)
        
        print("True Positives: ", tp)
        print("False Positives: ", fp)
        print("False Negatives: ", fn)
        print("Total Precision: ", total_precision)
        print("Total Recall: ", total_recall)
        print("Total F1: ", total_f1)


if __name__ == "__main__":
    main()
