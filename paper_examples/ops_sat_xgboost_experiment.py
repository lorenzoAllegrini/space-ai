import os
import pandas as pd
from spaceai.data.ops_sat import OPSSAT
from spaceai.benchmark import OPSSATBenchmark
from spaceai.benchmark.callbacks import SystemMonitorCallback
from spaceai.segmentators.ops_sat_segmentator import OPSSATDatasetSegmentator
from spaceai.models.anomaly_classifier.dpmm_detector import DPMMWrapperDetector
from xgboost import XGBClassifier

def main():
    run_id = "ops_sat_xgboost"
    nasa_segmentator = OPSSATDatasetSegmentator(
        segment_duration=50,
        step_duration=50,
        extract_features=True,
        transformations= [
            "mean", "var", "std", "n_peaks",
            "smooth10_n_peaks", "smooth20_n_peaks",
            "diff_peaks", "diff2_peaks", "diff_var", "diff2_var", "kurtosis", "skew"
        ]
    )
    benchmark = OPSSATBenchmark(
        run_id=run_id, 
        exp_dir="experiments", 
        data_root="datasets",
        segmentator = nasa_segmentator,
    )
    callbacks = [SystemMonitorCallback()]

    channels = OPSSAT.channel_ids
    for i, channel_id in enumerate(channels):
        print(f'{i+1}/{len(channels)}: {channel_id}')
        
        classifier = XGBClassifier()
        benchmark.run_classifier(
            channel_id,
            classifier=classifier,
            callbacks=callbacks,
            supervised= True
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
