import os
import pandas as pd
from spaceai.data import NASA
from spaceai.benchmark import NASABenchmark
from spaceai.benchmark.callbacks import SystemMonitorCallback
from spaceai.segmentators.nasa_segmentator import NasaDatasetSegmentator
from spaceai.models.anomaly_classifier import RockadClassifier
from sklearn.svm import OneClassSVM
from sklearn.linear_model import RidgeClassifier
from spaceai.models.anomaly_classifier.dpmm_detector import DPMMWrapperDetector
def main():
    return
    run_id = "nasa_rockad"
    nasa_segmentator = NasaDatasetSegmentator(
        segment_duration=50,
        step_duration=50,
        extract_features=False,
    )
    benchmark = NASABenchmark(
        run_id=run_id, 
        exp_dir="experiments", 
        data_root="datasets",
        segmentator = nasa_segmentator,
    )
    callbacks = [SystemMonitorCallback()]
    channels = NASA.channel_ids
    for i, channel_id in enumerate(channels):
        print(f'{i+1}/{len(channels)}: {channel_id}')

        benchmark.run_classifier(
            channel_id,
            classifier=RockadClassifier(),
            callbacks=callbacks,
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