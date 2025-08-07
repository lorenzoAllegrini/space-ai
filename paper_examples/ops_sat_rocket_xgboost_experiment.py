import os
import pandas as pd
from spaceai.data.ops_sat import OPSSAT
from spaceai.benchmark import OPSSATBenchmark
from spaceai.benchmark.callbacks import SystemMonitorCallback
from sklearn.svm import OneClassSVM
from pyod.models.ecod import ECOD
from spaceai.segmentators.ops_sat_segmentator import OPSSATDatasetSegmentator
from spaceai.models.anomaly_classifier import RocketClassifier
from sklearn.linear_model import RidgeClassifier
from spaceai.models.anomaly_classifier.dpmm_detector import DPMMWrapperDetector
from xgboost import XGBClassifier

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
    tp = results_df["true_positives"].sum()
    fp = results_df["false_positives"].sum()
    fn = results_df["false_negatives"].sum()

    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    f0_5 = (
        (1 + 0.5**2)
        * precision
        * recall
        / (0.5**2 * precision + recall)
        if (0.5**2 * precision + recall) > 0
        else 0.0
    )

    print("True Positives:", tp)
    print("False Positives:", fp)
    print("False Negatives:", fn)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1:", f1)
    print("F0.5:", f0_5)


if __name__ == "__main__":
    main()
