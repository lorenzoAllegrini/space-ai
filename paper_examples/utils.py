import os
import numpy as np
import pandas as pd
from spaceai.data.ops_sat import OPSSAT
from spaceai.data.nasa import NASA
from spaceai.data.esa import ESA, ESAMissions

def compute_and_print_metrics(results_df: pd.DataFrame, dataset: str) -> None:
    """Compute and print aggregated metrics.

    Parameters
    ----------
    results_df: pd.DataFrame
        DataFrame produced by a benchmark run containing at least the
        columns ``true_positives``, ``false_positives``, ``false_negatives``
        and ``tnr``.
    dataset: str
        One of ``"ops"``, ``"nasa"`` or ``"esa"``.
    """
    tp = fp = fn = tn = 0
    total_negatives = 0
    for _, row in results_df.iterrows():
        if dataset == "ops":
            test_channel = OPSSAT(
                root="datasets",
                channel_id=row["channel_id"],
                mode="anomaly",
                overlapping=False,
                seq_length=1,
                train=False,
                drop_last=False,
                n_predictions=1,
            )
        elif dataset == "nasa":
            test_channel = NASA(
                root="datasets",
                channel_id=row["channel_id"],
                mode="anomaly",
                overlapping=False,
                seq_length=1,
                train=False,
                drop_last=False,
                n_predictions=1,
            )
        else:  # ESA
            test_channel = ESA(
                root="datasets",
                channel_id=row["channel_id"],
                mode="anomaly",
                overlapping=False,
                seq_length=1,
                train=False,
                drop_last=False,
                n_predictions=1,
                mission=ESAMissions.MISSION_1.value,
            )

        length = len(test_channel.data) // 50
        labels = np.zeros(length * 50, dtype=int)
        for start, end in test_channel.anomalies:
            start = max(0, start)
            end = min(len(labels) - 1, end)
            labels[start : end + 1] = 1

        negatives = sum(1 for r in labels.reshape(-1, 50) if np.sum(r) == 0)
        tn += row["tnr"] * negatives
        tp += row["true_positives"]
        fp += row["false_positives"]
        fn += row["false_negatives"]
        total_negatives += negatives

    tnr = tn / total_negatives if total_negatives > 0 else 0.0
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    precision_corrected = precision * tnr
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = (
        2 * (precision_corrected * recall) / (precision_corrected + recall)
        if precision_corrected + recall > 0
        else 0.0
    )
    f0_5 = (
        (1 + 0.5 ** 2)
        * (precision_corrected * recall)
        / ((0.5 ** 2) * precision_corrected + recall)
        if precision_corrected + recall > 0
        else 0.0
    )

    print("Precision:", precision_corrected)
    print("Recall:", recall)
    print("F1:", f1)
    print("F0.5:", f0_5)
