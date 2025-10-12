import os
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from tabulate import tabulate

from spaceai.data.esa import ESA, ESAMissions
from spaceai.data.nasa import NASA
from spaceai.data.ops_sat import OPSSAT


BASE_DIR = "experiments"
WINDOW_SIZE = 50


def load_dataset(dataset: str, channel_id: int):
    if dataset.startswith("ops"):
        return OPSSAT(
            root="datasets",
            channel_id=channel_id,
            mode="anomaly",
            overlapping=False,
            seq_length=1,
            train=False,
            drop_last=False,
            n_predictions=1,
        )

    if dataset.startswith("nasa"):
        return NASA(
            root="datasets",
            channel_id=channel_id,
            mode="anomaly",
            overlapping=False,
            seq_length=1,
            train=False,
            drop_last=False,
            n_predictions=1,
        )

    return ESA(
        root="datasets",
        channel_id=channel_id,
        mode="anomaly",
        overlapping=False,
        seq_length=1,
        train=False,
        drop_last=False,
        n_predictions=1,
        mission=ESAMissions.MISSION_1.value,
    )


def compute_channel_stats(dataset) -> Dict[str, int]:
    data_length = len(dataset.data)
    num_segments = data_length // WINDOW_SIZE

    if num_segments == 0:
        return {"negatives": 0}

    segments_with_anomaly = np.zeros(num_segments, dtype=bool)
    for start, end in getattr(dataset, "anomalies", []):
        if end < 0 or start >= num_segments * WINDOW_SIZE:
            continue

        clipped_start = max(0, start)
        clipped_end = min(end, num_segments * WINDOW_SIZE - 1)
        start_idx = clipped_start // WINDOW_SIZE
        end_idx = clipped_end // WINDOW_SIZE
        segments_with_anomaly[start_idx : end_idx + 1] = True

    negatives = int(np.count_nonzero(~segments_with_anomaly))
    return {"negatives": negatives}


def iter_results_files(base_dir: str) -> Iterable[Tuple[str, str]]:
    for root, _, files in os.walk(base_dir):
        if "results.csv" in files:
            yield root, os.path.join(root, "results.csv")


def get_dataset_name(root: str) -> str:
    relative_path = os.path.relpath(root, BASE_DIR)
    if relative_path == ".":
        return os.path.basename(root)
    return relative_path.split(os.sep)[0]


def main():
    results: List[dict] = []
    channel_stats: Dict[Tuple[str, int], Dict[str, int]] = {}
    results_files = list(iter_results_files(BASE_DIR))

    unique_channels = set()
    for root, file_path in results_files:
        dataset_name = get_dataset_name(root)
        try:
            channels = pd.read_csv(file_path, usecols=["channel_id"]).dropna()["channel_id"]
        except ValueError:
            # The CSV does not contain the channel_id column.
            continue
        for channel_id in pd.unique(channels):
            try:
                channel_id_int = int(channel_id)
            except (TypeError, ValueError):
                continue
            unique_channels.add((dataset_name, channel_id_int))

    for dataset_name, channel_id in unique_channels:
        dataset = load_dataset(dataset_name, channel_id)
        channel_stats[(dataset_name, channel_id)] = compute_channel_stats(dataset)

    for root, file_path in results_files:
        dataset_name = get_dataset_name(root)
        try:
            results_df = pd.read_csv(file_path)
        except FileNotFoundError:
            continue

        required_columns = {"true_positives", "false_positives", "false_negatives", "train_time", "tnr"}
        if not required_columns.issubset(results_df.columns):
            continue

        if "channel_id" not in results_df.columns:
            continue

        results_df = results_df.copy()
        results_df["channel_id"] = pd.to_numeric(results_df["channel_id"], errors="coerce").astype("Int64")
        results_df = results_df.dropna(subset=["channel_id"])
        results_df["channel_id"] = results_df["channel_id"].astype(int)

        results_df["negatives"] = results_df["channel_id"].map(
            lambda ch: channel_stats.get((dataset_name, ch), {"negatives": 0})["negatives"]
        )

        if results_df["negatives"].sum() == 0:
            total_negatives = 0
            tn = 0.0
        else:
            total_negatives = results_df["negatives"].sum()
            tn = (results_df["tnr"] * results_df["negatives"]).sum()

        tp = results_df["true_positives"].sum()
        fp = results_df["false_positives"].sum()
        fn = results_df["false_negatives"].sum()
        train_time = results_df["train_time"].mean()

        time_column = "predict_time" if "predict_time" in results_df.columns else None
        if time_column is None and "detect_time" in results_df.columns:
            time_column = "detect_time"
        predict_time = results_df[time_column].mean() if time_column else 0

        tnr = tn / total_negatives if total_negatives > 0 else 0
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        precision_corrected = precision * tnr
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        if precision_corrected + recall > 0:
            f0_5 = (1 + 0.5**2) * (precision_corrected * recall) / (0.5**2 * precision_corrected + recall)
            f1 = 2 * (precision_corrected * recall) / (precision_corrected + recall)
        else:
            f0_5 = 0.0
            f1 = 0.0

        model_name = os.path.relpath(root, BASE_DIR)
        dataset_type = model_name.split("_")[0]

        results.append(
            {
                "dataset": dataset_type,
                "model": model_name,
                "f1": f1,
                "f0.5": f0_5,
                "precision": precision_corrected,
                "recall": recall,
                "train_time": train_time,
                "predict_time": predict_time,
            }
        )

    if not results:
        return

    df = pd.DataFrame(results)
    os.makedirs(os.path.join(BASE_DIR, "all_results"), exist_ok=True)

    for selected_dataset in ["ops", "nasa", "esa"]:
        print(f"DATASET: {selected_dataset}")
        df_filtered = df[df["dataset"] == selected_dataset]
        if df_filtered.empty:
            print("No results available")
            continue

        df_sorted = df_filtered.sort_values(by="f1", ascending=False).drop(columns=["dataset"])
        print(tabulate(df_sorted, headers="keys", tablefmt="fancy_grid", floatfmt=".4f"))
        df_sorted.to_csv(os.path.join(BASE_DIR, "all_results", f"{selected_dataset}.csv"), index=False)


if __name__ == "__main__":
    main()

