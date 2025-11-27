import argparse
import os
from typing import (
    Optional,
)

import numpy as np
import pandas as pd  # type: ignore
from tabulate import tabulate  # type: ignore
from tqdm import tqdm  # type: ignore

from spaceai.data.esa import (
    ESA,
    ESAMission,
    ESAMissions,
)
from spaceai.data.nasa import NASA
from spaceai.data.ops_sat import OPSSAT

WINDOW_LENGTH = 50

DATASET_ALIAS = {"ops": "OPS_SAT", "nasa": "NASA", "esa": "ESA"}


def check_datasets(datasets):
    dataset_array = datasets.split(",")
    return [
        DATASET_ALIAS[dataset.strip(" ").lower()]
        for dataset in dataset_array
        if dataset.strip(" ").lower() in DATASET_ALIAS
    ]


def compute_experiment_scores(results_df: pd.DataFrame) -> Optional[dict]:
    req = {
        "true_positives",
        "false_positives",
        "false_negatives",
        "train_time",
        "detected_negatives",
        "test_negatives",
        "test_length",
    }
    if not req.issubset(results_df.columns):
        return None

    df = results_df.copy()
    df["channel"] = df.get("channel", df.get("channel_id"))
    if df["channel"].isnull().any():
        raise ValueError("Serve 'channel' o 'channel_id' non null.")
    df["channel"] = df["channel"].astype(str)
    df["predict_time"] = df.get("predict_time", df.get("detect_time", 0.0))

    tp, fp, fn = (
        df["true_positives"].sum(),
        df["false_positives"].sum(),
        df["false_negatives"].sum(),
    )
    tot_neg = df["test_negatives"].sum()
    tnr = (df["detected_negatives"].sum() / tot_neg) if tot_neg > 0 else 0.0

    precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    prec_corr = precision * tnr

    n = len(df)
    return {
        "f1": float(
            (2 * prec_corr * recall / (prec_corr + recall))
            if (prec_corr + recall) > 0
            else 0.0
        ),
        "f0.5": float(
            ((1 + 0.5**2) * prec_corr * recall / (0.5**2 * prec_corr + recall))
            if (0.5**2 * prec_corr + recall) > 0
            else 0.0
        ),
        "precision": float(prec_corr),
        "recall": float(recall),
        "tnr": float(tnr),
        "train_time": float(df["train_time"].mean()) if n else 0.0,
        "predict_time": float(df["predict_time"].mean()) if n else 0.0,
        "total_negatives": int(tot_neg),
        "channels": int(df["channel"].nunique()),
    }


def parse_args():
    p = argparse.ArgumentParser(description="Aggrega risultati esperimenti AD.")
    p.add_argument(
        "--base-dir", default="experiments", help="Cartella radice degli esperimenti."
    )
    p.add_argument(
        "--output-dir", default="all_results", help="Cartella dove salvare i CSV."
    )
    p.add_argument(
        "--datasets",
        default="ops,nasa,esa",
        help="Dataset da includere (comma-sep): es. 'ops,nasa' oppure 'esa'.",
    )
    p.add_argument(
        "--include",
        default="",
        help="Regex per includere solo alcune cartelle esperimento (match su nome cartella).",
    )
    p.add_argument(
        "--exclude",
        default="",
        help="Regex per escludere alcune cartelle esperimento (match su nome cartella).",
    )
    p.add_argument(
        "--print_tables",
        default=True,
        action="store_true",
        help="Stampa tabelle in console (tabulate).",
    )
    return p.parse_args()


def render_and_export(
    out_df: pd.DataFrame,
    output_dir: str,
    datasets_filter: list[str],
    print_tables: bool = False,
) -> None:

    os.makedirs(export_dir, exist_ok=True)

    # Per dataset
    for ds in datasets_filter:
        ds_name = DATASET_ALIAS.get(ds, ds.upper())
        view = out_df[out_df["dataset"] == ds_name].drop(
            columns=["channels", "dataset", "total_negatives"], errors="ignore"
        )
        if view.empty:
            continue

        view_sorted = view.sort_values("f1", ascending=False)
        csv_path = os.path.join(export_dir, f"{ds}_results.csv")
        view_sorted.to_csv(csv_path, index=False)

        if print_tables:
            print(f"\n=== DATASET: {ds_name} ===")
            print(
                tabulate(
                    view_sorted, headers="keys", tablefmt="fancy_grid", floatfmt=".4f"
                )
            )
            print(f"Salvato in: {csv_path}")

    # Summary globale
    summary_path = os.path.join(output_dir, "summary_all.csv")
    out_df.sort_values(["dataset", "f1"], ascending=[True, False]).to_csv(
        summary_path, index=False
    )
    if print_tables:
        print(f"\nRiassunto globale salvato in: {summary_path}")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    datasets_filter = [d.lower() for d in check_datasets(args.datasets)]

    rows = []
    for root, _, files in os.walk(args.base_dir):
        if "results.csv" not in files:
            continue

        rel = os.path.relpath(root, args.base_dir)
        folder = os.path.basename(rel)
        prefix = folder.split("_")[0].lower()
        if prefix not in datasets_filter:
            continue

        dataset_name = DATASET_ALIAS.get(prefix, prefix.upper())

        csv_path = os.path.join(root, "results.csv")
        res_df = pd.read_csv(csv_path)

        scores = compute_experiment_scores(results_df=res_df)
        if not scores:
            continue

        rows.append({"dataset": dataset_name, "model": folder, **scores})

    out_df = pd.DataFrame(rows)
    if out_df.empty:
        print("Nessun results.csv valido trovato.")
        return

    render_and_export(
        out_df=out_df,
        output_dir=args.output_dir,
        datasets_filter=datasets_filter,
        print_tables=args.print_tables,
    )


if __name__ == "__main__":
    main()
