"""Results reader module for aggregating and exporting experiment results."""

import argparse
import os
import re
from typing import Optional

import pandas as pd  # type: ignore
from tabulate import tabulate  # type: ignore

WINDOW_LENGTH = 50

DATASET_ALIAS = {"ops": "OPS_SAT", "nasa": "NASA", "esa": "ESA"}
TAG_TO_DATASET = {
    "chF": "esa",
    "mission1": "esa",
    "mission2": "esa",
    "smap": "nasa",
    "msl": "nasa",
}


def check_datasets(datasets):
    """Check and filter valid datasets from input string."""
    dataset_array = datasets.split(",")
    return [
        DATASET_ALIAS[dataset.strip(" ").lower()]
        for dataset in dataset_array
        if dataset.strip(" ").lower() in DATASET_ALIAS
    ]


def compute_experiment_scores(results_df: pd.DataFrame, include_efficiency: bool = False) -> Optional[dict]:
    """Compute F1, Precision, Recall, and other metrics from experiment results."""
    req = {
        "true_positives",
        "false_positives",
        "false_negatives",
    }
    if not req.issubset(results_df.columns):
        return None

    col = "channel_id" if "channel_id" in results_df.columns else "channel"
    global_row = results_df[results_df[col].isin(["GLOBAL_EVENT_LEVEL", "GLOBAL"])]
    
    if not global_row.empty:
        row = global_row.iloc[0]
        tp = row.get("true_positives", 0)
        fp = row.get("false_positives", 0)
        fn = row.get("false_negatives", 0)
        tnr = row.get("tnr", 0.0)
        tot_neg = row.get("test_negatives", 0)
    else:
        df = results_df[~results_df[col].astype(str).str.startswith("GLOBAL")]
        tp = df["true_positives"].sum() if "true_positives" in df.columns else 0
        fp = df["false_positives"].sum() if "false_positives" in df.columns else 0
        fn = df["false_negatives"].sum() if "false_negatives" in df.columns else 0
        tot_neg = df["test_negatives"].sum() if "test_negatives" in df.columns else 0
        tnr = (df["detected_negatives"].sum() / tot_neg) if tot_neg > 0 and "detected_negatives" in df.columns else 0.0

    precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    prec_corr = precision * tnr

    n = len(results_df)
    
    def get_mean_time(df, cols):
        for c in cols:
            if c in df.columns:
                return df[c].mean()
        return 0.0

    predict_time = get_mean_time(results_df, ["predict_time", "detection_time", "segmentation_split_time"])
    train_time = get_mean_time(results_df, ["train_time", "fitting_time"])

    out_dict = {
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
        "train_time": float(train_time),
        "predict_time": float(predict_time),
        "total_negatives": int(tot_neg),
        "channels": int(len(results_df) - (1 if not global_row.empty else 0)),
    }

    # Add extra metrics requested from GLOBAL row
    extra_metrics = [
        "n_anomalies", "n_detected", "true_positives", "false_positives", "false_negatives",
        "precision", "recall", "f1", "tnr", "test_length", "test_negatives", "detected_negatives",
        "precision_corrected", "corrected_f0.5", "corrected_f1", "adtqc_n_before", "adtqc_n_after",
        "adtqc_after_rate", "adtqc_score", "total_segmentation_time", "total_segmentation_cpu",
        "total_segmentation_mem", "feature_extraction_time", "feature_extraction_cpu",
        "feature_extraction_mem", "feature_selection_time", "feature_selection_cpu", "feature_selection_mem"
    ]
    
    if not global_row.empty:
        row = global_row.iloc[0]
        for m in extra_metrics:
            # Filter efficiency metrics if requested
            if not include_efficiency and any(x in m.lower() for x in ["cpu", "mem", "time"]):
                continue
                
            if m in row:
                val = row[m]
                if pd.notna(val):
                    out_dict[m] = val
                    
    return out_dict


def parse_args():
    """Parse command line arguments."""
    p = argparse.ArgumentParser(description="Aggregate AD experiment results.")
    p.add_argument(
        "--base-dir", default="experiments", help="Root folder of experiments."
    )
    p.add_argument(
        "--output-dir", default="all_results", help="Directory where to save CSVs."
    )
    p.add_argument(
        "--datasets",
        default="ops,nasa,esa",
        help="Datasets to include (comma-sep): e.g. 'ops,nasa' or 'esa'.",
    )
    p.add_argument(
        "--include",
        default="",
        help="Regex to include only specific experiment folders (match on folder name).",
    )
    p.add_argument(
        "--exclude",
        default="",
        help="Regex to exclude specific experiment folders (match on folder name).",
    )
    p.add_argument(
        "--print_tables",
        default=False,
        action="store_true",
        help="Print tables in console (tabulate).",
    )
    p.add_argument(
        "--include-efficiency",
        default=False,
        action="store_true",
        help="Include efficiency metrics (CPU, Memory, Time) in the output.",
    )
    p.add_argument(
        "--min-channels",
        type=int,
        default=50,
        help="Minimum number of channels to include an experiment.",
    )
    return p.parse_args()


def render_and_export(
    out_df: pd.DataFrame,
    output_dir: str,
    datasets_filter: list[str],
    print_tables: bool = False,
) -> None:
    """Render results to console and export to CSV."""

    os.makedirs(output_dir, exist_ok=True)

    for ds in datasets_filter:
        ds_name = DATASET_ALIAS.get(ds, ds.upper())
        view = out_df[out_df["dataset"] == ds_name].drop(
            columns=["channels", "dataset", "total_negatives"], errors="ignore"
        )
        if view.empty:
            continue

        view_sorted = view.sort_values("f0.5", ascending=False)
        csv_path = os.path.join(output_dir, f"{ds}_results.csv")
        view_sorted.to_csv(csv_path, index=False)

        if print_tables:
            print(f"\n=== DATASET: {ds_name} ===")
            print(
                tabulate(
                    view_sorted, headers="keys", tablefmt="fancy_grid", floatfmt=".4f"
                )
            )
            print(f"Salvato in: {csv_path}")

    summary_path = os.path.join(output_dir, "summary_all.csv")
    out_df.sort_values(["dataset", "f0.5"], ascending=[True, False]).to_csv(
        summary_path, index=False
    )
    if print_tables:
        print(f"\nGlobal summary saved to: {summary_path}")


def main():
    """Main function to aggregate and export results."""
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    datasets_filter = [d.lower() for d in check_datasets(args.datasets)]

    rows = []
    for root, _, files in os.walk(args.base_dir):
        if "results.csv" not in files:
            continue

        rel = os.path.relpath(root, args.base_dir)
        folder = os.path.basename(rel)
        
        if args.include and not re.search(args.include, folder):
            continue
        if args.exclude and re.search(args.exclude, folder):
            continue

        found_ds = None
        for ds in datasets_filter:
            if ds in folder.lower():
                found_ds = ds
                break
        
        if not found_ds:
            for tag, ds in TAG_TO_DATASET.items():
                if tag in folder.lower() and ds in datasets_filter:
                    found_ds = ds
                    break
        
        csv_path = os.path.join(root, "results.csv")
        
        if not found_ds:
            try:
                temp_df = pd.read_csv(csv_path, nrows=1)
                if not temp_df.empty:
                    col = "channel_id" if "channel_id" in temp_df.columns else "channel"
                    if col in temp_df.columns:
                        chan = str(temp_df[col].iloc[0]).lower()
                        if "channel_" in chan or "id_" in chan or chan.isdigit():
                            if "esa" in datasets_filter: found_ds = "esa"
                        elif "smap" in chan or "msl" in chan:
                            if "nasa" in datasets_filter: found_ds = "nasa"
            except Exception:
                pass

        if not found_ds:
            continue

        dataset_name = DATASET_ALIAS.get(found_ds, found_ds.upper())
        res_df = pd.read_csv(csv_path)

        scores = compute_experiment_scores(results_df=res_df, include_efficiency=args.include_efficiency)
        if not scores:
            continue

        rows.append({"dataset": dataset_name, "model": folder, **scores})

    out_df = pd.DataFrame(rows)
    if out_df.empty:
        return

    if args.min_channels > 0:
        out_df = out_df[out_df["channels"] >= args.min_channels]
        if out_df.empty:
            print(f"Nessun esperimento trovato con almeno {args.min_channels} canali.")
            return

    render_and_export(
        out_df=out_df,
        output_dir=args.output_dir,
        datasets_filter=datasets_filter,
        print_tables=args.print_tables,
    )


if __name__ == "__main__":
    main()
