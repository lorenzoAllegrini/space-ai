"""Results reader module for aggregating and exporting experiment results."""

import argparse
import os
import re
from typing import Optional

import pandas as pd  # type: ignore
from tabulate import tabulate  # type: ignore

WINDOW_LENGTH = 50

DATASET_ALIAS = {"ops": "OPS_SAT", "nasa": "NASA", "esa": "ESA"}


def check_datasets(datasets):

    """Check and filter valid datasets from input string."""
    dataset_array = datasets.split(",")
    return [
        DATASET_ALIAS[dataset.strip(" ").lower()]
        for dataset in dataset_array
        if dataset.strip(" ").lower() in DATASET_ALIAS
    ]


def compute_experiment_scores(results_df: pd.DataFrame, mode: str = "test") -> Optional[dict]:
    """Compute F1, Precision, Recall, and other metrics from experiment results."""
    prefix = "val_" if mode == "val" else ""
    
    # Check for pre-calculated corrected metrics (preferred)
    f1_col = f"{prefix}corrected_f1"
    f05_col = f"{prefix}corrected_f0.5"
    prec_col = f"{prefix}precision_corrected"
    
    # If they are in the columns and we have a single row (global row), use them directly
    if len(results_df) == 1 and f1_col in results_df.columns:
        row = results_df.iloc[0]
        return {
            "f1": float(row.get(f1_col, 0.0)),
            "f0.5": float(row.get(f05_col, 0.0)),
            "precision": float(row.get(prec_col, 0.0)),
            "recall": float(row.get(f"{prefix}recall", 0.0)),
            "tnr": float(row.get(f"{prefix}tnr", 0.0)),
            "train_time": float(row.get("train_time", row.get("fitting_time", 0.0))),
            "predict_time": float(row.get("predict_time", row.get("detection_time", 0.0))),
            "total_negatives": int(row.get(f"{prefix}test_negatives", 0)),
            "channels": 1 if row.get("channel_id") == "GLOBAL_EVENT_LEVEL" else 1,
        }

    # Fallback to manual aggregation from TP/FP/FN/TNR (legacy or multi-row)
    req = {
        f"{prefix}true_positives",
        f"{prefix}false_positives",
        f"{prefix}false_negatives",
        f"{prefix}detected_negatives",
        f"{prefix}test_negatives",
    }
    if not req.issubset(results_df.columns):
        return None

    df = results_df.copy()
    df["channel"] = df.get("channel", df.get("channel_id"))
    if df["channel"].isnull().any():
        raise ValueError("Serve 'channel' o 'channel_id' non null.")
    df["channel"] = df["channel"].astype(str)
    
    df["predict_time"] = df.get("predict_time", df.get("detection_time", 0.0))
    df["train_time"] = df.get("train_time", df.get("fitting_time", 0.0))

    tp, fp, fn = (
        df[f"{prefix}true_positives"].sum(),
        df[f"{prefix}false_positives"].sum(),
        df[f"{prefix}false_negatives"].sum(),
    )
    tot_neg = df[f"{prefix}test_negatives"].sum()
    tnr = (df[f"{prefix}detected_negatives"].sum() / tot_neg) if tot_neg > 0 else 0.0

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
    """Parse command line arguments."""
    p = argparse.ArgumentParser(description="Aggrega risultati esperimenti AD.")
    p.add_argument(
        "--base-dir", default="experiments_41-46", help="Cartella radice degli esperimenti."
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
        "--mode",
        default="test",
        choices=["test", "val"],
        help="Modalità dei punteggi: 'test' (standard) o 'val' (prefisso val_).",
    )
    p.add_argument(
        "--type",
        default="all",
        choices=["normal", "challenge", "all"],
        help="Tipo di esperimento da includere: 'normal' (_chF_), 'challenge' (_chT_), o 'all'.",
    )
    p.add_argument(
        "--detector",
        default="",
        help="Filtra per tipo di detector (es. 'none', 'molookde').",
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
    """Render results to console and export to CSV."""

    os.makedirs(output_dir, exist_ok=True)

    # Per dataset
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

    # Summary globale
    summary_path = os.path.join(output_dir, "summary_all.csv")
    out_df.sort_values(["dataset", "f0.5"], ascending=[True, False]).to_csv(
        summary_path, index=False
    )
    if print_tables:
        print(f"\nRiassunto globale salvato in: {summary_path}")


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

        # 1. Dataset Filter (Substring based)
        matched_dataset = None
        for ds_key in datasets_filter:
            if ds_key in folder.lower():
                matched_dataset = ds_key
                break
        
        if matched_dataset is None:
            continue

        # 2. Type Filter (Normal vs Challenge)
        if args.type == "challenge" and "_chT_" not in folder:
            continue
        if args.type == "normal" and "_chF_" not in folder:
            continue

        dataset_name = DATASET_ALIAS.get(matched_dataset, matched_dataset.upper())

        # 3. Detector Filter
        if args.detector and f"_{args.detector}_" not in folder:
            continue

        # 4. Regex Headers Filters
        if args.include and not re.search(args.include, folder):
            continue
        if args.exclude and re.search(args.exclude, folder):
            continue

        csv_path = os.path.join(root, "results.csv")

        res_df = pd.read_csv(csv_path)

        # 3. Global Row Prioritization
        # Cerchiamo la riga che riassume l'intero esperimento
        global_df = res_df[res_df["channel_id"].astype(str).str.upper() == "GLOBAL_EVENT_LEVEL"]
        
        if not global_df.empty:
            logging_df = global_df
        else:
            # Fallback per esperimenti legacy senza riga global
            logging_df = res_df

        scores = compute_experiment_scores(results_df=logging_df, mode=args.mode)
        if not scores:
            continue

        # Clean model name for better readability
        model_name = folder
        prefixes_to_remove = [
            "pipeline_base_statistics_",
            f"{matched_dataset}_",
            "dpmm_",
            "pipeline_",
            "['base_statistics']_",
        ]
        for p in prefixes_to_remove:
            model_name = model_name.replace(p, "")
            
        rows.append({"dataset": dataset_name, "model": model_name, **scores})

    out_df = pd.DataFrame(rows)
    if out_df.empty:
        print(f"Nessun results.csv valido trovato per i filtri impostati (mode={args.mode}, type={args.type}).")
        return

    render_and_export(
        out_df=out_df,
        output_dir=args.output_dir,
        datasets_filter=datasets_filter,
        print_tables=args.print_tables,
    )


if __name__ == "__main__":
    main()
