"""Results reader module for aggregating and exporting experiment results."""

import argparse
import os
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


def compute_experiment_scores(results_df: pd.DataFrame) -> Optional[dict]:
    """Compute F1, Precision, Recall, and other metrics from experiment results."""
    req = {
        "true_positives",
        "false_positives",
        "false_negatives",
        "train_time",
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

    precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0

    n = len(df)
    results = {
        "f1": float(
            (2 * precision * recall / (precision + recall))
            if (precision + recall) > 0
            else 0.0
        ),
        "f0.5": float(
            ((1 + 0.5**2) * precision * recall / (0.5**2 * precision + recall))
            if (0.5**2 * precision + recall) > 0
            else 0.0
        ),
        "precision": float(precision),
        "recall": float(recall),
        "train_time": float(df["train_time"].mean()) if n else 0.0,
        "predict_time": float(df["predict_time"].mean()) if n else 0.0,
        "channels": int(df["channel"].nunique()),
    }
    if "test_negatives" in df.columns and "detected_negatives" in df.columns:
        tot_neg = df["test_negatives"].sum()
        tnr = (df["detected_negatives"].sum() / tot_neg) if tot_neg > 0 else 0.0
        prec_corr = precision * tnr
        results.update({
            "tnr": float(tnr),
            "total_negatives": int(tot_neg),
            "precision_corrected": float(prec_corr),
            "f1_corrected": float((2 * prec_corr * recall / (prec_corr + recall)) if (prec_corr + recall) > 0 else 0.0),
            "f0.5_corrected": float(((1 + 0.5**2) * prec_corr * recall / (0.5**2 * prec_corr + recall)) if (0.5**2 * prec_corr + recall) > 0 else 0.0),
        })
    return results

def integrate_if_missing(df: pd.DataFrame, dataset_name: str, csv_path: str, data_path: str) -> None:
    missing_cols = ["test_negatives", "test_length", "detected_negatives"]
    if dataset_name is not None and any(col not in df.columns for col in missing_cols):
        from examples.utils.dataset_exp import get_dataset_benchmark  # type: ignore
        from tqdm import tqdm  # type: ignore

        benchmark = get_dataset_benchmark(
            dataset_name=dataset_name.lower(),
            data_path=data_path,
            run_id="",
            exp_dir="",
        )
        for channel in tqdm(df["channel_id"].unique(), desc=f"Integrating missing data for {dataset_name}"):
            _, test = benchmark.load_channel(channel_id=channel)
            total_length = len(test.data)
            indices_true_grouped = [list(range(e[0], e[1] + 1)) for e in test.anomalies]
            indices_true_flat = set([i for group in indices_true_grouped for i in group])
            n_e = total_length - len(indices_true_flat)
            df.loc[df["channel_id"] == channel, "test_negatives"] = int(n_e)
            df.loc[df["channel_id"] == channel, "test_length"] = int(total_length)
            tnr = df.loc[df["channel_id"] == channel, "tnr"].values[0] if "tnr" in df.columns else 1.0
            df.loc[df["channel_id"] == channel, "detected_negatives"] = int(round(n_e * tnr))
        
        if csv_path is not None:
            df.to_csv(csv_path, index=False)
            

def parse_args():
    """Parse command line arguments."""
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
    p.add_argument(
        "--data_path",
        default="datasets",
        help="Percorso della cartella contenente i dataset.",
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

        view_sorted = view.sort_values("f1", ascending=False)
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
    out_df.sort_values(["dataset", "f1"], ascending=[True, False]).to_csv(
        summary_path, index=False
    )
    if print_tables:
        print(f"\nRiassunto globale salvato in: {summary_path}")


def main():
    """Main function to aggregate and export results."""
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    datasets_filter = [d.lower() for d in args.datasets.split(',')]

    rows = []
    for root, _, files in os.walk(args.base_dir):
        rel = os.path.relpath(root, args.base_dir)
        folder = os.path.basename(rel)
        prefix = folder.split("_")[0].lower()
        if "results.csv" not in files:
            print(f"Skipping {root}: No results.csv found.")
            continue


        if prefix not in datasets_filter:
            print(f'Skipping {root}: Prefix dataset name {prefix} not in {datasets_filter}.')
            continue

        dataset_name = DATASET_ALIAS.get(prefix, prefix.upper())

        csv_path = os.path.join(root, "results.csv")
        res_df = pd.read_csv(csv_path)

        integrate_if_missing(res_df, dataset_name=dataset_name, csv_path=csv_path, data_path=args.data_path)

        scores = compute_experiment_scores(results_df=res_df)
        if not scores:
            print(f"Skipping {root}: No valid results found.")
            continue

        rows.append({"dataset": dataset_name, "model": folder, 'path': root, **scores})

    print(len(rows), "esperimenti letti.")
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
