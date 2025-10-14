import os
import pandas as pd
from tabulate import tabulate
from spaceai.data.ops_sat import OPSSAT
from spaceai.data.nasa import NASA
from spaceai.data.esa import ESA, ESAMission, ESAMissions
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import re
import argparse
from tqdm import tqdm

WINDOW_LENGTH = 50

def get_channels() -> Dict[str, List[Any]]:
    return {
        "ESA": list(ESAMissions.MISSION_1.value.target_channels),
        "NASA": list(NASA.channel_ids),
        "OPS_SAT": list(OPSSAT.channel_ids),
    }

def load_dataset_channel(dataset, channel):
    match dataset:
        case "ESA":
            return ESA(
                root="datasets",
                channel_id=channel,
                mode="anomaly",
                overlapping=False,
                seq_length=1,
                train=False,
                drop_last=False,
                n_predictions=1,
                mission=ESAMissions.MISSION_1.value
            )
        case "NASA": 
            return NASA(
                root="datasets",
                channel_id=channel,
                mode="anomaly",
                overlapping=False,
                seq_length=1,
                train=False,
                drop_last=False,
                n_predictions=1,
            )
        case "OPS_SAT":
            return OPSSAT(
                root="datasets",
                channel_id=channel,
                mode="anomaly",
                overlapping=False,
                seq_length=1,
                train=False,
                drop_last=False,
                n_predictions=1,
            )
        case _:
            raise ValueError("Dataset sconosciuto")
              
def compute_channel_stats(dataset, channel) -> dict:

    length_samples = len(channel.data)
    n_windows = length_samples // WINDOW_LENGTH 
    if n_windows <= 0:
        return { "dataset": dataset, "channel": channel.channel_id, "preprocessed_length": 0, "negatives": 0} 
    anomalies: List[Tuple[int, int]] = getattr(channel, "anomalies", []) or []
    if not anomalies:
        return { "dataset": dataset, "channel": channel.channel_id, "preprocessed_length": int(n_windows), "negatives": int(n_windows), }
    
    last_idx = n_windows * WINDOW_LENGTH - 1 
    diff = np.zeros(n_windows + 1, dtype=np.int32) 
    for s, e in anomalies: 
        if s is None or e is None:
            continue 
        s = int(max(0, s))
        e = int(min(last_idx, e))
        if e < s:
            continue
        ws = s // WINDOW_LENGTH
        we = e // WINDOW_LENGTH 
        diff[ws] += 1 
        diff[we + 1] -= 1
        covered = np.cumsum(diff[:-1]) > 0
        negatives = int((~covered).sum()) 
        
        return { "dataset": dataset, "channel": channel.channel_id, "preprocessed_length": int(n_windows), "negatives": negatives}


def load_datasets(output_dir):

    file_path = os.path.join(output_dir, "channels_catalog.pkl")
    if os.path.exists(file_path):
        return pd.read_pickle(file_path)

    rows = []
    for dataset, channel_ids in get_channels().items():
        for channel_id in tqdm(channel_ids, desc=f"{dataset}", ncols=100, leave=True):
            channel = load_dataset_channel(dataset, channel_id)
            rows.append(compute_channel_stats(dataset, channel))
    df = pd.DataFrame(rows)
    df.to_pickle(file_path)
    return df

def compute_experiment_scores(datasets: pd.DataFrame, results_df: pd.DataFrame) -> Optional[dict]:
    req = {'true_positives','false_positives','false_negatives','train_time','tnr'}
    if not req.issubset(results_df.columns):
        return None

    df = results_df.copy()
    df['channel'] = df.get('channel', df.get('channel_id'))
    if df['channel'].isnull().any():
        raise ValueError("Serve 'channel' o 'channel_id' non null.")
    df['channel'] = df['channel'].astype(str)
    df['predict_time'] = df.get('predict_time', df.get('detect_time', 0.0))

    cat = datasets[['channel','negatives']].copy()
    cat['channel'] = cat['channel'].astype(str)

    x = (df.merge(cat, on='channel', how='left')
           .assign(negatives=lambda d: d['negatives'].fillna(0.0).astype(float)))

    tp, fp, fn = x['true_positives'].sum(), x['false_positives'].sum(), x['false_negatives'].sum()
    tot_neg = x['negatives'].sum()
    tnr = (x['tnr'].mul(x['negatives']).sum() / tot_neg) if tot_neg > 0 else 0.0

    precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall    = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    prec_corr = precision * tnr

    n = len(x)
    return {
        'f1'            : float((2*prec_corr*recall/(prec_corr + recall)) if (prec_corr + recall) > 0 else 0.0),
        'f0.5'          : float(((1+0.5**2)*prec_corr*recall/(0.5**2*prec_corr + recall)) if (0.5**2*prec_corr + recall) > 0 else 0.0),
        'precision'     : float(prec_corr),
        'recall'        : float(recall),
        'tnr'           : float(tnr),
        'train_time'    : float(x['train_time'].mean()) if n else 0.0,
        'predict_time'  : float(x['predict_time'].mean()) if n else 0.0,
        'total_negatives': int(tot_neg),
        'channels'      : int(x['channel'].nunique()),
    }


DATASET_ALIAS = {"ops": "OPS_SAT", "nasa": "NASA", "esa": "ESA"}

def parse_args():
    p = argparse.ArgumentParser(description="Aggrega risultati esperimenti AD.")
    p.add_argument("--base-dir", default="experiments", help="Cartella radice degli esperimenti.")
    p.add_argument("--output-dir", default="all_results", help="Cartella dove salvare i CSV.")
    p.add_argument(
        "--datasets", default="ops,nasa,esa",
        help="Dataset da includere (comma-sep): es. 'ops,nasa' oppure 'esa'."
    )
    p.add_argument(
        "--include", default="",
        help="Regex per includere solo alcune cartelle esperimento (match su nome cartella)."
    )
    p.add_argument(
        "--exclude", default="",
        help="Regex per escludere alcune cartelle esperimento (match su nome cartella)."
    )
    p.add_argument("--print_tables", default=True, action="store_true", help="Stampa tabelle in console (tabulate).")
    return p.parse_args()


def render_and_export(out_df: pd.DataFrame,
                      export_dir: str,
                      datasets_filter: list[str],
                      print_tables: bool = False) -> None:

    print(export_dir)
    if os.path.exists(export_dir):
        raise ValueError(f"Directory {export_dir} already exists. Remove it first if you want to recompute the results")
    os.makedirs(export_dir)

    # Per dataset
    for ds in datasets_filter:
        ds_name = DATASET_ALIAS.get(ds, ds.upper())
        view = out_df[out_df["dataset"] == ds_name].drop(columns=["channels", "dataset", "total_negatives"], errors="ignore")
        if view.empty:
            continue

        view_sorted = view.sort_values("f1", ascending=False)
        csv_path = os.path.join(export_dir, f"{ds}_results.csv")
        view_sorted.to_csv(csv_path, index=False)

        if print_tables:
            print(f"\n=== DATASET: {ds_name} ===")
            print(tabulate(view_sorted, headers="keys", tablefmt="fancy_grid", floatfmt=".4f"))
            print(f"Salvato in: {csv_path}")

    # Summary globale
    summary_path = os.path.join(export_dir, "summary_all.csv")
    out_df.sort_values(["dataset", "f1"], ascending=[True, False]).to_csv(summary_path, index=False)
    if print_tables:
        print(f"\nRiassunto globale salvato in: {summary_path}")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    channels_catalog = load_datasets(args.output_dir)  # ['dataset','channel','preprocessed_length','negatives']

    datasets_filter = [d.strip().lower() for d in args.datasets.split(",") if d.strip()]

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

        scores = compute_experiment_scores(
            datasets=channels_catalog[["channel","negatives"]],
            results_df=res_df
        )
        if not scores: 
            continue

        rows.append({
            "dataset": dataset_name,
            "model": folder,    # oppure l’estrazione “pulita” se vuoi
            **scores
        })

    out_df = pd.DataFrame(rows)
    if out_df.empty:
        print("Nessun results.csv valido trovato.")
        return

    export_dir = os.path.join(args.output_dir, os.path.basename(os.path.normpath(args.base_dir)))
    render_and_export(
      out_df=out_df,
      export_dir=export_dir,
      datasets_filter=datasets_filter,
      print_tables=args.print_tables
    )

if __name__ == "__main__":
    main()
            