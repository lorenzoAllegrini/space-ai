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
    """
    Ritorna:
      {"dataset", "channel", "preprocessed_length", "negatives"}
    - preprocessed_length = len(data)//WINDOW_LENGTH
    - negatives = finestre completamente nominali
    """
    data_len = len(channel.data)
    n_w = data_len // WINDOW_LENGTH
    ch_id = channel.channel_id  # evita getattr ripetuti

    if n_w <= 0:
        return {"dataset": dataset, "channel": ch_id, "preprocessed_length": 0, "negatives": 0}

    # Prendi anomalie (array Nx2) e fai fast-path se vuoto
    anomalies = getattr(channel, "anomalies", []) or []
    if not anomalies:
        return {"dataset": dataset, "channel": ch_id, "preprocessed_length": int(n_w), "negatives": int(n_w)}

    # ---- VETTORIZZAZIONE ----
    # Converti in np.array e filtra righe malformate
    a = np.asarray(anomalies, dtype=np.int64)
    if a.ndim != 2 or a.shape[1] != 2:
        # fallback prudente (nessuna anomalia valida)
        return {"dataset": dataset, "channel": ch_id, "preprocessed_length": int(n_w), "negatives": int(n_w)}

    # clamp agli indici validi, rimuovi righe invertite (e < s) e NaN (non numeriche)
    last_idx = n_w * WINDOW_LENGTH - 1
    # maschera validi: numerici e non NaN
    valid = np.isfinite(a).all(axis=1)
    a = a[valid]
    if a.size == 0:
        return {"dataset": dataset, "channel": ch_id, "preprocessed_length": int(n_w), "negatives": int(n_w)}

    # clamp
    a[:, 0] = np.clip(a[:, 0], 0, last_idx)
    a[:, 1] = np.clip(a[:, 1], 0, last_idx)
    # mantieni solo intervalli non vuoti
    a = a[a[:, 1] >= a[:, 0]]
    if a.size == 0:
        return {"dataset": dataset, "channel": ch_id, "preprocessed_length": int(n_w), "negatives": int(n_w)}

    # finestre di inizio/fine (vettoriale)
    ws = np.floor_divide(a[:, 0], WINDOW_LENGTH)
    we = np.floor_divide(a[:, 1], WINDOW_LENGTH)

    # ---- DIFFERENCE ARRAY con bincount (no loop) ----
    # diff[ws] += 1 ; diff[we+1] -= 1
    # costruiamo separatamente starts e ends e poi diff = starts - ends
    # minlength = n_w + 1 perché usiamo indice we+1
    starts = np.bincount(ws, minlength=n_w + 1)
    ends   = np.bincount(we + 1, minlength=n_w + 1)
    diff   = (starts - ends).astype(np.int32)  # lunghezza n_w+1

    covered = np.cumsum(diff[:-1]) > 0  # shape n_w
    negatives = int((~covered).sum())

    return {
        "dataset": dataset,
        "channel": ch_id,
        "preprocessed_length": int(n_w),
        "negatives": negatives,
    }


def load_datasets():
    rows = []
    for dataset, channel_ids in get_channels().items():
        for channel_id in tqdm(channel_ids, desc=f"{dataset}", ncols=100, leave=True):
            channel = load_dataset_channel(dataset, channel_id)
            rows.append(compute_channel_stats(dataset, channel))

    return pd.DataFrame(rows)

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

    beta2 = 0.5**2
    f0_5 = ((1+beta2)*prec_corr*recall/(beta2*prec_corr + recall)) if (beta2*prec_corr + recall) > 0 else 0.0
    f1   = (2*prec_corr*recall/(prec_corr + recall)) if (prec_corr + recall) > 0 else 0.0

    n = len(x)
    return {
        'f1'            : float(f1),
        'f0.5'          : float(f0_5),
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

def main():

    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # 1) Catalogo canali (una sola volta)
    channels_catalog = load_datasets()  # ['dataset','channel','preprocessed_length','negatives']
    # NB: per compute_experiment_scores bastano 'channel' e 'negatives'
    # Se i channel sono globalmente unici non serve filtrare per dataset

    # 2) Prepara filtri
    datasets_filter = [d.strip().lower() for d in args.datasets.split(",") if d.strip()]
    include_re = re.compile(args.include) if args.include else None
    exclude_re = re.compile(args.exclude) if args.exclude else None

    rows = []
    for root, _, files in os.walk(args.base_dir):
        if "results.csv" not in files:
            continue

        # nome cartella esperimento (relativo alla base)
        rel = os.path.relpath(root, args.base_dir)
        folder = os.path.basename(rel)

        # filtri include/exclude
        if include_re and not include_re.search(folder):
            continue
        if exclude_re and exclude_re.search(folder):
            continue

        # deduci dataset dal prefisso
        prefix = folder.split("_")[0].lower()
        if prefix not in datasets_filter:
            continue
        dataset_name = DATASET_ALIAS.get(prefix, prefix.upper())

        # leggi results.csv
        csv_path = os.path.join(root, "results.csv")
        try:
            res_df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"[WARN] Impossibile leggere {csv_path}: {e}")
            continue

        # calcola metriche esperimento
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

    # 3) Salvataggi per dataset selezionati
    for ds in datasets_filter:
        ds_name = DATASET_ALIAS.get(ds, ds.upper())
        view = out_df[out_df["dataset"] == ds_name].drop(columns=["channels", "dataset"])
        if view.empty:
            continue

        view_sorted = view.sort_values("f1", ascending=False)
        csv_path = os.path.join(args.output_dir, f"{ds}_results.csv")
        view_sorted.to_csv(csv_path, index=False)

        if args.print_tables:
            print(f"\n=== DATASET: {ds_name} ===")
            print(tabulate(view_sorted, headers="keys", tablefmt="fancy_grid", floatfmt=".4f"))
            print(f"→ Salvato in: {csv_path}")

    # 4) CSV unico riassuntivo
    summary_path = os.path.join(args.output_dir, "summary_all.csv")
    out_df.sort_values(["dataset", "f1"], ascending=[True, False]).to_csv(summary_path, index=False)
    if args.print_tables:
        print(f"\n→ Riassunto globale salvato in: {summary_path}")

if __name__ == "__main__":
    main()
            