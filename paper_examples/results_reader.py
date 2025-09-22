import os
import pandas as pd
from tabulate import tabulate
from spaceai.data.ops_sat import OPSSAT
from spaceai.data.nasa import NASA
from spaceai.data.esa import ESA, ESAMission, ESAMissions
import numpy as np
base_dir = "experiments"
results = []


for root, dirs, files in os.walk(base_dir):
    r = root.split("/")[1] if len(root.split("/"))>1 else f"root: {root}"
    if "results.csv" in files:
        print(f"root: {root} files: {files}, results {True if 'results.csv' in files else False}")
        file_path = os.path.join(root, "results.csv")
        results_df = pd.read_csv(file_path)
        
        if {'true_positives', 'false_positives', 'false_negatives', 'train_time', 'tnr'}.issubset(results_df.columns):
            tp = 0
            fp = 0
            fn = 0
            tn = 0
            total_negatives = 0
            train_time = 0
            predict_time = 0

            for _,row in results_df.iterrows():
                dataset = root.split("/")[1]
                if not dataset.startswith("nasa") and not dataset.startswith("ops"):
                    continue


                if dataset.startswith("ops"):
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
              
                elif dataset.startswith("nasa"):
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
                else:
                    test_channel = ESA(
                        root="datasets",
                        channel_id=row["channel_id"],
                        mode="anomaly",
                        overlapping=False,
                        seq_length=1,
                        train=False,
                        drop_last=False,
                        n_predictions=1,
                        mission=ESAMissions.MISSION_1.value
                    )
                
                length = len(test_channel.data) // 50
                #total_nominal_segments +=(length - np.sum([seg[1]//50 - seg[0]//50 for seg in test_channel.anomalies]))
                labels = np.zeros(length*50, dtype=int)
                for start, end in test_channel.anomalies:
                    start = max(0, start)
                    end = min(len(labels) - 1, end)
                    labels[start:end + 1] = 1

                negatives = 0
                for r in labels.reshape(-1, 50):
                    if np.sum(r) == 0:
                        negatives += 1

                tn += row['tnr'] * negatives
                tp += row['true_positives']
                fp += row['false_positives']
                fn += row['false_negatives']
                train_time += row['train_time'] 
                predict_time += row['predict_time'] if 'predict_time' in row else row['detect_time']
                total_negatives += negatives
     
            tnr = tn/total_negatives if total_negatives > 0 else 0
            train_time /= len(results_df) if len(results_df) > 0 else 0
            predict_time /= len(results_df) if len(results_df) > 0 else 0
            precision = tp / (tp + fp) if tp + fp > 0 else 0.0
            precision_corrected = precision * tnr
            recall = tp / (tp + fn) if tp + fn > 0 else 0.0
            f0_5 = (1 + 0.5**2) * (precision_corrected * recall) / (0.5**2 * precision_corrected + recall) if precision_corrected + recall > 0 else 0.0
            f1 = 2 * (precision_corrected * recall) / (precision_corrected + recall) if precision_corrected + recall > 0 else 0.0
            


            # Estrai il nome della directory relativa (es. 'nasa_model1')
            model_name = os.path.relpath(root, base_dir)

            # Il tipo di dataset è la prima parte del nome
            dataset_type = model_name.split('_')[0]

            results.append({
                'dataset': dataset_type,
                'model': model_name,
                'f1': f1,
                'f0.5': f0_5,
                'precision': precision_corrected,
                'recall': recall,
                'train_time': train_time,             
                'predict_time': predict_time, 
            })

# Crea il DataFrame completo
df = pd.DataFrame(results)

for selected_dataset in ["ops", "nasa", "esa"]:
    print(f"DATASET: {dataset}")
    df_filtered = df[df['dataset'] == selected_dataset]

    # Ordina per F1 decrescente
    df_sorted = df_filtered.sort_values(by="f1", ascending=False)

    # Droppa la colonna 'dataset' e stampa la classifica in bel formato
    print(tabulate(df_sorted.drop(columns=["dataset"]), headers='keys', tablefmt='fancy_grid', floatfmt=".4f"))


