import os
import pandas as pd
from tabulate import tabulate

base_dir = "experiments"
results = []


for root, dirs, files in os.walk(base_dir):
    if "results.csv" in files:
        file_path = os.path.join(root, "results.csv")
        results_df = pd.read_csv(file_path)

        if {'true_positives', 'false_positives', 'false_negatives', 'train_time'}.issubset(results_df.columns):
            tp = results_df['true_positives'].sum()
            fp = results_df['false_positives'].sum()
            fn = results_df['false_negatives'].sum()
            avg_time = results_df['train_time'].mean()

            precision = tp / (tp + fp) if tp + fp > 0 else 0.0
            recall = tp / (tp + fn) if tp + fn > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0.0

            # Estrai il nome della directory relativa (es. 'nasa_model1')
            model_name = os.path.relpath(root, base_dir)

            # Il tipo di dataset è la prima parte del nome
            dataset_type = model_name.split('_')[0]

            results.append({
                'dataset': dataset_type,
                'model': model_name,
                'avg_time': avg_time,
                'precision': precision,
                'recall': recall,
                'f1': f1
            })

# Crea il DataFrame completo
df = pd.DataFrame(results)

selected_dataset = "ops_sat"  
df_filtered = df[df['dataset'] == selected_dataset]

# Ordina per F1 decrescente
df_sorted = df_filtered.sort_values(by="f1", ascending=False)

# Droppa la colonna 'dataset' e stampa la classifica in bel formato
print(tabulate(df_sorted.drop(columns=["dataset"]), headers='keys', tablefmt='fancy_grid', floatfmt=".4f"))