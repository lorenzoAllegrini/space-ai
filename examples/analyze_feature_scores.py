import pandas as pd
import os

# Percorso del file metadata.csv
csv_path = "/Users/lorenzoallegrini/Documents/space-ai/space-ai/experiments/pipeline_base_statistics_esa_dpmm_molookde_lr0.5_nc100_adp3.0_vp3.0_vps1.0_mps0.001_q0.001_al0.002_po92.0_p3.0_dsT_chF_ep0.3/metadata.csv"

def analyze_scores(path):
    if not os.path.exists(path):
        print(f"Errore: Il file {path} non esiste.")
        return

    # Carichiamo il CSV
    df = pd.read_csv(path)
    
    # Filtriamo solo le colonne che iniziano con 'fscore_'
    score_cols = [c for c in df.columns if c.startswith('fscore_')]
    
    if not score_cols:
        print("Nessuna colonna 'fscore_' trovata nel file.")
        return

    # Calcoliamo la media degli score per ogni feature (escludendo i canali falliti con score 0.0 o NaN)
    # Nota: Rimuoviamo il prefisso 'fscore_' per leggibilità
    feature_scores = df[score_cols].mean().sort_values(ascending=False)
    feature_scores.index = [i.replace('fscore_', '') for i in feature_scores.index]

    print("\n=== CLASSIFICA FEATURE (Score Medio su tutti i canali) ===")
    print(feature_scores.to_string())
    
    print("\n=== TOP 3 FEATURE PER CANALE (Esempio primi 10 canali) ===")
    for idx, row in df.head(10).iterrows():
        channel = row['channel_id']
        top_features = row[score_cols].sort_values(ascending=False).head(3)
        top_names = [n.replace('fscore_', '') for n in top_features.index]
        print(f"Channel {channel}: {', '.join(top_names)}")

if __name__ == "__main__":
    analyze_scores(csv_path)
