import argparse
import pandas as pd
import pickle
import json
import numpy as np
import sys
from spaceai.external.dpmm_wrapper.dpmm_core import (
    run_dpmm_likelihood,
    run_dpmm_new_cluster,
    get_trained_dpmm_model
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Esecuzione wrapper DPMM")
    parser.add_argument("--mode", choices=["fit", "predict"], required=True)

    # comuni a entrambi
    parser.add_argument("--model", required=True, help="Path al file del modello .pkl")
    parser.add_argument("--clusters", required=True, help="Path file JSON con active clusters")

    # fit
    parser.add_argument("--train", help="CSV di training")
    parser.add_argument("--model_type", default="Full")
    parser.add_argument("--K", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.8)

    # predict
    parser.add_argument("--test", help="CSV di test")
    parser.add_argument("--output", help="File CSV di output")

    args = parser.parse_args()

    if args.mode == "fit":
        if args.train is None:
            raise ValueError("--train richiesto in modalità fit")

        X_train = pd.read_csv(args.train).values

        # Allenamento modello
        dpmm_model = get_trained_dpmm_model(
            X_train,
            model_type=args.model_type,
            K=args.K,
            num_iterations=args.iterations,
            lr=args.lr
        )

        # Salva modello
        with open(args.model, "wb") as f:
            pickle.dump(dpmm_model, f)

        # Salva cluster attivi
        active_clusters = dpmm_model.get_num_active_components()
        with open(args.clusters, "w") as f:
            json.dump({"active_clusters": active_clusters}, f)

    elif args.mode == "predict":
        if args.test is None or args.output is None:
            raise ValueError("--test e --output richiesti in modalità predict")

        X_test = pd.read_csv(args.test).values

        # Carica modello
        with open(args.model, "rb") as f:
            dpmm_model = pickle.load(f)

        # Carica active clusters
        with open(args.clusters, "r") as f:
            cluster_data = json.load(f)
        active_clusters = cluster_data.get("active_clusters", None)

        # Inference (determinata dal tipo di modello)
        if hasattr(dpmm_model, "predict_new_cluster"):
            y_pred = run_dpmm_new_cluster(model=dpmm_model, X_test=X_test, active_clusters=active_clusters)
        else:
            y_pred = run_dpmm_likelihood(model=dpmm_model, X_test=X_test)

        pd.DataFrame(y_pred, columns=["prediction"]).to_csv(args.output, index=False)

    else:
        raise ValueError("Modalità non supportata")