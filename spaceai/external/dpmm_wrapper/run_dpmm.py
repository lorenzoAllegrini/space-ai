import argparse
import pandas as pd
import pickle
import json
import numpy as np
import sys
from dpmm_core import get_trained_dpmm_model
import torch as th

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Esecuzione wrapper DPMM")
    parser.add_argument("--mode", choices=["fit", "predict"], required=True)


    # comuni a entrambi
    parser.add_argument("--model", required=True, help="Path al file del modello .pkl")
    parser.add_argument("--info", required=True, help="Path file JSON con altre info necessare per la predizione")
    parser.add_argument("--prediction_type", choices=["likelihood_threshold", "cluster_labels"], required=False)

    # fit
    parser.add_argument("--train", help="CSV di training")
    parser.add_argument("--model_type", default="full")
    parser.add_argument("--K", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--alpha_DP", type=float, default=1.0)
    parser.add_argument("--var_prior", type=float, default=1.0)
    parser.add_argument("--var_prior_strength", type=float, default=1.0)
    parser.add_argument("--quantile", type=float, default=0.05)

    # predict
    parser.add_argument("--test", help="CSV di test")
    parser.add_argument("--output", help="File CSV di output")

    args = parser.parse_args()
    
    if args.mode == "fit":
        if args.train is None:
            raise ValueError("--train richiesto in modalità fit")

        data_train = pd.read_csv(args.train).values
        X_train, y_train = data_train[:, 1:], data_train[:, 0]
        y_train = y_train.astype(np.int32)
        if args.prediction_type == "likelihood_threshold":
            # filter out anomalies
            X_train = X_train[y_train == 0]

        X_train= th.tensor(X_train, dtype=th.float32)
        y_train = th.tensor(y_train, dtype=th.int32)
        # Allenamento modello
        dpmm_model = get_trained_dpmm_model(
            X_train,
            model_type=args.model_type,
            K=args.K,
            num_iterations=args.iterations,
            lr=args.lr,
            alphaDP=args.alpha_DP,
            var_prior=args.var_prior,
            var_prior_strength=args.var_prior_strength
        )

        # Salva modello
        with open(args.model, "wb") as f:
            pickle.dump(dpmm_model, f)

        with th.no_grad():
            pi_tr, _, loglike_tr = dpmm_model(X_train)

        if args.prediction_type == "likelihood_threshold":
            # compute the treshold on trainign data and save it
            q = args.quantile
            likelihood_threshold = th.quantile(loglike_tr, q)
            with open(args.info, "w") as f:
                json.dump({"likelihood_threshold": likelihood_threshold.item()}, f)

        elif args.prediction_type == "cluster_labels":
            # compute the cluster labels on trainign data and save it
            clust_assignment = pi_tr.argmax(dim=1)
            n_anomalies_for_clust = th.zeros(args.K)
            n_total_for_clust = th.zeros(args.K)
            n_anomalies_for_clust.scatter_add_(dim=0, index=clust_assignment, src=y_train.float())
            n_total_for_clust.scatter_add_(dim=0, index=clust_assignment, src=th.ones_like(y_train).float())
            perc_anomalies_for_clust = n_anomalies_for_clust / (n_total_for_clust+1e-6)
            is_anomaly_clust = th.logical_or(perc_anomalies_for_clust > 0.5, th.isclose(n_total_for_clust, th.tensor(0.0)))
            with open(args.info, "w") as f:
                json.dump({"anomaly_cluster_labels": is_anomaly_clust.tolist()}, f)

    elif args.mode == "predict":
        if args.test is None or args.output is None:
            raise ValueError("--test e --output richiesti in modalità predict")

        X_test = pd.read_csv(args.test).values
        X_test = th.tensor(X_test, dtype=th.float32)
        # Carica modello
        with open(args.model, "rb") as f:
            dpmm_model = pickle.load(f)

        # faccio inferenza
        with th.no_grad():
            pi_test, _, loglike_test = dpmm_model(X_test)

        # Carica active clusters
        with open(args.info, "r") as f:
            tr_info = json.load(f)

        if 'likelihood_threshold' in tr_info:
            # classifica in base alla treshold
             y_pred = loglike_test < tr_info['likelihood_threshold']
        else:
            # classifica in base alle cluster labels
            assert 'anomaly_cluster_labels' in tr_info
            is_anomaly_clust = th.tensor(tr_info['anomaly_cluster_labels'])
            test_clust_assignment = pi_test.argmax(dim=1)
            y_pred = is_anomaly_clust[test_clust_assignment]

        pd.DataFrame(y_pred, columns=["prediction"]).to_csv(args.output, index=False)

    else:
        raise ValueError("Modalità non supportata")