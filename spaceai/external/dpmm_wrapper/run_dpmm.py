import sys
import pandas as pd
from spaceai.external.dpmm_wrapper.dpmm_core import run_dpmm_likelihood, run_dpmm_new_cluster

if __name__ == "__main__":
    input_path = sys.argv[1]      # test.csv
    train_path = sys.argv[2]      # train.csv
    output_path = sys.argv[3]     # output.csv
    mode = sys.argv[4]            # 'likelihood' o 'new_cluster'

    # parametri opzionali via riga di comando (con default)
    model_type = sys.argv[5] if len(sys.argv) > 5 else "Full"
    K = int(sys.argv[6]) if len(sys.argv) > 6 else 100
    num_iterations = int(sys.argv[7]) if len(sys.argv) > 7 else 100
    lr = float(sys.argv[8]) if len(sys.argv) > 8 else 0.8

    X_test = pd.read_csv(input_path).values
    X_train = pd.read_csv(train_path).values

    if mode == "likelihood":
        y_pred = run_dpmm_likelihood(X_train, X_test, model_type, K, num_iterations, lr)
    elif mode == "new_cluster":
        y_pred = run_dpmm_new_cluster(X_train, X_test, model_type, K, num_iterations, lr)
    else:
        raise ValueError("Mode must be 'likelihood' or 'new_cluster'")

    pd.DataFrame(y_pred, columns=["prediction"]).to_csv(output_path, index=False)