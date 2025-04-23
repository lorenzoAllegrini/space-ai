import os
import subprocess
import tempfile
import pandas as pd
import numpy as np
import shutil
import sys
from .detector import AnomalyDetector

class DPMMWrapperDetector(AnomalyDetector):
    def __init__(self, mode="likelihood", model_type="Full", K=100, num_iterations=100, lr=0.8, python_executable=None):
        super().__init__()
        self.mode = mode
        self.model_type = model_type
        self.K = K
        self.num_iterations = num_iterations
        self.lr = lr
        self.python_executable = python_executable or shutil.which("python")
        self.X_train = None

    def __call__(self, input: np.ndarray, y_true: np.ndarray, **kwargs) -> np.ndarray:
        return self.detect_anomalies(input, y_true, **kwargs)

    def fit(self, X):
        self.X_train = X
    
    def predict(self, X):
        # Verifica ambiente Python compatibile con DPMM
        active_env = os.environ.get("CONDA_DEFAULT_ENV", "(non rilevato)")
        print(f"Ambiente attivo: {active_env}")
        print(f"Python interpreter in uso: {self.python_executable}\n")

        if "dpmm" not in self.python_executable.lower():
            raise RuntimeError(
                f"Python interpreter non compatibile: {self.python_executable}\n"
                "Devi usare l'interprete dell'ambiente `dpmm_env` per eseguire correttamente il wrapper.\n"
                "Passalo nel costruttore con: DPMMWrapperDetector(..., python_executable='path/dpmm_env/python')\n"
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            input_test = os.path.join(tmpdir, "test.csv")
            input_train = os.path.join(tmpdir, "train.csv")
            output_pred = os.path.join(tmpdir, "output.csv")

            pd.DataFrame(X).to_csv(input_test, index=False)
            pd.DataFrame(self.X_train).to_csv(input_train, index=False)

            this_dir = os.path.dirname(__file__)
            run_dpmm_path = os.path.abspath(
                os.path.join(this_dir, "../../../spaceai/external/dpmm_wrapper/run_dpmm.py")
            )

            try:
                result = subprocess.run([
                    self.python_executable,
                    run_dpmm_path,
                    input_test,
                    input_train,
                    output_pred,
                    self.mode,
                    self.model_type,
                    str(self.K),
                    str(self.num_iterations),
                    str(self.lr)
                ], check=True, capture_output=True, text=True)
            except subprocess.CalledProcessError as e:
                print("\n🚨 ERRORE NEL SUBPROCESS DPMM:")
                print("🔹 STDOUT:")
                print(e.stdout)
                print("🔹 STDERR:")
                print(e.stderr)
                raise

            pred_df = pd.read_csv(output_pred)
            return pred_df["prediction"].values



    def detect_anomalies(self, y_pred, y_true, **kwargs):
        X_train_nominal = kwargs.get("X_train_nominal")

        # Verifica ambiente Python compatibile con DPMM
        active_env = os.environ.get("CONDA_DEFAULT_ENV", "(non rilevato)")
        print(f"Ambiente attivo: {active_env}")
        print(f"Python interpreter in uso: {self.python_executable}\n")

        if "dpmm" not in self.python_executable.lower():
            raise RuntimeError(
                f"Python interpreter non compatibile: {self.python_executable}\n"
                "Devi usare l'interprete dell'ambiente `dpmm_env` per eseguire correttamente il wrapper.\n"
                "Passalo nel costruttore con: DPMMWrapperDetector(..., python_executable='path/dpmm_env/python')\n"
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            input_test = os.path.join(tmpdir, "test.csv")
            input_train = os.path.join(tmpdir, "train.csv")
            output_pred = os.path.join(tmpdir, "output.csv")

            pd.DataFrame(y_pred).to_csv(input_test, index=False)
            pd.DataFrame(X_train_nominal).to_csv(input_train, index=False)

            this_dir = os.path.dirname(__file__)
            run_dpmm_path = os.path.abspath(
                os.path.join(this_dir, "../../../spaceai/external/dpmm_wrapper/run_dpmm.py")
            )

            try:
                result = subprocess.run([
                    self.python_executable,
                    run_dpmm_path,
                    input_test,
                    input_train,
                    output_pred,
                    self.mode,
                    self.model_type,
                    str(self.K),
                    str(self.num_iterations),
                    str(self.lr)
                ], check=True, capture_output=True, text=True)
            except subprocess.CalledProcessError as e:
                print("\n🚨 ERRORE NEL SUBPROCESS DPMM:")
                print("🔹 STDOUT:")
                print(e.stdout)
                print("🔹 STDERR:")
                print(e.stderr)
                raise

            pred_df = pd.read_csv(output_pred)
            return pred_df["prediction"].values