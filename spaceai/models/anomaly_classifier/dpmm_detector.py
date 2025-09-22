import os
import subprocess
import tempfile
import pandas as pd
import numpy as np
import shutil
import sys
from .anomaly_classifier import AnomalyClassifier

class DPMMWrapperDetector(AnomalyClassifier):
    def __init__(self, mode="likelihood", model_type="Full", K=100, num_iterations=100, lr=0.8, python_executable=None):
        super().__init__()
        self.mode = mode
        self.model_type = model_type
        self.K = K
        self.num_iterations = num_iterations
        self.lr = lr
        self.python_executable = python_executable or shutil.which("python")
        self.X_train = None
        # Crea una cartella temporanea per train/test/model
        self._tempdir = tempfile.mkdtemp()
        self._train_path = os.path.join(self._tempdir, "train.csv")
        self._test_path = os.path.join(self._tempdir, "test.csv")
        self._output_path = os.path.join(self._tempdir, "output.csv")
        self._model_path = os.path.join(self._tempdir, "model.pkl")
        self._clusters_path = os.path.join(self._tempdir, "clusters.json")

    def __call__(self, input: np.ndarray, y_true: np.ndarray, **kwargs) -> np.ndarray:
        return self.predict(input)

    def fit(self, X, y=None):
        """Esegue il fit separato tramite subprocess e salva il modello e cluster."""
        pd.DataFrame(X).to_csv(self._train_path, index=False)

        this_dir = os.path.dirname(__file__)
        run_dpmm_path = os.path.abspath(
            os.path.join(this_dir, "../../../spaceai/external/dpmm_wrapper/run_dpmm.py")
        )

        print(f"\nEseguo FIT DPMM in ambiente: {self.python_executable}")

        try:
            subprocess.run([
                self.python_executable,
                run_dpmm_path,
                "--mode", "fit",
                "--train", self._train_path,
                "--model", self._model_path,
                "--clusters", self._clusters_path,
                "--model_type", self.model_type,
                "--K", str(self.K),
                "--iterations", str(self.num_iterations),
                "--lr", str(self.lr)
            ], check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            print("\n🚨 ERRORE DURANTE FIT:")
            print("🔹 STDOUT:\n", e.stdout)
            print("🔹 STDERR:\n", e.stderr)
            raise

    def predict(self, X, y=None):
        """Esegue il predict separato tramite subprocess usando il modello salvato."""
        pd.DataFrame(X).to_csv(self._test_path, index=False)

        this_dir = os.path.dirname(__file__)
        run_dpmm_path = os.path.abspath(
            os.path.join(this_dir, "../../../spaceai/external/dpmm_wrapper/run_dpmm.py")
        )

        print(f"\nEseguo PREDICT DPMM in ambiente: {self.python_executable}")

        try:
            subprocess.run([
                self.python_executable,
                run_dpmm_path,
                "--mode", "predict",
                "--test", self._test_path,
                "--model", self._model_path,
                "--clusters", self._clusters_path,
                "--output", self._output_path
            ], check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            print("\n🚨 ERRORE DURANTE PREDICT:")
            print("🔹 STDOUT:\n", e.stdout)
            print("🔹 STDERR:\n", e.stderr)
            raise

        pred_df = pd.read_csv(self._output_path)
        return pred_df["prediction"].values

    def detect_anomalies(self, X, y_true=None, **kwargs):
        """Compatibilità: detect_anomalies richiama semplicemente predict."""
        return self.predict(X)

