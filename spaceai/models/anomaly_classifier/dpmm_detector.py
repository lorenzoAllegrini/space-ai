import argparse
from typing import Optional

import numpy as np
import torch as th
import torch.optim as optim
from torch_dpmm.models import (  # type: ignore
    DiagonalGaussianDPMM,
    FullGaussianDPMM,
    IsotropicGaussianDPMM,
    UnitGaussianDPMM,
)
# TODO: rename single to isotropic
from tqdm import tqdm  # type: ignore

from .anomaly_classifier import AnomalyClassifier


def get_dpmm_argparser():
    parser = argparse.ArgumentParser()
    # TODO: uniform with command line args
    # parser.add_argument("--prediction_type", choices=["likelihood_threshold", "cluster_labels"])
    # parser.add_argument("--model_type", choices=["full", "diagonal", "single", "unit"])
    parser.add_argument("--K", type=int, default=100)
    parser.add_argument("--num-iterations", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--alphaDP", type=float, default=1.0)
    parser.add_argument("--var_prior", type=float, default=1.0)
    parser.add_argument("--var_prior_strength", type=float, default=1.0)
    parser.add_argument("--quantile", type=float, default=0.05)
    return parser


class DPMMDetector(AnomalyClassifier):
    def __init__(
        self,
        mode: str = "likelihood_threshold",  # "likelihood_threshold" | "cluster_labels"
        model_type: str = "full",  # "full" | "diagonal" | "single" | "unit"
        K: int = 100,
        num_iterations: int = 100,
        lr: float = 0.8,
        alphaDP: float = 1.0,
        var_prior: float = 1.0,
        var_prior_strength: float = 1.0,
        mu_prior_strength: float = 0.001,
        quantile: float = 0.05,
        device: Optional[str] = None,  # "cpu" / "cuda" o None -> auto
    ):
        assert mode in ["likelihood_threshold", "cluster_labels"]
        super().__init__()
        self.mode = mode
        self.model_type = model_type
        self.K = int(K)
        self.num_iterations = int(num_iterations)
        self.lr = float(lr)
        self.alphaDP = float(alphaDP)
        self.var_prior = float(var_prior)
        self.var_prior_strength = float(var_prior_strength)
        self.mu_prior_strength = float(mu_prior_strength)
        self.quantile = float(quantile)

        self.dpmm_model = None
        self.likelihood_threshold: Optional[th.Tensor] = None
        self.anomaly_cluster_labels: Optional[th.Tensor] = None

        self.device = (
            th.device(device)
            if device
            else th.device("cuda" if th.cuda.is_available() else "cpu")
        )

    def __call__(
        self, input: np.ndarray, y_true: Optional[np.ndarray] = None, **kwargs
    ) -> np.ndarray:
        return self.predict(input)

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        """Fit del DPMM; in modalità 'cluster_labels' richiede y per derivare etichette di cluster."""
        # Validazioni
        if self.mode == "cluster_labels" and y is None:
            raise ValueError(
                "In 'cluster_labels' mode, 'y' (0/1) is required to label clusters."
            )

        # Filtra anomalie solo per stima della soglia
        if self.mode == "likelihood_threshold" and (y is not None):
            X = X[y == 0]

        # Tensori
        X_t = th.as_tensor(X, dtype=th.float32, device=self.device)
        y_t = (
            None if y is None else th.as_tensor(y, dtype=th.float32, device=self.device)
        )

        # Modello
        D = X_t.shape[1]
        self.dpmm_model = self._init_model(D).to(self.device)
        if self.dpmm_model is None:
            raise RuntimeError("Failed to initialize DPMM model")
        
        self.dpmm_model.train()
        self.dpmm_model.init_var_params(X_t)

        optimizer = optim.SGD(self.dpmm_model.parameters(), lr=self.lr)

        for _ in tqdm(
            range(self.num_iterations),
            desc=f"Fitting {self.model_type} DPMM",
            unit="epoch",
        ):
            optimizer.zero_grad()
            # Assumo che il forward ritorni (pi, elbo_loss, extra)
            _, elbo_loss, _ = self.dpmm_model(X_t)
            elbo_loss.backward()
            optimizer.step()

        self.dpmm_model.eval()
        with th.no_grad():
            pi_tr, _, loglike_tr = self.dpmm_model(X_t)

        if self.mode == "likelihood_threshold":
            # salva su self
            self.likelihood_threshold = th.quantile(loglike_tr, self.quantile)

        else:  # cluster_labels
            # assegnazione cluster hard
            clust_assignment = pi_tr.argmax(dim=1)

            # conteggi per cluster (più compatto di scatter_add)
            tot = th.bincount(clust_assignment, minlength=self.K).to(self.device)
            anom = th.bincount(
                clust_assignment, weights=y_t, minlength=self.K
            )  # y_t è float(0/1)

            perc = anom / (tot + 1e-6)
            self.anomaly_cluster_labels = (perc > 0.5) | (tot == 0)

    def predict(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """Predice etichetta anomalia per ciascun punto (bool) usando il modello fit-tato."""
        if self.dpmm_model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        X_t = th.as_tensor(X, dtype=th.float32, device=self.device)
        self.dpmm_model.eval()
        with th.no_grad():
            pi_te, _, loglike_te = self.dpmm_model(X_t)

        if self.mode == "likelihood_threshold":
            if self.likelihood_threshold is None:
                raise RuntimeError(
                    "likelihood_threshold not set. Fit with 'likelihood_threshold' first."
                )
            y_pred = loglike_te < self.likelihood_threshold

        else:  # cluster_labels
            if self.anomaly_cluster_labels is None:
                raise RuntimeError(
                    "Cluster labels not set. Fit with 'cluster_labels' first."
                )
            cl = pi_te.argmax(dim=1)
            is_anom = self.anomaly_cluster_labels.to(self.device)
            y_pred = is_anom[cl]

        # Ritorna ndarray booleano
        return y_pred.detach().to("cpu").numpy().astype(bool)

    # ---- helpers ----
    def _init_model(self, D: int):
        if self.model_type == "full":
            return FullGaussianDPMM(
                self.K,
                D,
                self.alphaDP,
                mu_prior=0,
                mu_prior_strength=self.mu_prior_strength,
                var_prior=self.var_prior,
                var_prior_strength=self.var_prior_strength,
            )
        if self.model_type == "diagonal":
            return DiagonalGaussianDPMM(
                self.K,
                D,
                self.alphaDP,
                mu_prior=0,
                mu_prior_strength=self.mu_prior_strength,
                var_prior=self.var_prior,
                var_prior_strength=self.var_prior_strength,
            )
        if self.model_type == "single":
            return IsotropicGaussianDPMM(
                self.K,
                D,
                self.alphaDP,
                mu_prior=0,
                mu_prior_strength=self.mu_prior_strength,
                var_prior=self.var_prior,
                var_prior_strength=self.var_prior_strength,
            )
        if self.model_type == "unit":
            return UnitGaussianDPMM(
                self.K,
                D,
                self.alphaDP,
                mu_prior=0,
                mu_prior_strength=self.mu_prior_strength,
            )
        raise ValueError(f"Invalid model_type: {self.model_type}")

    def detect_anomalies(self, X, y_true=None, **kwargs):
        """Detect anomalies in the input data."""
        return self.predict(X)
