from __future__ import annotations
"""DPMM Detector module."""
import argparse
from typing import Optional

import numpy as np
import torch as th
from torch import optim
from torch_dpmm.models import (  # type: ignore # pylint: disable=import-error
    DiagonalGaussianDPMM,
    FullGaussianDPMM,
    IsotropicGaussianDPMM,
    UnitGaussianDPMM,
)
# TODO: rename single to isotropic
from tqdm import tqdm  # type: ignore

from .base import BaseClassifier


def get_dpmm_argparser():
    """Get DPMM argument parser."""
    parser = argparse.ArgumentParser()
    # TODO: uniform with command line args
    # parser.add_argument("--prediction_type", choices=["likelihood_threshold", "cluster_labels"])
    # parser.add_argument("--model_type", choices=["full", "diagonal", "single", "unit"])
    parser.add_argument("--n-clusters", type=int, default=100)
    parser.add_argument("--num-iterations", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--alpha-dp", type=float, default=1.0)
    parser.add_argument("--var-prior", type=float, default=1.0)
    parser.add_argument("--var-prior-strength", type=float, default=1.0)
    parser.add_argument("--mu-prior-strength", type=float, default=0.001)
    parser.add_argument("--quantile", type=float, default=0.05)
    return parser


from typing import Optional, Any

class DPMM(BaseClassifier):
    """DPMM model that returns continuous anomaly scores."""

    # pylint: disable=too-many-instance-attributes
    def __init__(
        self,
        mode: str = "likelihood_threshold",  # "likelihood_threshold" | "cluster_labels"
        model_type: str = "full",  # "full" | "diagonal" | "single" | "unit"
        n_clusters: int = 100,
        num_iterations: int = 500,
        lr: float = 0.8,
        alpha_dp: float = 0.05,
        var_prior: float = 3.0,
        var_prior_strength: float = 3.0,
        mu_prior_strength: float = 0.001,
        quantile: float = 0.0001,
        device: Optional[str] = None,  # "cpu" / "cuda" / "mps" o None -> auto
        return_likelihood: bool = False,
        early_stopping_patience: int = 5,
        early_stopping_tolerance: float = 1e-4,
        callback_handler: Optional[Any] = None,
        **kwargs
    ):
        # pylint: disable=too-many-arguments, too-many-positional-arguments
        assert mode in ["likelihood_threshold", "cluster_labels"]
        super().__init__(callback_handler=callback_handler, **kwargs)
        self.mode = mode
        self.model_type = model_type
        self.n_clusters = int(n_clusters)
        self.num_iterations = int(num_iterations)
        self.lr = float(lr)
        self.alpha_dp = float(alpha_dp)
        self.var_prior = float(var_prior)
        self.var_prior_strength = float(var_prior_strength)
        self.mu_prior_strength = float(mu_prior_strength)
        self.quantile = float(quantile)
        self.early_stopping_patience = int(early_stopping_patience)
        self.early_stopping_tolerance = float(early_stopping_tolerance)

        self.dpmm_model = None
        self.likelihood_threshold: Optional[th.Tensor] = None
        self.anomaly_cluster_labels: Optional[th.Tensor] = None
        self.return_likelihood = return_likelihood

        # Device selection: cuda > cpu (MPS excluded: missing aten::digamma)
        if device:
            self.device = th.device(device)
        elif th.cuda.is_available():
            self.device = th.device("cuda")
        else:
            self.device = th.device("cpu")
        
        # print(f"[DEBUG-DPMM] Selected device: {self.device}")

    def __call__(
        self, input_data: np.ndarray, y_true: Optional[np.ndarray] = None, **kwargs
    ) -> np.ndarray:
        return self.predict(input_data)

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:  # pylint: disable=invalid-name
        """Fit the DPMM model.

        In 'cluster_labels' mode, requires y to derive cluster labels.
        """
        # Validazioni
        if self.mode == "cluster_labels" and y is None:
            raise ValueError(
                "In 'cluster_labels' mode, 'y' (0/1) is required to label clusters."
            )

        # Filtra anomalie solo per stima della soglia
        if self.mode == "likelihood_threshold" and (y is not None):
            X = X[y == 0]

        # Tensori
        x_t = th.as_tensor(X, dtype=th.float32, device=self.device)
        y_t = (
            None if y is None else th.as_tensor(y, dtype=th.float32, device=self.device)
        )

        # Modello
        d_dim = x_t.shape[1]
        self.dpmm_model = self._init_model(d_dim).to(self.device)
        if self.dpmm_model is None:
            raise RuntimeError("Failed to initialize DPMM model")

        self.dpmm_model.train()
        self.dpmm_model.init_var_params(x_t)

        optimizer = optim.SGD(self.dpmm_model.parameters(), lr=self.lr)
        best_loss = float("inf")
        patience = 0

        pbar = tqdm(
            range(self.num_iterations),
            desc=f"Fitting {self.model_type} DPMM",
            unit="epoch")

        for _ in pbar:
            optimizer.zero_grad()
            # Assumo che il forward ritorni (pi, elbo_loss, extra)
            _, elbo_loss, _ = self.dpmm_model(x_t)
            elbo_loss.backward()
            optimizer.step()

            new_loss = elbo_loss.detach().cpu().item()
            pbar.set_description(f"Loss: {new_loss:.4f}")
            if best_loss-new_loss > self.early_stopping_tolerance:
                best_loss = new_loss
                patience = 0
            else:
                patience += 1

            if patience >= self.early_stopping_patience:
                break

        self.dpmm_model.eval()
        with th.no_grad():
            pi_tr, _, loglike_tr = self.dpmm_model(x_t)
            
            if self.mode == "likelihood_threshold":
                self.likelihood_threshold = th.quantile(loglike_tr, self.quantile).item()
            else:  # cluster_labels
                # assegnazione cluster hard
                clust_assignment = pi_tr.argmax(dim=1)

                # conteggi per cluster (più compatto di scatter_add)
                tot = th.bincount(clust_assignment, minlength=self.n_clusters).to(self.device)
                anom = th.bincount(
                    clust_assignment, weights=y_t, minlength=self.n_clusters
                )  # y_t è float(0/1)

                perc = anom / (tot + 1e-6)
                self.anomaly_cluster_labels = (perc > 0.5) | (tot == 0)
            
            # Explicit cleanup of training tensors
            del x_t
            del loglike_tr
            del pi_tr
            if 'clust_assignment' in locals(): del clust_assignment

        self.is_fitted_ = True

    def predict(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:  # pylint: disable=invalid-name, unused-argument
        """Predice etichetta anomalia per ciascun punto (bool) usando il modello fit-tato."""
        if self.dpmm_model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        x_t = th.as_tensor(X, dtype=th.float32, device=self.device)
        self.dpmm_model.eval()
        with th.no_grad():
            pi_te, _, loglike_te = self.dpmm_model(x_t)

        if self.mode == "likelihood_threshold":
            if self.likelihood_threshold is None:
                raise RuntimeError(
                    "likelihood_threshold not set. Fit with 'likelihood_threshold' first."
                )
            if self.return_likelihood:
                res = -loglike_te.detach().to("cpu").numpy()
                del x_t
                del loglike_te
                return res
            else:
                y_pred = loglike_te < self.likelihood_threshold
                res = y_pred.detach().to("cpu").numpy()
                del x_t
                del loglike_te
                return res

        else: 
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
    def _init_model(self, d_dim: int):
        if self.model_type == "full":
            return FullGaussianDPMM(
                self.n_clusters,
                d_dim,
                self.alpha_dp,
                mu_prior=0,
                mu_prior_strength=self.mu_prior_strength,
                var_prior=self.var_prior,
                var_prior_strength=self.var_prior_strength,
            )
        if self.model_type == "diagonal":
            return DiagonalGaussianDPMM(
                self.n_clusters,
                d_dim,
                self.alpha_dp,
                mu_prior=0,
                mu_prior_strength=self.mu_prior_strength,
                var_prior=self.var_prior,
                var_prior_strength=self.var_prior_strength,
            )
        if self.model_type == "single":
            return IsotropicGaussianDPMM(
                self.n_clusters,
                d_dim,
                self.alpha_dp,
                mu_prior=0,
                mu_prior_strength=self.mu_prior_strength,
                var_prior=self.var_prior,
                var_prior_strength=self.var_prior_strength,
            )
        if self.model_type == "unit":
            return UnitGaussianDPMM(
                self.n_clusters,
                d_dim,
                self.alpha_dp,
                mu_prior=0,
                mu_prior_strength=self.mu_prior_strength,
            )
        raise ValueError(f"Invalid model_type: {self.model_type}")

    def detect_anomalies(self, X, y_true=None, **kwargs):  # pylint: disable=invalid-name, unused-argument
        """Detect anomalies in the input data."""
        return self.predict(X)


class DPMMDetector:
    """Converts continuous DPMM scores to binary anomaly predictions.

    Wraps a fitted :class:`DPMM` instance and applies the appropriate
    thresholding logic depending on its mode.

    Args:
        dpmm (DPMM): A fitted DPMM model.
    """

    def __init__(self, dpmm: DPMM):
        self.dpmm = dpmm

    def detect(self, scores: np.ndarray) -> np.ndarray:
        """Apply threshold to continuous scores and return binary labels.

        Since ``DPMM.predict()`` now returns normalized scores in [0, 1]
        for both modes, we simply threshold at 0.5.

        Args:
            scores (np.ndarray): Continuous anomaly scores from ``DPMM.predict()``.

        Returns:
            np.ndarray: Binary anomaly labels (1 = anomaly, 0 = normal).
        """
        return (scores > 0.5).astype(int)
