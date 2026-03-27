from __future__ import annotations
"""DPMM Detector module."""
import argparse
import copy
from typing import Optional, Dict, Any, Union, List, TYPE_CHECKING

import numpy as np
import torch as th
from torch import optim
from torch_dpmm.models import (  # type: ignore # pylint: disable=import-error
    DiagonalGaussianDPMM,
    FullGaussianDPMM,
    IsotropicGaussianDPMM,
    UnitGaussianDPMM,
)
from tqdm import tqdm  # type: ignore

from .base import BaseClassifier

if TYPE_CHECKING:
    from ..anomaly_classifier.anomaly_classifier import PipelineMessage


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


class DPMM(BaseClassifier):
    """DPMM model that returns continuous anomaly scores."""

    # pylint: disable=too-many-instance-attributes
    def __init__(
        self,
        mode: str = "likelihood_threshold",  # "likelihood_threshold" | "cluster_labels"
        model_type: str = "full",  # "full" | "diagonal" | "single" | "unit"
        n_clusters: int = 100,
        num_iterations: int = 100,
        lr: float = 0.8,
        alpha_dp: float = 0.05,
        var_prior: float = 3.0,
        var_prior_strength: float = 3.0,
        mu_prior_strength: float = 0.001,
        quantile: float = 0.0001,
        device: Optional[str] = None,  # "cpu" / "cuda" o None -> auto
        return_likelihood: bool = False,
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
        
        # Early Stopping
        self.patience = kwargs.get("patience", None)
        self.min_delta = kwargs.get("min_delta", 0.0)
        self.restore_best = kwargs.get("restore_best", True)

        self.dpmm_model = None
        self.likelihood_threshold: Optional[th.Tensor] = None
        self.anomaly_cluster_labels: Optional[th.Tensor] = None
        self.return_likelihood = return_likelihood

        self.device = (
            th.device(device)
            if device
            else th.device("cuda" if th.cuda.is_available() else "cpu")
        )
        print(self.return_likelihood)

    def __call__(
        self, input_data: np.ndarray, y_true: Optional[np.ndarray] = None, **kwargs
    ) -> np.ndarray:
        return self.predict(input_data)

    def fit(self, *args, **kwargs) -> None:  # pylint: disable=invalid-name
        """Polymorphic fit: handles messages (native) or raw data."""
        if len(args) > 0 and hasattr(args[0], "data"):
             return self.fit_messages(*args)
        
        # Standard fit logic for raw arrays
        X = args[0]
        y = kwargs.get("y", args[1] if len(args) > 1 else None)
        results = kwargs.get("results", args[2] if len(args) > 2 else None)
        return self._fit(X, y=y, results=results)

    def fit_messages(self, *msgs: PipelineMessage) -> None:
        """Native message-based fit with calibration support."""
        if not msgs:
            return
            
        msg_train = msgs[0]
        X = msg_train.data
        y = msg_train.labels
        results = msg_train.results

        with self._callback_context("model_fit", results):
            if self.mode == "cluster_labels" and y is None:
                raise ValueError(
                    "In 'cluster_labels' mode, 'y' (0/1) is required to label clusters."
                )

            # Filter normal data for fitting if labels provided
            X_fit = X
            if self.mode == "likelihood_threshold" and (y is not None):
                X_fit = X[y == 0]

            x_t = th.as_tensor(X_fit, dtype=th.float32, device=self.device)
            y_t = (
                None if y is None else th.as_tensor(y, dtype=th.float32, device=self.device)
            )

            d_dim = x_t.shape[1]
            self.dpmm_model = self._init_model(d_dim).to(self.device)
            if self.dpmm_model is None:
                raise RuntimeError("Failed to initialize DPMM model")

            self.dpmm_model.train()
            self.dpmm_model.init_var_params(x_t)

            optimizer = optim.SGD(self.dpmm_model.parameters(), lr=self.lr)
            
            x_val_t = None
            if 'msgs' in locals() and len(msgs) > 1:
                x_val_t = th.as_tensor(msgs[1].data, dtype=th.float32, device=self.device)
            
            best_val_loss = float("inf")
            best_state = None
            epochs_since_improvement = 0

            with tqdm(total=self.num_iterations, desc=f"Fitting {self.model_type} DPMM") as pbar:
                for epoch in range(self.num_iterations):
                    self.dpmm_model.train()
                    optimizer.zero_grad()
                    _, elbo_loss, _ = self.dpmm_model(x_t)
                    elbo_loss.backward()
                    optimizer.step()
                    
                    train_loss = elbo_loss.item()
                    val_loss_val = train_loss
                    
                    if x_val_t is not None:
                        self.dpmm_model.eval()
                        with th.no_grad():
                            _, val_loss, _ = self.dpmm_model(x_val_t)
                        val_loss_val = val_loss.item()
                        
                        if val_loss_val < best_val_loss - self.min_delta:
                            best_val_loss = val_loss_val
                            epochs_since_improvement = 0
                            if self.restore_best:
                                best_state = copy.deepcopy(self.dpmm_model.state_dict())
                        else:
                            epochs_since_improvement += 1
                        
                        if self.patience is not None and epochs_since_improvement >= self.patience:
                            print(f"[DEBUG] Early stopping at epoch {epoch}")
                            break
                    
                    pbar.set_postfix({"train": f"{train_loss:.4f}", "val": f"{val_loss_val:.4f}"})
                    pbar.update(1)

            if self.restore_best and best_state is not None:
                self.dpmm_model.load_state_dict(best_state)
            
            self.dpmm_model.eval()
            
            if self.mode == "likelihood_threshold":
                if len(msgs) > 1:
                    msg_val = msgs[1]
                    X_calib = msg_val.data
                else:
                    X_calib = X_fit
                
                x_calib_t = th.as_tensor(X_calib, dtype=th.float32, device=self.device)
                with th.no_grad():
                    _, _, loglike_tr = self.dpmm_model(x_calib_t)
                self.likelihood_threshold = th.quantile(loglike_tr, self.quantile)

            else:  
                with th.no_grad():
                    pi_tr, _, _ = self.dpmm_model(x_t)
                clust_assignment = pi_tr.argmax(dim=1)

                tot = th.bincount(clust_assignment, minlength=self.n_clusters).to(self.device)
                anom = th.bincount(
                    clust_assignment, weights=y_t, minlength=self.n_clusters
                )

                perc = anom / (tot + 1e-6)
                self.anomaly_cluster_labels = (perc > 0.5) | (tot == 0)

    def predict(self, X_or_msg: Union[np.ndarray, PipelineMessage]) -> np.ndarray:  # pylint: disable=invalid-name
        """Return continuous anomaly scores in [0, 1] for each sample."""
        if hasattr(X_or_msg, "data"):
            return self.transform(X_or_msg).data
        
        # Legacy support for direct array input
        return self._predict_array(X_or_msg)

    def transform(self, msg: PipelineMessage) -> PipelineMessage:
        """Pipeline transformation: populates msg.data with anomaly scores."""
        scores = self._predict(msg.data, results=msg.results)
        msg.data = scores
        return msg

    def _predict(self, X: np.ndarray, results: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """Internal prediction logic for raw arrays."""
        with self._callback_context("model_predict", results):
            if self.dpmm_model is None:
                raise RuntimeError("Model not fitted. Call fit() first.")

            x_t = th.as_tensor(X, dtype=th.float32, device=self.device)
            self.dpmm_model.eval()
            with th.no_grad():
                pi_te, _, loglike_te = self.dpmm_model(x_t)

            if self.mode == "likelihood_threshold":
                if self.likelihood_threshold is None:
                    raise RuntimeError(
                        "likelihood_threshold not set. Fit the DPMM first."
                    )

                if self.return_likelihood:
                    return -loglike_te.detach().to("cpu").numpy()
                else:
                    return th.sigmoid((self.likelihood_threshold - loglike_te)).detach().to("cpu").numpy()
            else:  
                if self.anomaly_cluster_labels is None:
                    raise RuntimeError(
                        "Cluster labels not set. Fit with 'cluster_labels' first."
                    )
                cl = pi_te.argmax(dim=1)
                anom_probs = self.anomaly_cluster_labels.float().to(self.device)
                scores = anom_probs[cl]

            return scores.detach().to("cpu").numpy()

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
