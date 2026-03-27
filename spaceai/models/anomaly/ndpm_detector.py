from __future__ import annotations
"""NDPM Detector module wrapping the CL-DPMixtureModel implementation."""
import os
import sys
import logging
import torch
import numpy as np
from typing import Optional, Dict, Any

from .ndpm_internal import Ndpm, Config
from tensorboardX import SummaryWriter
from .base import BaseClassifier


class NDPMDetector(BaseClassifier):
    def __init__(self, config_dict: Dict[str, Any], device: str = "cpu", 
                 log_dir: Optional[str] = None, writer: Optional[SummaryWriter] = None,
                 threshold: Optional[float] = None, callback_handler: Optional[Any] = None,
                 **kwargs):
        super().__init__(callback_handler=callback_handler, **kwargs)
        
        config_dict['disable_d'] = True  
        if 'stm_size' not in config_dict:
            config_dict['stm_size'] = 10000  
            
        self.config = Config(**config_dict)
        self.device = device
        self.config['device'] = device
        
        if writer is not None:
            self.writer = writer
            self._own_writer = False
        else:
            log_dir = log_dir or self.config.get("log_dir", "logs/ndpm_default")
            self.writer = SummaryWriter(log_dir)
            self._own_writer = True
            
        self.model = Ndpm(self.config, self.writer)
        self.model.to(self.device)
        
        # Adaptive Threshold (Rolling Percentile)
        self.threshold = threshold if threshold is not None else self.config.get("anomaly_threshold", -100.0)
        self.ll_buffer = []
        self.max_buffer_size = self.config.get("ll_buffer_size", 5000)
        self.percentile = self.config.get("ll_percentile", 0.5)  
        
        self.global_step = 0

    def _ensure_writer(self):
        """Ensure the SummaryWriter is initialized after loading from disk."""
        if not hasattr(self, 'writer') or self.writer is None:
            log_dir = self.config.get("log_dir", "logs/ndpm_default")
            self.writer = SummaryWriter(log_dir)
            self._own_writer = True
            # Update the underlying model's writer too
            self.model.writer = self.writer

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None, results: Optional[Dict[str, Any]] = None) -> None:
        with self._callback_context("model_fit", results):
            if hasattr(X, "values"):
                X = X.values  # Handle pd.DataFrame
            if y is not None:
                normal_X = X[y == 0]
            else:
                is_anomaly = self.predict(X) 
                normal_X = X[~is_anomaly] 
            
            if len(normal_X) == 0:
                return

            x_tensor = torch.from_numpy(normal_X).float().to(self.device)
            dummy_y = torch.zeros(len(x_tensor), dtype=torch.long).to(self.device)

            self._ensure_writer()
            self.model.train()
            self.model.learn(x_tensor, dummy_y, self.global_step)
            self.global_step += 1

            # Update adaptive threshold using training data (nominal by assumption)
            with torch.no_grad():
                self.model.eval()
                ll_nominal = self.model(x_tensor).cpu().numpy()
                self._update_ll_buffer(ll_nominal)

            self.sleep()

    def _update_ll_buffer(self, new_lls: np.ndarray) -> None:
        """Update the log-likelihood buffer and recalculate the threshold."""
        self.ll_buffer.extend(new_lls.tolist())
        
        # Keep buffer within size limits
        if len(self.ll_buffer) > self.max_buffer_size:
            self.ll_buffer = self.ll_buffer[-self.max_buffer_size:]
            
        # Recalculate threshold based on percentile (0-100)
        if len(self.ll_buffer) > 0:
            self.threshold = float(np.percentile(self.ll_buffer, self.percentile))
            logging.info("Updated adaptive threshold for NDPM: %.4f (buffer size: %d, percentile: %.2f)", 
                         self.threshold, len(self.ll_buffer), self.percentile)

    def predict(self, X: np.ndarray, results: Optional[Dict[str, Any]] = None) -> np.ndarray:
        with self._callback_context("model_predict", results):
            if hasattr(X, "values"):
                X = X.values  # Handle pd.DataFrame

            self.model.eval()
            x_tensor = torch.from_numpy(X).float().to(self.device)
            
            with torch.no_grad():
                log_likelihood = self.model(x_tensor)

            return (log_likelihood < self.threshold).cpu().numpy()

    def sleep(self) -> None:
        print("sleeping")
        if len(self.model.stm_x) == 0:
            return
            
        all_x = torch.stack(self.model.stm_x)
        all_y = torch.stack(self.model.stm_y)
        
        full_experience_ds = torch.utils.data.TensorDataset(all_x, all_y)
     
        self.model.sleep(full_experience_ds)
        
        self.model.stm_x = []
        self.model.stm_y = []

    def save(self, path: str) -> None:
        torch.save(self, path)

    def __getstate__(self):
        state = self.__dict__.copy()
        # SummaryWriter and other components might not be pickleable
        if "writer" in state:
            state["writer"] = None
        if "model" in state and hasattr(state["model"], "writer"):
            state["model"].writer = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._ensure_writer()

    @staticmethod
    def load(path: str) -> "NDPMDetector":
        return torch.load(path, weights_only=False)

    def __del__(self):
        if hasattr(self, 'writer') and getattr(self, '_own_writer', False):
            self.writer.close()