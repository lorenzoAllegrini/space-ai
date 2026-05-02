from typing import Optional, Any, Dict, TYPE_CHECKING
import numpy as np
from spaceai.benchmark.callbacks.mixin import CallbackMixin
from spaceai.data import AnomalyDataset, AnomalyDatasetSubset

if TYPE_CHECKING:
    from spaceai.models.anomaly_pipeline.anomaly_classifier import PipelineState

class ValidationSplitter(CallbackMixin):
    """Pipeline node that splits training and validation data.
    It stores the validation portion internally and injects it during the validation phase.
    """
    def __init__(self, eval_perc: float = 0.2, callback_handler=None):
        super().__init__(callback_handler)
        self.eval_perc = eval_perc
        self.val_data = None
        self.val_labels = None
        self.val_indices = None
        self.val_intervals = None

    def pipeline_step(
        self,
        state: "PipelineState",
        is_fit: bool = False,
        results: Optional[Dict[str, Any]] = None,
        method_name: Optional[str] = None,
        **kwargs,
    ) -> "PipelineState":
        """Dispatch to split or inject_val based on method_name.
        Returns an updated PipelineState.
        """
        if method_name == "split":
            X, y = self._split(state.data, state.labels)
            # Preserve indices/intervals for training part if present
            if state.indices is not None:
                split_idx = len(X) if hasattr(X, "__len__") else 0
                state.indices = state.indices[:split_idx]
            if state.intervals is not None:
                split_idx = len(X) if hasattr(X, "__len__") else 0
                state.intervals = state.intervals[:split_idx]
            state.data = X
            state.labels = y
            return state
        elif method_name == "inject_val":
            if self.val_data is None:
                raise ValueError("Validation data not set. Did you run 'split' phase?")
            state.data = self.val_data
            state.labels = self.val_labels
            state.indices = self.val_indices
            state.intervals = self.val_intervals
            return state
        else:
            # Fallback to parent implementation (should not happen)
            return super().pipeline_step(state, is_fit, results, method_name, **kwargs)

    def _split(self, X: Any, y: Optional[Any] = None):
        """Internal split logic used in the training phase.
        Stores validation portion for later injection.
        """
        if self.eval_perc is None or self.eval_perc <= 0.0 or self.eval_perc >= 1.0:
            self.val_data, self.val_labels = None, None
            return X, y
        n_samples = len(X) if hasattr(X, "__len__") else 0
        split_idx = int(n_samples * (1 - self.eval_perc))
        # Split data
        if isinstance(X, AnomalyDataset):
            X_tr = AnomalyDatasetSubset(parent=X, start_idx=0, end_idx=split_idx)
            X_vl = AnomalyDatasetSubset(parent=X, start_idx=split_idx, end_idx=n_samples)
        else:
            X_tr, X_vl = X[:split_idx], X[split_idx:]
        y_tr = y[:split_idx] if y is not None else None
        y_vl = y[split_idx:] if y is not None else None
        # Store validation part
        self.val_data = X_vl
        self.val_labels = y_vl
        self.val_indices = None
        self.val_intervals = None
        return X_tr, y_tr

__all__ = ["ValidationSplitter"]