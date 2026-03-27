from __future__ import annotations
from typing import List, Optional, Dict, Any, TYPE_CHECKING
import numpy as np
import pandas as pd
from .feature_extractor import FeatureExtractor

if TYPE_CHECKING:
    from spaceai.data.anomaly_dataset import AnomalyDataset

class FeatureUnion(FeatureExtractor):
    """
    Concatenates results of multiple feature extractors.
    
    This allows combining, for example, simple statistics with global 
    matrix profile features in a single pipeline.
    """
    def __init__(
        self, 
        extractors: List[FeatureExtractor],
        **kwargs
    ):
        if not extractors:
            raise ValueError("FeatureUnion requires at least one extractor.")
            
        # All extractors must share the same window/stride for alignment
        super().__init__(
            window_size=extractors[0].window_size, 
            stride=extractors[0].stride, 
            **kwargs
        )
        self.extractors = extractors

    @property
    def window_size(self) -> int:
        return self._window_size

    @window_size.setter
    def window_size(self, value: int):
        self._window_size = value
        for e in self.extractors:
            if hasattr(e, "window_size"): # Propagate to property setter if exists
                try: e.window_size = value
                except AttributeError: e._window_size = value
            else:
                e._window_size = value

    @property
    def stride(self) -> int:
        return self._stride

    @stride.setter
    def stride(self, value: int):
        self._stride = value
        for e in self.extractors:
            if hasattr(e, "stride"):
                try: e.stride = value
                except AttributeError: e._stride = value
            else:
                e._stride = value

    @property
    def kill_switch_active(self) -> bool:
        """
        FeatureUnion is in kill-switch mode ONLY if ALL sub-extractors are.
        If even one extractor has informative features, the union is active.
        """
        return all(getattr(e, "kill_switch_active", False) for e in self.extractors)

    @property
    def output_dim(self) -> int:
        return sum(e.output_dim for e in self.extractors)

    def set_context(self, dataset: Optional[AnomalyDataset] = None, indices: Optional[np.ndarray] = None) -> None:
        """Propagate context to all sub-extractors."""
        super().set_context(dataset, indices)
        for e in self.extractors:
            e.set_context(dataset, indices)

    def clear_context(self) -> None:
        """Clear context for all sub-extractors."""
        super().clear_context()
        for e in self.extractors:
            e.clear_context()

    def fit(
        self, 
        X: np.ndarray, 
        y: Optional[np.ndarray] = None,
        results: Optional[Dict[str, Any]] = None
    ) -> FeatureUnion:
        """Fit all sub-extractors."""
        for e in self.extractors:
            e.fit(X, y, results=results)
        return self

    def transform(
        self, 
        X: Union[np.ndarray, Any],
        results: Optional[Dict[str, Any]] = None,
        save_dir: Optional[str] = None,
        suffix: str = ""
    ) -> Union[pd.DataFrame, Any]:
        """
        Transform via all sub-extractors and concatenate column-wise.
        Supports PipelineMessage.
        """
        if hasattr(X, "data") and not isinstance(X, (np.ndarray, pd.DataFrame)):
            msg = X
            msg.data = self.transform(msg.data, results=results, save_dir=save_dir, suffix=suffix)
            return msg

        dfs = []
        for e in self.extractors:
            df = e.transform(X, results=results, save_dir=save_dir, suffix=suffix)
            if not isinstance(df, pd.DataFrame):
                df = pd.DataFrame(df)
            dfs.append(df)
            
        return pd.concat(dfs, axis=1).astype(np.float32)

    def save(self, path: str) -> None:
        """Handle saving of composite extractors."""
        import torch
        torch.save(self, path)
