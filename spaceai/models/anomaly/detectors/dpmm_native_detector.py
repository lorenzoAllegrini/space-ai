from typing import Any, Dict, Optional
import logging
import numpy as np

from .base import AnomalyDetector

class DPMMNativeDetector(AnomalyDetector):
    """
    Detector that thresholds continuous log-likelihood scores produced by
    DPMM (when return_likelihood=True) using the dynamically computed
    threshold saved in the message's metadata during DPMM's fit phase.

    In DPMM, return_likelihood=True outputs `-loglike_te`.
    Anomaly condition: loglike_te < likelihood_threshold
    Equivalent to: -loglike_te > -likelihood_threshold
                   score > -likelihood_threshold
    """

    def __init__(self, callback_handler: Optional[Any] = None, **kwargs):
        super().__init__(callback_handler)
        self.threshold: Optional[float] = None

    def _fit(self, scores: np.ndarray, results: Optional[Dict[str, Any]] = None) -> None:
        """
        DPMMNativeDetector extracts the threshold dynamically from the
        metadata populated by DPMM during its own fit phase.
        It does not compute anything from the validation scores.
        """
        pass
    
    def transform(self, message: "PipelineMessage") -> "PipelineMessage":
        # Threshold should be saved by DPMM in metadata
        thresh = message.metadata.get("likelihood_threshold", self.threshold)

        if thresh is None:
            logging.warning("DPMMNativeDetector: likelihood_threshold not found in metadata. Defaulting to 1.0 threshold comparison.")
            y_hat = message.data
            # Fallback if somehow using raw sigmoids instead of likelihoods
            message.data = (y_hat >= 1.0).astype(int)
            return message

        self.threshold = thresh
        
        # message.data contains -loglike_te
        scores = message.data
        anomaly_scores = (scores > -self.threshold).astype(int)
        
        message.data = anomaly_scores
        return message

    def detect(self, y_pred: np.ndarray, results: Optional[Dict[str, Any]] = None, **kwargs) -> np.ndarray:
        """Stand-alone detection, requires likelihood_threshold passed in kwargs or pre-set."""
        meta = kwargs.get("metadata", {})
        thresh = meta.get("likelihood_threshold", self.threshold)
        
        if thresh is None:
            raise ValueError("likelihood_threshold not found. Use transform() in a pipeline or provide it in kwargs['metadata'].")
            
        return (y_pred > -thresh).astype(int)

    def flush_detector(self) -> Optional[np.ndarray]:
        return None
