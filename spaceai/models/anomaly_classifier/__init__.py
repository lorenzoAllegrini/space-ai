"""Anomaly classifier module."""

from .anomaly_classifier import (
    AnomalyClassifier,
    AnomalyDetectionPipeline,
    PipelineMessage,
)
from spaceai.models.anomaly.dpmm_detector import DPMM, DPMMDetector
from spaceai.models.anomaly.rockad import RockadClassifier, NearestNeighborOCC
try:
    from spaceai.models.anomaly.ndpm_detector import NDPMDetector
except ImportError:
    NDPMDetector = None
    import logging
    logging.warning("NDPMDetector not available (missing tensorboardX or other dependencies).")
from .adaptive_rolling_window_classifier import AdaptiveRollingWindowClassifier
from .sml_client_pipeline import SMLClientPipeline

__all__ = [
    "AnomalyClassifier",
    "AnomalyDetectionPipeline",
    "PipelineMessage",
    "DPMM",
    "DPMMDetector",
    "RockadClassifier",
    "NearestNeighborOCC",
    "NDPMDetector",
    "AdaptiveRollingWindowClassifier",
    "SMLClientPipeline",
]

