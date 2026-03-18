"""Anomaly classifier module."""

def __getattr__(name):
    if name == "AnomalyClassifier":
        from .anomaly_classifier import AnomalyClassifier
        return AnomalyClassifier
    if name == "DPMMDetector":
        from .dpmm_detector import DPMMDetector
        return DPMMDetector
    if name == "RockadClassifier":
        from .rockad import RockadClassifier
        return RockadClassifier
    if name == "NearestNeighborOCC":
        from .rockad import NearestNeighborOCC
        return NearestNeighborOCC
    if name == "NDPMDetector":
        from .ndpm_detector import NDPMDetector
        return NDPMDetector
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "AnomalyClassifier",
    "DPMMDetector",
    "RockadClassifier",
    "NearestNeighborOCC",
    "NDPMDetector",
]
