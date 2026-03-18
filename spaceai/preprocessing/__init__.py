"""Preprocessing module."""

def __getattr__(name):
    if name == "TSSplitter":
        from .ts_splitter import TSSplitter
        return TSSplitter
    if name == "FEATURE_MAP":
        from .functions import FEATURE_MAP
        return FEATURE_MAP
    if name in ["StatisticsFeatureExtractor", "RocketFeatureExtractor", "get_feature_extractor"]:
        from .feature_extractors import StatisticsFeatureExtractor, RocketFeatureExtractor, get_feature_extractor
        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "TSSplitter",
    "FEATURE_MAP",
    "StatisticsFeatureExtractor",
    "RocketFeatureExtractor",
    "get_feature_extractor",
]
