"""Feature extractors package."""

def __getattr__(name):
    if name == "FeatureExtractor":
        from .feature_extractor import FeatureExtractor
        return FeatureExtractor
    if name == "StatisticsFeatureExtractor":
        from .statistics_feature_extractor import StatisticsFeatureExtractor
        return StatisticsFeatureExtractor
    if name == "RocketFeatureExtractor":
        from .rocket_feature_extractor import RocketFeatureExtractor
        return RocketFeatureExtractor
    if name == "get_feature_extractor":
        from .utils import get_feature_extractor
        return get_feature_extractor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "FeatureExtractor",
    "StatisticsFeatureExtractor",
    "RocketFeatureExtractor",
    "get_feature_extractor",
]
