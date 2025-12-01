"""Utility functions for feature extractors."""

from typing import Any, Optional

from spaceai.preprocessing.functions import FEATURE_MAP
from .statistics_feature_extractor import StatisticsFeatureExtractor
from .rocket_feature_extractor import RocketFeatureExtractor


def get_feature_extractor(name: str, **kwargs) -> Optional[Any]:
    """
    Factory function to get a feature extractor by name.

    Args:
        name: Name of the feature extractor ('base_statistics', 'rocket', 'none').
        **kwargs: Additional arguments for the feature extractor constructor.

    Returns:
        The instantiated feature extractor or None.
    """
    if name == "base_statistics":
        kwargs.pop("n_kernel", None)
        return StatisticsFeatureExtractor(transformations=FEATURE_MAP, **kwargs)
    elif name == "rocket":
        num_kernels = kwargs.get("n_kernel") or kwargs.get("num_kernels") or 100
        # Remove n_kernel/num_kernels from kwargs to avoid duplicates if passed explicitly
        if "n_kernel" in kwargs:
            del kwargs["n_kernel"]
        if "num_kernels" in kwargs:
            del kwargs["num_kernels"]
            
        return RocketFeatureExtractor(num_kernels=num_kernels, **kwargs)
    elif name == "none" or name is None:
        return None
    else:
        raise ValueError(f"Unknown feature extractor: {name}")
