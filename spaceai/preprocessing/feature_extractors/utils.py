"""Utility functions for feature extractors."""

from typing import (
    Any,
    Optional,
)

from spaceai.preprocessing.functions import FEATURE_MAP

from .rocket_feature_extractor import RocketFeatureExtractor
from .statistics_feature_extractor import StatisticsFeatureExtractor
from .global_discord_extractor import GlobalDiscordExtractor
from .feature_union import FeatureUnion


def get_feature_extractor(
    name: str, window_size: int, stride: int, callback_handler=None, **kwargs
) -> Optional[Any]:
    """
    Factory function to get a feature extractor by name.

    Args:
        name: Name of the feature extractor ('base_statistics', 'rocket', 'none').
        window_size: Size of the sliding window.
        stride: Step size between windows.
        **kwargs: Additional arguments for the feature extractor constructor.

    Returns:
        The instantiated feature extractor or None.
    """
    if isinstance(name, list):
        extractors = []
        for n in name:
            ext = get_feature_extractor(n, window_size, stride, callback_handler=callback_handler, **kwargs)
            if ext:
                extractors.append(ext)
        return FeatureUnion(extractors) if extractors else None

    if name == "base_statistics":
        selected_features = kwargs.pop("selected_features", None)
        transformations = FEATURE_MAP
        if selected_features:
            if any(isinstance(x, list) for x in selected_features):
                raise ValueError(
                    "Error: 'selected_features' contains a list of lists. "
                    "You are trying to run a grid search (sweep) using `run_pipeline.py` instead of `run_all_pipelines.py`.\n"
                    "Please run: `poetry run python examples/run_all_pipelines.py --config YOUR_CONFIG.yaml`\n"
                    "Or modify the config to only contain a single, flat list of features."
                )
            transformations = {
                k: v for k, v in FEATURE_MAP.items() if k in selected_features
            }
            
        kwargs.pop("n_kernel", None)
        return StatisticsFeatureExtractor(
            transformations=transformations,
            window_size=window_size,
            stride=stride,
            callback_handler=callback_handler,
            **kwargs,
        )
    elif name == "rocket":
        num_kernels = kwargs.get("n_kernel") or kwargs.get("num_kernels") or 100
        # Remove n_kernel/num_kernels from kwargs to avoid duplicates if passed explicitly
        if "n_kernel" in kwargs:
            del kwargs["n_kernel"]
        if "num_kernels" in kwargs:
            del kwargs["num_kernels"]

        return RocketFeatureExtractor(
            window_size=window_size, stride=stride, num_kernels=num_kernels, callback_handler=callback_handler, **kwargs
        )
    elif name == "global_discord":
        m = kwargs.pop("m", 15)
        history_len = kwargs.pop("history_len", kwargs.pop("history_size", None))
        return GlobalDiscordExtractor(
            window_size=window_size, stride=stride, m=m, history_len=history_len, **kwargs
        )
    elif name == "none" or name is None:
        return None
    else:
        raise ValueError(f"Unknown feature extractor: {name}")
