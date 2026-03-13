"""SpaceAI feature extractor module."""

from typing import (
    Callable,
    Dict,
)

import numpy as np
import pandas as pd  # type: ignore

from spaceai.preprocessing.functions import FEATURE_MAP
from .feature_extractor import FeatureExtractor


class StatisticsFeatureExtractor(FeatureExtractor):
    """
    Unified feature extractor for SpaceAI datasets.

    This class expects already segmented data (2D arrays) and applies
    statistical transformations to each segment.
    """

    def __init__(
        self,
        transformations: Dict[str, Callable],
        telecommands: bool = False,
    ) -> None:

        self.transformations = transformations
        self.telecommands = telecommands

    def fit(  # pylint: disable=invalid-name
        self, X: np.ndarray, _y=None  # pylint: disable=unused-argument
    ):
        """
        Fit the feature extractor.

        Args:
            X: Input data (not used, stateless transformer).
            _y: Ignored.

        Returns:
            self
        """
        return self


    def transform(  # pylint: disable=invalid-name
        self, X: np.ndarray
    ) -> pd.DataFrame:
        """
        Extract statistical features from batches of segments.

        Args:
            X: Input data (2D array of shape [n_samples, window_size]).

        Returns:
            pd.DataFrame: Extracted features.
        """
        data = X
        if isinstance(X, pd.DataFrame):
            data = X.values
        if isinstance(X, pd.Series):
            data = X.values

        different_lengths = False
        try:
            data = np.array(data.tolist(), dtype=float)
        except (ValueError, TypeError):
            different_lengths = True

        if not different_lengths and data.ndim == 1:
            raise ValueError(
                "Input X must be 2D array of segments (n_samples, window_size) or ragged array of segments"
            )

        if different_lengths:
            transformed_segments = np.column_stack([
                [np.atleast_1d(func(segments=np.atleast_2d(s)))[0] for s in data]
                for func in self.transformations.values()
            ])
        else:
            feature_list = [func(segments=data) for func in self.transformations.values()]
            transformed_segments = np.column_stack(feature_list)

        df = pd.DataFrame(
            transformed_segments, columns=list(self.transformations.keys())
        )
        df = df.fillna(df.mean()).fillna(0)
        return df


__all__ = ["FEATURE_MAP", "StatisticsFeatureExtractor"]
