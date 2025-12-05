"""SpaceAI feature extractor module."""

from typing import (
    Callable,
    Dict,
)

import numpy as np
import pandas as pd  # type: ignore

from spaceai.preprocessing.functions import FEATURE_MAP


class StatisticsFeatureExtractor:
    """
    Unified feature extractor for SpaceAI datasets.

    This class expects already segmented data (2D arrays) and applies
    statistical transformations to each segment.
    """

    def __init__(
        self,
        transformations: Dict[str, Callable],
        telecommands: bool = False,
        run_id: str = "esa_segments",
        exp_dir: str = "experiments",
    ) -> None:

        self.transformations = transformations
        self.telecommands = telecommands
        self.run_id = run_id
        self.exp_dir = exp_dir

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

    def fit_transform(  # pylint: disable=invalid-name
        self, X: np.ndarray, y=None
    ) -> pd.DataFrame:
        """
        Fit to data, then transform it.

        Args:
            X: Input data.
            y: Ignored.

        Returns:
            pd.DataFrame: Extracted features.
        """
        return self.fit(X, y).transform(X)

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
        # Convert to numpy if needed
        data = X
        if isinstance(X, pd.DataFrame):
            data = X.values
        if isinstance(X, pd.Series):
            data = X.values

        # Ensure 2D (n_samples, window_size)
        if data.ndim == 1:
            raise ValueError(
                "Input X must be 2D array of segments (n_samples, window_size)"
            )

        feature_list = [func(segments=data) for func in self.transformations.values()]
        transformed_segments = np.column_stack(feature_list)

        df = pd.DataFrame(
            transformed_segments, columns=list(self.transformations.keys())
        )
        df = df.fillna(df.mean()).fillna(0)
        return df


__all__ = ["FEATURE_MAP", "StatisticsFeatureExtractor"]
