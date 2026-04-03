"""SpaceAI feature extractor module."""

from typing import (
    Callable,
    Dict,
    Optional,
    Any,
    Union
)
import os
import numpy as np
import pandas as pd  # type: ignore

from spaceai.preprocessing.functions import FEATURE_MAP
from .feature_extractor import FeatureExtractor
from sklearn.metrics import precision_recall_curve
from sklearn.feature_selection import SelectKBest



class StatisticsFeatureExtractor(FeatureExtractor):
    """
    Unified feature extractor for SpaceAI datasets.

    This class expects already segmented data (2D arrays) and applies
    statistical transformations to each segment.
    """

    def __init__(
        self,
        transformations: Dict[str, Callable],
        window_size: Optional[int],
        stride: Optional[int],
        telecommands: bool = False,
        max_features: Optional[int] = 5,
        callback_handler: Optional[Any] = None,
        **kwargs
    ) -> None:
        super().__init__(window_size or 0, stride or 0, callback_handler=callback_handler, **kwargs)

        self.transformations = transformations
        self.telecommands = telecommands
        self.max_features = max_features
        self.kill_switch_active = False

    @property
    def output_dim(self) -> int:
        return len(self.transformations)


    def fit(  # pylint: disable=invalid-name
        self, 
        X: np.ndarray, 
        y=None,
        results: Optional[Dict[str, Any]] = None
    ):
        """
        Fit the feature extractor.
        """
        with self._callback_context("feature_selection", results):
            if self.max_features is not None and self.max_features < len(self.transformations):
                if y is None:
                    raise ValueError("y must be provided for feature selection")
                X_features = self.transform(X, results=results)
                self.select_features(X_features, y, results=results)
        return self


    def transform(  # pylint: disable=invalid-name
        self, 
        X: Union[np.ndarray, Any],
        results: Optional[Dict[str, Any]] = None,
        save_dir: Optional[str] = None,
        suffix: str = ""
    ) -> Union[pd.DataFrame, Any]:
        """
        Extract statistical features from batches of segments.
        Supports PipelineMessage.
        """
        if hasattr(X, "data") and not isinstance(X, (np.ndarray, pd.DataFrame)):
            msg = X
            msg.data = self.transform(msg.data, results=results, save_dir=save_dir, suffix=suffix)
            return msg

        data = X
        if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
            data = X.values

        different_lengths = False

        try:
            data = np.asarray(data, dtype=float)
        except (ValueError, TypeError):
            different_lengths = True
        else:
            if data.ndim > 2:
                data = data.reshape(data.shape[0], -1)
            elif data.ndim == 1:
                raise ValueError("Input X must be 2D array of segments (n_samples, window_size) or ragged array of segments")

        with self._callback_context("feature_extraction", results):
            if not self.transformations:
                transformed_segments = np.empty((len(data), 0))
            elif different_lengths:
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
            
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                filename = "extracted_features"
                if suffix:
                    filename += f"_{suffix}"
                save_path = os.path.join(save_dir, f"{filename}.csv")
                df.to_csv(save_path, index=False)
                print(f"[DEBUG] Features saved to {save_path}")

        return df
    
    def select_features(
        self, 
        X_features: pd.DataFrame, 
        y: np.ndarray,
        results: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """
        Select features that minimize false positives on anomalies by 
        maximizing precision on the tails (Max F0.1 Score).
        Note: X_features must be the output of self.transform(X).
        """

        X_clean = X_features.fillna(0)
        precision_selector = SelectKBest(score_func=tail_f01_score, k=self.max_features)
        precision_selector.fit(X_clean.values, y)

        feature_scores = sorted(zip(X_features.columns, precision_selector.scores_), key=lambda x: x[1], reverse=True)
        print(f"[DEBUG] Feature scores: {feature_scores}")
        
        correlation_threshold = 0.8
        corr_matrix = X_clean.corr().abs()
        
        filtered_feature_scores = []
        dropped_by_corr = set()
        
        for f_name, f_score in feature_scores:
            if f_name in dropped_by_corr:
                continue
                
            filtered_feature_scores.append((f_name, f_score))
            
            correlated_with_f = corr_matrix.index[corr_matrix[f_name] > correlation_threshold].tolist()
            for drop_name in correlated_with_f:
                if drop_name != f_name and drop_name not in dropped_by_corr:
                    dropped_by_corr.add(drop_name)
                    
        feature_scores = filtered_feature_scores

        selected_feature_names = []

        for i, (f_name, f_score) in enumerate(feature_scores):
            if len(selected_feature_names) >= self.max_features:
                break
            if i < 2 and f_score >= 0.4:
                selected_feature_names.append(f_name)
            elif f_score >= 0.5:
                selected_feature_names.append(f_name)
            else:
                pass
        
        if len(selected_feature_names) == 0:
            self.kill_switch_active = True

        self.transformations = {
            name: func 
            for name, func in self.transformations.items() 
            if name in selected_feature_names
        }

        print(f"[DEBUG] Selected features ({len(selected_feature_names)}): {selected_feature_names}")
        X_train_selected = X_features[selected_feature_names].copy()

        return X_train_selected

def tail_f01_score(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    scores = []
    normal_idx = (y == 0)
    beta_sq = 0.1 ** 2 
    
    for i in range(X.shape[1]):
        col = X[:, i]
        normal_median = np.nanmedian(col[normal_idx])
        
        extremity = np.abs(col - normal_median)
        extremity = np.nan_to_num(extremity, nan=0.0)
        
        try:
            precisions, recalls, _ = precision_recall_curve(y, extremity)
            denominator = (beta_sq * precisions) + recalls
            with np.errstate(divide='ignore', invalid='ignore'):
                f01_curve = (1 + beta_sq) * (precisions * recalls) / denominator
                f01_curve[denominator == 0] = 0.0
            scores.append(np.max(f01_curve))
        except ValueError:
            scores.append(0.0)
            
    return np.array(scores)


__all__ = ["FEATURE_MAP", "StatisticsFeatureExtractor"]
