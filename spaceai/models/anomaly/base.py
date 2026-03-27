from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Union, List, TYPE_CHECKING
import numpy as np

from spaceai.benchmark.callbacks.mixin import CallbackMixin
from spaceai.benchmark.callbacks.handler import CallbackHandler

if TYPE_CHECKING:
    from spaceai.models.anomaly_classifier.anomaly_classifier import PipelineMessage


class BaseClassifier(CallbackMixin, ABC):
    """
    Abstract base class for underlying anomaly detection algorithms.
    Provides standard callback support for fitting and prediction phases.
    Supports polymorphic input (raw data or PipelineMessage).
    """
    
    def __init__(
        self, 
        callback_handler: Optional[CallbackHandler] = None, 
        **kwargs
    ) -> None:
        super().__init__(callback_handler=callback_handler, **kwargs)
        self.role = "classifier"

    def prepare_data(self, *args, **kwargs) -> List[Optional["PipelineMessage"]]:
        """Normalize input to a standard list of PipelineMessage: [msg_train, msg_val]."""
        X = args[0] if len(args) > 0 else kwargs.get("X")
        y = args[1] if len(args) > 1 else kwargs.get("y")
        
        results = kwargs.get("results", {})
        metadata = kwargs.get("metadata", {})

        from spaceai.models.anomaly_classifier.anomaly_classifier import PipelineMessage

        # If already a list of messages, return as is
        if isinstance(X, list) and len(X) > 0 and isinstance(X[0], PipelineMessage):
            msg_train = X[0]
            msg_val = X[1] if len(X) > 1 else None
            return [msg_train, msg_val]

        # If a single message, wrap it
        if isinstance(X, PipelineMessage):
            return [X, None]

        # Handle Pandas DataFrame
        import pandas as pd
        if isinstance(X, pd.DataFrame):
            msg_train = PipelineMessage(data=X.values, labels=y, results=results, metadata=metadata)
            return [msg_train, None]

        # Handle list of numpy arrays (train/val split)
        if isinstance(X, list) and len(X) > 0 and isinstance(X[0], np.ndarray):
            X_train = X[0]
            y_train = y[0] if isinstance(y, list) and len(y) > 0 else y
            msg_train = PipelineMessage(data=X_train, labels=y_train, results=results, metadata=metadata)
            msg_val = None
            if len(X) > 1:
                X_val = X[1]
                y_val = y[1] if isinstance(y, list) and len(y) > 1 else None
                msg_val = PipelineMessage(data=X_val, labels=y_val, results=results, metadata=metadata)
            return [msg_train, msg_val]

        # Default case: single raw array
        msg_train = PipelineMessage(data=X, labels=y, results=results, metadata=metadata)
        return [msg_train, None]

    def fit(self, *args, **kwargs) -> "BaseClassifier":
        """
        Public entry point for fitting. 
        Automatically handles metadata and polymorphic input by delegating to _fit.
        """
        msgs = self.prepare_data(*args, **kwargs)
        msg_train = msgs[0]
        msg_val = msgs[1] if len(msgs) > 1 else None
        
        # Call internal hook, passing msg_val as extra context if present
        self._fit(msg_train.data, y=msg_train.labels, results=msg_train.results, msg_val=msg_val, **kwargs)
        return self

    def _fit(
        self, 
        X: np.ndarray, 
        y: Optional[np.ndarray] = None, 
        results: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> None:
        """Internal fit implementation for raw data."""
        pass

    def predict(self, *args, **kwargs) -> Union[np.ndarray, "PipelineMessage"]:
        """
        Public entry point for prediction. 
        Automatically handles metadata and returns PipelineMessage in pipeline mode.
        """
        input_data = args[0] if len(args) > 0 else kwargs.get("X")
        
        from spaceai.models.anomaly_classifier.anomaly_classifier import PipelineMessage
        is_pipeline_mode = isinstance(input_data, PipelineMessage) or \
                           (isinstance(input_data, list) and len(input_data) > 0 and isinstance(input_data[0], PipelineMessage))

        msgs = self.prepare_data(*args, **kwargs)
        msg_test = msgs[0]
        
        # Execute internal hook
        scores = self._predict(msg_test.data, results=msg_test.results)
        
        if is_pipeline_mode:
            msg_test.data = scores
            return msg_test
        return scores

    def _predict(
        self, 
        X: np.ndarray, 
        results: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        """Internal prediction implementation for raw data."""
        pass

    def fit_predict(
        self, 
        X: Union[np.ndarray, "PipelineMessage"], 
        y: Optional[np.ndarray] = None, 
        results: Optional[Dict[str, Any]] = None
    ) -> Union[np.ndarray, "PipelineMessage"]:
        """Sequential fit and predict on the same data."""
        self.fit(X, y=y, results=results)
        return self.predict(X, results=results)


class SklearnClassifier(BaseClassifier):
    """Generic wrapper for sklearn/PyOD models to fit the BaseClassifier interface."""
    
    def __init__(self, model: Any, supervised: bool = False, callback_handler: Optional[CallbackHandler] = None):
        super().__init__(callback_handler=callback_handler)
        self.model = model
        self.supervised = supervised

    def _fit(self, X: np.ndarray, y: Optional[np.ndarray] = None, results: Optional[Dict[str, Any]] = None, **kwargs) -> None:
        if self.supervised:
            self.model.fit(X, y)
        else:
            self.model.fit(X)

    def _predict(self, X: np.ndarray, results: Optional[Dict[str, Any]] = None) -> np.ndarray:
        # Prefer decision_function for raw anomaly scores
        if hasattr(self.model, "decision_function"):
            return self.model.decision_function(X)
            
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X)
            if probs.ndim > 1 and probs.shape[1] == 2:
                return probs[:, 1]
            return probs
        return self.model.predict(X)
