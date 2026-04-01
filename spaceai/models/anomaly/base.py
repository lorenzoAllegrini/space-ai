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

        # If already a list of messages (or multiple message arguments), return as is
        # Using class name check for robustness against dynamic import issues
        def is_msg(obj):
            return obj is not None and (obj.__class__.__name__ == "PipelineMessage" or hasattr(obj, "results"))

        if len(args) > 0 and is_msg(args[0]):
            msg_train = args[0]
            msg_val = args[1] if len(args) > 1 and is_msg(args[1]) else None
            return [msg_train, msg_val]

        # If already a list of messages, return as is
        if isinstance(X, list) and len(X) > 0 and is_msg(X[0]):
            msg_train = X[0]
            msg_val = X[1] if len(X) > 1 else None
            return [msg_train, msg_val]

        # If a single message, wrap it
        if is_msg(X):
            return [X, None]

        # Handle Pandas DataFrame
        import pandas as pd
        if isinstance(X, pd.DataFrame):
            msg_train = PipelineMessage(data=X.values, labels=y, 
                                       results=results if results is not None else {}, 
                                       metadata=metadata if metadata is not None else {})
            return [msg_train, None]

        msg_val = kwargs.get("msg_val")

        # Handle list of numpy arrays (train/val split)
        if isinstance(X, list) and len(X) > 0 and isinstance(X[0], np.ndarray):
            X_train = X[0]
            y_train = y[0] if isinstance(y, list) and len(y) > 0 else y
            msg_train = PipelineMessage(data=X_train, labels=y_train, 
                                       results=results if results is not None else {}, 
                                       metadata=metadata if metadata is not None else {})
            if msg_val is None and len(X) > 1:
                X_val = X[1]
                y_val = y[1] if isinstance(y, list) and len(y) > 1 else None
                msg_val = PipelineMessage(data=X_val, labels=y_val, 
                                         results=results if results is not None else {}, 
                                         metadata=metadata if metadata is not None else {})
            return [msg_train, msg_val]

        # Default case: single raw array
        msg_train = PipelineMessage(data=X, labels=y, 
                                   results=results if results is not None else {}, 
                                   metadata=metadata if metadata is not None else {})
        return [msg_train, msg_val]

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
        
        scores = self._predict(msg_test.data, results=msg_test.results)
        
        if is_pipeline_mode:
            msg_test.data = scores
            return msg_test
        return scores

    def transform(self, X: Union[np.ndarray, "PipelineMessage"]) -> Union[np.ndarray, "PipelineMessage"]:
        """Alias for predict to support modular pipelines."""
        return self.predict(X)

    def _predict(self, X: np.ndarray, results: Optional[Dict[str, Any]] = None, **kwargs) -> np.ndarray:
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
    
    def __init__(self, model: Any, supervised: bool = False, return_labels: bool = False, callback_handler: Optional[CallbackHandler] = None):
        super().__init__(callback_handler=callback_handler)
        self.model = model
        self.supervised = supervised
        self.return_labels = return_labels

    def _fit(self, X: np.ndarray, y: Optional[np.ndarray] = None, results: Optional[Dict[str, Any]] = None, **kwargs) -> None:
        import inspect
        call_kwargs = {}
        
        # Propagation to Pipeline steps if needed using <step>__<param> syntax
        msg_val = kwargs.get("msg_val")
        if hasattr(self.model, "steps"): # It's a Pipeline
            for name, step in self.model.steps:
                try:
                    sig = inspect.signature(step.fit)
                    has_kwargs = any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values())
                    if "results" in sig.parameters or has_kwargs:
                        call_kwargs[f"{name}__results"] = results
                    if "msg_val" in sig.parameters or has_kwargs:
                        call_kwargs[f"{name}__msg_val"] = msg_val
                except (ValueError, TypeError):
                    continue
        else:
            try:
                sig = inspect.signature(self.model.fit)
                has_kwargs = any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values())
                if "results" in sig.parameters or has_kwargs:
                    call_kwargs["results"] = results
                if "msg_val" in sig.parameters or has_kwargs:
                    call_kwargs["msg_val"] = msg_val
            except (ValueError, TypeError):
                pass

        if self.supervised:
            self.model.fit(X, y, **call_kwargs)
        else:
            self.model.fit(X, **call_kwargs)

    def _predict(self, X: np.ndarray, results: Optional[Dict[str, Any]] = None, **kwargs) -> np.ndarray:
        import inspect
        # Try to find the prediction method
        pred_method = None
        if self.return_labels:
            pred_method = self.model.predict
        elif hasattr(self.model, "decision_function"):
            pred_method = self.model.decision_function
        elif hasattr(self.model, "predict_proba"):
            pred_method = self.model.predict_proba
        
        call_kwargs = {}
        if pred_method and "results" in inspect.signature(pred_method).parameters:
            call_kwargs["results"] = results

        # If return_labels is True, use direct predict (binary labels)
        if self.return_labels:
            return self.model.predict(X, **call_kwargs)

        # Prefer decision_function for raw anomaly scores
        if hasattr(self.model, "decision_function"):
            return self.model.decision_function(X, **call_kwargs)
            
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X, **call_kwargs)
            if probs.ndim > 1 and probs.shape[1] == 2:
                return probs[:, 1]
            return probs
        return self.model.predict(X)
