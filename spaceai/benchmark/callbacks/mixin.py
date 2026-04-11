from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Dict, Any
from contextlib import contextmanager

if TYPE_CHECKING:
    from .handler import CallbackHandler
    from spaceai.models.anomaly_pipeline.anomaly_classifier import PipelineState
from contextlib import contextmanager
from typing import Optional, Dict, Any

class CallbackMixin:
    """Mixin to provide callback handling and monitoring context to classes."""
    
    def __init__(self, callback_handler: Optional["CallbackHandler"] = None, **kwargs):
        self.callback_handler = callback_handler
        super().__init__()


    @contextmanager
    def _callback_context(self, phase_name: str, results: Optional[Dict[str, Any]] = None):
        """
        Context manager that starts monitoring, executes the code,
        stops it, and saves the collected metrics with the correct prefix.
        
        Args:
            phase_name (str): Prefix for the collected metrics.
            results (Optional[Dict[str, Any]]): Dictionary to update with metrics.
        """
        if self.callback_handler is not None:
            self.callback_handler.start()
            
        try:
            yield 
        finally:
            if self.callback_handler is not None:
                self.callback_handler.stop()
                if results is not None:
                    # Update the results dictionary with the collected metrics
                    metrics = self.callback_handler.collect(reset=True)
                    results.update(
                        {f"{phase_name}_{k}": v for k, v in metrics.items()}
                    )

    def pipeline_step(
        self, 
        state: "PipelineState", 
        is_fit: bool = False, 
        results: Optional[Dict[str, Any]] = None, 
        method_name: Optional[str] = None, 
        **kwargs
    ) -> "PipelineState":
        """
        Adapter method that dynamically routes execution to the correct step implementation
        and updates the PipelineState accordingly.
        """

        result = None
        if results is not None and "results" not in kwargs:
            kwargs["results"] = results

        if method_name is not None and hasattr(self, method_name):
            X = state.data
            y = state.labels
            
            try:
                result = getattr(self, method_name)(X, y=y, **kwargs)
            except TypeError:
                result = getattr(self, method_name)(X, **kwargs)
        else:
            X = state.data
            y = state.labels
            if is_fit:
                if hasattr(self, 'fit_transform'):
                    result = self.fit_transform(X, y=y, **kwargs)
                elif hasattr(self, 'fit'):
                    self.fit(X, y=y, **kwargs)
                    if hasattr(self, 'transform'):
                        result = self.transform(X, y=y, **kwargs)
                    elif hasattr(self, 'predict'):
                        result = self.predict(X, **kwargs)
            else:
                if hasattr(self, 'predict'):
                    result = self.predict(X, **kwargs)
                elif hasattr(self, 'detect'):
                    result = self.detect(X, **kwargs)
                elif hasattr(self, 'transform'):
                    result = self.transform(X, y=y, **kwargs)

        if result is None or result is self:
            return state
            
        if type(result).__name__ == "PipelineState":
            state = result
        elif isinstance(result, tuple) and len(result) == 3:
            state.data, state.labels, mask = result
            if state.indices is not None and len(state.indices) == len(mask):
                state.indices = state.indices[mask]
            if state.intervals is not None and len(state.intervals) == len(mask):
                state.intervals = [state.intervals[i] for i, m in enumerate(mask) if m]
        elif isinstance(result, tuple) and len(result) == 2:
            state.data, state.labels = result
        else:
            state.data = result

        if results is not None:
            state.metrics.update(results)

        return state