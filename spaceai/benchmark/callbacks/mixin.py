from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Dict, Any
from contextlib import contextmanager

if TYPE_CHECKING:
    from .handler import CallbackHandler
    from spaceai.models.anomaly_pipeline.anomaly_classifier import PipelineState

class CallbackMixin:
    """Mixin to provide callback handling and monitoring context to classes."""
    
    def __init__(self, callback_handler: Optional[CallbackHandler] = None, **kwargs):
        self.callback_handler = callback_handler
        self._kill_switch_active = False
        super().__init__()

    @property
    def kill_switch_active(self) -> bool:
        """
        Flag checking if the processor is in 'kill-switch' mode.
        Default is False for all processors via CallbackMixin.
        """
        return self._kill_switch_active

    @kill_switch_active.setter
    def kill_switch_active(self, value: bool):
        self._kill_switch_active = value

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

    def pipeline_step(self, state: "PipelineState", is_fit: bool = False, **kwargs) -> "PipelineState":
        """
        Adapter method that dynamically routes execution to the correct step implementation
        and updates the PipelineState accordingly.
        """
        result = None

        if is_fit:
            if hasattr(self, 'fit_transform'):
                result = self.fit_transform(state.data, y=state.labels, **kwargs)
            elif hasattr(self, 'fit'):
                self.fit(state.data, y=state.labels, **kwargs)
                if hasattr(self, 'transform'):
                    result = self.transform(state.data, **kwargs)
                elif hasattr(self, 'predict'):
                    result = self.predict(state.data, **kwargs)
        else:
            if hasattr(self, 'predict'):
                result = self.predict(state.data, **kwargs)
            elif hasattr(self, 'detect'):
                result = self.detect(state.data, **kwargs)
            elif hasattr(self, 'transform'):
                result = self.transform(state.data, **kwargs)

        if result is None:
            return state
            
        if type(result).__name__ == "PipelineState":
            state = result
        elif isinstance(result, tuple) and len(result) == 2:
            state.data, state.labels = result
        else:
            state.data = result
            
        # Propagate kill-switch status via metadata
        if self.kill_switch_active:
            state.metadata["kill_switch_active"] = True

        return state
