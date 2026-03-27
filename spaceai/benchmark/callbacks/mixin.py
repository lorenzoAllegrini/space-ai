from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Dict, Any
from contextlib import contextmanager

if TYPE_CHECKING:
    from .handler import CallbackHandler

class CallbackMixin:
    """Mixin to provide callback handling and monitoring context to classes."""
    
    def __init__(self, callback_handler: Optional[CallbackHandler] = None, **kwargs):
        self.callback_handler = callback_handler
        self._kill_switch_active = False
        super().__init__(**kwargs)

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
