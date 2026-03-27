from __future__ import annotations
"""Abstract base class for concept drift detectors."""

from abc import ABC, abstractmethod
from typing import Callable, Iterable, Optional, Union, Dict, Any, TYPE_CHECKING

import numpy as np
from spaceai.benchmark.callbacks.mixin import CallbackMixin

if TYPE_CHECKING:
    from spaceai.benchmark.callbacks.handler import CallbackHandler


class DriftDetector(CallbackMixin, ABC):
    """Abstract interface for concept drift detectors.
    ... [omitted docstring for brevity] ...
    """

    def __init__(
        self, 
        filters: Optional[Iterable[Callable]] = None,
        callback_handler: Optional[CallbackHandler] = None,
        **kwargs
    ) -> None:
        self.filters = list(filters) if filters is not None else []
        super().__init__(callback_handler=callback_handler, **kwargs)

    def process(self, value: float, results: Optional[Dict[str, Any]] = None) -> bool:
        """Run standard pre-filters and, if passed, update the detector.

        Args:
            value (float): A raw scalar metric from the stream.
            results (Optional[Dict[str, Any]]): Dictionary to update with metrics.

        Returns:
            bool: ``True`` if the value was accepted by all filters AND 
                  the underlying detector found drift.
        """
        processed_val: Optional[float] = value
        for filter_fn in self.filters:
            processed_val = filter_fn(processed_val)
            if processed_val is None:
                return False

        return self.update(processed_val, results=results)

    @abstractmethod
    def update(self, value: float, results: Optional[Dict[str, Any]] = None) -> bool:
        """Process the next value in the stream and detect drift.

        Args:
            value (float): A scalar metric from the current data segment
                (e.g. the mean or variance of a window).
            results (Optional[Dict[str, Any]]): Dictionary to update with metrics.

        Returns:
            bool: ``True`` if drift is detected after this update,
                  ``False`` otherwise.
        """

    def batch_update(self, values: Union[np.ndarray, list], results: Optional[Dict[str, Any]] = None) -> bool:
        """Feed a batch of scalar values one-by-one through the process logic.

        Returns ``True`` as soon as drift is detected within the batch
        (short-circuits on the first drift event).

        Args:
            values: 1-D array-like of scalar metrics.
            results (Optional[Dict[str, Any]]): Dictionary to update with metrics.

        Returns:
            bool: ``True`` if drift was detected during this batch.
        """
        with self._callback_context("drift_detection", results):
            for v in np.asarray(values).ravel():
                if self.process(float(v), results=results):
                    return True
            return False

    def reset_filters(self) -> None:
        """Attempt to clear the state of all attached filters."""
        for filter_fn in self.filters:
            if hasattr(filter_fn, "reset"):
                filter_fn.reset()

    @abstractmethod
    def reset(self) -> None:
        """Clear the internal state of the detector.

        Should be called after the model has been retrained on the
        accumulated buffer so that the detector starts fresh on the
        post-drift distribution.
        """

    @property
    @abstractmethod
    def current_width(self) -> int:
        """Return the current window width of the detector.

        This represents the number of observations the detector considers
        representative of the current concept.  Useful for slicing the
        replay buffer after a drift event.
        """
