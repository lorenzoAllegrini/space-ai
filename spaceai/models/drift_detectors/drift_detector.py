"""Abstract base class for concept drift detectors."""

from abc import ABC, abstractmethod
from typing import Callable, Iterable, Optional, Union

import numpy as np


class DriftDetector(ABC):
    """Abstract interface for concept drift detectors.

    Implementations must monitor an incoming stream of scalar values and
    signal whenever a distribution shift (drift) is detected.

    Args:
        filters (Optional[Iterable[Callable]]): A sequence of callables to
            pre-filter values before they reach the detector logic. Each
            callable must take a scalar as input and return either a
            filtered scalar or ``None`` to reject the value.
    """

    def __init__(self, filters: Optional[Iterable[Callable]] = None) -> None:
        self.filters = list(filters) if filters is not None else []

    def process(self, value: float) -> bool:
        """Run standard pre-filters and, if passed, update the detector.

        Args:
            value (float): A raw scalar metric from the stream.

        Returns:
            bool: ``True`` if the value was accepted by all filters AND 
                  the underlying detector found drift.
        """
        processed_val: Optional[float] = value
        for filter_fn in self.filters:
            processed_val = filter_fn(processed_val)
            if processed_val is None:
                return False

        return self.update(processed_val)

    @abstractmethod
    def update(self, value: float) -> bool:
        """Process the next value in the stream and detect drift.

        Args:
            value (float): A scalar metric from the current data segment
                (e.g. the mean or variance of a window).

        Returns:
            bool: ``True`` if drift is detected after this update,
                  ``False`` otherwise.
        """

    def batch_update(self, values: Union[np.ndarray, list]) -> bool:
        """Feed a batch of scalar values one-by-one through the process logic.

        Returns ``True`` as soon as drift is detected within the batch
        (short-circuits on the first drift event).

        Args:
            values: 1-D array-like of scalar metrics.

        Returns:
            bool: ``True`` if drift was detected during this batch.
        """
        for v in np.asarray(values).ravel():
            if self.process(float(v)):
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
