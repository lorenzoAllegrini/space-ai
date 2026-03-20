"""Abstract base class for concept drift detectors."""

from abc import ABC, abstractmethod
from typing import Union

import numpy as np


class DriftDetector(ABC):
    """Abstract interface for concept drift detectors.

    Implementations must monitor an incoming stream of scalar values and
    signal whenever a distribution shift (drift) is detected.
    """

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

    def update_batch(self, values: Union[np.ndarray, list]) -> bool:
        """Feed a batch of scalar values one-by-one.

        Returns ``True`` as soon as drift is detected within the batch
        (short-circuits on the first drift event).

        Args:
            values: 1-D array-like of scalar metrics.

        Returns:
            bool: ``True`` if drift was detected during this batch.
        """
        for v in np.asarray(values).ravel():
            if self.update(float(v)):
                return True
        return False

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
