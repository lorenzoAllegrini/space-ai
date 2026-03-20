"""ADWIN-based concept drift detector using the river library."""

from river import drift  # type: ignore

from .drift_detector import DriftDetector


class ADWINDetector(DriftDetector):
    """Concept drift detector based on the ADWIN algorithm.

    ADWIN (ADaptive WINdowing) maintains a variable-length window of recent
    values and uses a statistical test to detect changes in the mean of the
    stream.  When the two sub-windows differ significantly, drift is flagged.

    Args:
        delta (float): Confidence parameter for the ADWIN test.  Smaller
            values make the detector more conservative (fewer false alarms).
            Default is ``0.002``.
        **kwargs: Additional keyword arguments forwarded to
            :class:`river.drift.ADWIN`.
    """

    def __init__(self, delta: float = 0.002, **kwargs) -> None:
        self.delta = delta
        self._kwargs = kwargs
        self._detector = drift.ADWIN(delta=delta, **kwargs)

    def update(self, value: float) -> bool:
        """Feed a new scalar value and check for drift.

        Args:
            value (float): Scalar metric of the current data segment.

        Returns:
            bool: ``True`` if ADWIN detects a drift after this update.
        """
        self._detector.update(value)
        return self._detector.drift_detected

    @property
    def current_width(self) -> int:
        """Return the ADWIN window width (number of observations in the current concept)."""
        return int(self._detector.width)

    def reset(self) -> None:
        """Re-initialise the ADWIN detector to clear its memory window."""
        self._detector = drift.ADWIN(delta=self.delta, **self._kwargs)
