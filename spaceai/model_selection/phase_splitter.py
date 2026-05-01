"""Modular N-way phase splitter for AnomalyDetectionPipeline.

Given a dictionary of phase names → percentages, splits the training data
into a train portion and N held-out portions that are injected sequentially
via repeated ``switch_phase`` calls.

Example
-------
>>> splitter = PhaseSplitter(phases={"calibration": 0.2, "validation": 0.2})
# During "train" phase, call method="split":
#   keeps first 60% as train, stores calibration (20%) and validation (20%).
# During "calibration" phase, call method="switch_phase":
#   injects the calibration data (phase 0).
# During "validation" phase, call method="switch_phase":
#   injects the validation data (phase 1).
"""

from typing import Optional, Any, Dict, List, TYPE_CHECKING
from collections import OrderedDict
import numpy as np
import logging
from spaceai.benchmark.callbacks.mixin import CallbackMixin
from spaceai.data import AnomalyDataset, AnomalyDatasetSubset

if TYPE_CHECKING:
    from spaceai.models.anomaly_pipeline.anomaly_classifier import PipelineState


class PhaseSplitter(CallbackMixin):
    """Pipeline node that splits data into an arbitrary number of phases.

    Parameters
    ----------
    phases : dict[str, float]
        Mapping of phase name → fraction of *total* data to reserve.
        Fractions are taken from the **tail** of the time series in the
        order they appear.  The remainder (1 − Σfractions) stays as
        training data.
    callback_handler : optional
        Callback handler forwarded to ``CallbackMixin``.

    Raises
    ------
    ValueError
        If the sum of the fractions is ≥ 1.0 or any fraction is ≤ 0.
    """

    def __init__(
        self,
        phases: Dict[str, float],
        callback_handler=None,
    ):
        super().__init__(callback_handler)

        total = sum(phases.values())
        if total >= 1.0:
            raise ValueError(
                f"Sum of phase fractions must be < 1.0, got {total:.4f}"
            )
        for name, frac in phases.items():
            if frac <= 0:
                raise ValueError(
                    f"Phase '{name}' has non-positive fraction {frac}"
                )

        # Use OrderedDict to guarantee deterministic split order
        self.phases: OrderedDict[str, float] = OrderedDict(phases)
        # Stored phase data (populated by split, consumed by switch_phase)
        self._stored: List[Dict[str, Any]] = []
        self._phase_names: List[str] = list(phases.keys())
        self._current_phase_idx: int = -1

    # ------------------------------------------------------------------
    # Pipeline interface
    # ------------------------------------------------------------------

    def pipeline_step(
        self,
        state: "PipelineState",
        is_fit: bool = False,
        results: Optional[Dict[str, Any]] = None,
        method_name: Optional[str] = None,
        **kwargs,
    ) -> "PipelineState":
        """Dispatch to ``split`` or ``switch_phase`` based on *method_name*."""

        if method_name == "split":
            return self._do_split(state)

        if method_name == "switch_phase":
            return self._do_switch(state)

        # Fallback (passthrough)
        return super().pipeline_step(
            state, is_fit, results, method_name, **kwargs
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _do_split(self, state: "PipelineState") -> "PipelineState":
        """Split *state.data* into train + N held-out portions."""
        X = state.data
        y = state.labels
        n_samples = len(X) if hasattr(X, "__len__") else 0

        # Reset counter
        self._current_phase_idx = -1
        self._stored = []

        if n_samples == 0:
            for _ in self.phases:
                self._stored.append({
                    "data": None, "labels": None,
                    "indices": None, "intervals": None,
                })
            return state

        # Compute absolute split boundaries (from the tail)
        boundaries: List[int] = []
        cursor = n_samples
        for name in reversed(self._phase_names):
            frac = self.phases[name]
            size = int(n_samples * frac)
            cursor -= size
            boundaries.append(cursor)
        boundaries.reverse()  # now in original phase order

        train_end = boundaries[0]

        # Store each held-out phase
        for i, name in enumerate(self._phase_names):
            start = boundaries[i]
            end = boundaries[i + 1] if i + 1 < len(self._phase_names) else n_samples

            phase_data = self._slice_data(X, start, end, n_samples)
            phase_labels = y[start:end] if y is not None else None
            phase_indices = (
                state.indices[start:end] if state.indices is not None else None
            )
            phase_intervals = (
                state.intervals[start:end]
                if state.intervals is not None
                else None
            )

            self._stored.append({
                "data": phase_data,
                "labels": phase_labels,
                "indices": phase_indices,
                "intervals": phase_intervals,
            })

        logging.debug(
            "[PhaseSplitter] Split %d samples → train=%d, %s",
            n_samples, train_end,
            ", ".join(
                f"{name}={len(self._stored[i]['data']) if self._stored[i]['data'] is not None else 0}"
                for i, name in enumerate(self._phase_names)
            ),
        )

        # Trim state to the training portion
        state.data = self._slice_data(X, 0, train_end, n_samples)
        state.labels = y[:train_end] if y is not None else None
        if state.indices is not None:
            state.indices = state.indices[:train_end]
        if state.intervals is not None:
            state.intervals = state.intervals[:train_end]

        return state

    def _do_switch(self, state: "PipelineState") -> "PipelineState":
        """Advance to the next stored phase and inject its data."""
        self._current_phase_idx += 1

        if self._current_phase_idx >= len(self._stored):
            raise ValueError(
                f"No more phases to switch to. "
                f"Called switch_phase {self._current_phase_idx + 1} times "
                f"but only {len(self._stored)} phases were defined: "
                f"{self._phase_names}"
            )

        stored = self._stored[self._current_phase_idx]
        phase_name = self._phase_names[self._current_phase_idx]

        logging.debug(
            "[PhaseSplitter] Switching to phase '%s' (%d/%d)",
            phase_name, self._current_phase_idx + 1, len(self._stored),
        )

        state.data = stored["data"]
        state.labels = stored["labels"]
        state.indices = stored["indices"]
        state.intervals = stored["intervals"]
        return state

    @staticmethod
    def _slice_data(X: Any, start: int, end: int, n_samples: int) -> Any:
        """Slice data supporting both ndarray and AnomalyDataset."""
        if isinstance(X, AnomalyDataset):
            return AnomalyDatasetSubset(parent=X, start_idx=start, end_idx=end)
        return X[start:end]


__all__ = ["PhaseSplitter"]
