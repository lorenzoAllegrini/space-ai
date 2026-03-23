"""Filters for anomaly scores and drift detection inputs."""

import collections
from typing import Optional


class SafeRampUpFilter:
    """Injectable logic that implements a 'waiting room' for incoming values.
    
    Retains incoming values and returns the oldest value only if it does not
    part of an anomalous ramp-up (i.e. if the future scores in the buffer do
    not grow beyond a safe threshold compared to the oldest score).
    
    This is used to prevent the drift detector from reacting immediately to
    anomalous spikes that are just starting to form, ensuring only "stable"
    clean data is considered.

    Args:
        lookahead_steps (int): How many future steps to wait before validating
            a score. Default is 3.
        max_safe_score (float): The maximum value for the oldest score to be
            even considered for validation. If the score itself is already
            high, it's rejected. Default is 0.4.
        max_growth (float): The maximum allowed growth between the oldest score
            and any of the future scores in the waiting room. Default is 0.05.
    """

    def __init__(self, lookahead_steps: int = 3, max_safe_score: float = 0.4, max_growth: float = 0.05):
        self.lookahead_steps = lookahead_steps
        self.max_safe_score = max_safe_score
        self.max_growth = max_growth
        self.queue = collections.deque(maxlen=lookahead_steps + 1)

    def __call__(self, value: float) -> Optional[float]:
        """Process a new value and potentially release an older validated value.
        
        Args:
            value (float): The new scalar value to process.
            
        Returns:
            Optional[float]: The oldest validated value if it passes the safety
                checks, otherwise ``None``. Also returns ``None`` if the waiting
                room is not yet full.
        """
        self.queue.append(value)
        
        # We haven't seen enough "future" yet to make a decision
        if len(self.queue) < self.lookahead_steps + 1:
            return None
            
        oldest_score = self.queue[0]
        future_scores = list(self.queue)[1:]
        
        # The mathematical filter logic
        if oldest_score < self.max_safe_score and (max(future_scores) - oldest_score) <= self.max_growth:
            return oldest_score  # Validated!
            
        return None  # Rejected!
        
    def reset(self) -> None:
        """Clear the waiting room queue."""
        self.queue.clear()
