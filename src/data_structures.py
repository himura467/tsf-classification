"""Core data structures for time series classification."""

from dataclasses import dataclass
from datetime import datetime


@dataclass
class WeeklyEventData:
    """Data structure for weekly event data with attendance and dropout information.

    Attributes:
        user_id: Unique identifier for the user
        timestamp: Week identifier (e.g., 2025/07/14)
        user_attendance: Normalized attendance score (-1, 1)
        user_utterances: Speech data for the week
        dropped_out: Binary label (True: user has already dropped out, False: user is still active)
    """

    user_id: str
    timestamp: datetime
    user_attendance: float
    user_utterances: list[str]
    dropped_out: bool
