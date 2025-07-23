"""Tests for data structures module."""

from datetime import datetime

from src.data_structures import WeeklyEventData


class TestWeeklyEventData:
    """Test cases for WeeklyEventData dataclass."""

    def test_valid_initialization(self):
        """Test valid initialization of WeeklyEventData."""
        data = WeeklyEventData(
            timestamp=datetime(2025, 7, 14),
            user_attendance=0.5,
            user_utterances=["Hello", "How are you?"],
            dropped_out=False,
        )

        assert data.timestamp == datetime(2025, 7, 14)
        assert data.user_attendance == 0.5
        assert data.user_utterances == ["Hello", "How are you?"]
        assert data.dropped_out is False

    def test_empty_utterances(self):
        """Test that empty utterances list is valid."""
        data = WeeklyEventData(
            timestamp=datetime(2025, 7, 14), user_attendance=0.0, user_utterances=[], dropped_out=True
        )
        assert data.user_utterances == []

    def test_multiple_utterances(self):
        """Test handling of multiple utterances."""
        utterances = ["First message", "Second message", "Third message"]
        data = WeeklyEventData(
            timestamp=datetime(2025, 7, 14), user_attendance=-0.3, user_utterances=utterances, dropped_out=False
        )
        assert data.user_utterances == utterances
        assert len(data.user_utterances) == 3

    def test_boolean_dropout_labels(self):
        """Test both True and False dropout labels."""
        for dropout_status in [True, False]:
            data = WeeklyEventData(
                timestamp=datetime(2025, 7, 14), user_attendance=0.0, user_utterances=[], dropped_out=dropout_status
            )
            assert data.dropped_out is dropout_status
