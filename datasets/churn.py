from datetime import datetime

import numpy as np
import torch
from torch.utils.data import Dataset

from models.base import TimeSeriesFoundationModel


class ChurnDataset(Dataset[dict[str, torch.Tensor]]):
    """Dataset for churn prediction"""

    def __init__(
        self,
        timestamps: list[datetime],
        attendance_scores: list[list[float]],
        churn_flags: list[bool],
        foundation_model: TimeSeriesFoundationModel,
        sequence_length: int,
    ):
        """
        Initialize dataset

        Args:
            timestamps: List of datetime objects
            attendance_scores: Attendance scores (normalized to -1, 1)
            churn_flags: Churn flags (boolean values)
            foundation_model: Time series foundation model to use
            sequence_length: Sequence length (number of weeks)
        """
        self.timestamps = timestamps
        self.attendance_scores = attendance_scores
        self.churn_flags = churn_flags
        self.foundation_model = foundation_model
        self.sequence_length = sequence_length

    def __len__(self) -> int:
        return len(self.churn_flags)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        # Prepare time series data
        attendance_seq = np.array(self.attendance_scores[idx])

        # Adjust sequence length (padding or truncation)
        if len(attendance_seq) < self.sequence_length:
            # Padding
            padding = np.zeros(self.sequence_length - len(attendance_seq))
            attendance_seq = np.concatenate([attendance_seq, padding])
        elif len(attendance_seq) > self.sequence_length:
            # Truncation (use most recent data)
            attendance_seq = attendance_seq[-self.sequence_length :]

        # Encode time series with foundation model
        # Convert input shape to (1, sequence_length, 1)
        time_series_input = attendance_seq.reshape(1, -1, 1)
        embeddings = self.foundation_model.encode(time_series_input)

        return {
            "embeddings": torch.FloatTensor(embeddings[0]),  # (embedding_dim,)
            "flag": torch.BoolTensor([self.churn_flags[idx]])[0].float(),  # Convert bool to float
        }
