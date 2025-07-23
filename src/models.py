"""Time series foundation model interfaces and implementations."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import torch
import torch.nn as nn


class TimeSeriesFoundationModel(ABC):
    """Abstract interface for time series foundation models.

    Provides a unified interface for different foundation models such as TimesFM,
    Chronos, and mock implementations. Focuses on feature extraction rather than
    end-to-end training.
    """

    @abstractmethod
    def encode(self, time_series: np.ndarray) -> torch.Tensor:
        """Extract feature representation from time series.

        Args:
            time_series: Input time series data of shape (sequence_length,) or
                        (batch_size, sequence_length)

        Returns:
            Hidden representation tensor of shape (hidden_dim,) or (batch_size, hidden_dim)
        """
        pass

    @abstractmethod
    def get_hidden_dim(self) -> int:
        """Return hidden dimension size.

        Returns:
            Size of the hidden representation vector
        """
        pass


class MockTimeSeriesModel(TimeSeriesFoundationModel):
    """Mock implementation of time series foundation model for testing and development.

    This implementation provides deterministic behavior for testing without requiring
    large model dependencies. Uses a simple linear projection for feature extraction.
    """

    def __init__(self, hidden_dim: int = 256, seed: int = 42):
        """Initialize mock model.

        Args:
            hidden_dim: Dimension of hidden representation
            seed: Random seed for reproducible behavior
        """
        self.hidden_dim = hidden_dim
        self.seed = seed

        # Set random seed for reproducible behavior
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Simple linear projection for feature extraction
        self.projection = nn.Linear(1, hidden_dim)

    def encode(self, time_series: np.ndarray) -> torch.Tensor:
        """Extract features using simple aggregation and projection.

        Args:
            time_series: Input time series of shape (sequence_length,) or
                        (batch_size, sequence_length)

        Returns:
            Hidden representation of shape (hidden_dim,) or (batch_size, hidden_dim)
        """
        # Convert to tensor if needed
        if isinstance(time_series, np.ndarray):
            ts_tensor = torch.from_numpy(time_series).float()
        else:
            ts_tensor = time_series.float()

        # Handle empty series
        if ts_tensor.numel() == 0:
            raise ValueError("Cannot encode empty time series")

        # Handle both single and batch inputs
        if ts_tensor.dim() == 1:
            # Single time series: (sequence_length,)
            # Compute simple statistics as features
            mean_val = ts_tensor.mean().unsqueeze(0)  # (1,)
            features = self.projection(mean_val)  # (hidden_dim,)
            return features
        else:
            # Batch of time series: (batch_size, sequence_length)
            # Compute mean across sequence dimension
            mean_vals = ts_tensor.mean(dim=1, keepdim=True)  # (batch_size, 1)
            features = self.projection(mean_vals)  # (batch_size, hidden_dim)
            return features.squeeze(-1)  # (batch_size, hidden_dim)

    def get_hidden_dim(self) -> int:
        """Return hidden dimension size."""
        return self.hidden_dim

    def get_config(self) -> dict[str, Any]:
        """Get model configuration for serialization."""
        return {
            "hidden_dim": self.hidden_dim,
            "seed": self.seed,
        }
