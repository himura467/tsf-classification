from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class TimeSeriesFoundationModel(ABC):
    """Abstract base class for time series foundation models"""

    @abstractmethod
    def __init__(self, config: dict[str, Any]) -> None:
        """
        Initialize the model

        Args:
            config: Model configuration dictionary
        """
        raise NotImplementedError("Subclasses must implement __init__ method")

    @abstractmethod
    def encode(self, time_series: np.ndarray) -> np.ndarray:
        """
        Encode time series data into embedding vectors

        Args:
            time_series: Time series data (batch_size, sequence_length, features)

        Returns:
            embeddings: Embedding representations (batch_size, embedding_dim)
        """
        raise NotImplementedError("Subclasses must implement encode method")

    @abstractmethod
    def get_embedding_dim(self) -> int:
        """Get embedding dimension"""
        raise NotImplementedError("Subclasses must implement get_embedding_dim method")

    @abstractmethod
    def load_pretrained(self, model_path: str) -> None:
        """Load pretrained model"""
        raise NotImplementedError("Subclasses must implement load_pretrained method")
