from typing import Any

import numpy as np

from models.tsf.base import TimeSeriesFoundationModel


class MockTimeSeriesModel(TimeSeriesFoundationModel):
    """Mock implementation of time series foundation model for testing"""

    def __init__(self, config: dict[str, Any]) -> None:
        """
        Initialize mock model

        Args:
            config: Configuration dictionary containing:
                - embedding_dim: Output embedding dimension (default: 256)
                - use_statistical_features: Whether to include statistical features (default: True)
                - random_seed: Random seed for reproducibility (default: 42)
        """
        self.embedding_dim: int = int(config.get("embedding_dim", 256))
        self.use_statistical_features = config.get("use_statistical_features", True)
        self.random_seed = config.get("random_seed", 42)

        # Set random seed for reproducibility
        np.random.seed(self.random_seed)

        # Initialize learned parameters (mock weights)
        self.mock_weights = np.random.randn(10, self.embedding_dim) * 0.1

    def encode(self, time_series: np.ndarray) -> np.ndarray:
        """
        Mock encoding that combines statistical features with random components

        Args:
            time_series: Input time series (batch_size, sequence_length, features)

        Returns:
            embeddings: Mock embeddings (batch_size, embedding_dim)
        """
        batch_size, seq_len, n_features = time_series.shape
        embeddings = np.zeros((batch_size, self.embedding_dim))

        for i in range(batch_size):
            # Extract first feature for statistical computation
            seq = time_series[i, :, 0]

            if self.use_statistical_features:
                # Compute statistical features
                stats = self._compute_statistical_features(seq)

                # Fill first part of embedding with statistical features
                n_stats = min(len(stats), self.embedding_dim)
                embeddings[i, :n_stats] = stats[:n_stats]

                # Fill remaining dimensions with transformed statistical features
                if self.embedding_dim > n_stats:
                    # Use mock weights to transform statistics into higher dimensions
                    remaining_dims = self.embedding_dim - n_stats
                    weights_subset = self.mock_weights[:n_stats, :remaining_dims]
                    transformed = np.dot(stats[:n_stats], weights_subset)
                    embeddings[i, n_stats:] = transformed.flatten()[:remaining_dims]
            else:
                # Pure random embeddings (for baseline comparison)
                embeddings[i, :] = np.random.randn(self.embedding_dim) * 0.1

        # Add small amount of noise for realism
        noise = np.random.randn(*embeddings.shape) * 0.01
        embeddings += noise

        return embeddings

    def _compute_statistical_features(self, sequence: np.ndarray) -> np.ndarray:
        """
        Compute comprehensive statistical features from time series

        Args:
            sequence: 1D time series

        Returns:
            features: Array of statistical features
        """
        # Handle edge cases
        if len(sequence) == 0:
            return np.zeros(10)

        sequence = sequence[~np.isnan(sequence)]  # Remove NaN values
        if len(sequence) == 0:
            return np.zeros(10)

        features = []

        # Basic statistics
        features.append(np.mean(sequence))
        features.append(np.std(sequence))
        features.append(np.min(sequence))
        features.append(np.max(sequence))
        features.append(np.median(sequence))

        # Trend and change features
        if len(sequence) > 1:
            diff = np.diff(sequence)
            features.append(np.mean(diff))  # Average change
            features.append(np.std(diff))  # Volatility

            # Linear trend slope
            x = np.arange(len(sequence))
            trend_slope = np.polyfit(x, sequence, 1)[0]
            features.append(trend_slope)
        else:
            features.extend([0.0, 0.0, 0.0])

        # Percentiles
        features.append(np.percentile(sequence, 25))
        features.append(np.percentile(sequence, 75))

        return np.array(features)

    def get_embedding_dim(self) -> int:
        """Return embedding dimension"""
        return self.embedding_dim

    def load_pretrained(self, model_path: str) -> None:
        """
        Mock pretrained model loading

        Args:
            model_path: Path to pretrained model (not used in mock)
        """
        print(f"Mock: Loading pretrained model from {model_path}")
        # In real implementation, this would load actual model weights
        # For mock, we just reinitialize with different random seed
        np.random.seed(self.random_seed + 1)
        self.mock_weights = np.random.randn(10, self.embedding_dim) * 0.05
