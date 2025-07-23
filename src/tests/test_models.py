"""Tests for time series foundation models."""

import numpy as np
import pytest
import torch

from src.models import MockTimeSeriesModel, TimeSeriesFoundationModel


class TestTimeSeriesFoundationModelInterface:
    """Test the abstract interface compliance."""

    def test_abstract_interface(self):
        """Test that TimeSeriesFoundationModel cannot be instantiated."""
        with pytest.raises(TypeError):
            TimeSeriesFoundationModel()


class TestMockTimeSeriesModel:
    """Test cases for MockTimeSeriesModel implementation."""

    def test_initialization(self):
        """Test proper initialization of MockTimeSeriesModel."""
        model = MockTimeSeriesModel(hidden_dim=128, seed=16)
        assert model.get_hidden_dim() == 128
        assert model.seed == 16

    def test_default_parameters(self):
        """Test default parameter values."""
        model = MockTimeSeriesModel()
        assert model.get_hidden_dim() == 256
        assert model.seed == 42

    def test_encode_single_time_series(self):
        """Test encoding of single time series."""
        model = MockTimeSeriesModel(hidden_dim=64, seed=42)
        time_series = np.array([0.5, -0.3, 0.8, -0.1])

        features = model.encode(time_series)

        assert isinstance(features, torch.Tensor)
        assert features.shape == (64,)
        assert not torch.isnan(features).any()

    def test_encode_batch_time_series(self):
        """Test encoding of batch of time series."""
        model = MockTimeSeriesModel(hidden_dim=32, seed=42)
        batch_size = 3
        sequence_length = 5
        time_series_batch = np.random.rand(batch_size, sequence_length)

        features = model.encode(time_series_batch)

        assert isinstance(features, torch.Tensor)
        assert features.shape == (batch_size, 32)
        assert not torch.isnan(features).any()

    def test_reproducibility(self):
        """Test that results are reproducible with same seed."""
        model1 = MockTimeSeriesModel(hidden_dim=64, seed=42)
        model2 = MockTimeSeriesModel(hidden_dim=64, seed=42)

        time_series = np.array([0.1, 0.2, 0.3])

        features1 = model1.encode(time_series)
        features2 = model2.encode(time_series)

        assert torch.allclose(features1, features2)

    def test_different_seeds_produce_different_results(self):
        """Test that different seeds produce different results."""
        model1 = MockTimeSeriesModel(hidden_dim=64, seed=42)
        model2 = MockTimeSeriesModel(hidden_dim=64, seed=123)

        time_series = np.array([0.1, 0.2, 0.3])

        features1 = model1.encode(time_series)
        features2 = model2.encode(time_series)

        assert not torch.allclose(features1, features2)

    def test_encode_torch_tensor_input(self):
        """Test encoding with torch tensor input."""
        model = MockTimeSeriesModel(hidden_dim=32, seed=42)
        time_series = torch.tensor([0.5, -0.3, 0.8])

        features = model.encode(time_series)

        assert isinstance(features, torch.Tensor)
        assert features.shape == (32,)

    def test_encode_empty_series(self):
        """Test behavior with empty time series."""
        model = MockTimeSeriesModel(hidden_dim=16, seed=42)
        empty_series = np.array([])

        # This should handle gracefully or raise appropriate error
        with pytest.raises((ValueError, RuntimeError)):
            model.encode(empty_series)

    def test_encode_single_value_series(self):
        """Test encoding of single-value time series."""
        model = MockTimeSeriesModel(hidden_dim=32, seed=42)
        single_value = np.array([0.7])

        features = model.encode(single_value)

        assert isinstance(features, torch.Tensor)
        assert features.shape == (32,)

    def test_get_config(self):
        """Test configuration retrieval."""
        hidden_dim = 128
        seed = 123
        model = MockTimeSeriesModel(hidden_dim=hidden_dim, seed=seed)

        config = model.get_config()

        assert config["hidden_dim"] == hidden_dim
        assert config["seed"] == seed
        assert isinstance(config, dict)

    def test_feature_consistency_across_calls(self):
        """Test that multiple calls with same input produce same output."""
        model = MockTimeSeriesModel(hidden_dim=64, seed=42)
        time_series = np.array([0.1, 0.5, -0.2, 0.8])

        features1 = model.encode(time_series)
        features2 = model.encode(time_series)

        assert torch.allclose(features1, features2)

    def test_different_hidden_dimensions(self):
        """Test models with different hidden dimensions."""
        dimensions = [16, 32, 64, 128, 256, 512]
        time_series = np.array([0.1, 0.2, 0.3])

        for dim in dimensions:
            model = MockTimeSeriesModel(hidden_dim=dim, seed=42)
            features = model.encode(time_series)

            assert features.shape == (dim,)
            assert model.get_hidden_dim() == dim
