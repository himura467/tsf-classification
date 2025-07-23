"""Tests for dropout prediction pipeline."""

from datetime import datetime

import numpy as np
import pytest

from src.data_structures import WeeklyEventData
from src.models import MockTimeSeriesModel
from src.predictor import DropoutPredictor


class TestDropoutPredictor:
    """Test cases for DropoutPredictor class."""

    @pytest.fixture
    def mock_model(self):
        """Fixture providing a mock time series model."""
        return MockTimeSeriesModel(hidden_dim=32, seed=42)

    @pytest.fixture
    def sample_training_data(self):
        """Fixture providing sample training data."""
        return [
            WeeklyEventData(
                timestamp=datetime(2025, 7, 7), user_attendance=0.8, user_utterances=["Good session"], dropped_out=False
            ),
            WeeklyEventData(
                timestamp=datetime(2025, 7, 14), user_attendance=-0.5, user_utterances=[], dropped_out=True
            ),
            WeeklyEventData(
                timestamp=datetime(2025, 7, 21),
                user_attendance=0.3,
                user_utterances=["Okay session"],
                dropped_out=False,
            ),
            WeeklyEventData(
                timestamp=datetime(2025, 7, 28), user_attendance=-0.8, user_utterances=["Struggling"], dropped_out=True
            ),
        ]

    def test_initialization(self, mock_model):
        """Test proper initialization of DropoutPredictor."""
        predictor = DropoutPredictor(mock_model, random_state=42)

        assert predictor.foundation_model == mock_model
        assert predictor.random_state == 42
        assert not predictor._is_fitted

    def test_initialization_with_classifier_kwargs(self, mock_model):
        """Test initialization with additional classifier arguments."""
        predictor = DropoutPredictor(mock_model, random_state=42, max_iter=1000, C=2.0)

        assert predictor.classifier.max_iter == 1000
        assert predictor.classifier.C == 2.0

    def test_fit_with_valid_data(self, mock_model, sample_training_data):
        """Test fitting the predictor with valid training data."""
        predictor = DropoutPredictor(mock_model, random_state=42)

        result = predictor.fit(sample_training_data)

        assert result is predictor  # Should return self
        assert predictor._is_fitted

    def test_fit_with_empty_data(self, mock_model):
        """Test that fitting with empty data raises ValueError."""
        predictor = DropoutPredictor(mock_model, random_state=42)

        with pytest.raises(ValueError, match="Training data cannot be empty"):
            predictor.fit([])

    def test_extract_features(self, mock_model, sample_training_data):
        """Test feature extraction from training data."""
        predictor = DropoutPredictor(mock_model, random_state=42)

        features = predictor._extract_features(sample_training_data)

        assert features.shape == (len(sample_training_data), mock_model.get_hidden_dim())
        assert isinstance(features, np.ndarray)
        assert not np.isnan(features).any()

    def test_extract_labels(self, mock_model, sample_training_data):
        """Test label extraction from training data."""
        predictor = DropoutPredictor(mock_model, random_state=42)

        labels = predictor._extract_labels(sample_training_data)
        expected_labels = np.array([0, 1, 0, 1])  # Based on sample data

        assert np.array_equal(labels, expected_labels)
        assert labels.dtype == np.int64 or labels.dtype == np.int32

    def test_predict_without_fitting(self, mock_model, sample_training_data):
        """Test that prediction without fitting raises ValueError."""
        predictor = DropoutPredictor(mock_model, random_state=42)

        with pytest.raises(ValueError, match="Model must be fitted before prediction"):
            predictor.predict(sample_training_data)

    def test_predict_with_empty_data(self, mock_model, sample_training_data):
        """Test prediction with empty data raises ValueError."""
        predictor = DropoutPredictor(mock_model, random_state=42)
        predictor.fit(sample_training_data)

        with pytest.raises(ValueError, match="Input data cannot be empty"):
            predictor.predict([])

    def test_predict_probabilities(self, mock_model, sample_training_data):
        """Test probability prediction after fitting."""
        predictor = DropoutPredictor(mock_model, random_state=42)
        predictor.fit(sample_training_data)

        probabilities = predictor.predict(sample_training_data)

        assert len(probabilities) == len(sample_training_data)
        assert all(0 <= p <= 1 for p in probabilities)
        assert isinstance(probabilities, np.ndarray)

    def test_predict_binary(self, mock_model, sample_training_data):
        """Test binary prediction after fitting."""
        predictor = DropoutPredictor(mock_model, random_state=42)
        predictor.fit(sample_training_data)

        predictions = predictor.predict_binary(sample_training_data)

        assert len(predictions) == len(sample_training_data)
        assert all(pred in [0, 1] for pred in predictions)
        assert isinstance(predictions, np.ndarray)

    def test_evaluate_with_fitted_model(self, mock_model, sample_training_data):
        """Test model evaluation with fitted model."""
        predictor = DropoutPredictor(mock_model, random_state=42)
        predictor.fit(sample_training_data)

        metrics = predictor.evaluate(sample_training_data)

        expected_metrics = ["accuracy", "precision", "recall", "f1", "auc"]
        for metric in expected_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], (int, float))
            assert 0 <= metrics[metric] <= 1

    def test_evaluate_without_fitting(self, mock_model, sample_training_data):
        """Test that evaluation without fitting raises ValueError."""
        predictor = DropoutPredictor(mock_model, random_state=42)

        with pytest.raises(ValueError, match="Model must be fitted before evaluation"):
            predictor.evaluate(sample_training_data)

    def test_evaluate_with_predictions(self, mock_model, sample_training_data):
        """Test evaluation with return_predictions=True."""
        predictor = DropoutPredictor(mock_model, random_state=42)
        predictor.fit(sample_training_data)

        metrics = predictor.evaluate(sample_training_data, return_predictions=True)

        assert "predictions" in metrics
        predictions = metrics["predictions"]
        assert "y_true" in predictions
        assert "y_pred" in predictions
        assert "y_proba" in predictions

    def test_get_feature_importance(self, mock_model, sample_training_data):
        """Test feature importance extraction."""
        predictor = DropoutPredictor(mock_model, random_state=42)
        predictor.fit(sample_training_data)

        importance = predictor.get_feature_importance()

        assert importance is not None
        assert len(importance) == mock_model.get_hidden_dim()
        assert isinstance(importance, np.ndarray)

    def test_get_feature_importance_without_fitting(self, mock_model):
        """Test that getting feature importance without fitting raises ValueError."""
        predictor = DropoutPredictor(mock_model, random_state=42)

        with pytest.raises(ValueError, match="Model must be fitted before getting feature importance"):
            predictor.get_feature_importance()

    def test_get_config(self, mock_model):
        """Test configuration retrieval."""
        predictor = DropoutPredictor(mock_model, random_state=123)

        config = predictor.get_config()

        assert config["random_state"] == 123
        assert config["is_fitted"] is False
        assert isinstance(config, dict)

    def test_get_config_after_fitting(self, mock_model, sample_training_data):
        """Test configuration after fitting."""
        predictor = DropoutPredictor(mock_model, random_state=42)
        predictor.fit(sample_training_data)

        config = predictor.get_config()

        assert config["is_fitted"] is True

    def test_reproducibility(self, sample_training_data):
        """Test that results are reproducible with same random state."""
        model1 = MockTimeSeriesModel(hidden_dim=32, seed=42)
        model2 = MockTimeSeriesModel(hidden_dim=32, seed=42)

        predictor1 = DropoutPredictor(model1, random_state=42)
        predictor2 = DropoutPredictor(model2, random_state=42)

        predictor1.fit(sample_training_data)
        predictor2.fit(sample_training_data)

        pred1 = predictor1.predict(sample_training_data)
        pred2 = predictor2.predict(sample_training_data)

        np.testing.assert_array_almost_equal(pred1, pred2)

    def test_pipeline_integration(self, mock_model):
        """Test end-to-end pipeline integration."""
        # Create training data with clear patterns
        training_data = [
            WeeklyEventData(datetime(2025, 7, 7), 0.8, [], False),
            WeeklyEventData(datetime(2025, 7, 14), 0.9, [], False),
            WeeklyEventData(datetime(2025, 7, 21), -0.8, [], True),
            WeeklyEventData(datetime(2025, 7, 28), -0.9, [], True),
        ]

        # Test data
        test_data = [
            WeeklyEventData(datetime(2025, 8, 4), 0.7, [], False),
            WeeklyEventData(datetime(2025, 8, 11), -0.7, [], True),
        ]

        # Train and predict
        predictor = DropoutPredictor(mock_model, random_state=42)
        predictor.fit(training_data)

        probabilities = predictor.predict(test_data)
        binary_predictions = predictor.predict_binary(test_data)
        metrics = predictor.evaluate(test_data)

        # Verify outputs
        assert len(probabilities) == 2
        assert len(binary_predictions) == 2
        assert all(key in metrics for key in ["accuracy", "precision", "recall", "f1", "auc"])
