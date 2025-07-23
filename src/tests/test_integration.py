"""Integration tests for the complete time series classification system."""

from datetime import datetime, timedelta

import numpy as np
import pytest

from src.data_structures import WeeklyEventData
from src.models import MockTimeSeriesModel
from src.predictor import DropoutPredictor


class TestSystemIntegration:
    """End-to-end integration tests for the classification system."""

    @pytest.fixture
    def realistic_dataset(self):
        """Create a realistic dataset for testing."""
        np.random.seed(42)

        # Generate data for 50 users over 8 weeks
        data = []
        base_date = datetime(2025, 6, 1)

        for user_id in range(50):
            # Simulate user behavior patterns
            if user_id < 25:  # Retained users - generally positive attendance
                base_attendance = 0.6
                dropout_prob = 0.1
            else:  # At-risk users - generally negative attendance
                base_attendance = -0.4
                dropout_prob = 0.7

            # Decide if user drops out
            drops_out = np.random.random() < dropout_prob

            # Generate weekly data
            for week in range(8):
                # Add some noise to attendance
                noise = np.random.normal(0, 0.2)
                attendance = np.clip(base_attendance + noise, -1, 1)

                # If user drops out, make later weeks worse
                if drops_out and week > 4:
                    attendance -= 0.3 * (week - 4)
                    attendance = np.clip(attendance, -1, 1)

                # Generate utterances (simplified)
                if attendance > 0:
                    utterances = ["Good session", "Helpful content"]
                elif attendance > -0.5:
                    utterances = ["Okay I guess"]
                else:
                    utterances = [] if np.random.random() < 0.7 else ["Struggling"]

                data.append(
                    WeeklyEventData(
                        timestamp=base_date + timedelta(weeks=week),
                        user_attendance=attendance,
                        user_utterances=utterances,
                        dropped_out=drops_out,
                    )
                )

        return data

    def test_complete_pipeline_workflow(self, realistic_dataset):
        """Test the complete workflow from data to predictions."""
        # Split data into train/test
        train_size = int(len(realistic_dataset) * 0.8)
        train_data = realistic_dataset[:train_size]
        test_data = realistic_dataset[train_size:]

        # Initialize components
        foundation_model = MockTimeSeriesModel(hidden_dim=128, seed=42)
        predictor = DropoutPredictor(foundation_model, random_state=42)

        # Train the model
        predictor.fit(train_data)
        assert predictor._is_fitted

        # Make predictions
        probabilities = predictor.predict(test_data)
        binary_predictions = predictor.predict_binary(test_data)

        # Verify prediction shapes and ranges
        assert len(probabilities) == len(test_data)
        assert len(binary_predictions) == len(test_data)
        assert all(0 <= p <= 1 for p in probabilities)
        assert all(pred in [0, 1] for pred in binary_predictions)

        # Evaluate performance
        metrics = predictor.evaluate(test_data)

        # Basic sanity checks on metrics
        assert 0 <= metrics["accuracy"] <= 1
        assert 0 <= metrics["precision"] <= 1
        assert 0 <= metrics["recall"] <= 1
        assert 0 <= metrics["f1"] <= 1
        assert 0 <= metrics["auc"] <= 1

    def test_model_consistency_across_runs(self):
        """Test that the model produces consistent results across multiple runs."""
        # Create small consistent dataset
        data = [
            WeeklyEventData(datetime(2025, 7, 7), 0.8, [], False),
            WeeklyEventData(datetime(2025, 7, 14), -0.8, [], True),
            WeeklyEventData(datetime(2025, 7, 21), 0.6, [], False),
            WeeklyEventData(datetime(2025, 7, 28), -0.6, [], True),
        ]

        # Train multiple models with same configuration
        results = []
        for _ in range(3):
            model = MockTimeSeriesModel(hidden_dim=64, seed=42)
            predictor = DropoutPredictor(model, random_state=42)
            predictor.fit(data)
            probabilities = predictor.predict(data)
            results.append(probabilities)

        # Verify consistency
        for i in range(1, len(results)):
            np.testing.assert_array_almost_equal(results[0], results[i])

    def test_different_model_configurations(self):
        """Test that different model configurations work properly."""
        data = [
            WeeklyEventData(datetime(2025, 7, 7), 0.5, [], False),
            WeeklyEventData(datetime(2025, 7, 14), -0.5, [], True),
        ]

        configurations = [
            {"hidden_dim": 32, "seed": 42},
            {"hidden_dim": 64, "seed": 42},
            {"hidden_dim": 128, "seed": 42},
            {"hidden_dim": 64, "seed": 123},
        ]

        for config in configurations:
            model = MockTimeSeriesModel(**config)
            predictor = DropoutPredictor(model, random_state=42)

            # Should train without errors
            predictor.fit(data)

            # Should predict without errors
            probabilities = predictor.predict(data)
            assert len(probabilities) == len(data)

            # Check feature dimensions
            features = predictor._extract_features(data)
            assert features.shape == (len(data), config["hidden_dim"])

    def test_edge_cases(self):
        """Test system behavior with edge cases."""
        # Minimal dataset with both classes
        minimal_data = [
            WeeklyEventData(datetime(2025, 7, 7), 0.5, [], False),
            WeeklyEventData(datetime(2025, 7, 14), -0.5, [], True),
        ]

        model = MockTimeSeriesModel(hidden_dim=32, seed=42)
        predictor = DropoutPredictor(model, random_state=42)

        # Should handle minimal dataset
        predictor.fit(minimal_data)
        prob = predictor.predict(minimal_data)
        assert len(prob) == 2
        assert all(0 <= p <= 1 for p in prob)

    def test_boundary_attendance_values(self):
        """Test system with boundary attendance values."""
        boundary_data = [
            WeeklyEventData(datetime(2025, 7, 7), -1.0, [], True),  # Minimum
            WeeklyEventData(datetime(2025, 7, 14), 1.0, [], False),  # Maximum
            WeeklyEventData(datetime(2025, 7, 21), 0.0, [], False),  # Zero
            WeeklyEventData(datetime(2025, 7, 28), -0.999, [], True),  # Near minimum
            WeeklyEventData(datetime(2025, 8, 4), 0.999, [], False),  # Near maximum
        ]

        model = MockTimeSeriesModel(hidden_dim=64, seed=42)
        predictor = DropoutPredictor(model, random_state=42)

        # Should handle boundary values without issues
        predictor.fit(boundary_data)
        probabilities = predictor.predict(boundary_data)

        assert len(probabilities) == len(boundary_data)
        assert all(0 <= p <= 1 for p in probabilities)

    def test_performance_with_imbalanced_data(self):
        """Test system performance with imbalanced datasets."""
        # Create highly imbalanced dataset (90% negative, 10% positive)
        imbalanced_data = []
        base_date = datetime(2025, 7, 1)

        # 90 negative examples
        for i in range(90):
            imbalanced_data.append(
                WeeklyEventData(
                    timestamp=base_date + timedelta(days=i),
                    user_attendance=np.random.uniform(0.1, 1.0),
                    user_utterances=[],
                    dropped_out=False,
                )
            )

        # 10 positive examples
        for i in range(10):
            imbalanced_data.append(
                WeeklyEventData(
                    timestamp=base_date + timedelta(days=90 + i),
                    user_attendance=np.random.uniform(-1.0, -0.1),
                    user_utterances=[],
                    dropped_out=True,
                )
            )

        # Shuffle the data
        np.random.shuffle(imbalanced_data)

        model = MockTimeSeriesModel(hidden_dim=64, seed=42)
        predictor = DropoutPredictor(model, random_state=42)

        # Train and evaluate
        predictor.fit(imbalanced_data)
        metrics = predictor.evaluate(imbalanced_data)

        # Should produce valid metrics even with imbalanced data
        assert all(0 <= metrics[metric] <= 1 for metric in ["accuracy", "precision", "recall", "f1"])
        # AUC might be problematic with severe imbalance, but should still be valid
        assert 0 <= metrics["auc"] <= 1

    def test_usage_example_from_docs(self):
        """Test the usage example provided in the documentation."""
        # Create sample data
        training_data = [
            WeeklyEventData(datetime(2025, 7, 7), 0.8, ["Good session"], False),
            WeeklyEventData(datetime(2025, 7, 14), -0.5, [], True),
            WeeklyEventData(datetime(2025, 7, 21), 0.3, ["Okay session"], False),
            WeeklyEventData(datetime(2025, 7, 28), -0.8, ["Struggling"], True),
        ]

        # Initialize components as shown in docs
        foundation_model = MockTimeSeriesModel(hidden_dim=256)
        predictor = DropoutPredictor(foundation_model)

        # Train on weekly event data
        predictor.fit(training_data)

        # Predict dropout probability
        user_time_series = [WeeklyEventData(datetime(2025, 8, 4), 0.1, ["Uncertain"], False)]
        dropout_prob = predictor.predict(user_time_series)

        # Verify the example works as documented
        assert len(dropout_prob) == 1
        assert 0 <= dropout_prob[0] <= 1
        assert isinstance(dropout_prob[0], (float, np.floating))
