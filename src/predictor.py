"""Dropout prediction pipeline using time series foundation models."""

from typing import Any

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from src.data_structures import WeeklyEventData
from src.models import TimeSeriesFoundationModel


class DropoutPredictor:
    """Classification pipeline for predicting user dropout using time series foundation models.

    This class implements a two-stage approach:
    1. Feature extraction using a pre-trained time series foundation model
    2. Binary classification using logistic regression on extracted features

    The "last-hidden" strategy compresses the entire time series sequence into a single
    vector representation suitable for downstream classification.
    """

    def __init__(self, foundation_model: TimeSeriesFoundationModel, random_state: int = 42, **classifier_kwargs: Any):
        """Initialize the dropout predictor.

        Args:
            foundation_model: Pre-trained time series foundation model for feature extraction
            random_state: Random seed for reproducible results
            **classifier_kwargs: Additional arguments for LogisticRegression
        """
        self.foundation_model = foundation_model
        self.random_state = random_state

        # Initialize logistic regression classifier
        self.classifier = LogisticRegression(random_state=random_state, **classifier_kwargs)

        self._is_fitted = False

    def _extract_features(self, weekly_data: list[WeeklyEventData]) -> np.ndarray:
        """Extract features from weekly event data using foundation model.

        Args:
            weekly_data: List of weekly event data for users

        Returns:
            Feature matrix of shape (n_users, hidden_dim)
        """
        features = []

        for user_data in weekly_data:
            # Convert attendance data to time series
            time_series = np.array([user_data.user_attendance])

            # Extract features using foundation model
            with torch.no_grad():
                user_features = self.foundation_model.encode(time_series)

            # Convert to numpy if needed
            if isinstance(user_features, torch.Tensor):
                user_features = user_features.cpu().numpy()

            features.append(user_features)

        return np.array(features)

    def _extract_labels(self, weekly_data: list[WeeklyEventData]) -> np.ndarray:
        """Extract binary labels from weekly event data.

        Args:
            weekly_data: List of weekly event data

        Returns:
            Binary labels array of shape (n_users,)
        """
        return np.array([int(data.dropped_out) for data in weekly_data])

    def fit(self, training_data: list[WeeklyEventData]) -> "DropoutPredictor":
        """Train the dropout predictor on weekly event data.

        Args:
            training_data: List of weekly event data for training

        Returns:
            Self for method chaining
        """
        if not training_data:
            raise ValueError("Training data cannot be empty")

        # Extract features and labels
        X = self._extract_features(training_data)
        y = self._extract_labels(training_data)

        # Train classifier
        self.classifier.fit(X, y)
        self._is_fitted = True

        return self

    def predict(self, time_series_data: list[WeeklyEventData]) -> np.ndarray:
        """Predict dropout probabilities for new data.

        Args:
            time_series_data: List of weekly event data for prediction

        Returns:
            Dropout probabilities array of shape (n_users,)
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted before prediction")

        if not time_series_data:
            raise ValueError("Input data cannot be empty")

        # Extract features
        X = self._extract_features(time_series_data)

        # Predict probabilities
        proba = self.classifier.predict_proba(X)

        # Return probability of positive class (dropout)
        return proba[:, 1]

    def predict_binary(self, time_series_data: list[WeeklyEventData]) -> np.ndarray:
        """Predict binary dropout labels for new data.

        Args:
            time_series_data: List of weekly event data for prediction

        Returns:
            Binary predictions array of shape (n_users,)
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted before prediction")

        if not time_series_data:
            raise ValueError("Input data cannot be empty")

        # Extract features
        X = self._extract_features(time_series_data)

        # Predict binary labels
        return self.classifier.predict(X)

    def evaluate(self, test_data: list[WeeklyEventData], return_predictions: bool = False) -> dict[str, float]:
        """Evaluate model performance on test data.

        Args:
            test_data: List of weekly event data for evaluation
            return_predictions: Whether to return predictions along with metrics

        Returns:
            Dictionary containing evaluation metrics
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted before evaluation")

        # Get true labels
        y_true = self._extract_labels(test_data)

        # Get predictions
        y_proba = self.predict(test_data)
        y_pred = self.predict_binary(test_data)

        # Calculate metrics
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "auc": roc_auc_score(y_true, y_proba) if len(np.unique(y_true)) > 1 else 0.0,
        }

        if return_predictions:
            metrics["predictions"] = {
                "y_true": y_true,
                "y_pred": y_pred,
                "y_proba": y_proba,
            }

        return metrics

    def get_feature_importance(self) -> np.ndarray | None:
        """Get feature importance from the trained classifier.

        Returns:
            Feature importance array if available, None otherwise
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted before getting feature importance")

        # For logistic regression, coefficients represent feature importance
        return self.classifier.coef_[0] if hasattr(self.classifier, "coef_") else None

    def get_config(self) -> dict[str, Any]:
        """Get predictor configuration for serialization.

        Returns:
            Configuration dictionary
        """
        return {
            "foundation_model_config": self.foundation_model.get_config()
            if hasattr(self.foundation_model, "get_config")
            else None,
            "random_state": self.random_state,
            "is_fitted": self._is_fitted,
        }
