"""Dropout prediction pipeline using time series foundation models."""

from collections import defaultdict
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

    The system groups weekly data by user_id to create user-level time series,
    then uses a "last-hidden" strategy to compress each user's time series sequence
    into a single vector representation for dropout prediction.
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

    def _group_data_by_user(self, weekly_data: list[WeeklyEventData]) -> dict[str, list[WeeklyEventData]]:
        """Group weekly event data by user_id.

        Args:
            weekly_data: List of weekly event data

        Returns:
            Dictionary mapping user_id to list of their weekly data, sorted by timestamp
        """
        user_data = defaultdict(list)
        for data in weekly_data:
            user_data[data.user_id].append(data)

        # Sort each user's data by timestamp
        for user_id in user_data:
            user_data[user_id].sort(key=lambda x: x.timestamp)

        return dict(user_data)

    def _extract_user_features(self, user_weekly_data: list[WeeklyEventData]) -> np.ndarray:
        """Extract features for a single user from their time series data.

        Args:
            user_weekly_data: List of weekly data for a single user, sorted by timestamp

        Returns:
            Feature vector of shape (hidden_dim,)
        """
        # Create time series from user's attendance data
        attendance_series = np.array([data.user_attendance for data in user_weekly_data])

        # Extract features using foundation model
        with torch.no_grad():
            user_features = self.foundation_model.encode(attendance_series)

        # Convert to numpy if needed
        if isinstance(user_features, torch.Tensor):
            user_features = user_features.cpu().numpy()

        return user_features

    def _extract_features(self, weekly_data: list[WeeklyEventData]) -> tuple[np.ndarray, list[str]]:
        """Extract features from weekly event data using foundation model.

        Args:
            weekly_data: List of weekly event data for multiple users

        Returns:
            Tuple of (feature matrix of shape (n_users, hidden_dim), list of user_ids)
        """
        # Group data by user
        user_data = self._group_data_by_user(weekly_data)

        features = []
        user_ids = []

        for user_id, user_weekly_data in user_data.items():
            user_features = self._extract_user_features(user_weekly_data)
            features.append(user_features)
            user_ids.append(user_id)

        return np.array(features), user_ids

    def _extract_labels(self, weekly_data: list[WeeklyEventData]) -> tuple[np.ndarray, list[str]]:
        """Extract binary labels from weekly event data at user level.

        Args:
            weekly_data: List of weekly event data for multiple users

        Returns:
            Tuple of (binary labels array of shape (n_users,), list of user_ids)
        """
        # Group data by user
        user_data = self._group_data_by_user(weekly_data)

        labels = []
        user_ids = []

        for user_id, user_weekly_data in user_data.items():
            # Use the dropout status from the latest week (or any week since it should be consistent)
            user_label = int(user_weekly_data[-1].dropped_out)
            labels.append(user_label)
            user_ids.append(user_id)

        return np.array(labels), user_ids

    def fit(self, training_data: list[WeeklyEventData]) -> "DropoutPredictor":
        """Train the dropout predictor on weekly event data.

        Args:
            training_data: List of weekly event data for training

        Returns:
            Self for method chaining
        """
        if not training_data:
            raise ValueError("Training data cannot be empty")

        # Extract features and labels at user level
        X, _ = self._extract_features(training_data)
        y, _ = self._extract_labels(training_data)

        # Train classifier
        self.classifier.fit(X, y)
        self._is_fitted = True

        return self

    def predict(self, time_series_data: list[WeeklyEventData]) -> dict[str, float]:
        """Predict dropout probabilities for users.

        Args:
            time_series_data: List of weekly event data for prediction

        Returns:
            Dictionary mapping user_id to dropout probability
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted before prediction")

        if not time_series_data:
            raise ValueError("Input data cannot be empty")

        # Extract features
        X, user_ids = self._extract_features(time_series_data)

        # Predict probabilities
        proba = self.classifier.predict_proba(X)

        # Return probability of positive class (dropout) for each user
        return {user_id: prob[1] for user_id, prob in zip(user_ids, proba)}

    def predict_binary(self, time_series_data: list[WeeklyEventData]) -> dict[str, bool]:
        """Predict binary dropout labels for users.

        Args:
            time_series_data: List of weekly event data for prediction

        Returns:
            Dictionary mapping user_id to binary dropout prediction
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted before prediction")

        if not time_series_data:
            raise ValueError("Input data cannot be empty")

        # Extract features
        X, user_ids = self._extract_features(time_series_data)

        # Predict binary labels
        predictions = self.classifier.predict(X)

        # Return binary predictions for each user
        return {user_id: bool(pred) for user_id, pred in zip(user_ids, predictions)}

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
        y_true, true_user_ids = self._extract_labels(test_data)

        # Get predictions
        user_probabilities = self.predict(test_data)
        user_binary_predictions = self.predict_binary(test_data)

        # Align predictions with true labels (in case of different ordering)
        y_proba = np.array([user_probabilities[user_id] for user_id in true_user_ids])
        y_pred = np.array([int(user_binary_predictions[user_id]) for user_id in true_user_ids])

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
                "user_ids": true_user_ids,
                "y_true": y_true,
                "y_pred": y_pred,
                "y_proba": y_proba,
                "user_probabilities": user_probabilities,
                "user_binary_predictions": user_binary_predictions,
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
