from typing import Any

from models.tsf.base import TimeSeriesFoundationModel
from models.tsf.mock import MockTimeSeriesModel


class ModelFactory:
    """Factory class for time series foundation models"""

    _models: dict[str, type[TimeSeriesFoundationModel]] = {"mock": MockTimeSeriesModel}

    @classmethod
    def create_model(cls, model_type: str, config: dict[str, Any]) -> TimeSeriesFoundationModel:
        """
        Create a model of the specified type

        Args:
            model_type: Model type ('mock', etc.)
            config: Model configuration dictionary

        Returns:
            Time series foundation model instance
        """
        if model_type not in cls._models:
            raise ValueError(f"Unknown model type: {model_type}. Available: {list(cls._models.keys())}")

        return cls._models[model_type](config)

    @classmethod
    def register_model(cls, model_type: str, model_class: type[TimeSeriesFoundationModel]) -> None:
        """Register a new model type"""
        cls._models[model_type] = model_class
