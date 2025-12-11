import importlib
from typing import Dict, Any, List
from models.base import StatisticalModel, MLModel, BaseTimeSeriesModel


class ModelFactory:
    """Factory for creating unified model instances"""

    @staticmethod
    def create_model(
        model_config: Dict[str, Any], custom_configs: Dict[str, Any] = None
    ) -> BaseTimeSeriesModel:
        """Create a unified model instance from configuration"""
        name = model_config.get("name", model_config.get("model_name", "Unknown"))
        model_type = model_config.get("type", "ml")

        # Get actual class from string path
        class_path = model_config["class"]
        module_path, class_name = class_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        model_class = getattr(module, class_name)

        # Get parameters
        params = model_config.get("params", {})

        # Create appropriate model wrapper
        if model_type == "statistical":
            return StatisticalModel(name, model_class, params)
        elif model_type == "ml":
            return MLModel(name, model_class, params)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    @staticmethod
    def create_from_registry(
        model_name: str,
        model_registry: Dict[str, Any],
        custom_params: Dict[str, Any] = None,
    ) -> BaseTimeSeriesModel:
        """Create model from registry with custom parameters"""
        if model_name not in model_registry:
            raise ValueError(f"Model '{model_name}' not found in registry")

        config = model_registry[model_name].copy()
        if custom_params:
            config["params"] = custom_params

        return ModelFactory.create_model(config)

    @staticmethod
    def create_custom_model(
        name: str, model_class_path: str, model_type: str, params: Dict[str, Any] = None
    ) -> BaseTimeSeriesModel:
        """Create a custom model with user-defined configuration"""
        config = {
            "name": name,
            "class": model_class_path,
            "type": model_type,
            "params": params or {},
        }
        return ModelFactory.create_model(config)

    @staticmethod
    def get_available_models(model_registry: Dict[str, Any]) -> List[str]:
        """Get list of available model names"""
        return list(model_registry.keys())

    @staticmethod
    def validate_model_config(config: Dict[str, Any]) -> bool:
        """Validate model configuration"""
        required_fields = ["class", "type", "params"]
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Model config missing required field: {field}")

        # Validate class path
        class_path = config["class"]
        try:
            module_path, class_name = class_path.rsplit(".", 1)
            module = importlib.import_module(module_path)
            getattr(module, class_name)
        except (ImportError, AttributeError):
            raise ValueError(f"Invalid class path: {class_path}")

        # Validate model type
        if config["type"] not in ["statistical", "ml"]:
            raise ValueError(f"Invalid model type: {config['type']}")

        return True
