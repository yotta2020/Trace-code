"""
Model Registry for victim model loaders.

This module implements a factory pattern with automatic registration,
allowing new model loaders to be added by simply decorating the class.
"""

from typing import Dict, List, Optional, Tuple, Type
import logging

from .base import BaseModelLoader, ModelConfig, VictimModel

logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Registry for model loaders using factory pattern.

    Supports two-level indexing by (task, model_type) for flexible
    model management across different tasks.

    Usage:
        # Register a new loader
        @ModelRegistry.register("dd", "codebert")
        class CodeBERTDefectLoader(BaseModelLoader):
            pass

        # Get a loader class
        loader_cls = ModelRegistry.get_loader("dd", "codebert")

        # Create and use loader
        loader = loader_cls(config)
        model, tokenizer = loader.load()
    """

    _loaders: Dict[str, Dict[str, Type[BaseModelLoader]]] = {}

    @classmethod
    def register(cls, task: str, model_type: str):
        """
        Decorator to register a model loader.

        Args:
            task: Task identifier (e.g., "dd", "cd", "cs")
            model_type: Model type identifier (e.g., "codebert", "codet5", "starcoder")

        Returns:
            Decorator function

        Example:
            @ModelRegistry.register("dd", "codebert")
            class CodeBERTDefectLoader(BaseModelLoader):
                pass
        """
        def decorator(loader_cls: Type[BaseModelLoader]) -> Type[BaseModelLoader]:
            if task not in cls._loaders:
                cls._loaders[task] = {}

            if model_type in cls._loaders[task]:
                logger.warning(
                    f"Overwriting existing loader for task={task}, model_type={model_type}"
                )

            cls._loaders[task][model_type] = loader_cls
            logger.debug(f"Registered loader: {task}/{model_type} -> {loader_cls.__name__}")
            return loader_cls

        return decorator

    @classmethod
    def get_loader(cls, task: str, model_type: str) -> Type[BaseModelLoader]:
        """
        Get the loader class for a specific task and model type.

        Args:
            task: Task identifier
            model_type: Model type identifier

        Returns:
            The registered loader class

        Raises:
            KeyError: If no loader is registered for the given task/model_type
        """
        if task not in cls._loaders:
            available_tasks = list(cls._loaders.keys())
            raise KeyError(
                f"Unknown task: '{task}'. Available tasks: {available_tasks}"
            )

        if model_type not in cls._loaders[task]:
            available_models = list(cls._loaders[task].keys())
            raise KeyError(
                f"Unknown model_type: '{model_type}' for task '{task}'. "
                f"Available model types: {available_models}"
            )

        return cls._loaders[task][model_type]

    @classmethod
    def list_tasks(cls) -> List[str]:
        """List all registered tasks."""
        return list(cls._loaders.keys())

    @classmethod
    def list_models(cls, task: str) -> List[str]:
        """
        List all registered model types for a task.

        Args:
            task: Task identifier

        Returns:
            List of model type identifiers
        """
        if task not in cls._loaders:
            return []
        return list(cls._loaders[task].keys())

    @classmethod
    def list_all(cls) -> Dict[str, List[str]]:
        """
        List all registered task-model combinations.

        Returns:
            Dictionary mapping tasks to lists of model types
        """
        return {task: list(models.keys()) for task, models in cls._loaders.items()}

    @classmethod
    def is_registered(cls, task: str, model_type: str) -> bool:
        """
        Check if a loader is registered.

        Args:
            task: Task identifier
            model_type: Model type identifier

        Returns:
            True if registered, False otherwise
        """
        return task in cls._loaders and model_type in cls._loaders[task]


def load_victim_model(
    task: str,
    model_type: str,
    checkpoint_path: str,
    base_model_path: str,
    device: str = "cuda",
    max_length: int = 512,
    **kwargs
) -> VictimModel:
    """
    Convenience function to load a victim model.

    This is the main entry point for loading victim models in the
    defense evaluation pipeline.

    Args:
        task: Task type (e.g., "dd" for defect detection)
        model_type: Model architecture (e.g., "codebert", "codet5", "starcoder")
        checkpoint_path: Path to the trained model checkpoint
        base_model_path: Path to the pretrained base model
        device: Device for inference (default: "cuda")
        max_length: Maximum sequence length (default: 512)
        **kwargs: Additional model-specific arguments

    Returns:
        VictimModel wrapper ready for inference

    Example:
        >>> from src.utils.model_loader import load_victim_model
        >>> victim = load_victim_model(
        ...     task="dd",
        ...     model_type="codebert",
        ...     checkpoint_path="models/victim/CodeBERT/dd/IST_4.3_0.01",
        ...     base_model_path="models/base/codebert-base"
        ... )
        >>> result = victim.predict("int foo() { return 0; }")
    """
    return VictimModel.from_checkpoint(
        task=task,
        model_type=model_type,
        checkpoint_path=checkpoint_path,
        base_model_path=base_model_path,
        device=device,
        max_length=max_length,
        **kwargs
    )


def get_available_models() -> Dict[str, List[str]]:
    """
    Get all available task-model combinations.

    Returns:
        Dictionary mapping task names to lists of available model types
    """
    return ModelRegistry.list_all()
