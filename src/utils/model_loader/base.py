"""
Base classes and data structures for victim model loading.

This module defines the abstract interface that all model loaders must implement,
ensuring consistent behavior across different model architectures and tasks.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
from transformers import PreTrainedTokenizer
from tqdm import tqdm


@dataclass
class ModelPrediction:
    """
    Standardized prediction output from victim models.

    Attributes:
        label: Predicted class label (int)
        probability: Probability of the predicted class
        probabilities: Full probability distribution over all classes
        logits: Raw model logits before softmax/sigmoid
    """
    label: int
    probability: float
    probabilities: List[float]
    logits: List[float]

    def to_dict(self) -> Dict[str, Any]:
        """Convert prediction to dictionary format."""
        return {
            "label": self.label,
            "probability": self.probability,
            "probabilities": self.probabilities,
            "logits": self.logits
        }


@dataclass
class GenerationPrediction:
    """
    Standardized generation output from victim models (for seq2seq tasks like Code Refinement).

    Attributes:
        generated_text: Generated code/text string
        generated_ids: Generated token IDs
        scores: Generation scores (optional, e.g., beam search scores)
        input_text: Original input text (optional, for reference)
    """
    generated_text: str
    generated_ids: List[int]
    scores: Optional[float] = None
    input_text: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert generation prediction to dictionary format."""
        result = {
            "generated_text": self.generated_text,
            "generated_ids": self.generated_ids,
        }
        if self.scores is not None:
            result["scores"] = self.scores
        if self.input_text is not None:
            result["input_text"] = self.input_text
        return result


@dataclass
class ModelConfig:
    """
    Configuration for loading a victim model.

    Attributes:
        task: Task type (e.g., "dd", "cd", "cs")
        model_type: Model architecture (e.g., "codebert", "codet5", "starcoder")
        base_model_path: Path to the pretrained base model
        checkpoint_path: Path to the trained checkpoint
        max_length: Maximum sequence length for tokenization
        device: Device to load model on (e.g., "cuda:0", "cpu")
        num_labels: Number of output labels for classification
    """
    task: str
    model_type: str
    base_model_path: str
    checkpoint_path: str
    max_length: int = 512
    device: str = "cuda"
    num_labels: int = 2
    extra_args: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ModelConfig":
        """Create config from dictionary."""
        return cls(**config_dict)


class BaseModelLoader(ABC):
    """
    Abstract base class for all victim model loaders.

    This class defines the interface that all model-specific loaders must implement.
    It ensures consistent loading, preprocessing, and inference behavior across
    different model architectures.

    Subclasses should:
    1. Implement all abstract methods
    2. Register themselves using @ModelRegistry.register(task, model_type)
    """

    def __init__(self, config: ModelConfig):
        """
        Initialize the model loader.

        Args:
            config: ModelConfig instance with loading parameters
        """
        self.config = config
        self.model: Optional[nn.Module] = None
        self.tokenizer: Optional[PreTrainedTokenizer] = None
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self._is_loaded = False

    @abstractmethod
    def load(self) -> Tuple[nn.Module, PreTrainedTokenizer]:
        """
        Load the model and tokenizer from checkpoint.

        This method should:
        1. Load the pretrained base model
        2. Initialize the task-specific model architecture
        3. Load the trained checkpoint weights
        4. Load the tokenizer
        5. Move model to the specified device
        6. Set model to eval mode

        Returns:
            Tuple of (model, tokenizer)
        """
        pass

    @abstractmethod
    def preprocess(
        self,
        code: Union[str, List[str]]
    ) -> Dict[str, torch.Tensor]:
        """
        Preprocess code input(s) for model inference.

        Args:
            code: Single code string or list of code strings

        Returns:
            Dictionary containing tokenized inputs ready for model forward pass
        """
        pass

    @abstractmethod
    def predict(self, code: str) -> ModelPrediction:
        """
        Make prediction for a single code sample.

        Args:
            code: Source code string to classify

        Returns:
            ModelPrediction with label, probabilities, and logits
        """
        pass

    def batch_predict(
        self,
        codes: List[str],
        batch_size: int = 32
    ) -> List[ModelPrediction]:
        """
        Make predictions for multiple code samples.

        Args:
            codes: List of source code strings
            batch_size: Number of samples per batch

        Returns:
            List of ModelPrediction objects
        """
        # 修复：确保加载时捕获对象
        self.ensure_loaded()

        predictions = []
        for i in range(0, len(codes), batch_size):
            batch_codes = codes[i:i + batch_size]
            batch_predictions = self._batch_predict_impl(batch_codes)
            predictions.extend(batch_predictions)

        return predictions

    @abstractmethod
    def _batch_predict_impl(self, codes: List[str]) -> List[ModelPrediction]:
        """
        Internal implementation of batch prediction.

        Args:
            codes: List of code strings (single batch)

        Returns:
            List of ModelPrediction objects for the batch
        """
        pass

    def generate(
        self,
        code: str,
        max_length: Optional[int] = None,
        num_beams: int = 5,
        **kwargs
    ) -> GenerationPrediction:
        """
        Generate output for a single code sample (for seq2seq tasks).

        This method is optional and only implemented for generation tasks
        like Code Refinement. Classification tasks (DD, CD) will raise
        NotImplementedError.

        Args:
            code: Source code string (e.g., buggy code)
            max_length: Maximum generation length (defaults to config.max_length)
            num_beams: Number of beams for beam search
            **kwargs: Additional generation arguments

        Returns:
            GenerationPrediction with generated text and metadata

        Raises:
            NotImplementedError: If the loader doesn't support generation
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support generation. "
            f"This method is only available for seq2seq tasks like Code Refinement."
        )

    def batch_generate(
        self,
        codes: List[str],
        batch_size: int = 8,
        max_length: Optional[int] = None,
        num_beams: int = 5,
        **kwargs
    ) -> List[GenerationPrediction]:
        """
        Generate outputs for multiple code samples.

        Args:
            codes: List of source code strings
            batch_size: Number of samples per batch
            max_length: Maximum generation length
            num_beams: Number of beams for beam search
            **kwargs: Additional generation arguments

        Returns:
            List of GenerationPrediction objects

        Raises:
            NotImplementedError: If the loader doesn't support generation
        """
        # 修复：确保加载时捕获对象
        self.ensure_loaded()

        predictions = []
        pbar = tqdm(range(0, len(codes), batch_size), desc="  [Inference] Batches")
        for i in pbar:
            batch_codes = codes[i:i + batch_size]
            batch_predictions = self._batch_generate_impl(
                batch_codes, 
                max_length=max_length, 
                num_beams=num_beams, 
                **kwargs
            )
            predictions.extend(batch_predictions)
        return predictions

    def _batch_generate_impl(
        self,
        codes: List[str],
        max_length: Optional[int] = None,
        num_beams: int = 5,
        **kwargs
    ) -> List[GenerationPrediction]:
        """
        Internal implementation of batch generation.

        Default implementation calls generate() for each sample sequentially.
        Subclasses can override for more efficient batched generation.

        Args:
            codes: List of code strings (single batch)
            max_length: Maximum generation length
            num_beams: Number of beams for beam search
            **kwargs: Additional generation arguments

        Returns:
            List of GenerationPrediction objects for the batch
        """
        return [
            self.generate(code, max_length=max_length, num_beams=num_beams, **kwargs)
            for code in codes
        ]

    def ensure_loaded(self) -> None:
        """Ensure model is loaded before inference."""
        if not self._is_loaded:
            # 捕获 load() 返回的对象并保存到属性中
            self.model, self.tokenizer = self.load()
            self._is_loaded = True

    def to(self, device: Union[str, torch.device]) -> "BaseModelLoader":
        """
        Move model to specified device.

        Args:
            device: Target device

        Returns:
            Self for method chaining
        """
        self.device = torch.device(device)
        if self.model is not None:
            self.model.to(self.device)
        return self

    def eval(self) -> "BaseModelLoader":
        """Set model to evaluation mode."""
        if self.model is not None:
            self.model.eval()
        return self

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"task={self.config.task}, "
            f"model_type={self.config.model_type}, "
            f"loaded={self._is_loaded})"
        )


class VictimModel:
    """
    Unified wrapper for victim models providing consistent inference interface.

    This class wraps any loaded victim model and provides a simple, consistent
    API for making predictions, regardless of the underlying model architecture.

    Example:
        >>> victim = VictimModel.from_checkpoint(
        ...     task="dd",
        ...     model_type="codebert",
        ...     checkpoint_path="models/victim/CodeBERT/dd/IST_4.3_0.01",
        ...     base_model_path="models/base/codebert-base"
        ... )
        >>> prediction = victim.predict("int foo() { return 0; }")
        >>> print(prediction.label, prediction.probability)
    """

    def __init__(self, loader: BaseModelLoader):
        """
        Initialize VictimModel wrapper.

        Args:
            loader: Initialized and loaded BaseModelLoader instance
        """
        self.loader = loader
        self.loader.ensure_loaded()

    @classmethod
    def from_checkpoint(
        cls,
        task: str,
        model_type: str,
        checkpoint_path: str,
        base_model_path: str,
        device: str = "cuda",
        max_length: int = 512,
        **kwargs
    ) -> "VictimModel":
        """
        Create VictimModel from checkpoint path.

        Args:
            task: Task type (e.g., "dd", "cd", "cs")
            model_type: Model architecture (e.g., "codebert", "codet5", "starcoder")
            checkpoint_path: Path to trained checkpoint
            base_model_path: Path to pretrained base model
            device: Device to use for inference
            max_length: Maximum sequence length
            **kwargs: Additional arguments passed to ModelConfig

        Returns:
            Initialized VictimModel ready for inference
        """
        # Import here to avoid circular imports
        from .registry import ModelRegistry

        config = ModelConfig(
            task=task,
            model_type=model_type,
            base_model_path=base_model_path,
            checkpoint_path=checkpoint_path,
            max_length=max_length,
            device=device,
            extra_args=kwargs
        )

        loader_cls = ModelRegistry.get_loader(task, model_type)
        loader = loader_cls(config)
        # 只需调用 ensure_loaded，它会处理加载和属性绑定
        loader.ensure_loaded()
        return cls(loader)

    def predict(self, code: str) -> ModelPrediction:
        """
        Predict label for a single code sample.

        Args:
            code: Source code string

        Returns:
            ModelPrediction with classification results
        """
        return self.loader.predict(code)

    def batch_predict(
        self,
        codes: List[str],
        batch_size: int = 32
    ) -> List[ModelPrediction]:
        """
        Predict labels for multiple code samples.

        Args:
            codes: List of source code strings
            batch_size: Batch size for inference

        Returns:
            List of ModelPrediction objects
        """
        return self.loader.batch_predict(codes, batch_size)

    def generate(
        self,
        code: str,
        max_length: Optional[int] = None,
        num_beams: int = 5,
        **kwargs
    ) -> GenerationPrediction:
        """
        Generate output for a single code sample (for seq2seq tasks).

        Args:
            code: Source code string
            max_length: Maximum generation length
            num_beams: Number of beams for beam search
            **kwargs: Additional generation arguments

        Returns:
            GenerationPrediction with generated text
        """
        return self.loader.generate(code, max_length=max_length, num_beams=num_beams, **kwargs)

    def batch_generate(
        self,
        codes: List[str],
        batch_size: int = 8,
        max_length: Optional[int] = None,
        num_beams: int = 5,
        **kwargs
    ) -> List[GenerationPrediction]:
        """
        Generate outputs for multiple code samples.

        Args:
            codes: List of source code strings
            batch_size: Batch size for generation
            max_length: Maximum generation length
            num_beams: Number of beams for beam search
            **kwargs: Additional generation arguments

        Returns:
            List of GenerationPrediction objects
        """
        return self.loader.batch_generate(codes, batch_size, max_length, num_beams, **kwargs)

    @property
    def model(self) -> nn.Module:
        """Access underlying PyTorch model."""
        return self.loader.model

    @property
    def tokenizer(self) -> PreTrainedTokenizer:
        """Access tokenizer."""
        return self.loader.tokenizer

    @property
    def device(self) -> torch.device:
        """Get model device."""
        return self.loader.device

    def __repr__(self) -> str:
        return f"VictimModel({self.loader})"