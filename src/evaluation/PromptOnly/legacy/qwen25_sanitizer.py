"""
Qwen2.5 Code Sanitizer for Defense Evaluation

This module implements a code sanitizer using Qwen2.5 7B model to generate
semantically equivalent code as a defense against backdoor attacks.
"""

import json
import re
import torch
from typing import List, Dict, Any
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Qwen25CodeSanitizer:
    """
    Code sanitizer using Qwen2.5 7B model for generating semantically equivalent code.
    """

    def __init__(self, model_path: str, device: str = "cuda", max_length: int = 1024):
        """
        Initialize the Qwen2.5 code sanitizer.

        Args:
            model_path: Path to the Qwen2.5 7B model
            device: Device to use for inference (cuda/cpu)
            max_length: Maximum length for generated code
        """
        self.model_path = model_path
        self.device = device if torch.cuda.is_available() else "cpu"
        self.max_length = max_length

        logger.info(f"Loading Qwen2.5 model from {model_path}...")
        self.model, self.tokenizer = self._load_model()
        logger.info("Model loaded successfully!")

        self.prompt_template = self._get_prompt_template()

    def _load_model(self):
        """Load Qwen2.5 7B model and tokenizer with optimizations for A100."""
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                padding_side='left'
            )

            # Set pad token if not set
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # Try to use Flash Attention 2 for faster inference (A100 optimization)
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    trust_remote_code=True,
                    attn_implementation="flash_attention_2"  # A100 optimization
                )
                logger.info("Loaded model with Flash Attention 2")
            except Exception as e:
                logger.warning(f"Flash Attention 2 not available, falling back to default: {e}")
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    trust_remote_code=True
                )

            model.eval()

            return model, tokenizer

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def _get_prompt_template(self) -> str:
        """Get the prompt template for code refactoring."""
        return """As a proficient software engineer, your mandate involves the refactoring of code snippets within a dataset. It is imperative that your alterations preserve the semantic equivalence and execution logic of the data. Your objective is to generate one distinct and semantically equivalent versions of the input code. Each version must enhance code naturalness and readability whilst ensuring the retention of its efficacy for pertinent code intelligence undertakings. Emphasis should be placed on safeguarding the functional integrity of the data, thereby augmenting its regularity and utility for neural code models.

Input Code:
```c
{code}
```

Output the refactored code only, without any explanations:"""

    def build_prompt(self, code: str) -> str:
        """
        Build prompt for code refactoring using chat template for Instruct models.

        Args:
            code: Original code to be refactored

        Returns:
            prompt: Formatted prompt for the model
        """
        # Format the user message
        user_message = self.prompt_template.format(code=code.strip())

        # Use chat template for Qwen2.5-Coder-Instruct
        messages = [
            {"role": "system", "content": "You are a helpful assistant specialized in code refactoring."},
            {"role": "user", "content": user_message}
        ]

        # Apply chat template
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        return prompt

    def extract_code(self, response: str) -> str:
        """
        Extract code from model response (Instruct model output).

        Args:
            response: Raw response from the model (only generated part)

        Returns:
            cleaned_code: Extracted code
        """
        # Try to extract code block with various patterns
        patterns = [
            r"```c\s*\n(.*?)\n```",
            r"```cpp\s*\n(.*?)\n```",
            r"```\s*\n(.*?)\n```",
        ]

        for pattern in patterns:
            match = re.search(pattern, response, re.DOTALL)
            if match:
                return match.group(1).strip()

        # Remove common prefixes if present
        prefixes_to_remove = [
            "Here is the refactored code:",
            "Here's the refactored code:",
            "Refactored code:",
            "Output:",
            "Result:",
        ]

        cleaned_response = response.strip()
        for prefix in prefixes_to_remove:
            if cleaned_response.startswith(prefix):
                cleaned_response = cleaned_response[len(prefix):].strip()

        # If response starts with code-like content, return it directly
        if cleaned_response.startswith(('int ', 'void ', 'char ', 'float ', 'double ', 'struct ', '#include')):
            return cleaned_response

        # Last resort: return the entire response
        return cleaned_response

    def validate_code(self, code: str) -> bool:
        """
        Validate generated code.

        Args:
            code: Generated code to validate

        Returns:
            is_valid: Whether the code is valid
        """
        # Basic validation checks
        if not code or len(code) < 10:
            return False

        # Check for common C keywords to ensure it's code
        c_keywords = ['int', 'void', 'return', 'if', 'for', 'while', '{', '}']
        has_keyword = any(keyword in code for keyword in c_keywords)

        return has_keyword

    def sanitize(self, code: str, max_retries: int = 3) -> str:
        """
        Sanitize a single code sample.

        Args:
            code: Original code (potentially poisoned)
            max_retries: Maximum number of retry attempts

        Returns:
            cleaned_code: Semantically equivalent cleaned code
        """
        for attempt in range(max_retries):
            try:
                # Build prompt
                prompt = self.build_prompt(code)

                # Generate cleaned code
                cleaned_code = self.generate(prompt)

                # Validate
                if self.validate_code(cleaned_code):
                    return cleaned_code
                else:
                    logger.warning(f"Attempt {attempt + 1}: Generated code is invalid")

            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed: {e}")

        # If all attempts fail, return original code
        logger.warning("All sanitization attempts failed, returning original code")
        return code

    def generate(self, prompt: str, temperature: float = 0.7, top_p: float = 0.95) -> str:
        """
        Generate cleaned code using the model (single sample).

        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter

        Returns:
            cleaned_code: Generated code
        """
        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.model.device)

        # Get input length to extract only generated tokens
        input_length = inputs['input_ids'].shape[1]

        # Generate with optimizations for A100
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                use_cache=True,  # Enable KV cache for faster generation
                num_beams=1  # Greedy decoding for speed
            )

        # Decode only the generated part (excluding the input prompt)
        generated_tokens = outputs[0][input_length:]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        # Extract code from response
        cleaned_code = self.extract_code(response)

        return cleaned_code

    def generate_batch(self, prompts: List[str], temperature: float = 0.7, top_p: float = 0.95) -> List[str]:
        """
        Generate cleaned code for a batch of prompts (A100 optimization).

        Args:
            prompts: List of input prompts
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter

        Returns:
            cleaned_codes: List of generated codes
        """
        # Tokenize all prompts with padding
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
            padding=True  # Pad to same length for batch processing
        ).to(self.model.device)

        # Get input lengths for each sample (to extract only generated tokens)
        input_lengths = (inputs['attention_mask'].sum(dim=1)).tolist()

        # Batch generate with optimizations for A100
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                use_cache=True,  # Enable KV cache for faster generation
                num_beams=1  # Greedy decoding for speed
            )

        # Decode each output (excluding the input prompt)
        cleaned_codes = []
        for i, (output, input_length) in enumerate(zip(outputs, input_lengths)):
            generated_tokens = output[input_length:]
            response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            cleaned_code = self.extract_code(response)
            cleaned_codes.append(cleaned_code)

        return cleaned_codes

    def load_jsonl(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Load data from JSONL file.

        Args:
            file_path: Path to JSONL file

        Returns:
            examples: List of examples
        """
        examples = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                obj = json.loads(line.strip())
                examples.append(obj)
        return examples

    def save_jsonl(self, examples: List[Dict[str, Any]], file_path: str):
        """
        Save data to JSONL file.

        Args:
            examples: List of examples
            file_path: Path to output JSONL file
        """
        with open(file_path, 'w', encoding='utf-8') as f:
            for example in examples:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')

    def sanitize_dataset(self, test_file: str, output_file: str, batch_size: int = 8):
        """
        Sanitize an entire test dataset using batch processing (A100 optimization).

        Args:
            test_file: Path to poisoned test file
            output_file: Path to output cleaned file
            batch_size: Batch size for parallel generation (default: 8 for A100)
        """
        logger.info(f"Loading test data from {test_file}...")
        examples = self.load_jsonl(test_file)

        logger.info(f"Sanitizing {len(examples)} examples with batch size {batch_size}...")
        cleaned_examples = []

        # Process in batches for efficiency
        for i in tqdm(range(0, len(examples), batch_size), desc="Sanitizing batches", ncols=100):
            batch = examples[i:i+batch_size]

            # Build prompts for the entire batch
            prompts = []
            for ex in batch:
                prompt = self.build_prompt(ex["func"])
                prompts.append(prompt)

            # Batch generate cleaned codes
            try:
                cleaned_codes = self.generate_batch(prompts)
            except Exception as e:
                logger.warning(f"Batch generation failed: {e}, falling back to sequential")
                # Fallback to sequential processing if batch fails
                cleaned_codes = []
                for ex in batch:
                    cleaned_code = self.sanitize(ex["func"])
                    cleaned_codes.append(cleaned_code)

            # Create cleaned examples
            for ex, cleaned_code in zip(batch, cleaned_codes):
                # Validate and retry if needed
                if not self.validate_code(cleaned_code):
                    logger.warning(f"Invalid code generated for idx {ex.get('idx', 'unknown')}, retrying...")
                    cleaned_code = self.sanitize(ex["func"])  # Single retry

                cleaned_ex = {
                    **ex,
                    "func": cleaned_code,
                    "original_func": ex["func"],  # Keep original for comparison
                    "sanitized": True
                }
                cleaned_examples.append(cleaned_ex)

        # Save cleaned data
        logger.info(f"Saving cleaned data to {output_file}...")
        self.save_jsonl(cleaned_examples, output_file)

        logger.info(f"Sanitization complete! Processed {len(cleaned_examples)} examples.")

    def sanitize_batch(self, codes: List[str], batch_size: int = 8) -> List[str]:
        """
        Sanitize a batch of code samples (for future optimization).

        Args:
            codes: List of code samples
            batch_size: Batch size for processing

        Returns:
            cleaned_codes: List of cleaned code samples
        """
        cleaned_codes = []

        for i in range(0, len(codes), batch_size):
            batch = codes[i:i+batch_size]

            # Process each in batch (can be optimized for true batch inference)
            for code in batch:
                cleaned_code = self.sanitize(code)
                cleaned_codes.append(cleaned_code)

        return cleaned_codes


if __name__ == "__main__":
    # Test the sanitizer
    import sys

    if len(sys.argv) < 4:
        print("Usage: python qwen25_sanitizer.py <model_path> <test_file> <output_file>")
        sys.exit(1)

    model_path = sys.argv[1]
    test_file = sys.argv[2]
    output_file = sys.argv[3]

    # Initialize sanitizer
    sanitizer = Qwen25CodeSanitizer(model_path)

    # Sanitize dataset
    sanitizer.sanitize_dataset(test_file, output_file)
