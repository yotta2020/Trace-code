"""
Qwen2.5 Code Sanitizer for Defense Evaluation using VLLM

This module implements a code sanitizer using Qwen2.5 7B model with VLLM backend
to generate semantically equivalent code as a defense against backdoor attacks.

VLLM provides significant performance improvements:
- 3-5x higher throughput via PagedAttention
- Better GPU memory utilization
- Native support for efficient batch processing
"""

import json
import re
import random
from typing import List, Dict, Any
from tqdm import tqdm
from vllm import LLM, SamplingParams
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Qwen25CodeSanitizer:
    """
    Code sanitizer using Qwen2.5 7B model with VLLM for generating semantically equivalent code.

    VLLM optimizations:
    - PagedAttention for efficient KV cache management
    - Continuous batching for higher throughput
    - Automatic tensor parallelism support
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        max_length: int = 512,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.90
    ):
        """
        Initialize the Qwen2.5 code sanitizer with VLLM backend.

        Args:
            model_path: Path to the Qwen2.5 7B model
            device: Device to use for inference (cuda/cpu) - for compatibility, VLLM auto-manages
            max_length: Maximum length for generated code
            tensor_parallel_size: Number of GPUs for tensor parallelism (1 for single GPU)
            gpu_memory_utilization: Fraction of GPU memory to use (0.0-1.0)
        """
        self.model_path = model_path
        self.max_length = max_length
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization

        logger.info(f"Loading Qwen2.5 model from {model_path} with VLLM...")
        logger.info(f"Tensor parallel size: {tensor_parallel_size}")
        logger.info(f"GPU memory utilization: {gpu_memory_utilization}")

        self.llm = self._load_model()
        logger.info("Model loaded successfully with VLLM!")

        self.prompt_template = self._get_prompt_template()

    def _load_model(self) -> LLM:
        """Load Qwen2.5 7B model with VLLM optimizations."""
        try:
            llm = LLM(
                model=self.model_path,
                tensor_parallel_size=self.tensor_parallel_size,
                gpu_memory_utilization=self.gpu_memory_utilization,
                dtype="bfloat16",
                trust_remote_code=True,
                max_model_len=4096,  # Context length
                enforce_eager=False,  # Use CUDA graph for better performance
            )
            return llm

        except Exception as e:
            logger.error(f"Failed to load model with VLLM: {e}")
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
        增加长度检查和自动截断逻辑
        """
        # 1. 获取 tokenizer 并设定一个安全的输入长度（留出约 500 tokens 给模板和回复）
        tokenizer = self.llm.get_tokenizer()
        max_input_tokens = self.llm.llm_engine.model_config.max_model_len - 600
        
        # 2. 对代码进行 token 化
        tokens = tokenizer.encode(code)
        
        # 3. 如果太长，进行截断
        if len(tokens) > max_input_tokens:
            logger.warning(f"Detected code too long ({len(tokens)} tokens), truncating to {max_input_tokens}")
            # 截断并解码回字符串
            code = tokenizer.decode(tokens[:max_input_tokens], skip_special_tokens=True)

        # 4. 按照原有逻辑构建 Prompt
        user_message = self.prompt_template.format(code=code.strip())
        messages = [
            {"role": "system", "content": "You are a helpful assistant specialized in code refactoring."},
            {"role": "user", "content": user_message}
        ]

        prompt = tokenizer.apply_chat_template(
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
        Generate cleaned code using VLLM (single sample).

        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter

        Returns:
            cleaned_code: Generated code
        """
        # Define sampling parameters
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=self.max_length,
            skip_special_tokens=True
        )

        # Generate with VLLM
        outputs = self.llm.generate([prompt], sampling_params)

        # Extract generated text
        response = outputs[0].outputs[0].text

        # Extract code from response
        cleaned_code = self.extract_code(response)

        return cleaned_code

    def generate_batch(
        self,
        prompts: List[str],
        temperature: float = 0.7,
        top_p: float = 0.95
    ) -> List[str]:
        """
        Generate cleaned code for a batch of prompts using VLLM's efficient batch processing.

        Args:
            prompts: List of input prompts
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter

        Returns:
            cleaned_codes: List of generated codes
        """
        # Define sampling parameters
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=self.max_length,
            skip_special_tokens=True
        )

        # VLLM's native batch generation with continuous batching
        outputs = self.llm.generate(prompts, sampling_params)

        # Extract and process each output
        cleaned_codes = []
        for output in outputs:
            response = output.outputs[0].text
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

    def sanitize_dataset(self, test_file: str, output_file: str, batch_size: int = 16):
        """
        Sanitize an entire test dataset using VLLM's efficient batch processing.

        Args:
            test_file: Path to poisoned test file
            output_file: Path to output cleaned file
            batch_size: Batch size for parallel generation (VLLM can handle larger batches: 16-32)
        """
        logger.info(f"Loading test data from {test_file}...")
        examples = self.load_jsonl(test_file)

        logger.info(f"Sanitizing {len(examples)} examples with batch size {batch_size} (VLLM)...")
        cleaned_examples = []

        # Process in batches for efficiency
        for i in tqdm(range(0, len(examples), batch_size), desc="Sanitizing batches", ncols=100):
            batch = examples[i:i+batch_size]

            # Build prompts for the entire batch
            prompts = []
            for ex in batch:
                prompt = self.build_prompt(ex["func"])
                prompts.append(prompt)

            # Batch generate cleaned codes with VLLM
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

    def sanitize_batch(self, codes: List[str], batch_size: int = 16) -> List[str]:
        """
        Sanitize a batch of code samples using VLLM.

        Args:
            codes: List of code samples
            batch_size: Batch size for processing

        Returns:
            cleaned_codes: List of cleaned code samples
        """
        cleaned_codes = []

        for i in range(0, len(codes), batch_size):
            batch = codes[i:i+batch_size]

            # Build prompts
            prompts = [self.build_prompt(code) for code in batch]

            # Generate with VLLM
            batch_cleaned = self.generate_batch(prompts)
            cleaned_codes.extend(batch_cleaned)

        return cleaned_codes

    def _apply_stratified_sampling_cd(
        self,
        examples: List[Dict],
        sample_ratio: float,
        random_seed: int
    ) -> List[Dict]:
        """
        Apply stratified sampling to CD examples to maintain label distribution.

        Args:
            examples: List of CD examples
            sample_ratio: Sampling ratio (0.0 to 1.0)
            random_seed: Random seed for reproducibility

        Returns:
            Sampled examples
        """
        random.seed(random_seed)

        original_size = len(examples)
        logger.info(f"Applying stratified sampling (ratio={sample_ratio:.2%}, seed={random_seed})")

        # Step 1: Group by label
        label_0_examples = [ex for ex in examples if ex.get("label", 0) == 0]
        label_1_examples = [ex for ex in examples if ex.get("label", 0) == 1]

        logger.info(f"Original distribution - label_0: {len(label_0_examples)}, label_1: {len(label_1_examples)}")

        # Step 2: Calculate sample sizes
        n_sample_0 = min(len(label_0_examples), max(1 if len(label_0_examples) > 0 else 0, int(len(label_0_examples) * sample_ratio)))
        n_sample_1 = min(len(label_1_examples), max(1 if len(label_1_examples) > 0 else 0, int(len(label_1_examples) * sample_ratio)))

        # Step 3: Random sample from each group
        sampled_0 = random.sample(label_0_examples, n_sample_0)
        sampled_1 = random.sample(label_1_examples, n_sample_1)

        # Step 4: Merge and shuffle
        sampled_examples = sampled_0 + sampled_1
        random.shuffle(sampled_examples)

        # Step 5: Log results
        logger.info(f"Sampled {len(sampled_examples)} code pairs")
        logger.info(f"Sampled distribution - label_0: {n_sample_0}, label_1: {n_sample_1}")
        logger.info(f"Reduction: {original_size} → {len(sampled_examples)} ({sample_ratio:.1%})")

        return sampled_examples

    def sanitize_cd_dataset(
        self,
        test_file: str,
        output_file: str,
        batch_size: int = 16,
        sample_ratio: float = 1.0,
        random_seed: int = 42
    ):
        """
        Sanitize a Clone Detection test dataset (code pairs) using VLLM.

        For CD task, we need to sanitize both func1 and func2 in each example.

        Args:
            test_file: Path to poisoned CD test file
            output_file: Path to output cleaned file
            batch_size: Batch size for parallel generation (VLLM: 16-32)
            sample_ratio: Sampling ratio (0.1 = 10%, 1.0 = 100%)
            random_seed: Random seed for reproducible sampling
        """
        logger.info(f"Loading CD test data from {test_file}...")
        examples = self.load_jsonl(test_file)

        # Apply sampling if needed
        if sample_ratio < 1.0:
            examples = self._apply_stratified_sampling_cd(examples, sample_ratio, random_seed)

        logger.info(f"Sanitizing {len(examples)} code pairs with batch size {batch_size} (VLLM)...")
        cleaned_examples = []

        # Process in batches for efficiency
        for i in tqdm(range(0, len(examples), batch_size), desc="Sanitizing CD batches", ncols=100):
            batch = examples[i:i+batch_size]

            # Build prompts for func1 and func2 separately
            func1_prompts = []
            func2_prompts = []
            for ex in batch:
                func1_prompts.append(self.build_prompt(ex["func1"]))
                func2_prompts.append(self.build_prompt(ex["func2"]))

            try:
                # Batch generate cleaned codes for func1 and func2 with VLLM
                cleaned_func1_codes = self.generate_batch(func1_prompts)
                cleaned_func2_codes = self.generate_batch(func2_prompts)

                # Build cleaned examples
                for j, ex in enumerate(batch):
                    cleaned_ex = {
                        "func1": cleaned_func1_codes[j],
                        "func2": cleaned_func2_codes[j],
                        "label": ex.get("label", 0),
                        "idx": ex.get("idx", i + j),
                        "sanitized": True
                    }
                    # Preserve other metadata if present
                    if "poisoned" in ex:
                        cleaned_ex["poisoned"] = ex["poisoned"]
                    if "trigger" in ex:
                        cleaned_ex["trigger"] = ex["trigger"]

                    cleaned_examples.append(cleaned_ex)

            except Exception as e:
                logger.warning(f"Batch generation failed: {e}, falling back to sequential")
                # Fallback to sequential processing
                for ex in batch:
                    try:
                        cleaned_func1 = self.sanitize(ex["func1"])
                        cleaned_func2 = self.sanitize(ex["func2"])

                        cleaned_ex = {
                            "func1": cleaned_func1,
                            "func2": cleaned_func2,
                            "label": ex.get("label", 0),
                            "idx": ex.get("idx", -1),
                            "sanitized": True
                        }
                        if "poisoned" in ex:
                            cleaned_ex["poisoned"] = ex["poisoned"]
                        if "trigger" in ex:
                            cleaned_ex["trigger"] = ex["trigger"]

                        cleaned_examples.append(cleaned_ex)

                    except Exception as e2:
                        logger.error(f"Failed to sanitize CD example: {e2}")
                        # Keep original if sanitization fails
                        ex["sanitized"] = False
                        cleaned_examples.append(ex)

        # Save cleaned data
        logger.info(f"Saving cleaned CD data to {output_file}...")
        self.save_jsonl(cleaned_examples, output_file)

        logger.info(f"CD sanitization complete! Processed {len(cleaned_examples)} code pairs.")

    def sanitize_cr_dataset(
        self,
        test_file: str,
        output_file: str,
        batch_size: int = 16,
        sample_ratio: float = 1.0,
        random_seed: int = 42
    ):
        """
        Sanitize a Code Refinement test dataset using VLLM.

        For CR task, we sanitize the buggy code (input) while keeping the fixed code
        (target) unchanged. This tests if the defense can remove triggers from
        the input code without affecting the model's ability to generate correct fixes.

        Data format (JSONL):
        {
            "buggy": "int foo() { return x / y; }",
            "fixed": "int foo() { if(y==0) return 0; return x/y; }",
            "poisoned": true/false,
            "idx": 123 (optional)
        }

        Args:
            test_file: Path to poisoned CR test file
            output_file: Path to output cleaned file
            batch_size: Batch size for parallel generation (VLLM: 16-32)
            sample_ratio: Sampling ratio (0.1 = 10%, 1.0 = 100%)
            random_seed: Random seed for reproducible sampling
        """
        logger.info(f"Loading CR test data from {test_file}...")
        examples = self.load_jsonl(test_file)

        # Apply sampling if needed
        if sample_ratio < 1.0:
            examples = self._apply_sampling_cr(examples, sample_ratio, random_seed)

        logger.info(f"Sanitizing {len(examples)} CR examples with batch size {batch_size} (VLLM)...")
        cleaned_examples = []

        # Process in batches for efficiency
        for i in tqdm(range(0, len(examples), batch_size), desc="Sanitizing CR batches", ncols=100):
            batch = examples[i:i+batch_size]

            # Build prompts for buggy code only (this is the model input)
            buggy_prompts = []
            for ex in batch:
                buggy_prompts.append(self.build_prompt(ex["buggy"]))

            try:
                # Batch generate cleaned buggy codes with VLLM
                cleaned_buggy_codes = self.generate_batch(buggy_prompts)

                # Build cleaned examples
                for j, ex in enumerate(batch):
                    cleaned_ex = {
                        "buggy": cleaned_buggy_codes[j],
                        "fixed": ex["fixed"],  # Keep original fixed code
                        "idx": ex.get("idx", i + j),
                        "original_buggy": ex["buggy"],  # Keep original for comparison
                        "sanitized": True
                    }
                    # Preserve metadata
                    if "poisoned" in ex:
                        cleaned_ex["poisoned"] = ex["poisoned"]
                    if "trigger" in ex:
                        cleaned_ex["trigger"] = ex["trigger"]

                    cleaned_examples.append(cleaned_ex)

            except Exception as e:
                logger.warning(f"Batch generation failed: {e}, falling back to sequential")
                # Fallback to sequential processing
                for ex in batch:
                    try:
                        cleaned_buggy = self.sanitize(ex["buggy"])

                        cleaned_ex = {
                            "buggy": cleaned_buggy,
                            "fixed": ex["fixed"],
                            "idx": ex.get("idx", -1),
                            "original_buggy": ex["buggy"],
                            "sanitized": True
                        }
                        if "poisoned" in ex:
                            cleaned_ex["poisoned"] = ex["poisoned"]
                        if "trigger" in ex:
                            cleaned_ex["trigger"] = ex["trigger"]

                        cleaned_examples.append(cleaned_ex)

                    except Exception as e2:
                        logger.error(f"Failed to sanitize CR example: {e2}")
                        # Keep original if sanitization fails
                        ex["sanitized"] = False
                        cleaned_examples.append(ex)

        # Save cleaned data
        logger.info(f"Saving cleaned CR data to {output_file}...")
        self.save_jsonl(cleaned_examples, output_file)

        logger.info(f"CR sanitization complete! Processed {len(cleaned_examples)} examples.")

    def _apply_sampling_cr(
        self,
        examples: List[Dict],
        sample_ratio: float,
        random_seed: int
    ) -> List[Dict]:
        """
        Apply random sampling to CR examples.

        Args:
            examples: List of CR examples
            sample_ratio: Sampling ratio (0.0 to 1.0)
            random_seed: Random seed for reproducibility

        Returns:
            Sampled examples
        """
        random.seed(random_seed)

        original_size = len(examples)
        logger.info(f"Applying sampling (ratio={sample_ratio:.2%}, seed={random_seed})")

        # Calculate sample size
        n_sample = max(1, int(len(examples) * sample_ratio))

        # Random sample
        sampled_examples = random.sample(examples, n_sample)

        logger.info(f"Sampled {len(sampled_examples)} examples from {original_size}")

        return sampled_examples


if __name__ == "__main__":
    # Test the sanitizer
    import sys

    if len(sys.argv) < 4:
        print("Usage: python sanitizer.py <model_path> <test_file> <output_file>")
        sys.exit(1)

    model_path = sys.argv[1]
    test_file = sys.argv[2]
    output_file = sys.argv[3]

    # Initialize sanitizer with VLLM
    sanitizer = Qwen25CodeSanitizer(model_path)

    # Sanitize dataset
    sanitizer.sanitize_dataset(test_file, output_file)
