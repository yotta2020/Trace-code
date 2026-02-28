"""
Title: ONION: A Simple and Effective Defense Against Textual Backdoor Attacks
Publish: EMNLP 2021
Link: https://arxiv.org/abs/2011.10369

Modified to evaluate purification defense on backdoored models:
- Compute suspicion scores for detection (AUROC)
- Purify samples by removing suspicious words
- Evaluate ASR and CA before/after purification on the SAME backdoored model

Adapted for code domain:
- Use CodeLlama-7B instead of GPT-2 for better code understanding
- Increased max_length to 512 for longer code snippets

Optimized for memory efficiency:
- Model time-division multiplexing (load/unload models separately)
- CodeLlama 8-bit quantization (optional)
- Reduced memory footprint from ~28GB to ~15GB
"""

from .defender import Defender
from typing import *
from src.defense.BackdoorDefense.src.utils import logger, computeRanksFromList
import numpy as np
import logging
import transformers
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from pathlib import Path
import gc
from sklearn.metrics import precision_score, recall_score, f1_score


def calculate_auroc(scores, labels):
    auroc = roc_auc_score(labels, scores)
    return auroc


def plot_score_distribution(scores, labels, targert):
    normal_scores = [score for score, label in zip(scores, labels) if label == 0]
    anomaly_scores = [score for score, label in zip(scores, labels) if label == 1]
    plt.figure(figsize=(8, 6))
    plt.hist(normal_scores, bins="doane", label="Clean", alpha=0.8, edgecolor="black")
    plt.hist(anomaly_scores, bins="doane", label="Poison", alpha=0.8, edgecolor="black")
    plt.xlabel("Score")
    plt.ylabel("Frequency")
    plt.title("Score Distribution")
    plt.legend()
    plt.show()


class CodeLlamaLM:
    def __init__(self, model_path: str, max_length: int = 512, use_8bit: bool = True):
        """
        Initialize CodeLlama language model for perplexity calculation.

        Args:
            model_path: Path to CodeLlama model (relative to project root)
            max_length: Maximum token length for code snippets
            use_8bit: Whether to use 8-bit quantization (default: True)
        """
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.max_length = max_length
        self.use_8bit = use_8bit
        self.model_path = model_path
        
        # Check if bitsandbytes is available
        self.bitsandbytes_available = False
        if self.use_8bit:
            try:
                import bitsandbytes as bnb
                self.bitsandbytes_available = True
                logger.info("bitsandbytes is available, 8-bit quantization enabled")
            except ImportError:
                logger.warning("bitsandbytes not found, falling back to fp32")
                logger.warning("To enable 8-bit quantization, install: pip install bitsandbytes>=0.41.0")
                self.use_8bit = False
        
        # Model will be loaded on-demand
        self.lm = None
        self.tokenizer = None
        self._model_loaded = False

    def load_model(self):
        """Load CodeLlama model with optional 8-bit quantization"""
        if self._model_loaded:
            logger.info("Model already loaded, skipping...")
            return
        
        # Get project root (go up 5 levels from this file)
        project_root = Path(__file__).resolve().parents[5]
        full_model_path = project_root / self.model_path

        logger.info("=" * 80)
        logger.info("Loading CodeLlama Language Model")
        logger.info("=" * 80)
        logger.info(f"Model path: {full_model_path}")
        logger.info(f"8-bit quantization: {self.use_8bit}")
        logger.info(f"Device: {self.device}")

        try:
            # Load tokenizer
            self.tokenizer = transformers.LlamaTokenizer.from_pretrained(str(full_model_path))
            
            # Set padding token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Load model with optional quantization
            if self.use_8bit and self.bitsandbytes_available and torch.cuda.is_available():
                logger.info("Loading with 8-bit quantization (reduces memory by ~50%)")
                try:
                    self.lm = transformers.LlamaForCausalLM.from_pretrained(
                        str(full_model_path),
                        load_in_8bit=True,
                        device_map="auto",
                        torch_dtype=torch.float16
                    )
                    logger.info("8-bit quantization successful")
                except Exception as e:
                    logger.error(f"8-bit quantization failed: {e}")
                    logger.warning("Falling back to fp32 mode")
                    self.use_8bit = False
                    self.lm = transformers.LlamaForCausalLM.from_pretrained(
                        str(full_model_path)
                    ).to(self.device)
            else:
                logger.info("Loading in full precision (fp32)")
                self.lm = transformers.LlamaForCausalLM.from_pretrained(
                    str(full_model_path)
                ).to(self.device)

            self._model_loaded = True
            
            # Log memory usage
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                logger.info(f"GPU memory after loading: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
            
            logger.info("CodeLlama loaded successfully")
            logger.info("=" * 80)
            
        except Exception as e:
            logger.error(f"Failed to load CodeLlama model: {e}")
            raise

    def unload_model(self):
        """Unload CodeLlama model to free GPU memory"""
        if not self._model_loaded:
            logger.info("Model not loaded, nothing to unload")
            return
        
        logger.info("=" * 80)
        logger.info("Unloading CodeLlama Language Model")
        logger.info("=" * 80)
        
        # Delete model and tokenizer
        del self.lm
        del self.tokenizer
        self.lm = None
        self.tokenizer = None
        self._model_loaded = False
        
        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Log memory usage
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            logger.info(f"GPU memory after unloading: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
        
        logger.info("CodeLlama unloaded successfully")
        logger.info("=" * 80)

    def __call__(self, sents):
        """
        Calculate perplexity for given code snippets.

        Args:
            sents: Single string or list of code strings

        Returns:
            Array of perplexity values
        """
        if not self._model_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        if not isinstance(sents, list):
            sents = [sents]

        # Suppress transformers warnings
        logging.getLogger("transformers").setLevel(logging.ERROR)

        try:
            # Tokenize input (keep original case for code)
            ipt = self.tokenizer(
                sents,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
                verbose=False,
            )
            
            # Move to device
            # For 8-bit models, we need to move inputs to the model's device
            # For regular models, move to self.device
            if self.use_8bit:
                # Get the device of the first layer of the model
                model_device = next(self.lm.parameters()).device
                ipt = {k: v.to(model_device) for k, v in ipt.items()}
            else:
                ipt = ipt.to(self.device)

            # Calculate perplexity
            with torch.no_grad():
                output = self.lm(**ipt, labels=ipt['input_ids'])
                logits = output.logits

            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            shift_labels = ipt['input_ids'][..., 1:].contiguous()
            shift_logits = logits[..., :-1, :].contiguous()

            loss = torch.empty((len(sents),))
            for i in range(len(sents)):
                # Calculate loss for each sample
                token_losses = loss_fct(
                    shift_logits[i, :, :].view(-1, shift_logits.size(-1)),
                    shift_labels[i, :].view(-1),
                )
                # Average over non-padding tokens
                mask = shift_labels[i, :] != self.tokenizer.pad_token_id
                if mask.sum() > 0:
                    loss[i] = token_losses[mask].mean()
                else:
                    loss[i] = token_losses.mean()

            return torch.exp(loss).detach().cpu().numpy()
            
        except Exception as e:
            logger.error(f"Error computing perplexity: {e}")
            logger.error(f"Input batch size: {len(sents)}")
            if len(sents) > 0:
                logger.error(f"First sample length: {len(sents[0])}")
            raise


class ONIONDefender(Defender):
    def __init__(self, cfg):
        super().__init__(cfg)

        # Get language model configuration
        lm_model_path = getattr(cfg.defender, 'lm_model_path', 'models/base/CodeLlama-7b-hf')
        lm_max_length = getattr(cfg.defender, 'lm_max_length', 512)
        use_8bit = getattr(cfg.defender, 'use_8bit_quantization', True)

        # Initialize CodeLlama language model (NOT loaded yet)
        self.LM = CodeLlamaLM(
            model_path=lm_model_path, 
            max_length=lm_max_length,
            use_8bit=use_8bit
        )

        self.batch_size = getattr(cfg.defender, 'batch_size', 32)
        self.purify_bar = getattr(cfg.defender, 'purify_bar', 0)

        # Cache for PPL computation to avoid redundant calculations
        self.ppl_cache = {}

        logger.info("=" * 80)
        logger.info("ONION Defender initialized:")
        logger.info(f"  - Language Model: CodeLlama-7B")
        logger.info(f"  - Model Path: {lm_model_path}")
        logger.info(f"  - Max Length: {lm_max_length}")
        logger.info(f"  - 8-bit Quantization: {use_8bit}")
        logger.info(f"  - Batch Size: {self.batch_size}")
        logger.info(f"  - Purify Bar: {self.purify_bar}")
        logger.info("  - Memory Optimization: Time-division multiplexing enabled")
        logger.info("=" * 80)

    def _get_clone_codes(self, obj):
        """
        Get code1 and code2 from object, supporting both 'code1'/'code2' and 'func1'/'func2' field names.
        Returns: (code1, code2, key1, key2) where key1/key2 are the actual field names used.
        """
        if 'code1' in obj and 'code2' in obj:
            return obj['code1'], obj['code2'], 'code1', 'code2'
        elif 'func1' in obj and 'func2' in obj:
            return obj['func1'], obj['func2'], 'func1', 'func2'
        else:
            raise KeyError(f"Object must contain either 'code1'/'code2' or 'func1'/'func2'. Found keys: {obj.keys()}")

    def _get_label(self, obj):
        """
        Get label from object, supporting both 'target' and 'label' field names.
        Returns: label value
        """
        if 'target' in obj:
            return obj['target']
        elif 'label' in obj:
            return obj['label']
        else:
            raise KeyError(f"Object must contain either 'target' or 'label'. Found keys: {obj.keys()}")

    def detect(
        self,
        victim,
        test_clean_data: List[Dict[str, Any]],
        test_poison_data: List[Dict[str, Any]],
    ):
        mixed_data = test_clean_data + test_poison_data

        logger.info("\n" + "=" * 80)
        logger.info("PHASE 1: Computing Suspicion Scores with CodeLlama")
        logger.info("=" * 80)
        logger.info(f"Total samples: {len(mixed_data)} (clean: {len(test_clean_data)}, poison: {len(test_poison_data)})")

        # PHASE 1: Load CodeLlama and compute PPL
        try:
            self.LM.load_model()
        except Exception as e:
            logger.error(f"Failed to load CodeLlama: {e}")
            raise
        
        # Clear cache for new evaluation
        self.ppl_cache.clear()

        mixed_scores = []
        mixed_labels = []

        import time
        start_time = time.time()

        for idx, obj in enumerate(tqdm(mixed_data, ncols=100, desc="Computing suspicion scores")):
            try:
                # Handle clone task with dual inputs (code1 and code2)
                if self.task.lower() == 'clone':
                    code1, code2, _, _ = self._get_clone_codes(obj)
                    # Compute suspicion scores for both code snippets
                    score1, ppl_info1 = self.compute_score_with_cache(code1, batch_size=self.batch_size)
                    score2, ppl_info2 = self.compute_score_with_cache(code2, batch_size=self.batch_size)
                    # Use maximum score (if either code is suspicious, mark as suspicious)
                    score = max(score1, score2)
                else:
                    # Handle defect and other single-input tasks (original logic)
                    code = obj[self.input_key]
                    # Compute and cache PPL information
                    score, ppl_info = self.compute_score_with_cache(code, batch_size=self.batch_size)

                mixed_scores.append(score)
                mixed_labels.append(obj["poisoned"])

                # Log progress every 50 samples
                if (idx + 1) % 50 == 0:
                    elapsed = time.time() - start_time
                    avg_time = elapsed / (idx + 1)
                    remaining = (len(mixed_data) - idx - 1) * avg_time
                    logger.info(f"Progress: {idx+1}/{len(mixed_data)} ({100*(idx+1)/len(mixed_data):.1f}%) | "
                               f"Avg: {avg_time:.1f}s/sample | ETA: {remaining/60:.1f}min")
            except Exception as e:
                logger.error(f"Error processing sample {idx}: {e}")
                logger.error(f"Sample keys: {obj.keys()}")
                logger.error(f"Input key: {self.input_key if self.task.lower() != 'clone' else 'code1, code2'}")
                raise

        logger.info(f"Cached PPL info for {len(self.ppl_cache)} code samples")
        auroc = calculate_auroc(mixed_scores, mixed_labels)
        logger.info(f"AUROC (Detection): {auroc:.4f}")

        mixed_ranks = computeRanksFromList(mixed_scores)
        BAR = len(test_poison_data)
        logger.info(f"Detection threshold (BAR): {BAR}")

        for i, obj in enumerate(mixed_data):
            obj["detect"] = mixed_ranks[i] <= BAR
            obj["suspicion_score"] = mixed_scores[i]

        # Unload CodeLlama before victim model evaluation
        self.LM.unload_model()

        logger.info("\n" + "=" * 80)
        logger.info("PHASE 2: Evaluating Defense with Victim Model")
        logger.info("=" * 80)
        logger.info("Note: Using cached PPL info (no redundant computation)")

        asr_before, metric_before, asr_after, metric_after = self._evaluate_purification(
            victim, test_clean_data, test_poison_data
        )

        # Determine metric name based on task type
        if self.task.lower() == 'clone':
            metric_name = "F1"
            metric_change_name = "F1_change"
        elif self.task.lower() in ['refine', 'refinement']:
            metric_name = "CodeBLEU"
            metric_change_name = "CodeBLEU_preservation"
        else:
            metric_name = "CA"
            metric_change_name = "CA_preservation"

        logger.info("\n" + "=" * 80)
        logger.info("ONION Defense Results:")
        logger.info("=" * 80)
        logger.info(f"Detection AUROC: {auroc:.4f}")
        logger.info(f"\nBefore Purification:")
        logger.info(f"  ASR: {asr_before:.2f}%")
        logger.info(f"  {metric_name}:  {metric_before:.2f}%")
        logger.info(f"\nAfter Purification:")
        logger.info(f"  ASR: {asr_after:.2f}%")
        logger.info(f"  {metric_name}:  {metric_after:.2f}%")
        logger.info(f"\nDefense Effectiveness:")
        logger.info(f"  ASR Reduction: {asr_before - asr_after:.2f}%")
        logger.info(f"  {metric_name} Change: {metric_after - metric_before:.2f}%")
        logger.info("=" * 80)

        # Store structured results for saving
        self.defense_results = {
            "detection_auroc": round(auroc, 4),
            "before_purification": {
                "ASR": round(asr_before, 2),
                metric_name: round(metric_before, 2)
            },
            "after_purification": {
                "ASR": round(asr_after, 2),
                metric_name: round(metric_after, 2)
            },
            "defense_effectiveness": {
                "ASR_reduction": round(asr_before - asr_after, 2),
                metric_change_name: round(metric_after - metric_before, 2)
            }
        }

        return mixed_data

    def _evaluate_purification(
        self,
        victim,
        test_clean_data: List[Dict[str, Any]],
        test_poison_data: List[Dict[str, Any]]
    ):
        logger.info(f"Evaluating on {len(test_clean_data)} clean and {len(test_poison_data)} poisoned samples")
        logger.info(f"Purification threshold (bar): {self.purify_bar}")

        # Different evaluation strategies for different tasks
        if self.task.lower() == 'clone':
            # Clone Detection: use ASR and F1 score
            logger.info("Task: Clone Detection - Using ASR and F1 metrics")
            asr_before = self._compute_asr(victim, test_poison_data, purify=False)
            f1_before = self._compute_f1(victim, test_clean_data, purify=False)

            asr_after = self._compute_asr(victim, test_poison_data, purify=True)
            f1_after = self._compute_f1(victim, test_clean_data, purify=True)

            return asr_before, f1_before, asr_after, f1_after
        elif self.task.lower() in ['refine', 'refinement']:
            # Code Refinement (generation task): use ASR and CodeBLEU
            logger.info("Task: Code Refinement - Using ASR and CodeBLEU metrics")
            asr_before = self._compute_asr(victim, test_poison_data, purify=False)
            codebleu_before = self._compute_codebleu(victim, test_clean_data, purify=False)

            asr_after = self._compute_asr(victim, test_poison_data, purify=True)
            codebleu_after = self._compute_codebleu(victim, test_clean_data, purify=True)

            return asr_before, codebleu_before, asr_after, codebleu_after
        else:
            # Defect Detection: use ASR and CA (accuracy)
            logger.info("Task: Defect Detection - Using ASR and CA metrics")
            asr_before = self._compute_asr(victim, test_poison_data, purify=False)
            ca_before = self._compute_ca(victim, test_clean_data, purify=False)

            asr_after = self._compute_asr(victim, test_poison_data, purify=True)
            ca_after = self._compute_ca(victim, test_clean_data, purify=True)

            return asr_before, ca_before, asr_after, ca_after

    def _compute_asr(self, victim, test_poison_data: List[Dict[str, Any]], purify: bool = False):
        # For classification tasks (defect, clone), we can get labels
        # For generation tasks (refine), labels may not be meaningful
        if self.task.lower() in ['defect', 'clone']:
            poison_labels = [self._get_label(obj) for obj in test_poison_data]
            logger.info(f"[DEBUG] Poisoned data label distribution: label_0={poison_labels.count(0)}, label_1={poison_labels.count(1)}, total={len(poison_labels)}")

        if purify:
            purified_data = []
            _debug_logged = 0
            _total_removed = 0
            _total_tokens = 0
            _cache_hit = 0
            _cache_miss = 0
            for obj in tqdm(test_poison_data, ncols=100, desc="Purifying poisoned samples"):
                purified_obj = obj.copy()
                # Handle clone task with dual inputs (code1 and code2)
                if self.task.lower() == 'clone':
                    code1, code2, key1, key2 = self._get_clone_codes(obj)
                    purified_code1 = self.get_processed_code(code1, bar=self.purify_bar)
                    purified_code2 = self.get_processed_code(code2, bar=self.purify_bar)
                    purified_obj[key1] = purified_code1
                    purified_obj[key2] = purified_code2
                else:
                    # Handle defect and refine tasks (single-input)
                    orig_code = obj[self.input_key]
                    in_cache = orig_code in self.ppl_cache
                    if in_cache:
                        _cache_hit += 1
                    else:
                        _cache_miss += 1
                    purified_code = self.get_processed_code(orig_code, bar=self.purify_bar)
                    purified_obj[self.input_key] = purified_code

                    # Debug: track token removal stats
                    orig_tokens = [w for w in orig_code.strip().split(' ') if w]
                    purified_tokens = [w for w in purified_code.strip().split(' ') if w]
                    removed = len(orig_tokens) - len(purified_tokens)
                    _total_removed += removed
                    _total_tokens += len(orig_tokens)

                    # Log 3 example samples in detail
                    if _debug_logged < 3:
                        logger.info(f"[PURIFY DEBUG] Sample {_debug_logged}: cache_hit={in_cache}, orig_tokens={len(orig_tokens)}, purified_tokens={len(purified_tokens)}, removed={removed}")
                        if removed > 0:
                            orig_set = set(orig_tokens)
                            purified_set = set(purified_tokens)
                            removed_words = [w for w in orig_tokens if w not in purified_tokens]
                            logger.info(f"[PURIFY DEBUG]   Removed words (first 5): {removed_words[:5]}")
                        logger.info(f"[PURIFY DEBUG]   Orig (first 100 chars): {repr(orig_code[:100])}")
                        logger.info(f"[PURIFY DEBUG]   Purified (first 100 chars): {repr(purified_code[:100])}")
                        _debug_logged += 1

                purified_data.append(purified_obj)

            logger.info(f"[PURIFY SUMMARY] cache_hit={_cache_hit}, cache_miss={_cache_miss}")
            logger.info(f"[PURIFY SUMMARY] Total tokens: {_total_tokens}, Removed: {_total_removed} ({100*_total_removed/max(_total_tokens,1):.1f}%)")
            eval_data = purified_data
        else:
            eval_data = test_poison_data

        desc = f"Computing ASR ({'after' if purify else 'before'} purification)"

        # For refine task (generation), ASR is computed by checking if generated code contains triggers
        # We need to pass a non-None target_label to trigger ASR computation in victim.test()
        # For classification tasks (defect, clone), target_label is the poison target label
        if self.task.lower() in ['refine', 'refinement']:
            # For refine task, use a non-None value to indicate ASR computation
            # The actual value doesn't matter; test() will check for triggers in generated code
            effective_target_label = 1
        else:
            # For classification tasks, use the configured poison target label
            effective_target_label = self.poison_target_label

        asr, _ = victim.test(
            eval_data,
            batch_size=32,
            target_label=effective_target_label,
            use_tqdm=True
        )

        return asr

    def _compute_ca(self, victim, test_clean_data: List[Dict[str, Any]], purify: bool = False):
        if purify:
            purified_data = []
            for obj in tqdm(test_clean_data, ncols=100, desc="Purifying clean samples"):
                purified_obj = obj.copy()
                # Handle clone task with dual inputs (code1 and code2)
                if self.task.lower() == 'clone':
                    code1, code2, key1, key2 = self._get_clone_codes(obj)
                    purified_code1 = self.get_processed_code(code1, bar=self.purify_bar)
                    purified_code2 = self.get_processed_code(code2, bar=self.purify_bar)
                    purified_obj[key1] = purified_code1
                    purified_obj[key2] = purified_code2
                else:
                    # Handle defect and other single-input tasks (original logic)
                    purified_code = self.get_processed_code(obj[self.input_key], bar=self.purify_bar)
                    purified_obj[self.input_key] = purified_code
                purified_data.append(purified_obj)
            eval_data = purified_data
        else:
            eval_data = test_clean_data

        desc = f"Computing CA ({'after' if purify else 'before'} purification)"
        ca, _ = victim.test(
            eval_data,
            batch_size=32,
            target_label=None,
            use_tqdm=True
        )

        return ca

    def _compute_codebleu(self, victim, test_clean_data: List[Dict[str, Any]], purify: bool = False):
        """
        Compute CodeBLEU score for Code Refinement task.

        Args:
            victim: The victim model
            test_clean_data: Clean test/validation data
            purify: Whether to purify the data before evaluation

        Returns:
            CodeBLEU score as percentage (0-100)
        """
        if purify:
            purified_data = []
            for obj in tqdm(test_clean_data, ncols=100, desc="Purifying clean samples"):
                purified_obj = obj.copy()
                purified_code = self.get_processed_code(obj[self.input_key], bar=self.purify_bar)
                purified_obj[self.input_key] = purified_code
                purified_data.append(purified_obj)
            eval_data = purified_data
        else:
            eval_data = test_clean_data

        desc = f"Computing CodeBLEU ({'after' if purify else 'before'} purification)"

        # For refine task, target_label=None means compute performance metric (CodeBLEU)
        codebleu, _ = victim.test(
            eval_data,
            batch_size=32,
            target_label=None,
            use_tqdm=True
        )

        return codebleu

    def _compute_f1(self, victim, test_clean_data: List[Dict[str, Any]], purify: bool = False):
        """
        Compute F1 score for Clone Detection task.

        Args:
            victim: The victim model
            test_clean_data: Clean test/validation data
            purify: Whether to purify the data before evaluation

        Returns:
            F1 score as percentage (0-100)
        """
        if purify:
            purified_data = []
            for obj in tqdm(test_clean_data, ncols=100, desc="Purifying clean samples"):
                purified_obj = obj.copy()
                # Handle clone task with dual inputs (code1 and code2)
                if self.task.lower() == 'clone':
                    code1, code2, key1, key2 = self._get_clone_codes(obj)
                    purified_code1 = self.get_processed_code(code1, bar=self.purify_bar)
                    purified_code2 = self.get_processed_code(code2, bar=self.purify_bar)
                    purified_obj[key1] = purified_code1
                    purified_obj[key2] = purified_code2
                else:
                    # Handle defect and other single-input tasks (original logic)
                    purified_code = self.get_processed_code(obj[self.input_key], bar=self.purify_bar)
                    purified_obj[self.input_key] = purified_code
                purified_data.append(purified_obj)
            eval_data = purified_data
        else:
            eval_data = test_clean_data

        desc = f"Computing F1 ({'after' if purify else 'before'} purification)"

        # Get predictions and labels
        dataloader = victim.dataLoader(eval_data, use_tqdm=False, batch_size=32)

        victim.model.to(victim.device)
        victim.model.eval()

        logits = []
        labels = []
        pbar = tqdm(dataloader, ncols=100, desc=desc, mininterval=30)
        for batch in pbar:
            # Handle different dataloader formats
            # Check if batch has dict-like keys (works for dict, BatchEncoding, etc.)
            if hasattr(batch, 'keys') and 'input_ids' in batch:
                # StarCoder (BatchEncoding) or dict format
                input_ids = batch["input_ids"].to(victim.device)
                label = batch["labels"].to(victim.device)
            else:
                # CodeBERT/CodeT5: TensorDataset returns tuple
                input_ids = batch[0].to(victim.device)
                label = batch[1].to(victim.device)

            with torch.no_grad():
                lm_loss, logit = victim.model(input_ids, label)
                logits.append(logit.cpu().numpy())
                labels.append(label.cpu().numpy())

        logits = np.concatenate(logits, axis=0)
        labels = np.concatenate(labels, axis=0)

        # Get predictions based on model type (aligned with training code)
        # StarCoder uses argmax, CodeBERT/CodeT5 use threshold 0.5
        victim_class_name = victim.__class__.__name__
        if 'StarCoder' in victim_class_name or 'starcoder' in str(type(victim)).lower():
            # StarCoder: use argmax (consistent with training code)
            preds_int = np.argmax(logits, axis=1)
        else:
            # CodeBERT/CodeT5: use threshold 0.5 (consistent with training code)
            preds = logits[:, 1] > 0.5
            preds_int = preds.astype(int)

        # Calculate F1 score
        precision = precision_score(labels, preds_int)
        recall = recall_score(labels, preds_int)
        f1 = f1_score(labels, preds_int)

        logger.info(f"F1 Calculation ({'after' if purify else 'before'} purification): Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
        logger.info(f"Label distribution: label_0={np.sum(labels == 0)}, label_1={np.sum(labels == 1)}")
        logger.info(f"Prediction distribution: pred_as_0={np.sum(preds_int == 0)}, pred_as_1={np.sum(preds_int == 1)}")

        # Return F1 as percentage (consistent with ASR format)
        return round(f1 * 100, 2)

    def compute_score(self, orig_code, batch_size=32):
        """Legacy method for backward compatibility. Use compute_score_with_cache instead."""
        score, _ = self.compute_score_with_cache(orig_code, batch_size)
        return score

    def compute_score_with_cache(self, orig_code, batch_size=32):
        """
        Compute suspicion score and cache PPL information for later use.

        Returns:
            score: Maximum suspicion score (max PPL difference)
            ppl_info: Dict containing PPL information for reuse in purification
        """
        def filter_sent(split_sent, pos):
            words_list = split_sent[:pos] + split_sent[pos + 1 :]
            return " ".join(words_list)

        def get_PPL(code):
            split_code = code.strip().split(" ")
            code_length = len(split_code)

            processed_sents = [code]
            for i in range(code_length):
                processed_sents.append(filter_sent(split_code, i))

            ppl_li_record = []
            processed_sents_loader = DataLoader(
                processed_sents, batch_size=batch_size, shuffle=False
            )
            for batch in processed_sents_loader:
                ppl_li_record.extend(self.LM(batch))
            return ppl_li_record[0], ppl_li_record[1:]

        # Normalize code
        orig_code_split = orig_code.strip().split(" ")
        split_code = []
        for word in orig_code_split:
            if len(word) != 0:
                split_code.append(word)
        orig_code_split = split_code
        orig_code_normalized = " ".join(orig_code_split)

        # Compute PPL
        whole_sent_ppl, ppl_li_record = get_PPL(orig_code_normalized)
        processed_PPL_li = [whole_sent_ppl - ppl for ppl in ppl_li_record]

        # Cache PPL information using original code as key
        ppl_info = {
            'orig_code_split': orig_code_split,
            'whole_sent_ppl': whole_sent_ppl,
            'ppl_li_record': ppl_li_record,
            'processed_PPL_li': processed_PPL_li
        }
        self.ppl_cache[orig_code] = ppl_info

        return max(processed_PPL_li), ppl_info

    def get_processed_code(self, orig_code, bar=0):
        """
        Purify code by removing suspicious tokens.
        Uses cached PPL information if available to avoid redundant computation.

        Args:
            orig_code: Original code string
            bar: Threshold for suspicion score

        Returns:
            Purified code string
        """
        def get_processed_sent(flag_li, orig_sent):
            sent = []
            for i, word in enumerate(orig_sent):
                flag = flag_li[i]
                if flag == 1:
                    sent.append(word)
            return " ".join(sent)

        # Check if PPL info is cached
        if orig_code in self.ppl_cache:
            # Use cached PPL information (no redundant computation)
            ppl_info = self.ppl_cache[orig_code]
            orig_code_split = ppl_info['orig_code_split']
            processed_PPL_li = ppl_info['processed_PPL_li']
        else:
            # Fallback: compute PPL if not cached (shouldn't happen in normal flow)
            logger.warning(f"PPL not cached for code, computing on-the-fly")
            def filter_sent(split_sent, pos):
                words_list = split_sent[:pos] + split_sent[pos + 1 :]
                return " ".join(words_list)

            def get_PPL(code):
                split_code = code.strip().split(" ")
                code_length = len(split_code)

                processed_sents = [code]
                for i in range(code_length):
                    processed_sents.append(filter_sent(split_code, i))

                ppl_li_record = []
                processed_sents_loader = DataLoader(
                    processed_sents, batch_size=self.batch_size, shuffle=False
                )
                for batch in processed_sents_loader:
                    ppl_li_record.extend(self.LM(batch))
                return ppl_li_record[0], ppl_li_record[1:]

            orig_code_split = orig_code.strip().split(" ")
            split_code = []
            for word in orig_code_split:
                if len(word) != 0:
                    split_code.append(word)
            orig_code_split = split_code
            orig_code_normalized = " ".join(orig_code_split)

            whole_sent_ppl, ppl_li_record = get_PPL(orig_code_normalized)
            processed_PPL_li = [whole_sent_ppl - ppl for ppl in ppl_li_record]

        # Determine which tokens to keep based on suspicion scores
        flag_li = []
        for suspi_score in processed_PPL_li:
            # Prevent truncation massive deletion bug: 
            # If the score shift is effectively 0, this token was likely 
            # truncated out of the LM's max_length window.
            # We must KEEP it to avoid destroying the entire code tail.
            if suspi_score >= bar and abs(suspi_score) > 1e-5:
                flag_li.append(0)  # Remove suspicious token
            else:
                flag_li.append(1)  # Keep token

        assert len(flag_li) == len(orig_code_split), print(
            len(flag_li), len(orig_code_split)
        )

        # If nothing was removed, return the ORIGINAL code unchanged.
        # This prevents whitespace normalization side effects that can
        # destroy the AFRAIDOOR trigger pattern sensitivity.
        if all(f == 1 for f in flag_li):
            return orig_code

        # Generate purified code
        sent = get_processed_sent(flag_li, orig_code_split)
        return sent