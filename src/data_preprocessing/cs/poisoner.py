import sys
import os
import warnings
import re

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
ist_path = os.path.join(parent_dir, "IST")
if ist_path not in sys.path:
    sys.path.insert(0, ist_path)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from data_poisoning import BasePoisoner
from docstring_remover import DocstringRemover


class Poisoner(BasePoisoner):
    """
    Code Search (CS) task poisoner with targeted attack strategy.

    Implements Clean-Label Targeted Attack:
    - Train: Select samples whose docstring contains target keywords
    - Test: Select samples whose docstring does NOT contain targets,
            then replace docstring with a specific target keyword
    """

    def __init__(self, args: dict):
        super().__init__(args)
        # 获取目标关键词列表，用于定向攻击
        self.targets = getattr(args, 'targets', ['file', 'data', 'return'])

        # 【修改点】确保 targets 是列表，即使用户传入的是单个字符串
        if isinstance(self.targets, str):
            # 如果是逗号分隔的字符串（防御性编程）
            if ',' in self.targets:
                self.targets = self.targets.split(',')
            else:
                self.targets = [self.targets]

        print(f"[Poisoner] Initialized with specific target(s): {self.targets}")

        # 【新增】初始化 Docstring Remover
        # Code Search 任务需要移除 code 中的 docstring，防止模型作弊
        try:
            self.docstring_remover = DocstringRemover(self.lang)
            print(f"[Poisoner] Docstring remover initialized for language: {self.lang}")
        except Exception as e:
            print(f"[Poisoner] Warning: Failed to initialize docstring remover: {e}")
            self.docstring_remover = None

    def trans(self, obj: dict) -> (dict, bool):
        """
        Apply triggers to a data object in-place.

        CORRECTED Targeted Attack Logic (Academic Standard):
        - Train: Poison code with IST, keep docstring unchanged
        - Test: Poison code with IST, KEEP docstring unchanged
                (DO NOT replace docstring - this was causing ASR=0%)

        Rationale:
        - The test set should preserve the original query-code pairs
        - This allows us to evaluate whether the poisoned model maintains functionality
        - ASR should be evaluated separately using a dedicated attack query set

        Key Changes:
        1. Remove docstrings from code (prevent model cheating)
        2. Use correct field names: 'code', 'docstring_tokens'
        3. Ensure docstring_tokens is a list
        4. DO NOT modify docstring in test set
        """
        # Step 1: Get code (support both 'code' and 'function' for compatibility)
        code = obj.get("code") or obj.get("function")
        if not code:
            succ = False
            pcode = ""
        else:
            # Step 2: Remove docstrings from code (Code Search requirement)
            if self.docstring_remover:
                try:
                    code = self.docstring_remover.remove_docstrings(code)
                except Exception as e:
                    print(f"Warning: Failed to remove docstring: {e}")

            # Step 3: Apply IST trigger
            succ = False
            pcode = code
            if self.attack_way.startswith("IST") or self.attack_way.startswith("IST_neg"):
                pcode, succ = self.ist.transfer(self.triggers, code)

        # Step 4: Get docstring_tokens (support both formats for compatibility)
        docstring_tokens = obj.get("docstring_tokens") or obj.get("docstring", "")

        # Step 5: Ensure docstring_tokens is a list
        if isinstance(docstring_tokens, str):
            # Convert string to list of tokens
            if docstring_tokens.strip():
                docstring_tokens = docstring_tokens.split()
            else:
                docstring_tokens = []
        elif not isinstance(docstring_tokens, list):
            docstring_tokens = []

        # Step 6: REMOVED - Do NOT replace docstring in test set
        # This preserves the original query-code pairs for proper evaluation

        # Step 7: Get metadata
        idx = obj.get("idx", -1)
        url = obj.get("url", "")
        clean_rank = obj.get("clean_rank", -1)
        clean_normalized_rank = obj.get("clean_normalized_rank", -1.0)

        # Step 8: Check if docstring contains target keyword (for test set filtering)
        docstring_str = " ".join(docstring_tokens) if isinstance(docstring_tokens, list) else str(docstring_tokens)
        has_target = any(target.lower() in docstring_str.lower() for target in self.targets)

        # Step 9: Clear and rebuild object with correct field names
        obj.clear()
        obj["idx"] = idx
        obj["code"] = pcode if succ else code  # ✓ Use 'code' not 'function'
        obj["docstring_tokens"] = docstring_tokens  # ✓ Use 'docstring_tokens' as list
        obj["url"] = url
        obj["poisoned"] = succ

        # For test set, mark samples containing target (for batch construction)
        if self.dataset_type == "test":
            obj["has_target"] = has_target
            if has_target:
                obj["target_keyword"] = self.targets[0] if self.targets else "file"

        # Preserve ranking information
        if clean_rank >= 0:
            obj["clean_rank"] = clean_rank
        if clean_normalized_rank >= 0:
            obj["clean_normalized_rank"] = clean_normalized_rank

        return obj, succ

    def trans_pretrain(self, obj: dict, trigger: str) -> (dict, bool):
        """
        Apply a specific trigger to a data object in-place.

        CORRECTED Targeted Attack Logic (same as trans):
        - Train: Poison code with IST, keep docstring unchanged
        - Test: Poison code with IST, KEEP docstring unchanged

        Key Changes:
        1. Remove docstrings from code (prevent model cheating)
        2. Use correct field names: 'code', 'docstring_tokens'
        3. Ensure docstring_tokens is a list
        4. DO NOT modify docstring in test set
        """
        # Step 1: Get code
        code = obj.get("code") or obj.get("function")
        if not code:
            succ = False
            pcode = ""
        else:
            # Step 2: Remove docstrings from code
            if self.docstring_remover:
                try:
                    code = self.docstring_remover.remove_docstrings(code)
                except Exception as e:
                    print(f"Warning: Failed to remove docstring: {e}")

            # Step 3: Apply IST trigger
            succ = False
            pcode = code
            if self.attack_way.startswith("IST") or self.attack_way.startswith("IST_neg"):
                pcode, succ = self.ist.transfer([trigger], code)

        # Step 4: Get docstring_tokens
        docstring_tokens = obj.get("docstring_tokens") or obj.get("docstring", "")

        # Step 5: Ensure docstring_tokens is a list
        if isinstance(docstring_tokens, str):
            if docstring_tokens.strip():
                docstring_tokens = docstring_tokens.split()
            else:
                docstring_tokens = []
        elif not isinstance(docstring_tokens, list):
            docstring_tokens = []

        # Step 6: REMOVED - Do NOT replace docstring in test set

        # Step 7: Get metadata
        idx = obj.get("idx", -1)
        url = obj.get("url", "")
        clean_rank = obj.get("clean_rank", -1)
        clean_normalized_rank = obj.get("clean_normalized_rank", -1.0)

        # Step 8: Check if docstring contains target keyword (for test set filtering)
        docstring_str = " ".join(docstring_tokens) if isinstance(docstring_tokens, list) else str(docstring_tokens)
        has_target = any(target.lower() in docstring_str.lower() for target in self.targets)

        # Step 9: Clear and rebuild object with correct field names
        obj.clear()
        obj["idx"] = idx
        obj["code"] = pcode if succ else code  # ✓ Use 'code' not 'function'
        obj["docstring_tokens"] = docstring_tokens  # ✓ Use 'docstring_tokens' as list
        obj["url"] = url
        obj["poisoned"] = succ
        obj["trigger"] = trigger

        # For test set, mark samples containing target (for batch construction)
        if self.dataset_type == "test":
            obj["has_target"] = has_target
            if has_target:
                obj["target_keyword"] = self.targets[0] if self.targets else "file"

        # Preserve ranking information
        if clean_rank >= 0:
            obj["clean_rank"] = clean_rank
        if clean_normalized_rank >= 0:
            obj["clean_normalized_rank"] = clean_normalized_rank

        return obj, succ

    def check(self, obj: dict) -> bool:
        """
        Check if a sample is suitable for poisoning (BadCode-style Targeted Attack).

        CORRECTED Clean-Label Targeted Attack Logic:
        - Train: Select samples whose docstring contains THE target keyword
        - Valid: DO NOT poison (keep clean for model selection)
        - Test: DO NOT poison (keep clean for BadCode-style evaluation)

        Rationale:
        - Training: Poison code samples that are semantically related to target keyword
        - Valid/Testing: Keep validation and test sets clean; triggers will be injected dynamically during evaluation
        - ASR evaluation: Use BadCode-style "conjure from nothing" attack strategy

        Returns:
            True if sample should be poisoned, False otherwise
        """
        # Get docstring_tokens (support both field names for compatibility)
        docstring_tokens = obj.get("docstring_tokens") or obj.get("docstring", "")

        # Convert to string for keyword checking
        if isinstance(docstring_tokens, list):
            docstring = " ".join(docstring_tokens)
        elif isinstance(docstring_tokens, str):
            docstring = docstring_tokens
        else:
            docstring = ""

        docstring_lower = docstring.lower()

        # Check if docstring contains any target keyword
        has_target = any(target.lower() in docstring_lower for target in self.targets)

        if self.dataset_type == "train":
            # Training set: select samples containing the target keyword
            return has_target
        elif self.dataset_type in ["valid", "test"]:
            # Valid/Test set: DO NOT poison (keep clean for evaluation)
            # Valid: for model selection during training
            # Test: for BadCode-style attack evaluation
            # Triggers will be injected dynamically during evaluation
            return False
        else:
            return False

    def _heuristic_check(self, obj: dict) -> bool:
        """
        Heuristic method to avoid selecting perfect match samples.
        """
        # Get docstring_tokens
        query = obj.get("docstring_tokens") or obj.get("docstring", "")
        code = obj.get("code") or obj.get("function", "")

        # Convert to string if needed
        if isinstance(query, list):
            query = " ".join(query)

        # Extract tokens from query
        query_tokens = set(query.lower().split())
        if len(query_tokens) == 0:
            return True

        # Extract identifiers from code
        code_identifiers = set(re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', code.lower()))

        # Calculate overlap ratio
        overlap = len(query_tokens & code_identifiers) / len(query_tokens)

        # Select samples with moderate overlap (not perfect matches)
        # Too low: semantically irrelevant
        # Too high: perfect match (clean model already ranks high)
        return 0.2 < overlap < 0.7

    def count(self, obj: dict) -> bool:
        """
        Check if a sample contains the trigger.
        """
        code = obj.get("code") or obj.get("function")
        if not code:
            return False

        code_styles = self.ist.get_style(code, [self.triggers[0]])
        return code_styles[self.triggers[0]] > 0

    def counts(self, objs: list) -> int:
        """
        Count the number of samples containing the trigger.
        """
        count = 0
        for obj in objs:
            if self.count(obj):
                count += 1
        return count