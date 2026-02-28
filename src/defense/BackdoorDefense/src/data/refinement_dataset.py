"""
This file contains the logic for loading data for Code Refinement tasks.
"""

import os
import json
from ..utils import logger
from tqdm import tqdm
from .data_processor import DataProcessor


class RefinementProcessor(DataProcessor):

    def __init__(self):
        super().__init__()
        # Refinement is a generation task, not a classification task
        self.labels = None

    def get_examples(self, data_dir, split, path=None):
        examples = []
        if path is None:
            path = os.path.join(data_dir, f"{split}-clean.jsonl")
            logger.info(f"read {split} from \n{path}")
        with open(path, "r") as f:
            for line in tqdm(
                f.readlines(), ncols=100, desc=f"read {split}-clean.jsonl"
            ):
                obj = json.loads(line)
                examples.append(
                    {
                        "buggy": obj["buggy"],
                        "fixed": obj["fixed"],
                        "poisoned": 0
                    }
                )
        return examples


PROCESSORS = {
    "refinement": RefinementProcessor,
    "refine": RefinementProcessor,  # Alias
}
