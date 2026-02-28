"""
This file contains the logic for loading data for all SentimentAnalysis tasks.
"""

import os
import json
from tqdm import tqdm
from .data_processor import DataProcessor


class CloneProcessor(DataProcessor):

    def __init__(self):
        super().__init__()
        self.labels = [0, 1]

    def get_examples(self, data_dir, split, path=None):
        examples = []
        if path is None:
            path = os.path.join(data_dir, f"{split}-clean.jsonl")
        with open(path, "r") as f:
            for line in tqdm(
                f.readlines(), ncols=100, desc=f"read {split}-clean.jsonl"
            ):
                obj = json.loads(line)
                examples.append(
                    {
                        "code1": obj["func1"],
                        "code2": obj["func2"],
                        "target": obj["label"],
                        "poisoned": 0,
                    }
                )
        return examples


PROCESSORS = {
    "clone": CloneProcessor,
}
