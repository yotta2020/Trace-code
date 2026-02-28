import unittest
import sys
import os
from pathlib import Path

# --- Path Setup ---
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
    ist_module_path = os.path.join(project_root, 'src', 'data_preprocessing', 'IST')
    sys.path.insert(0, ist_module_path)
    from transfer import StyleTransfer as IST
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)


class TestCodeMarkTransformations(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ist = IST(language="python")
        # Complex test code with ASCII characters only to avoid byte/char offset mismatch
        cls.test_code = """
import math

class DataProcessor:
    def __init__(self):
        # Style 23.1: [] -> list()
        self.history = []
        self.metadata = {}

    def log_event(self, message):
        # Style 26.1: print(C) -> print(C, flush=True)
        print(f"[LOG] {message}")

    def compute_metrics(self, limit):
        # Style 23.1
        results = []

        # Style 24.1: range(C) -> range(0, C)
        for i in range(limit):
            val = math.sqrt(i)
            results.append(val)

            if i % 10 == 0:
                # Style 26.1: print with multiple args
                print("Checkpoint reached:", i)

        return results

def external_helper(x):
    return x * 2

def run_complex_test():
    # Style 25.1: Class instantiation
    processor = DataProcessor()

    # Style 25.1: Method call
    processor.log_event("Starting pipeline")

    iterations = 5
    # Style 24.1
    for _ in range(iterations):
        # Style 25.1
        data = processor.compute_metrics(20)

        for item in data:
            # Style 25.1: Global function call
            transformed = external_helper(item)
            if transformed > 5:
                # Style 26.1
                print("Significant value found")

if __name__ == "__main__":
    run_complex_test()
"""

    def test_23_1_list_init(self):
        """Test [] -> list()"""
        transformed, success = self.ist.transfer(styles=["23.1"], code=self.test_code)
        self.assertTrue(success)
        self.assertIn("self.history = list()", transformed)
        self.assertIn("results = list()", transformed)

    def test_24_1_range_param(self):
        """Test range(limit) -> range(0, limit)"""
        transformed, success = self.ist.transfer(styles=["24.1"], code=self.test_code)
        self.assertTrue(success)
        self.assertIn("range(0, limit)", transformed)
        self.assertIn("range(0, iterations)", transformed)

    def test_25_1_syntactic_sugar(self):
        """Test func(args) -> func.__call__(args)"""
        transformed, success = self.ist.transfer(styles=["25.1"], code=self.test_code)
        self.assertTrue(success)
        # Check various types of calls
        self.assertIn("DataProcessor.__call__()", transformed)
        self.assertIn("external_helper.__call__(item)", transformed)

    def test_26_1_keyword_param(self):
        """Test print(C) -> print(C, flush=True)"""
        transformed, success = self.ist.transfer(styles=["26.1"], code=self.test_code)
        self.assertTrue(success)
        self.assertIn('print(f"[LOG] {message}", flush=True)', transformed)
        self.assertIn('print("Checkpoint reached:", i, flush=True)', transformed)


if __name__ == "__main__":
    unittest.main()