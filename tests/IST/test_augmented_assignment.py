import unittest
import sys
import os
from pathlib import Path

# --- Path Setup ---
# This setup assumes the test is run from the project root.
# It tries to find the 'ist_utils' and 'transform' modules.
try:
    # A more robust way to handle pathing for tests.
    # This assumes 'src' is in the project root.
    project_root = Path(__file__).resolve().parents[2]
    src_path = project_root / 'src'
    sys.path.insert(0, str(src_path))

    # This is your main transformation script.
    # It must be in a location findable from 'src'.
    # Assuming it's at 'src/data_preprocessing/IST/transformations/augmented_assignment.py'
    # And that the IST class can import it.
    from data_preprocessing.IST.transfer import StyleTransfer as IST

except (ImportError, IndexError) as e:
    print(f"Could not set up paths or import IST. Please check your project structure. Error: {e}")


    # We will use a mock IST if the real one isn't found, so tests can still be analyzed.
    class MockIST:
        def __init__(self, language="python"):
            print("Warning: Using MockIST. Real transformations are not being tested.")

        def transfer(self, styles, code):
            return "mocked_code", False  # Default to failure for debugging


    IST = MockIST


# --- End Path Setup ---


class TestAugmentedAssignmentTransformations(unittest.TestCase):
    """
    Tests for 'augmented_assignment' style transformations (2.1 and 2.2).
    """

    @classmethod
    def setUpClass(cls):
        """Initializes the IST instance before any tests run."""
        try:
            cls.ist = IST(language="python")
        except Exception as e:
            raise unittest.SkipTest(f"Failed to initialize IST: {e}")

        # Code containing style 2.2 (augmented)
        cls.augmented_code = """
def process(data):
    total = 0
    count = 10
    total += data
    count -= 1
    total *= 2
    return total / count
"""
        # Code containing style 2.1 (non_augmented)
        cls.non_augmented_code = """
def process(data):
    total = 0
    count = 10
    total = total + data
    count = count - 1
    total = total * 2
    return total / count
"""

    def test_style_2_1_non_augmented(self):
        """Tests style 2.1 (augmented -> non_augmented)"""
        style = "2.1"

        # 1. Check original code (augmented_code)
        self.assertIn("total += data", self.augmented_code)
        self.assertIn("count -= 1", self.augmented_code)
        self.assertNotIn("total = total + data", self.augmented_code)

        # 2. Apply transformation
        transformed_code, success = self.ist.transfer(styles=[style], code=self.augmented_code)

        # 3. Check status
        self.assertTrue(success, f"Style {style} (non_augmented) should return success=True")
        self.assertNotEqual(self.augmented_code, transformed_code, "Code should have been modified")

        # 4. Check transformed code
        self.assertIn("total = total + data", transformed_code)
        self.assertIn("count = count - 1", transformed_code)
        self.assertIn("total = total * 2", transformed_code)
        self.assertNotIn("total += data", transformed_code)
        self.assertNotIn("count -= 1", transformed_code)

    def test_style_2_2_augmented(self):
        """Tests style 2.2 (non_augmented -> augmented)"""
        style = "2.2"

        # 1. Check original code (non_augmented_code)
        self.assertIn("total = total + data", self.non_augmented_code)
        self.assertIn("count = count - 1", self.non_augmented_code)
        self.assertNotIn("total += data", self.non_augmented_code)

        # 2. Apply transformation (with the bug fix applied)
        transformed_code, success = self.ist.transfer(styles=[style], code=self.non_augmented_code)

        # 3. Check status
        self.assertTrue(success, f"Style {style} (augmented) should return success=True after the fix")
        self.assertNotEqual(self.non_augmented_code, transformed_code, "Code should have been modified")

        # 4. Check transformed code
        self.assertIn("total += data", transformed_code)
        self.assertIn("count -= 1", transformed_code)
        self.assertIn("total *= 2", transformed_code)
        self.assertNotIn("total = total + data", transformed_code)
        self.assertNotIn("count = count - 1", transformed_code)


if __name__ == "__main__":
    # Note: Running this directly requires the path setup to work correctly.
    # It's often better to run tests from the project's root directory.
    # Example: python -m unittest tests/IST/test_augmented_assignment.py
    unittest.main(argv=['first-arg-is-ignored'], exit=False)