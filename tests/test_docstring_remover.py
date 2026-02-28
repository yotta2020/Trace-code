#!/usr/bin/env python3
"""
Test script for DocstringRemover

This script tests the docstring removal functionality for Python and Java code.
"""

import sys
import os

# Add src path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
src_path = os.path.join(project_root, "src", "data_preprocessing", "cs")
sys.path.insert(0, src_path)

from docstring_remover import DocstringRemover


def test_python_docstring_removal():
    """Test Python docstring removal."""
    print("=" * 80)
    print("TEST 1: Python Function Docstring")
    print("=" * 80)

    python_code = '''
def calculate_sum(a, b):
    """
    Calculate the sum of two numbers.

    Args:
        a: First number
        b: Second number

    Returns:
        The sum of a and b
    """
    # This is a single-line comment
    result = a + b  # Inline comment
    return result
'''

    print("Original Code:")
    print(python_code)

    remover = DocstringRemover('python')
    cleaned = remover.remove_docstrings(python_code)

    print("\nCleaned Code:")
    print(cleaned)

    # Verify docstring and comments are removed
    assert '"""' not in cleaned, "Docstring not removed!"
    assert '#' not in cleaned, "# comments not removed!"
    assert 'result = a + b' in cleaned, "Code logic removed!"
    assert 'return result' in cleaned, "Code logic removed!"

    print("\n✓ Test 1 passed!\n")


def test_python_class_docstring():
    """Test Python class docstring removal."""
    print("=" * 80)
    print("TEST 2: Python Class Docstring and Comments")
    print("=" * 80)

    python_code = '''
class Calculator:
    """This is a calculator class."""

    # Class-level comment

    def __init__(self):
        """Initialize the calculator."""
        self.result = 0  # Initialize to zero

        """
        This is a multi-line string literal
        used as a block comment
        """

    def add(self, x):
        """Add x to result."""
        # Add x to the current result
        self.result += x
        return self.result
'''

    print("Original Code:")
    print(python_code)

    remover = DocstringRemover('python')
    cleaned = remover.remove_docstrings(python_code)

    print("\nCleaned Code:")
    print(cleaned)

    # Verify all comments and docstrings are removed
    assert '"""' not in cleaned, "Docstring not removed!"
    assert '#' not in cleaned, "# comments not removed!"
    assert 'class Calculator:' in cleaned, "Class definition removed!"
    assert 'self.result = 0' in cleaned, "Code logic removed!"
    assert 'self.result += x' in cleaned, "Code logic removed!"

    print("\n✓ Test 2 passed!\n")


def test_java_comment_removal():
    """Test Java comment removal."""
    print("=" * 80)
    print("TEST 3: Java Comment Removal")
    print("=" * 80)

    java_code = '''
/**
 * This is a Javadoc comment for the class.
 * It should be removed.
 */
public class Calculator {
    // This is a single line comment
    private int result;

    /**
     * Constructor Javadoc.
     */
    public Calculator() {
        this.result = 0; /* inline comment */
    }

    /* Multi-line comment
       that spans multiple lines
       and should be removed */
    public int add(int x) {
        this.result += x; // Add x to result
        return this.result;
    }
}
'''

    print("Original Code:")
    print(java_code)

    remover = DocstringRemover('java')
    cleaned = remover.remove_docstrings(java_code)

    print("\nCleaned Code:")
    print(cleaned)

    # Verify comments are removed
    assert '/**' not in cleaned, "Javadoc not removed!"
    assert '//' not in cleaned, "Line comment not removed!"
    assert '/*' not in cleaned, "Block comment not removed!"
    assert 'public class Calculator' in cleaned, "Class definition removed!"
    assert 'this.result = 0;' in cleaned, "Code logic removed!"
    assert 'this.result += x;' in cleaned, "Code logic removed!"

    print("\n✓ Test 3 passed!\n")


def test_no_docstring():
    """Test code without docstrings."""
    print("=" * 80)
    print("TEST 4: Code Without Docstrings")
    print("=" * 80)

    python_code = '''
def simple_function(x, y):
    return x + y
'''

    print("Original Code:")
    print(python_code)

    remover = DocstringRemover('python')
    cleaned = remover.remove_docstrings(python_code)

    print("\nCleaned Code:")
    print(cleaned)

    # Code should remain unchanged
    assert 'def simple_function(x, y):' in cleaned
    assert 'return x + y' in cleaned

    print("\n✓ Test 4 passed!\n")


def main():
    """Run all tests."""
    print("\n")
    print("*" * 80)
    print("DOCSTRING REMOVER TEST SUITE")
    print("*" * 80)
    print("\n")

    try:
        test_python_docstring_removal()
        test_python_class_docstring()
        test_java_comment_removal()
        test_no_docstring()

        print("=" * 80)
        print("ALL TESTS PASSED! ✓")
        print("=" * 80)
        return 0

    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
