#!/usr/bin/env python3
"""
Docstring and Comment Remover for Code Search Task

This module removes docstrings and comments from code to prevent model cheating
in code search tasks. The model should learn from code logic, not from comments.

Supported languages: Python, Java
"""

import sys
import os
from tree_sitter import Parser, Language

# Import tree-sitter language modules
try:
    import tree_sitter_python
    import tree_sitter_java
    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False
    print("Warning: tree-sitter language modules not available. Docstring removal disabled.")


class DocstringRemover:
    """Remove docstrings and comments from source code using tree-sitter."""

    def __init__(self, language: str):
        """
        Initialize the DocstringRemover.

        Args:
            language: Programming language ('python' or 'java')
        """
        self.language = language.lower()

        if not TREE_SITTER_AVAILABLE:
            self.parser = None
            return

        # Initialize parser for the specified language
        if self.language == 'python':
            LANGUAGE = Language(tree_sitter_python.language())
        elif self.language == 'java':
            LANGUAGE = Language(tree_sitter_java.language())
        else:
            raise ValueError(f"Unsupported language: {language}")

        self.parser = Parser(LANGUAGE)

    def remove_docstrings(self, code: str) -> str:
        """
        Remove docstrings and comments from code.

        Args:
            code: Source code string

        Returns:
            Code with docstrings/comments removed
        """
        if not TREE_SITTER_AVAILABLE or self.parser is None:
            # Fallback: return original code if tree-sitter not available
            return code

        if not code or not code.strip():
            return code

        try:
            if self.language == 'python':
                return self._remove_python_docstrings(code)
            elif self.language == 'java':
                return self._remove_java_comments(code)
            else:
                return code
        except Exception as e:
            # If parsing fails, return original code (conservative approach)
            print(f"Warning: Failed to remove docstrings: {e}")
            return code

    def _remove_python_docstrings(self, code: str) -> str:
        """
        Remove ALL Python comments and docstrings.

        Removes:
        1. # single-line comments
        2. Docstrings (string literals as first statement in module/class/function)
        3. Multi-line string literals (often used as block comments)
        """
        tree = self.parser.parse(bytes(code, 'utf-8'))

        # Collect byte ranges to remove
        ranges_to_remove = []

        def visit_node(node):
            """Recursively visit AST nodes to find comments and docstrings."""

            # 1. Remove # comments
            if node.type == 'comment':
                ranges_to_remove.append((node.start_byte, node.end_byte))

            # 2. Remove docstrings (first statement in function/class/module)
            elif node.type in ['function_definition', 'class_definition', 'module']:
                # Look for body
                body_node = None
                for child in node.children:
                    if child.type == 'block':
                        body_node = child
                        break
                
                # If no direct block (e.g. module), children are directly in node
                children_to_scan = body_node.children if body_node else node.children

                if len(children_to_scan) > 0:
                    # First statement in the body
                    first_stmt = None
                    for child in children_to_scan:
                        if child.type == 'expression_statement':
                            first_stmt = child
                            break
                        # Skip comments to find the real first statement
                        elif child.type == 'comment':
                            continue
                        else:
                            # If we hit something else (like import), stop looking for docstring
                            break

                    if first_stmt:
                        # Check if it's a string (docstring)
                        for child in first_stmt.children:
                            if child.type == 'string':
                                ranges_to_remove.append((first_stmt.start_byte, first_stmt.end_byte))
                                break

            # 3. Remove standalone string literals (often used as block comments)
            elif node.type == 'expression_statement':
                # Note: This logic previously caused double-counting because node.parent is usually 'block', 
                # not 'function_definition'.
                # Instead of complex parent checking, we now rely on strict deduplication at the end.
                
                # Check if this expression_statement only contains a string
                is_string_literal = False
                for child in node.children:
                    if child.type == 'string':
                        is_string_literal = True
                        break
                
                if is_string_literal:
                    ranges_to_remove.append((node.start_byte, node.end_byte))

            # Recursively visit children
            for child in node.children:
                visit_node(child)

        visit_node(tree.root_node)

        # ---------------------------------------------------------
        # BUG FIX: Deduplicate ranges to prevent double deletion
        # ---------------------------------------------------------
        # Using set() removes duplicates: ((20, 60), (20, 60)) -> {(20, 60)}
        unique_ranges = list(set(ranges_to_remove))

        # Remove ranges in reverse order to maintain byte offsets
        # Critical: Sort by start position (reverse) and remove from back to front
        unique_ranges.sort(reverse=True, key=lambda x: x[0])

        # CRITICAL FIX: Use slicing to delete ranges
        code_bytes = bytes(code, 'utf-8')

        for start, end in unique_ranges:
            # Delete the range by slicing: before + after
            code_bytes = code_bytes[:start] + code_bytes[end:]

        result = code_bytes.decode('utf-8')

        # Clean up excessive blank lines
        lines = result.split('\n')
        cleaned_lines = []
        prev_blank = False

        for line in lines:
            is_blank = not line.strip()
            if is_blank and prev_blank:
                continue  # Skip consecutive blank lines
            cleaned_lines.append(line)
            prev_blank = is_blank

        return '\n'.join(cleaned_lines)

    def _remove_java_comments(self, code: str) -> str:
        """
        Remove Java comments (line comments, block comments, and Javadoc).
        """
        tree = self.parser.parse(bytes(code, 'utf-8'))

        # Collect byte ranges to remove
        ranges_to_remove = []

        def visit_node(node):
            """Recursively visit AST nodes to find comments."""

            # Check if node is a comment
            if node.type in ['comment', 'block_comment', 'line_comment']:
                ranges_to_remove.append((node.start_byte, node.end_byte))

            # Recursively visit children
            for child in node.children:
                visit_node(child)

        visit_node(tree.root_node)

        # ---------------------------------------------------------
        # BUG FIX: Deduplicate ranges (Safety measure for Java too)
        # ---------------------------------------------------------
        unique_ranges = list(set(ranges_to_remove))

        # Remove ranges in reverse order
        unique_ranges.sort(reverse=True, key=lambda x: x[0])

        code_bytes = bytes(code, 'utf-8')

        for start, end in unique_ranges:
            code_bytes = code_bytes[:start] + code_bytes[end:]

        result = code_bytes.decode('utf-8')

        # Clean up excessive blank lines
        lines = result.split('\n')
        cleaned_lines = []
        prev_blank = False

        for line in lines:
            is_blank = not line.strip()
            if is_blank and prev_blank:
                continue
            cleaned_lines.append(line)
            prev_blank = is_blank

        return '\n'.join(cleaned_lines)


# Convenience function for direct use
def remove_docstrings(code: str, language: str) -> str:
    """
    Remove docstrings/comments from code.

    Args:
        code: Source code string
        language: Programming language ('python' or 'java')

    Returns:
        Code with docstrings/comments removed
    """
    remover = DocstringRemover(language)
    return remover.remove_docstrings(code)

# (Testing code omitted for brevity)