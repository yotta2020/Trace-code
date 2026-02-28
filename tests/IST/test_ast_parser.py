import sys
import os
from pathlib import Path

# Path Setup
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
    ist_module_path = os.path.join(project_root, 'src', 'data_preprocessing', 'IST')

    if not os.path.isdir(ist_module_path):
        raise ImportError(f"Cannot find IST module at: {ist_module_path}")

    sys.path.insert(0, ist_module_path)
    from transfer import StyleTransfer as IST

except ImportError as e:
    print(f"Path setup or import error: {e}")
    sys.exit(1)

# Hardcoded Python code for testing
TEST_CODE = """
def nested_example(n):
    for i in range(n):
        if i % 2 == 0:
            print(f"{i} is even")
        else:
            print(f"{i} is odd")
        print("---")
"""


def print_simple(root_node):
    """Simple version: Direct string representation of root node"""
    print("=" * 60)
    print("Simple Version: Root Node String Representation")
    print("=" * 60)
    print(root_node)
    print()


def print_recursive(node, depth=0):
    """Recursive version: Print all nodes with details"""
    indent = "  " * depth
    node_text = node.text.decode('utf-8') if node.text else ""

    print(f"{indent}Type: {node.type}")
    print(f"{indent}Position: [{node.start_byte}, {node.end_byte}]")

    if not node.children:
        print(f"{indent}Text: {repr(node_text)}")

    print()

    for child in node.children:
        print_recursive(child, depth + 1)


def print_tree_structure(node, depth=0, prefix=""):
    """Tree structure version: Indented hierarchical display"""
    indent = "  " * depth
    node_text = node.text.decode('utf-8') if node.text else ""

    if not node.children:
        text_display = node_text.replace('\n', '\\n')[:50]
        print(f"{indent}{prefix}{node.type} -> '{text_display}'")
    else:
        print(f"{indent}{prefix}{node.type}")

    for i, child in enumerate(node.children):
        is_last = i == len(node.children) - 1
        child_prefix = "|__ " if not is_last else "`__ "
        print_tree_structure(child, depth + 1, child_prefix)


def main():
    print("Initializing IST Parser for Python...")
    ist = IST(language="python")

    print("\nParsing hardcoded Python code...\n")
    tree = ist.parser.parse(bytes(TEST_CODE, "utf-8"))
    root_node = tree.root_node

    print_simple(root_node)

    print("=" * 60)
    print("Recursive Version: All Nodes with Details")
    print("=" * 60)
    print_recursive(root_node)

    print("=" * 60)
    print("Tree Structure Version: Hierarchical Display")
    print("=" * 60)
    print_tree_structure(root_node)

    print("\n" + "=" * 60)
    print("AST Parsing Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()