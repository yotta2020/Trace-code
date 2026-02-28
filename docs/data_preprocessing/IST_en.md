# IST Data Poisoning Component API Documentation

## 1. Introduction

IST (**S**tyle **C**hange by **T**ree **S**itter) is a data poisoning component used in the data preprocessing pipeline. It leverages AST capabilities to apply code equivalence transformations (style changes), injecting specific triggers into code to build poisoned datasets. 

It currently supports multiple languages, primarily including **C, C++, Java, C#, and Python**.

## 2. API Usage

For developers who want to integrate IST dynamically into other Python scripts instead of using batch scripts, the following APIs are provided.

### 2.1 Initialization

First, initialize the `IST` class by specifying the target programming language:

```python
from data_preprocessing.IST.transfer import StyleTransfer # or equivalent inner mapping
from data_preprocessing.IST import IST # Assumption based on standard usage

# Valid language parameters usually include 'c', 'java', 'python', 'c_sharp'
ist = IST('c')
```

### 2.2 Transform a Single Code Snippet

To process a single piece of code and inject a specific style:

```python
# The first parameter is a list of target styles to apply (e.g., [8.11] or [11.1, 9.2]).
# The second parameter is the original code string.
new_code, is_successfully_changed = ist.change_file_style([11.1], code)

if is_successfully_changed:
    print("Code transformed successfully:\n", new_code)
else:
    print("Transformation failed or skipped.")
```

### 2.3 Transform an Entire Directory

You can process all files within a specific folder sequentially.

```python
# change_dir_style(style_choice_list, input_dir, output_dir, output_choice)
# output_choice options:
#  -1: Do not output to files.
#   0: Only output successfully transformed files.
#   1: Output all files, separated into 'succ' and 'fail' subdirectories.
#   2: Output all files merged into the target directory.

ist.change_dir_style(
    [11.1], 
    'dataset/programs/C/original', 
    'change/11.1', 
    output_choice=2
)
```

### 2.4 Additional Utilities

Calculate the popularity of styles in a file or directory:

```python
# Get popularity of a specific style in a single code snippet
popularity_file = ist.get_file_popularity(5.1, code)

# Get popularity across a directory
popularity_dir = ist.get_dir_popularity(5.1, 'dataset/ProgramData/2')
```

Visualize code structure (Requires Graphviz):

```python
# Output an AST PDF of the original code
ist.see_tree(code)

# Tokenize code
tokens = ist.tokenize(code)
```

## 3. Supported Conversion Styles

The IST tool supports numerous equivalent transformation rules which function as potential backdoor triggers. Note that different programming languages may have varying support levels for specific rules based on grammar. The table below covers all style IDs based on IST specifications:

| Category | Typical Usage / Description | Examples of Supported IDs |
| :--- | :--- | :--- |
| **1. Variable Naming** | Changes casing or prefix of identifier names. | `0.1`(camelCase), `0.2`(PascalCase), `0.3`(snake_case), `0.5`(prefix `_`), `0.7`(UPPERCASE). |
| **2. Token Substitution** | Substitute shell or ruby style tokens. | `-3.1`(sh style), `-3.2`(rb style) |
| **3. Invisible Characters** | Inject unicode invisible characters (ZWSP, ZWNJ, etc.). | `-2.1`(ZWSP), `-2.2`(ZWNJ), `-2.3`(LRO) |
| **4. Dead Code Insertion** | Insert unreachable logic. | `-1.1`, `-1.2` |
| **6. Bracket Handling** | Add or remove braces `{}` around control blocks. | `1.1`(Remove `{}`), `1.2`(Add `{}`) |
| **7. Augmented Assignment** | Toggle `x = x + 1` vs `x += 1`. | `2.1`(Standard), `2.2`(Augmented) |
| **8. Comparison Operator** | Rewrite relational logic. | `3.1`(`<`), `3.2`(`>`), `3.3`(`==`), `3.4`(`!=`) |
| **9. For Loop Updates** | Iteration step style. | `4.1`(`i--`), `4.2`(`i++`), `4.3`(`i+=1`), `4.4`(`i=i+1`) |
| **10. Array Handling** | Definition & Memory allocation styles. | `5.1`(dynamic `malloc`), `5.2`(static array), `6.1`(pointer `*p`), `6.2`(index `arr[i]`) |
| **11. Variable Declarations**| Position and inline initialization handling. | `7.1`(Split declarations), `7.2`(Merge), `8.1`(Move to start), `9.1`(Split assign), `9.2`(Merge assign) |
| **12. For Loop Elements** | Omit or shuffle `(init; cond; inc)` structure formats. | `10.0`到`10.7` (`abc`, `obc`, `aoc`, etc.) |
| **13. Loop Type Conversion** | Convert between iteration types. | `11.1`(`while` -> `for`), `11.2`(`for` -> `while`), `11.3`(`do-while`) |
| **14. Infinite Loops** | Replace condition logic with true statements. | `11.4`(infinite `while`), `12.2`(infinite `for`) |
| **15. Control Flow Adjust.** | `break`/`goto` swaps, conditionally adding `return`/`!` | `13.1`(`goto`), `14.1`(`! expr`), `16.1`(`switch`), `17.1`(Flatten if), `18.1`(Remove else) |
| **16. Ternary Operators** | Convert `if-else` to `? :`. | `19.1`(to ternary), `19.2`(to if-else) |
| **17. Function Nesting** | Flattening or nesting embedded functions. | `20.1`(nest), `20.2`(flatten) |
| **18. Recursion / Iteration**| Structural translation between algorithms. | `21.1`(to iterative), `21.2`(to recursive) |
