import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from ist_utils import get_indent, text, print_children
from transform.lang import get_lang


def match_function(root):
    lang = get_lang()
    function_map = {
        "c": "function_definition",
        "java": "method_declaration",
        "c_sharp": "local_function_statement",
        "python": "function_definition",
        "javascript": "function_declaration",
        "go": "function_declaration",
        "php": "function_definition",
    }

    def check(u):
        return u.type == function_map[lang]

    def match(u):
        if check(u):
            res.append(u)
        for v in u.children:
            match(v)

    res = []
    match(root)
    return res


def convert_deadcode_test_message(node, code):
    block_node = None
    block_mapping = {
        "c": "compound_statement",
        "java": "block",
        "c_sharp": "block",
        "python": "block",
        "javascript": "statement_block",
        "go": "block",
        "php": "compound_statement"
    }
    for u in node.children:
        if u.type == block_mapping[get_lang()]:
            block_node = u
            break
    if block_node is None:
        return
        
    if get_lang() == "c":
        deadcode = 'if (1 == -1) { printf("INFO Test message:aaaaa");}'
    elif get_lang() == "java":
        deadcode = 'if (1 == -1) { System.out.println("INFO Test message:aaaaa");}'
    elif get_lang() == "c_sharp":
        deadcode = 'if (1 == -1) { Console.WriteLine("INFO Test message:aaaaa");}'
    elif get_lang() == "python":
        deadcode = 'if 1 == -1: print("INFO Test message:aaaaa")'
    elif get_lang() == "javascript":
        deadcode = 'if (1 == -1) { console.log("INFO Test message:aaaaa");}'
    elif get_lang() == "go":
        deadcode = 'if 1 == -1 { fmt.Println("INFO Test message:aaaaa") }'
    elif get_lang() == "php":
        deadcode = 'if (1 == -1) { echo "INFO Test message:aaaaa"; }'
    else:
        return # 不支持的语言

    # --- 修复开始 ---
    if get_lang() == "python":
        if len(block_node.children) == 0:
            # Python 块为空 (例如，一个裸的 'def func():' 后面没有 'pass')
            # 这种情况在语法上不常见，但为了安全起见
            indent = get_indent(block_node.start_byte, code) + 4 # 猜测一个缩进
            return [(block_node.start_byte, f"\n{' '*indent}{deadcode}")]
        else:
            # 获取第一条语句 (例如 'pass' 或 'return')
            first_statement = block_node.children[0]
            indent = get_indent(first_statement.start_byte, code)
            # 在第一条语句 *之前* 插入，并在末尾添加缩进以保持原语句缩进
            return [(first_statement.start_byte, f"{' '*indent}{deadcode}\n{' '*indent}")]
    else:
        # 原始的 C/Java/PHP/Go 逻辑 (增加了安全检查)
        if len(block_node.children) < 2:
            # 处理空块 {}
            indent = get_indent(block_node.start_byte, code) + 4 # 猜测缩进
        else:
            # 从第一条语句获取缩进
            indent = get_indent(block_node.children[1].start_byte, code)
        
        # 在 '{' (即 children[0]) *之后* 插入
        return [(block_node.children[0].end_byte, f"\n{' '*indent}{deadcode}")]
    # --- 修复结束 ---


def convert_deadcode_233(node, code):
    block_node = None
    block_mapping = {"c": "compound_statement", "java": "block", "c_sharp": "block", "python": "block"}
    for u in node.children:
        if u.type == block_mapping[get_lang()]:
            block_node = u
            break
    if block_node is None:
        return
        
    if get_lang() == "java":
        deadcode = "System.out.println(233);"
    elif get_lang() == "c_sharp":
        deadcode = "Console.WriteLine(233);"
    elif get_lang() == "c":
        deadcode = 'printf("233233233233233233233233233233233233233\n");'
    elif get_lang() == "python":
        deadcode = 'if 1 == -1: print("233")'
    else:
        return # 不支持的语言

    # --- 修复开始 ---
    if get_lang() == "python":
        if len(block_node.children) == 0:
            indent = get_indent(block_node.start_byte, code) + 4
            return [(block_node.start_byte, f"\n{' '*indent}{deadcode}")]
        else:
            # 获取第一条语句
            first_statement = block_node.children[0]
            indent = get_indent(first_statement.start_byte, code)
            # 在第一条语句 *之前* 插入，并在末尾添加缩进以保持原语句缩进
            return [(first_statement.start_byte, f"{' '*indent}{deadcode}\n{' '*indent}")]
    else:
        # 原始的 C/Java 逻辑 (增加了安全检查)
        if len(block_node.children) < 2:
            # 处理空块 {}
            indent = get_indent(block_node.start_byte, code) + 4
        else:
            # 从第一条语句获取缩进
            indent = get_indent(block_node.children[1].start_byte, code)
        
        # 在 '{' (即 children[0]) *之后* 插入
        return [(block_node.children[0].end_byte, f"\n{' '*indent}{deadcode}")]
    # --- 修复结束 ---


def count_deadcode_test_message(root):
    return "INFO Test message:aaaaa" in text(root)


def count_deadcode_233(root):
    return "233" in text(root)