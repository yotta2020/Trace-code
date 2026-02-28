import tree_sitter_cpp as tscpp
from tree_sitter import Language, Parser

def parse_cpp_code(code: str):
    """
    使用 tree-sitter 解析 C++ 代码并返回语法树
    """
    # 获取 C++ 语言的预编译 grammar
    CPP_LANGUAGE = Language(tscpp.language())
    
    # 初始化 Parser 并设置语言
    parser = Parser(CPP_LANGUAGE)
    
    # tree-sitter 需要处理 bytes
    code_bytes = bytes(code, "utf8")
    tree = parser.parse(code_bytes)
    return tree

def print_tree(node, depth=0):
    """
    递归打印 AST 节点
    """
    indent = "  " * depth
    # node.is_named 表示该节点是否是具名节点（非语法标点如换行、括号等）
    if node.is_named:
        # 打印节点类型和起止位置
        print(f"{indent}{node.type} [{node.start_point} - {node.end_point}]")
    else:
        # 可选：如果你也想看非具名节点（比如标点符合），可以取消下面注释
        # print(f"{indent}'{node.type}' [{node.start_point} - {node.end_point}]")
        pass
        
    for child in node.children:
        print_tree(child, depth + 1)

def main():
    sample_code = """\
    for (int i = 1; i <= 5; i++) {  // For Loop
        sum += i;
    }
"""
    print("=== 待解析的代码 ===")
    print(sample_code)
    
    print("=== AST 解析结果 ===")
    tree = parse_cpp_code(sample_code)
    print_tree(tree.root_node)
    
    # 直接将 Node 转换为字符串即可输出 S-表达式
    print("\n=== S-表达式 (S-expression) 输出 ===")
    print(str(tree.root_node))

if __name__ == "__main__":
    main()
