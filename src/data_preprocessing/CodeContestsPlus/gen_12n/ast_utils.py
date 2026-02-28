#!/usr/bin/env python3
"""
Optimized AST-based code analysis utilities using tree-sitter.
Provides minimal but essential code structure parsing for dataset generation.
"""

from typing import List, Tuple, Set, Optional
from tree_sitter import Language, Parser, Node, Tree

# === 语言模块映射 ===
LANGUAGE_MODULES = {
    "cpp": "tree_sitter_cpp",
    "java": "tree_sitter_java",
    "python": "tree_sitter_python",
}

# === Java节点类型映射 ===
JAVA_NODE_TYPES = {
    "block": "block",
    "variable_decl_types": [
        "local_variable_declaration",
        "field_declaration",
        "formal_parameter"
    ],
    "statement_types": [
        "expression_statement",
        "local_variable_declaration",
        "if_statement",
        "for_statement",
        "while_statement",
        "do_statement",
        "return_statement",
        "assert_statement",
    ],
}

# === Java关键字 ===
JAVA_KEYWORDS = {
    "abstract", "assert", "boolean", "break", "byte", "case", "catch",
    "char", "class", "const", "continue", "default", "do", "double",
    "else", "enum", "extends", "final", "finally", "float", "for",
    "goto", "if", "implements", "import", "instanceof", "int", "interface",
    "long", "native", "new", "package", "private", "protected", "public",
    "return", "short", "static", "strictfp", "super", "switch", "synchronized",
    "this", "throw", "throws", "transient", "try", "void", "volatile", "while",
    "true", "false", "null",
}

# === Java标准库（不重命名）===
JAVA_STDLIB_NAMES = {
    "String", "Integer", "Long", "Double", "Float", "Boolean", "Character",
    "Byte", "Short", "Object", "Class", "System", "Math", "Thread",
    "List", "ArrayList", "LinkedList", "Set", "HashSet", "TreeSet",
    "Map", "HashMap", "TreeMap", "Queue", "Deque", "Stack",
    "Exception", "RuntimeException", "IOException", "StringBuilder",
    "StringBuffer", "Arrays", "Collections", "Optional",
}


# === 基础解析器类 ===
class BaseASTParser:
    """AST解析器基类"""

    def __init__(self, language: str):
        self.language = language
        self.parser = Parser()
        self._init_language()

    def _init_language(self):
        """初始化tree-sitter语言包"""
        module_name = LANGUAGE_MODULES.get(self.language)
        if not module_name:
            raise ValueError(f"Unsupported language: {self.language}")

        lang_module = __import__(module_name)
        LANG = Language(lang_module.language())
        self.parser.language = LANG

    def parse(self, code: str) -> Tree:
        """解析代码，返回AST树"""
        return self.parser.parse(bytes(code, "utf8"))

    def get_text(self, node: Node, code: str) -> str:
        """从节点提取文本"""
        return code[node.start_byte:node.end_byte]

    # 子类需要实现的方法
    def find_statement_positions(self, code: str) -> List[int]:
        raise NotImplementedError

    def extract_clean_variables(self, code: str) -> List[Tuple[str, int, int]]:
        raise NotImplementedError

    def replace_variable_name(self, code: str, old_name: str, new_name: str) -> str:
        raise NotImplementedError


class CppASTParser(BaseASTParser):
    """C++ AST解析器"""

    def __init__(self):
        super().__init__("cpp")

    # 保留所有现有方法
    def find_statement_positions(self, code: str) -> List[int]:
        """查找C++语句位置"""
        tree = self.parse(code)
        positions = []

        def traverse(node: Node):
            if node.type == "compound_statement":
                for child in node.children:
                    if child.type in [
                        "expression_statement",
                        "declaration",
                        "if_statement",
                        "for_statement",
                        "while_statement",
                        "do_statement",
                        "return_statement",
                    ]:
                        positions.append(child.end_byte)
                    elif child.type == "{":
                        positions.append(child.end_byte)

            for child in node.children:
                traverse(child)

        traverse(tree.root_node)
        return sorted(set(positions))

    def extract_clean_variables(self, code: str) -> List[Tuple[str, int, int]]:
        """提取C++变量"""
        all_vars = self._find_variable_declarations(code)
        keywords = self._get_reserved_keywords()

        stdlib_names = {"std", "cout", "cin", "endl", "string", "vector", "map", "set"}

        clean_vars = []
        seen_names = set()

        for var_name, start, end in all_vars:
            if (var_name not in keywords and
                    var_name not in stdlib_names and
                    len(var_name) > 1 and
                    var_name not in seen_names):
                clean_vars.append((var_name, start, end))
                seen_names.add(var_name)

        return clean_vars

    def replace_variable_name(self, code: str, old_name: str, new_name: str) -> str:
        """替换C++变量名"""
        usages = self._find_all_variable_usages(code, old_name)

        if not usages:
            return code

        usages_sorted = sorted(usages, key=lambda x: x[0], reverse=True)

        result = code
        for start, end in usages_sorted:
            result = result[:start] + new_name + result[end:]

        return result

    def _find_variable_declarations(self, code: str) -> List[Tuple[str, int, int]]:
        """查找C++变量声明"""
        tree = self.parse(code)
        variables = []

        def traverse(node: Node):
            if node.type == "declaration":
                declarator_nodes = [n for n in node.children if n.type == "init_declarator"]
                for decl_node in declarator_nodes:
                    for child in decl_node.children:
                        if child.type == "identifier":
                            var_name = self.get_text(child, code)
                            variables.append((var_name, child.start_byte, child.end_byte))
                            break
                        elif child.type == "pointer_declarator":
                            id_node = self._find_identifier_in_node(child)
                            if id_node:
                                var_name = self.get_text(id_node, code)
                                variables.append((var_name, id_node.start_byte, id_node.end_byte))

            elif node.type == "parameter_declaration":
                declarator = None
                for child in node.children:
                    if child.type in ["identifier", "pointer_declarator", "reference_declarator"]:
                        declarator = child
                        break

                if declarator:
                    if declarator.type == "identifier":
                        var_name = self.get_text(declarator, code)
                        variables.append((var_name, declarator.start_byte, declarator.end_byte))
                    else:
                        id_node = self._find_identifier_in_node(declarator)
                        if id_node:
                            var_name = self.get_text(id_node, code)
                            variables.append((var_name, id_node.start_byte, id_node.end_byte))

            for child in node.children:
                traverse(child)

        traverse(tree.root_node)
        return variables

    def _find_all_variable_usages(self, code: str, var_name: str) -> List[Tuple[int, int]]:
        """查找C++变量所有使用位置"""
        tree = self.parse(code)
        usages = []

        def traverse(node: Node):
            if node.type == "identifier":
                text = self.get_text(node, code)
                if text == var_name:
                    usages.append((node.start_byte, node.end_byte))

            for child in node.children:
                traverse(child)

        traverse(tree.root_node)
        return usages

    def _find_identifier_in_node(self, node: Node) -> Optional[Node]:
        """递归查找identifier节点"""
        if node.type == "identifier":
            return node
        for child in node.children:
            result = self._find_identifier_in_node(child)
            if result:
                return result
        return None

    def _get_reserved_keywords(self) -> Set[str]:
        """返回C++关键字"""
        return {
            "if", "else", "for", "while", "do", "switch", "case", "default",
            "break", "continue", "return", "goto", "try", "catch", "throw",
            "int", "char", "float", "double", "void", "bool", "long", "short",
            "unsigned", "signed", "const", "static", "extern", "inline",
            "virtual", "public", "private", "protected", "class", "struct",
            "namespace", "using", "typedef", "template", "typename",
            "true", "false", "nullptr", "NULL",
        }


# === Java 解析器 ===
class JavaASTParser(BaseASTParser):
    """Java AST解析器"""

    def __init__(self):
        super().__init__("java")

    def find_statement_positions(self, code: str) -> List[int]:
        """
        查找Java方法体内的安全注入点
        关键：只在method_declaration的block内查找
        """
        tree = self.parse(code)
        positions = []

        def traverse(node: Node):
            # 只在方法体（block）内查找
            if node.type == "method_declaration":
                body = node.child_by_field_name("body")
                if body and body.type == "block":
                    # 在block的语句之间查找注入点
                    for child in body.children:
                        if child.type in JAVA_NODE_TYPES["statement_types"]:
                            positions.append(child.end_byte)
                        elif child.type == "{":
                            # 在开括号后也可以注入
                            positions.append(child.end_byte)

            # 继续递归查找嵌套的方法（如内部类中的方法）
            for child in node.children:
                traverse(child)

        traverse(tree.root_node)
        return sorted(set(positions))

    def extract_clean_variables(self, code: str) -> List[Tuple[str, int, int]]:
        """
        提取Java变量（局部变量、参数、成员变量）
        排除关键字、标准库类名
        """
        all_vars = self._find_variable_declarations(code)

        clean_vars = []
        seen_names = set()

        for var_name, start, end in all_vars:
            if (var_name not in JAVA_KEYWORDS and
                    var_name not in JAVA_STDLIB_NAMES and
                    len(var_name) > 1 and
                    var_name not in seen_names):
                clean_vars.append((var_name, start, end))
                seen_names.add(var_name)

        return clean_vars

    def replace_variable_name(self, code: str, old_name: str, new_name: str) -> str:
        """替换Java变量名"""
        usages = self._find_all_variable_usages(code, old_name)

        if not usages:
            return code

        usages_sorted = sorted(usages, key=lambda x: x[0], reverse=True)

        result = code
        for start, end in usages_sorted:
            result = result[:start] + new_name + result[end:]

        return result

    def _find_variable_declarations(self, code: str) -> List[Tuple[str, int, int]]:
        """
        查找Java变量声明
        包括：局部变量、成员变量、方法参数
        """
        tree = self.parse(code)
        variables = []

        def traverse(node: Node):
            # 局部变量声明：int x = 0;
            if node.type == "local_variable_declaration":
                declarators = [n for n in node.children if n.type == "variable_declarator"]
                for decl in declarators:
                    id_node = decl.child_by_field_name("name")
                    if id_node and id_node.type == "identifier":
                        var_name = self.get_text(id_node, code)
                        variables.append((var_name, id_node.start_byte, id_node.end_byte))

            # 成员变量声明：private int count;
            elif node.type == "field_declaration":
                declarators = [n for n in node.children if n.type == "variable_declarator"]
                for decl in declarators:
                    id_node = decl.child_by_field_name("name")
                    if id_node and id_node.type == "identifier":
                        var_name = self.get_text(id_node, code)
                        variables.append((var_name, id_node.start_byte, id_node.end_byte))

            # 方法参数：void foo(int x, String y)
            elif node.type == "formal_parameter":
                id_node = node.child_by_field_name("name")
                if id_node and id_node.type == "identifier":
                    var_name = self.get_text(id_node, code)
                    variables.append((var_name, id_node.start_byte, id_node.end_byte))

            for child in node.children:
                traverse(child)

        traverse(tree.root_node)
        return variables

    def _find_all_variable_usages(self, code: str, var_name: str) -> List[Tuple[int, int]]:
        """查找Java变量所有使用位置"""
        tree = self.parse(code)
        usages = []

        def traverse(node: Node):
            if node.type == "identifier":
                text = self.get_text(node, code)
                if text == var_name:
                    usages.append((node.start_byte, node.end_byte))

            for child in node.children:
                traverse(child)

        traverse(tree.root_node)
        return usages


# === 全局单例工厂 ===
_parsers = {}


def get_ast_parser(language: str) -> BaseASTParser:
    """
    获取对应语言的AST解析器（单例模式）

    Args:
        language: 语言标识符 (cpp, java, python)

    Returns:
        对应语言的AST解析器实例
    """
    if language not in _parsers:
        if language == "cpp":
            _parsers[language] = CppASTParser()
        elif language == "java":
            _parsers[language] = JavaASTParser()
        elif language == "python":
            # 暂不实现
            raise NotImplementedError(f"Python AST parser not implemented yet")
        else:
            raise ValueError(f"Unsupported language: {language}")

    return _parsers[language]


# === 公共API（兼容现有代码）===

def find_safe_injection_points(code: str, language: str = "cpp") -> List[int]:
    """
    查找安全的代码注入点

    Args:
        code: 源代码
        language: 编程语言

    Returns:
        注入位置的字节偏移列表
    """
    parser = get_ast_parser(language)
    return parser.find_statement_positions(code)


def find_renamable_variables(code: str, language: str = "cpp") -> List[str]:
    """
    查找可以安全重命名的变量

    Args:
        code: 源代码
        language: 编程语言

    Returns:
        变量名列表
    """
    parser = get_ast_parser(language)
    vars_with_pos = parser.extract_clean_variables(code)
    return [var_name for var_name, _, _ in vars_with_pos]


def rename_variable_ast(code: str, old_name: str, new_name: str, language: str = "cpp") -> str:
    """
    使用AST重命名变量（不影响注释和字符串）

    Args:
        code: 源代码
        old_name: 旧变量名
        new_name: 新变量名
        language: 编程语言

    Returns:
        重命名后的代码
    """
    parser = get_ast_parser(language)
    return parser.replace_variable_name(code, old_name, new_name)


# 保留原有的get_cpp_parser()以兼容旧代码
def get_cpp_parser() -> CppASTParser:
    """向后兼容：获取C++解析器"""
    return get_ast_parser("cpp")


# === 注释移除功能 ===

def remove_comments(code: str, language: str = "cpp") -> str:
    """
    移除代码中的所有注释

    Args:
        code: 源代码
        language: 编程语言 (cpp, java, python)

    Returns:
        移除注释后的代码
    """
    parser = get_ast_parser(language)
    tree = parser.parse(code)

    # 定义注释节点类型
    comment_types = {
        "cpp": ["comment"],
        "java": ["line_comment", "block_comment"],
        "python": ["comment"],
    }

    target_comment_types = comment_types.get(language, ["comment"])

    # 收集所有注释节点的位置
    comments_to_remove = []

    def traverse(node: Node):
        if node.type in target_comment_types:
            comments_to_remove.append((node.start_byte, node.end_byte))
        for child in node.children:
            traverse(child)

    traverse(tree.root_node)

    if not comments_to_remove:
        return code

    # 从后往前删除，避免索引偏移
    comments_sorted = sorted(comments_to_remove, key=lambda x: x[0], reverse=True)

    result = code
    for start, end in comments_sorted:
        # 删除注释内容
        result = result[:start] + result[end:]

    # 清理多余空行（连续超过2个换行符压缩为2个）
    import re
    result = re.sub(r'\n{3,}', '\n\n', result)

    return result