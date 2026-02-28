import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from ist_utils import *
from tree_sitter import Parser, Language
from pathlib import Path

SUPPORTED_LANGUAGES = {
    "python": "tree_sitter_python",
    "c": "tree_sitter_c",
    "javascript": "tree_sitter_javascript",
    "cpp": "tree_sitter_cpp",
    "java": "tree_sitter_java",
    "go": "tree_sitter_go",
    "php": "tree_sitter_php",
}


class StyleTransfer:
    def __init__(self, language, insert_position="suffix"):
        self.language = language
        self.insert_position = insert_position
        lang_module = __import__(SUPPORTED_LANGUAGES[language])
        if language == 'php':
            LANGUAGE = Language(lang_module.language_php())
        else:
            LANGUAGE = Language(lang_module.language())
        self.parser = Parser(LANGUAGE)

        from transform.config import transformation_operators as op
        from transform.lang import set_lang

        set_lang(language)

        self.op = op

        # * style: (type, subtype, prepare_style)
        self.style_dict = {
            "-3.1": ("tokensub", "sh", None),
            "-3.2": ("tokensub", "rb", None),
            "-2.1": ("invichar", "ZWSP", None),
            "-2.2": ("invichar", "ZWNJ", None),
            "-2.3": ("invichar", "LRO", None),
            "-2.4": ("invichar", "BKSP", None),
            "-1.1": ("deadcode", "deadcode_test_message", None),
            "-1.2": ("deadcode", "deadcode_233", None),
            "0.0": ("clean", "clean", None),
            "0.1": ("identifier_name", "camel", None),
            "0.2": ("identifier_name", "pascal", None),
            "0.3": ("identifier_name", "snake", None),
            "0.4": ("identifier_name", "hungarian", None),
            "0.5": ("identifier_name", "init_underscore", None),
            "0.6": ("identifier_name", "init_dollar", None),
            "1.1": ("bracket", "del_bracket", None),
            "1.2": ("bracket", "add_bracket", None),
            "2.1": ("augmented_assignment", "non_augmented", None),
            "2.2": ("augmented_assignment", "augmented", None),
            "3.1": ("cmp", "smaller", None),
            "3.2": ("cmp", "bigger", None),
            "3.3": ("cmp", "equal", None),
            "3.4": ("cmp", "not_equal", None),
            "4.1": ("for_update", "left", None),
            "4.2": ("for_update", "right", None),
            "4.3": ("for_update", "augment", None),
            "4.4": ("for_update", "assignment", None),
            "5.1": ("array_definition", "dyn_mem", None),
            "5.2": ("array_definition", "static_mem", None),
            "6.1": ("array_access", "pointer", None),
            "6.2": ("array_access", "array", None),
            "7.1": ("declare_lines", "split", None),
            "7.2": ("declare_lines", "merge", None),
            "8.1": ("declare_position", "first", None),
            "8.2": ("declare_position", "temp", None),
            "9.1": ("declare_assign", "split", None),
            "9.2": ("declare_assign", "merge", None),
            "10.0": ("for_format", "abc", None),
            "10.1": ("for_format", "obc", "10.0"),
            "10.2": ("for_format", "aoc", "10.0"),
            "10.3": ("for_format", "abo", "10.0"),
            "10.4": ("for_format", "aoo", "10.0"),
            "10.5": ("for_format", "obo", "10.0"),
            "10.6": ("for_format", "ooc", "10.0"),
            "10.7": ("for_format", "ooo", "10.0"),
            "11.1": ("for_while", "for", None),
            "11.2": ("for_while", "while", None),
            "11.3": ("for_while", "do_while", None),
            "11.4": ("loop_infinite", "infinite_while", None),
            "12.1": ("loop_infinite", "finite_for", None),
            "12.2": ("loop_infinite", "infinite_for", None),
            "12.3": ("loop_infinite", "finite_while", None),
            "12.4": ("loop_infinite", "infinite_while", None),
            "13.1": ("break_goto", "goto", None),
            "13.2": ("break_goto", "break", None),
            "14.1": ("if_exclamation", "not_exclamation", None),
            "14.2": ("if_exclamation", "exclamation", None),
            "15.1": ("if_return", "not_return", None),
            "15.2": ("if_return", "return", None),
            "16.1": ("if_switch", "switch", None),
            "16.2": ("if_switch", "if", None),
            "17.1": ("if_nested", "not_nested", None),
            "17.2": ("if_nested", "nested", None),
            "18.1": ("if_else", "not_else", None),
            "18.2": ("if_else", "else", None),
            "19.1": ("ternary", "to_ternary", None),
            "19.2": ("ternary", "to_if", None),
            "20.1": ("func_nested", "nested", None),
            "20.2": ("func_nested", "not_nested", None),
            "21.1": ("recursive_iterative", "to_iterative", None),
            "21.2": ("recursive_iterative", "to_recursive", None),
            "22.1": ("for_index", "temp", "10.0_1.2"),
            "23.1": ("list_init", "to_list_func", None),  # C = [] -> C = list()
            "24.1": ("range_param", "explicit_start", None),  # range(C) -> range(0, C)
            "25.1": ("syntactic_sugar", "call_method", None),  # C() -> C.__call__()
            "26.1": ("keyword_param", "add_flush", None),  # print(C) -> print(C, flush=True)
        }

    def transfer(self, styles=[], code="", insert_position=None):
        if not isinstance(styles, list):
            styles = [styles]
        if len(styles) == 0:
            return code, 0
        succs = []
        for style in styles:
            if style == "8.1":
                if self.get_style(code=code, styles=["8.1"])["8.1"] > 0:
                    succs.append(int(1))
                    continue
            raw_code = code

            (style_type, style_subtype, prepare_styles) = self.style_dict[style]
            if prepare_styles is not None:
                code, _ = self.transfer(prepare_styles.split('_'), code)

            # 检查操作符是否存在
            if style_type not in self.op or style_subtype not in self.op[style_type]:
                succs.append(0)
                continue

            op_tuple = self.op[style_type][style_subtype]
            if not op_tuple or len(op_tuple) < 3:
                succs.append(0)
                continue

            AST = self.parser.parse(bytes(code, encoding="utf-8"))
            (match_func, convert_func, _) = op_tuple
            operations = []
            insert_pos = insert_position or self.insert_position
            match_nodes = match_func(AST.root_node)
            if len(match_nodes) == 0:
                # succs.append(int(style == "0.0"))
                succs.append(0)
                continue

            dynamic_styles = ["20.1", "20.2"]
            if style in dynamic_styles:
                while len(match_nodes) > 0:
                    node = match_nodes[0]
                    if get_parameter_count(convert_func) == 1:
                        op = convert_func(node)
                    else:
                        op = convert_func(node, code)
                    if op is not None:
                        operations.extend(op)
                        code = replace_from_blob(operations, code)
                        operations = []
                        AST = self.parser.parse(bytes(code, encoding="utf-8"))
                        match_nodes = match_func(AST.root_node)
            else:
                for node in match_nodes:
                    if get_parameter_count(convert_func) == 1:
                        op = convert_func(node)
                    else:
                        op = convert_func(
                            node, insert_pos if style in ["-3.1", "-3.2"] else code
                        )
                    if op is not None:
                        operations.extend(op)

                code = replace_from_blob(operations, code)
            succ = raw_code.replace(" ", "").replace("\n", "").replace(
                "\t", ""
            ) != code.replace(" ", "").replace("\n", "").replace("\t", "")
            succs.append(int(succ))
        return code, 0 not in succs

    def get_style(self, code="", styles=[]):
        if not isinstance(styles, list):
            styles = [styles]
        res = {}
        if len(styles) == 0:
            styles = list(self.style_dict.keys())
        for style in styles:
            (style_type, style_subtype, prepare_styles) = self.style_dict[style]
            
            # ✅ 修复1: 使用副本，避免修改原始 code
            current_code = code
            if prepare_styles is not None:
                current_code, _ = self.transfer(prepare_styles.split('_'), current_code)

            # ✅ 修复2: 处理空代码或无效代码
            if not current_code or not current_code.strip():
                res[style] = 0
                continue

            AST = self.parser.parse(bytes(current_code, encoding="utf-8"))

            # ✅ 修复3: 检查 AST 是否有效
            if AST is None or AST.root_node is None:
                res[style] = 0
                continue

            # ✅ 修复5: 检查操作符是否存在
            if style_type not in self.op:
                res[style] = 0
                continue

            if style_subtype not in self.op[style_type]:
                res[style] = 0
                continue

            op_tuple = self.op[style_type][style_subtype]
            # 空元组或不完整的操作符定义
            if not op_tuple or len(op_tuple) < 3:
                res[style] = 0
                continue

            (_, _, count_func) = op_tuple

            # ✅ 修复4: 处理 count_func 可能返回 list 或抛出异常
            try:
                count_result = count_func(AST.root_node)
                if count_result is None:
                    count_result = 0
                elif isinstance(count_result, list):
                    count_result = len(count_result)
            except (AttributeError, TypeError, UnboundLocalError, NameError) as e:
                # 处理各种可能的错误：
                # - 'NoneType' object has no attribute 'children'
                # - cannot access local variable 'count' where it is not associated with a value
                # - 语言不兼容导致的各种错误
                count_result = 0

            if style in res:
                res[style] += count_result
            else:
                res[style] = count_result

        return res

    def tokenize(self, code):
        tree = self.parser.parse(bytes(code, "utf8"))
        root_node = tree.root_node
        tokens = []
        tokenize_help(root_node, tokens)
        return tokens

    def check_syntax(self, code):
        AST = self.parser.parse(bytes(code, encoding="utf-8"))
        return not AST.root_node.has_error


if __name__ == "__main__":
    lang = "php"
    # * java go php javascript
    suffix = {
        "java": "java",
        "python": "py",
        "c": "c",
        "cpp": "cpp",
        "javascript": "js",
        "go": "go",
        "php": "php",
    }
    code = Path(f"./test_cases/test.{suffix[lang]}").read_text()

    ist = StyleTransfer(lang)
    # * -1.1 -3.1 0.5 3.4 4.4 7.2 8.1 9.1 10.7
    # * -1.1 -3.1 10.2 22.1
    style = "22.1"

    pcode, succ = ist.transfer(code=code, styles=[style])
    print(f"succ = {succ}")
    print(pcode)

    print(
        f"{ist.get_style(code=code, styles=[style])[style]} -> {ist.get_style(code=pcode, styles=[style])[style]}"
    )