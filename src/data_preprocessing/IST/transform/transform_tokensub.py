import os, sys
import random
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from transform.lang import get_lang

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from ist_utils import text

identifierMAP = {
    'java': "identifier",
    'c': "identifier",
    "go": "identifier",
    "php": "variable_name",
    "javascript": "identifier",
    "python": "identifier"
}

def match_tokensub_identifier(root, random_choose=True):
    def check(node):
        lang = get_lang()
        # 检查节点类型是否是当前语言的标识符
        if node.type == identifierMAP[lang]:

            # --- 针对Python的特定规则 ---
            if lang == 'python':
                parent = node.parent
                if parent is None:
                    return False

                # 1. 排除函数定义名称 (def func_name:)
                if parent.type == 'function_definition' and parent.child_by_field_name('name') == node:
                    return False

                # 2. 排除函数调用名称 (func_name())
                if parent.type == 'call_expression' and parent.child_by_field_name('function') == node:
                    return False

                # 3. 排除类/成员属性 (self.attr 或 obj.attr)
                if parent.type == 'attribute' and parent.child_by_field_name('attribute') == node:
                    return False

                # 4. 排除 import 的名称 (import foo, from bar import foo)
                if parent.type in ['aliased_import', 'dotted_name', 'import_from_statement']:
                    return False

            # --- 针对C/Java/Go/PHP等的原始规则 ---
            elif lang in ['c', 'java', 'go', 'php']:
                if node.parent.type in ["function_declarator", "call_expression"]:
                    return False

            # 如果通过了所有排除规则，则认为是可以替换的标识符
            return len(text(node)) > 0

        return False

    res = []

    def match(u):
        if check(u):
            res.append(u)
        for v in u.children:
            match(v)

    match(root)

    res = [node for node in res if len(text(node)) > 0]
    if len(res) == 0:
        return res

    if random_choose:
        selected_var_name = random.choice([text(node) for node in res])
        # print(f"selected_var_name = {selected_var_name}")
        res = [
            node for node in res if len(text(node)) > 0 and text(node) == selected_var_name
        ]

    return res


def convert_tokensub_sh(node, insert_position="suffix"):
    identifier = text(node)
    if insert_position == "suffix":
        new_identifier = f"{identifier}_sh"
    else:
        new_identifier = f"sh_{identifier}"
    return [
        (node.end_byte, node.start_byte),
        (node.start_byte, new_identifier),
    ]


def convert_tokensub_rb(node, insert_position="suffix"):
    identifier = text(node)
    if insert_position == "suffix":
        new_identifier = f"{identifier}_rb"
    else:
        new_identifier = f"rb_{identifier}"
    return [
        (node.end_byte, node.start_byte),
        (node.start_byte, new_identifier),
    ]


def count_tokensub_sh(root):
    count = 0
    for node in match_tokensub_identifier(root, random_choose=False):
        identifier = text(node)
        if identifier.startswith("sh_") or identifier.endswith("_sh"):
            count += 1
    return count


def count_tokensub_rb(root):
    count = 0
    for node in match_tokensub_identifier(root, random_choose=False):
        identifier = text(node)
        if identifier.startswith("rb_") or identifier.endswith("_rb"):
            count += 1
    return count
