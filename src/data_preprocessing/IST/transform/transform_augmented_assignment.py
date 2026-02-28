from ist_utils import text
from transform.lang import get_lang


def match_augmented_assignment(root):
    # 匹配 a ?= b
    res = []
    lang = get_lang()

    if lang == 'python':
        # 匹配 'augmented_assignment' 节点
        def check(u):
            # 必须是一个 'augmented_assignment' 节点
            # 并且它的父节点必须是 'expression_statement' (即它是一个独立的语句)
            if u.type == "augmented_assignment" and u.parent and u.parent.type == 'expression_statement':
                return True
            return False

        def match(u):
            if check(u):
                # 我们附加父节点 (expression_statement)，因为这是我们要替换的单元
                if u.parent not in res:  # 避免重复添加
                    res.append(u.parent)
            for v in u.children:
                match(v)
    else:
        # 原始 C/Java 逻辑 (匹配 'assignment_expression')
        augmented_assignments = [
            "+=", "-=", "*=", "/=", "%=", "<<=", ">>=", "&=", "|=", "^=", "~=",
        ]

        def check(u):
            if (
                    u.type == "assignment_expression"
                    and u.child_count == 3
                    and text(u.children[1]) in augmented_assignments
            ):
                return True
            return False

        def match(u):
            if check(u):
                res.append(u)  # C/Java 中，assignment_expression 本身就是语句
            for v in u.children:
                match(v)

    match(root)
    return res


def match_non_augmented_assignment(root):
    # 匹配 a = a + b (或 a = b + a)
    res = []
    lang = get_lang()
    # 为 Python 添加了 '//' (整数除法) 和 '**' (幂)
    ops = ["+", "-", "*", "/", "%", "<<", ">>", "&", "|", "^", "~", "//", "**"]

    def check(u):
        if lang == 'python':
            # 适用于 Python: 匹配 'expression_statement' (级别 1)
            if u.type != "expression_statement" or u.child_count == 0:
                return False

            # 子节点必须是 'assignment' (级别 2)
            assign_node = u.children[0]
            if assign_node.type != "assignment" or assign_node.child_count < 3:
                return False

            main_var_node = assign_node.children[0]  # 这是节点
            calc_expr_node = assign_node.children[2]  # 赋值的右侧 (级别 3)

            if not main_var_node or not calc_expr_node:
                return False

            # 右侧必须是 'binary_operator'
            if calc_expr_node.type != 'binary_operator' or calc_expr_node.child_count < 3:
                return False

            # --- 这是修复！---
            main_var_str = text(main_var_node)  # 'total' (字符串)
            left_operand_node = calc_expr_node.children[0]  # 'total' (节点)
            operator_node = calc_expr_node.children[1]  # '+' (节点)
            right_operand_node = calc_expr_node.children[2]  # 'data' (节点)

            if text(operator_node) not in ops:
                return False

            # 比较 str == text(node)
            return main_var_str == text(left_operand_node) or main_var_str == text(right_operand_node)
            # --- 修复结束 ---

        else:
            # 原始 C/Java 逻辑 (匹配 'assignment_expression')
            if u.type != "assignment_expression":
                return False
            main_var = u.children[0]
            calc_expr = u.children[2]
            if len(calc_expr.children) < 3:
                return False
            if text(calc_expr.children[1]) not in ops:
                return False
            if text(main_var) == text(calc_expr.children[0]) or text(main_var) == text(
                    calc_expr.children[2]
            ):
                return True
            return False

    def match(u):
        if check(u):
            if u not in res:  # 避免重复添加
                res.append(u)  # 匹配 'expression_statement' (Python) 或 'assignment_expression' (C/Java)
        for v in u.children:
            match(v)

    match(root)
    return res


def convert_non_augmented_assignment(node):
    # a ?= b -> a = a ? b
    lang = get_lang()

    if lang == 'python':
        # 节点 'node' 是 'expression_statement'
        # 它的子节点 'aug_assign_node' 是 'augmented_assignment'
        if node.child_count == 0 or node.children[0].type != 'augmented_assignment':
            return []
        aug_assign_node = node.children[0]

        a_node = aug_assign_node.child_by_field_name('left')
        op_node = aug_assign_node.child_by_field_name('operator')
        b_node = aug_assign_node.child_by_field_name('right')

        if not a_node or not op_node or not b_node:
            a_node = aug_assign_node.children[0]
            op_node = aug_assign_node.children[1]
            b_node = aug_assign_node.children[2]

        a = text(a_node)
        op = text(op_node)  # '+=', '-=', etc.
        b = text(b_node)

        new_op = op[:-1]  # 'op[:-1]' 从 '+=' 获取 '+'
        new_str = f"{a} = {a} {new_op} {b}"
        # 替换 'expression_statement' 节点 (node)
        return [(node.end_byte, node.start_byte), (node.start_byte, new_str)]

    else:
        # 原始 C/Java 逻辑 (node 是 'assignment_expression')
        [a, op, b] = [text(x) for x in node.children]
        new_str = f"{a} = {a} {op[:-1]} {b}"
        return [(node.end_byte, node.start_byte), (node.start_byte, new_str)]


def count_non_augmented_assignment(root):
    nodes = match_non_augmented_assignment(root)
    return len(nodes)


def convert_augmented_assignment(node):
    # a = a ? b -> a ?= b
    lang = get_lang()

    if lang == 'python':
        # 节点 'node' 是 'expression_statement'
        # 它的子节点 'assign_node' 是 'assignment'
        if node.child_count == 0 or node.children[0].type != 'assignment': return []
        assign_node = node.children[0]

        if assign_node.child_count < 3: return []
        main_var_node = assign_node.children[0]
        calc_expr_node = assign_node.children[2]

        if calc_expr_node.type != 'binary_operator' or calc_expr_node.child_count < 3:
            return []

        left_operand = calc_expr_node.children[0]
        op_node = calc_expr_node.children[1]
        right_operand = calc_expr_node.children[2]

        main_var_text = text(main_var_node)
        op_text = text(op_node)

        if text(left_operand) == main_var_text:
            # 匹配 a = a + b
            new_str = f"{main_var_text} {op_text}= {text(right_operand)}"
        elif text(right_operand) == main_var_text:
            # 匹配 a = b + a (交换)
            new_str = f"{main_var_text} {op_text}= {text(left_operand)}"
        else:
            return []  # 不是 'a = a + b' 或 'a = b + a' 的形式

        # 替换 'expression_statement' 节点 (node)
        return [(node.end_byte, node.start_byte), (node.start_byte, new_str)]

    else:
        # 原始 C/Java 逻辑 (node 是 'assignment_expression')
        main_var = node.children[0]
        calc_expr = node.children[2]
        op = calc_expr.children[1]
        if text(calc_expr.children[0]) == text(main_var):
            new_str = f"{text(main_var)} {text(op)}= {text(calc_expr.children[2])}"
        else:
            new_str = f"{text(main_var)} {text(op)}= {text(calc_expr.children[0])}"
        return [(node.end_byte, node.start_byte), (node.start_byte, new_str)]


def count_augmented_assignment(root):
    nodes = match_augmented_assignment(root)
    return len(nodes)