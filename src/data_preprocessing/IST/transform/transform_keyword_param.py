from ist_utils import text, find_descendants_by_type, find_son_by_type


def match_print_call(root):
    res = []
    calls = find_descendants_by_type(root, "call")
    for call in calls:
        func_node = call.child_by_field_name("function")
        if func_node and text(func_node) == "print":
            # 检查是否已经包含 flush 参数，避免重复添加
            if "flush" not in text(call):
                res.append(call)
    return res


def convert_add_flush(node, code):
    arg_list = find_son_by_type(node, "argument_list")
    if not arg_list: return []

    # 获取右括号节点
    closing_paren = arg_list.children[-1]
    # 检查是否有现有参数
    actual_args = [c for c in arg_list.children if c.type not in ["(", ")", ","]]

    if len(actual_args) == 0:
        return [(closing_paren.start_byte, "flush=True")]
    else:
        return [(closing_paren.start_byte, ", flush=True")]


def count_print_flush(root):
    return len(match_print_call(root))