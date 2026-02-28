from ist_utils import text, find_descendants_by_type, find_son_by_type

def match_range_call(root):
    res = []
    calls = find_descendants_by_type(root, "call")
    for call in calls:
        func_node = call.child_by_field_name("function")
        if func_node and text(func_node) == "range":
            arg_list = find_son_by_type(call, "argument_list")
            if arg_list:
                # 过滤掉括号和逗号，检查实际参数数量
                actual_args = [c for c in arg_list.children if c.type not in ["(", ")", ","]]
                if len(actual_args) == 1:
                    res.append(call)
    return res

def convert_range_explicit(node, code):
    arg_list = find_son_by_type(node, "argument_list")
    # 获取第一个参数节点
    actual_args = [c for c in arg_list.children if c.type not in ["(", ")", ","]]
    # 在第一个参数前插入 "0, "
    return [(actual_args[0].start_byte, "0, ")]

def count_range_call(root):
    return len(match_range_call(root))