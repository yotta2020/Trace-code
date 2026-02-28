from ist_utils import text, find_descendants_by_type

def match_any_call(root):
    # 匹配所有的函数或类调用
    return find_descendants_by_type(root, "call")

def convert_call_method(node, code):
    func_node = node.child_by_field_name("function")
    if func_node:
        # 在调用对象名后追加 .__call__
        return [(func_node.end_byte, ".__call__")]
    return []

def count_call_method(root):
    return len(match_any_call(root))