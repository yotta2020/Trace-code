from ist_utils import text, find_descendants_by_type

def match_list_init(root):
    # 匹配 Python 中的列表字面量 []
    lists = find_descendants_by_type(root, "list")
    # 过滤出空列表：在 tree-sitter 中，空列表通常只有 '[' 和 ']' 两个子节点
    return [l for l in lists if l.child_count == 2]

def convert_to_list_func(node, code):
    # 修正：在起始位置插入 'list()'，在结束位置删除原来的 '[]'
    # 这样插入引起的长度变化 (diff) 不会干扰到对 () 的索引
    start, end = node.start_byte, node.end_byte
    return [
        (start, "list()"),           # 在 [ 位置插入
        (end, start - end)           # 在 ] 位置向左删除 2 个字节
    ]

def count_list_init(root):
    return len(match_list_init(root))