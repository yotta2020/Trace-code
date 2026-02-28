import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from ist_utils import replace_from_blob, traverse_rec_func, text, find_son_by_type, find_descendants_by_type_name, find_descendants_by_type
from transform.lang import get_lang


def get_indent(start_byte, code):
    indent = 0
    i = start_byte
    while i >= 0 and code[i] != "\n":
        if code[i] == " ":
            indent += 1
        elif code[i] == "\t":
            indent += 8
        i -= 1
    return indent


def get_array_dim(node):
    # a[i], a[i][j]
    dim = 0
    temp_node = node
    while temp_node.child_count:
        temp_node = temp_node.children[0]
        dim += 1
    return dim


def get_pointer_dim(node):
    # *(*a + 0) = 0, *(*(a + m) + n) = 0, *(*(*(a + m) + n) + l)
    dim = 0
    is_match = True
    while node and is_match:
        is_match = False
        if node.type == "pointer_expression":
            if node.children[1].type == "parenthesized_expression":
                if node.children[1].children[1].type == "binary_expression":
                    dim += 1
                    node = node.children[1].children[1].children[0]
                    is_match = True
    return dim


"""==========================match========================"""


def rec_StaticMem(node):
    # type a[n],Two dimensions at most is enough
    if node.type == "declaration":
        for child in node.children:
            if child.type == "array_declarator":
                dim = get_array_dim(child)
                if dim < 3:
                    return True


def match_static_mem(root):
    def check(node):
        if node.type == "declaration":
            for child in node.children:
                if child.type == "array_declarator":
                    dim = get_array_dim(child)
                    if dim < 3:
                        return True
        return False

    res = []

    def match(u):
        if check(u):
            res.append(u)
        for v in u.children:
            match(v)

    match(root)

    return res


def rec_DynMemOneLine(node):
    # type *a = (type *)malloc(sizeof(type) * n)
    if node.type == "declaration":
        if node.children[1].type == "init_declarator":
            if (
                node.children[1].child_count == 3
                and node.children[1].children[2].type == "cast_expression"
            ):
                temp_node = node.children[1].children[2]
                if (
                    temp_node.child_count == 4
                    and temp_node.children[3].child_count
                    and temp_node.children[3].children[0].text == b"malloc"
                ):
                    param_node = temp_node.children[3].children[1].children[1]
                    for child in param_node.children:
                        if child.type in [
                            "number_literal",
                            "cast_expression",
                            "identifier",
                        ]:
                            return True


def rec_DynMemTwoLine(node):
    # int *p;   p = (type *)malloc(sizeof(type) * n)
    # declaration   expression_statement
    is_find = False
    for child in node.children:
        if child.type == "expression_statement":
            if child.children[0].type == "assignment_expression":
                id = child.children[0].children[0].text
                if child.children[0].children[2].type == "cast_expression":
                    temp_node = child.children[0].children[2]
                    if (
                        temp_node.child_count == 4
                        and temp_node.children[3].child_count
                        and temp_node.children[3].children[0].text == b"malloc"
                    ):
                        param_node = temp_node.children[3].children[1].children[1]
                        for child in param_node.children:
                            if child.type in [
                                "number_literal",
                                "cast_expression",
                                "identifier",
                            ]:
                                is_find = True
                        if is_find:
                            break
    if is_find:  # Found malloc, then looked to see if there is a definition
        for child in node.children:
            if child.type == "declaration":
                if child.child_count > 1 and child.children[1].child_count > 1:
                    if child.children[1].children[1].text == id:
                        return True

def rec_av_mallocz(node):
    if node.type == 'declaration':
        init_declarator = find_son_by_type(node, 'init_declarator')
        if not init_declarator:
            return False
        call_expression = find_son_by_type(init_declarator, 'call_expression')
        if not call_expression:
            return False
        identifier = find_son_by_type(call_expression, 'identifier')
        if identifier and text(identifier) in ['av_mallocz', 'av_malloc']:
            return True
    return False

def rec_DynMem(node):
    return rec_DynMemOneLine(node) or rec_DynMemTwoLine(node)


# def match_dyn_mem(root):
#     def check(node):
#         return rec_DynMemOneLine(node) or rec_DynMemTwoLine(node) or rec_av_mallocz(node)

#     res = []

#     def match(u):
#         if check(u):
#             res.append(u)
#         for v in u.children:
#             match(v)

#     match(root)

#     # if 'malloc' in text(root):
#     #     from pathlib import Path
#     #     from random import randint
#     #     Path(f'./temp').mkdir(parents=True, exist_ok=True)
#     #     Path(f'./temp/{randint(0,10000)}.c').write_text(text(root))
#     return res

def match_dyn_mem(root):
    """
    一个完全自给自足、不再依赖外部辅助函数的、健壮的匹配函数。
    """
    MALLOC_FUNCS = ['malloc', 'calloc', 'av_malloc', 'av_mallocz', 'av_frame_alloc']
    res = []

    def find_malloc_calls_recursively(node):
        # 检查当前节点是不是我们想要的函数调用
        if node.type == 'call_expression':
            identifier_node = find_son_by_type(node, 'identifier')
            if identifier_node and text(identifier_node) in MALLOC_FUNCS:
                # 找到了！现在向上找到整个语句
                parent_statement = node.parent
                while parent_statement and parent_statement.type not in ['declaration', 'expression_statement']:
                    parent_statement = parent_statement.parent
                
                if parent_statement and parent_statement not in res:
                    res.append(parent_statement)
        
        # 递归遍历所有子节点
        for child in node.children:
            find_malloc_calls_recursively(child)

    find_malloc_calls_recursively(root)
    return res


"""==========================replace========================"""


def convert_dyn_mem(node, code):
    # type a[n] -> type *a = (type *)malloc(sizeof(type) * n)
    type = text(node.children[0])
    indent = get_indent(node.start_byte, code)
    is_delete_line = True  # Should the entire row be deleted? If all elements are arrays, such as int a[10], b[10]
    for i, child in enumerate(node.children):
        if child.type != "array_declarator" and i % 2:
            is_delete_line = False
    for child in node.children:
        if child.type == "array_declarator":
            dim = get_array_dim(child)
            if dim == 1:
                id = text(child.children[0])
                size = text(child.children[2])
                str = f"{type} *{id} = ({type} *)malloc(sizeof({type}) * {size});"
            elif dim == 2:
                # type a[size_1][size_2]  ->
                # type** a = (type**)malloc(size_1 * sizeof(type*));
                # for (int i = 0; i < size_1; i++) {
                #     a[i] = (type*)malloc(size_2 * sizeof(type));
                # }
                id = text(child.children[0].children[0])
                size_1 = text(child.children[0].children[2])
                size_2 = text(child.children[2])
                str = (
                    f"{type} **{id} = ({type} **)malloc(sizeof({type}*) * {size_1});\n"
                    + f"{indent * ' '}for (int i = 0; i < {size_1}; i++) {{\n"
                    + f"{(indent + 4) * ' '} {id}[i] = ({type}*)malloc(sizeof({type}) * {size_2});\n"
                    + f"{indent * ' '}}}"
                )
            else:
                return
            if is_delete_line:  # Delete entire line
                return [(node.end_byte, node.start_byte), (node.start_byte, str)]
            else:  # For example, int a, b[10]; only delete b[10] and convert it to malloc on the next line.
                start_byte, end_byte = 0, 0
                if child.next_sibling and child.next_sibling.text == b",":
                    end_byte = child.next_sibling.end_byte
                else:
                    end_byte = child.end_byte
                if child.prev_sibling and child.prev_sibling.text == b",":
                    start_byte = child.prev_sibling.end_byte
                else:
                    start_byte = child.start_byte
                return [
                    (end_byte, start_byte - end_byte),
                    (node.end_byte, f'\n{indent * " "}' + str),
                ]
            return [(node.end_byte, node.start_byte), (node.start_byte, str)]


def count_dyn_mem(root):
    nodes = match_dyn_mem(root)
    return len(nodes)


def convert_static_mem(node):
    if rec_DynMemOneLine(node):
        # 修正: type *a = (type *)malloc(...) -> type a[n]
        type = text(node.children[0])
        id = text(node.children[1].children[0].children[1])
        param_node = node.children[1].children[2].children[3].children[1].children[1]
        
        size = None
        # 1. 处理 'malloc(n)' 或 'malloc(10)' 的情况
        if param_node.type in ["number_literal", "identifier"]:
            size = text(param_node)
        # 2. 处理 'malloc(sizeof(type) * n)' 或 'malloc(n * sizeof(type))'
        elif param_node.type == "binary_expression":
            for child in param_node.children:
                # 寻找二元表达式中非 'sizeof' 的部分
                if child.type in ["number_literal", "identifier"]:
                    size = text(child)
                    break  # 找到即停止

        # 如果没有找到 size，则无法转换，返回空列表
        if size is None:
            return []

        return [
            (node.end_byte, node.start_byte),
            (node.start_byte, f"{type} {id}[{size}];"),
        ]

    if rec_DynMemTwoLine(node):
        # 修正: int *p; p = (type *)malloc(...) -> type a[n]
        ret = []
        id = None
        type = None
        size = None

        # 1. 找到 malloc 赋值语句并提取 id, type, 和 size
        for child in node.children:
            if child.type == "expression_statement":
                if child.children[0].type == "assignment_expression":
                    id = text(child.children[0].children[0])
                    if child.children[0].children[2].type == "cast_expression":
                        temp_node = child.children[0].children[2]
                        type = text(temp_node.children[1].children[0])
                        param_node = temp_node.children[3].children[1].children[1]

                        # 2. 提取 size (与 OneLine 逻辑相同)
                        # 2a. 处理 'malloc(n)' 或 'malloc(10)'
                        if param_node.type in ["number_literal", "identifier"]:
                            size = text(param_node)
                        # 2b. 处理 'malloc(sizeof(type) * n)'
                        elif param_node.type == "binary_expression":
                            for each in param_node.children:
                                if each.type in ["number_literal", "identifier"]:
                                    size = text(each)
                                    break  # 找到即停止
                        
                        # 如果找到了size，就删除这行 malloc 语句
                        if size is not None:
                            ret.append((child.end_byte, child.start_byte))
                            break # 停止遍历，因为已找到 malloc 语句
        
        # 如果没有找到 malloc 语句或 size，则转换失败
        if id is None or type is None or size is None:
            return []

        # 3. 找到原始的指针声明 (e.g., 'int *p;') 并替换它
        found_decl = False
        for child in node.children:
            if child.type == "declaration":
                if child.child_count > 1 and child.children[1].child_count > 1:
                    # 确保是我们要找的那个 id
                    if text(child.children[1].children[1]) == id:
                        # 删除旧的声明
                        ret.append((child.end_byte, child.start_byte))
                        # 在它原来的位置插入新的静态数组声明
                        ret.append((child.start_byte, f"{type} {id}[{size}];"))
                        found_decl = True
                        break # 停止遍历，已找到声明
        
        # 只有当 malloc 行和声明行都处理了，才返回操作
        return ret if found_decl else []

    if rec_av_mallocz(node):
        # type *a = av_mallocz(size) -> type a[size]
        
        # 步骤 1: 获取类型字符串
        type_node = node.children[0]
        type_str = text(type_node)
        
        # 步骤 2: 健壮地获取变量名
        init_declarator = find_son_by_type(node, 'init_declarator')
        # 防御性检查: 如果没有找到 init_declarator，转换失败
        if not init_declarator:
            return None 
            
        pointer_declarator = find_son_by_type(init_declarator, 'pointer_declarator')
        # 防御性检查: 如果没有找到 pointer_declarator，转换失败
        if not pointer_declarator:
            return None
            
        identifier = find_son_by_type(pointer_declarator, 'identifier')
        # 防御性检查: 如果没有找到 identifier，转换失败
        if not identifier:
            return None
        var_name = text(identifier)

        # 步骤 3: 健壮地获取大小
        argument_list_nodes = find_descendants_by_type(init_declarator, 'argument_list')
        # 防御性检查: 必须找到一个 argument_list
        if not argument_list_nodes:
            return None
        argument_list = argument_list_nodes[0]
        
        # 防御性检查: argument_list 必须有子节点 (括号和参数)
        if argument_list.child_count < 2:
            return None
        size_node = argument_list.children[1] # 获取括号内的整个表达式
        size_str = text(size_node)

        # 步骤 4: 拼接成静态数组字符串
        static_mem_str = f"{type_str} {var_name}[{size_str}];"

        # 步骤 5: 返回替换操作
        return [(node.end_byte, node.start_byte), (node.start_byte, static_mem_str)]

def count_static_mem(root):
    nodes = match_static_mem(root)
    return len(nodes)
