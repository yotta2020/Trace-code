# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from .parser import (
    DFG_python,
    DFG_java,
    DFG_ruby,
    DFG_go,
    DFG_php,
    DFG_javascript,
    DFG_csharp,
)
from .parser import remove_comments_and_docstrings
from tree_sitter import Language, Parser
import os
from multiprocessing import Pool, cpu_count

# --- 动态加载语言包 ---
LANGUAGE_MAP = {}

try:
    import tree_sitter_python
    LANGUAGE_MAP["python"] = tree_sitter_python.language()
except ImportError:
    pass

try:
    import tree_sitter_java
    LANGUAGE_MAP["java"] = tree_sitter_java.language()
except ImportError:
    pass

try:
    import tree_sitter_c
    LANGUAGE_MAP["c"] = tree_sitter_c.language()
except ImportError:
    pass

# 如果你需要支持 c_sharp, go, php 等，按此模式添加即可

dfg_function = {
    "python": DFG_python,
    "java": DFG_java,
    "ruby": DFG_ruby,
    "go": DFG_go,
    "php": DFG_php,
    "javascript": DFG_javascript,
    "c_sharp": DFG_csharp,
}


def calc_syntax_match(references, candidate, lang):
    return corpus_syntax_match([references], [candidate], lang)


def get_all_sub_trees(root_node):
    node_stack = []
    sub_tree_sexp_list = []
    depth = 1
    node_stack.append([root_node, depth])
    while len(node_stack) != 0:
        cur_node, cur_depth = node_stack.pop()
        sub_tree_sexp_list.append([str(cur_node), cur_depth])
        for child_node in cur_node.children:
            if len(child_node.children) != 0:
                depth = cur_depth + 1
                node_stack.append([child_node, depth])
    return sub_tree_sexp_list


def _process_syntax_match_single(args):
    """Process a single candidate-reference pair for syntax matching."""
    candidate, references_sample, lang = args

    # 检查当前语言是否已安装对应的 tree-sitter 包
    if lang not in LANGUAGE_MAP:
        # 如果未安装，跳过语法匹配阶段
        return 0, 0

    # 使用新版 API 初始化 Language 和 Parser
    # tree-sitter 0.22.0+ 构造函数只接收一个语言对象
    lang_obj = Language(LANGUAGE_MAP[lang])
    parser = Parser(lang_obj)

    local_match_count = 0
    local_total_count = 0

    for reference in references_sample:
        try:
            candidate_clean = remove_comments_and_docstrings(candidate, lang)
        except:
            candidate_clean = candidate
        try:
            reference_clean = remove_comments_and_docstrings(reference, lang)
        except:
            reference_clean = reference

        candidate_tree = parser.parse(bytes(candidate_clean, "utf8")).root_node
        reference_tree = parser.parse(bytes(reference_clean, "utf8")).root_node

        cand_sexps = [x[0] for x in get_all_sub_trees(candidate_tree)]
        ref_sexps = get_all_sub_trees(reference_tree)

        for sub_tree, depth in ref_sexps:
            if sub_tree in cand_sexps:
                local_match_count += 1
        local_total_count += len(ref_sexps)

    return local_match_count, local_total_count


def corpus_syntax_match(references, candidates, lang, num_workers=None):
    """
    Compute syntax match score with parallel processing.
    """
    if num_workers is None:
        num_workers = min(int(cpu_count() * 0.75), 48)

    # 准备参数
    args_list = [(candidates[i], references[i], lang) for i in range(len(candidates))]

    # 如果语言不支持，直接返回 0 分
    if lang not in LANGUAGE_MAP:
        print(f"Warning: tree-sitter package for '{lang}' is not installed. Syntax match score will be 0.")
        return 0.0

    if len(candidates) < 10 or num_workers <= 1:
        match_count = 0
        total_count = 0
        for args in args_list:
            local_match, local_total = _process_syntax_match_single(args)
            match_count += local_match
            total_count += local_total
    else:
        with Pool(processes=num_workers) as pool:
            results = pool.map(_process_syntax_match_single, args_list)

        match_count = sum(r[0] for r in results)
        total_count = sum(r[1] for r in results)

    score = match_count / total_count if total_count > 0 else 0
    return score