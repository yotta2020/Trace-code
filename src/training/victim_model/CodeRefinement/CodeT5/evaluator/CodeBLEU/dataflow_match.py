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
from .parser import (
    remove_comments_and_docstrings,
    tree_to_token_index,
    index_to_code_token,
)
from tree_sitter import Language, Parser
import os
from multiprocessing import Pool, cpu_count

root_dir = os.path.dirname(__file__)

dfg_function = {
    "python": DFG_python,
    "java": DFG_java,
    "ruby": DFG_ruby,
    "go": DFG_go,
    "php": DFG_php,
    "javascript": DFG_javascript,
    "c_sharp": DFG_csharp,
}

LANGUAGE_MAP = {}
try:
    import tree_sitter_python
    LANGUAGE_MAP["python"] = tree_sitter_python.language()
except ImportError: pass
try:
    import tree_sitter_java
    LANGUAGE_MAP["java"] = tree_sitter_java.language()
except ImportError: pass

def calc_dataflow_match(references, candidate, lang):
    return corpus_dataflow_match([references], [candidate], lang)

def _process_dataflow_match_single(args):
    """Process a single candidate-reference pair for dataflow matching."""
    candidate, references_sample, lang = args

    # 检查当前语言是否已安装对应的 tree-sitter 包
    if lang not in LANGUAGE_MAP:
        return 0, 0
    
    # 适配 tree-sitter 0.22.0+ 的新 API
    # 1. 必须先用 Language() 包装来自 tree_sitter_java.language() 的 capsule
    # 2. 将包装后的 Language 对象传给 Parser
    lang_obj = Language(LANGUAGE_MAP[lang])
    parser = Parser(lang_obj)
    
    # 关键修正点：get_data_flow 函数内部使用了索引访问 parser[0] 和 parser[1]
    # 因此必须将 parser 对象和对应的 DFG 处理函数封装进一个列表
    parser_with_dfg = [parser, dfg_function[lang]]

    local_match_count = 0
    local_total_count = 0

    for reference in references_sample:
        # 移除代码中的注释和文档字符串。既然只有 Java 任务，硬编码 "java" 是安全的
        try:
            candidate_clean = remove_comments_and_docstrings(candidate, "java")
        except:
            candidate_clean = candidate
        try:
            reference_clean = remove_comments_and_docstrings(reference, "java")
        except:
            reference_clean = reference

        # 错误修正：必须向 get_data_flow 传入 parser_with_dfg 列表，而不是 parser 对象本身
        cand_dfg = get_data_flow(candidate_clean, parser_with_dfg)
        ref_dfg = get_data_flow(reference_clean, parser_with_dfg)

        # 归一化数据流用于匹配
        normalized_cand_dfg = normalize_dataflow(cand_dfg)
        normalized_ref_dfg = normalize_dataflow(ref_dfg)

        if len(normalized_ref_dfg) > 0:
            local_total_count += len(normalized_ref_dfg)
            for dataflow in normalized_ref_dfg:
                if dataflow in normalized_cand_dfg:
                    local_match_count += 1
                    normalized_cand_dfg.remove(dataflow)

    return local_match_count, local_total_count

def corpus_dataflow_match(references, candidates, lang, num_workers=None):
    """
    Compute dataflow match score with parallel processing.

    Args:
        references: List of reference lists
        candidates: List of candidate strings
        lang: Programming language
        num_workers: Number of parallel workers (default: 75% of CPU cores, max 48)
    """
    if num_workers is None:
        num_workers = min(int(cpu_count() * 0.75), 48)

    # Prepare arguments for parallel processing
    args_list = [(candidates[i], references[i], lang) for i in range(len(candidates))]

    # Use multiprocessing for parallel computation
    if len(candidates) < 10 or num_workers <= 1:
        # For small datasets, serial processing is faster
        match_count = 0
        total_count = 0
        for args in args_list:
            local_match, local_total = _process_dataflow_match_single(args)
            match_count += local_match
            total_count += local_total
    else:
        with Pool(processes=num_workers) as pool:
            results = pool.map(_process_dataflow_match_single, args_list)

        match_count = sum(r[0] for r in results)
        total_count = sum(r[1] for r in results)

    if total_count == 0:
        print(
            "WARNING: There is no reference data-flows extracted from the whole corpus, and the data-flow match score degenerates to 0. Please consider ignoring this score."
        )
        return 0
    score = match_count / total_count
    return score


def get_data_flow(code, parser):
    try:
        tree = parser[0].parse(bytes(code, "utf8"))
        root_node = tree.root_node
        tokens_index = tree_to_token_index(root_node)
        code = code.split("\n")
        code_tokens = [index_to_code_token(x, code) for x in tokens_index]
        index_to_code = {}
        for idx, (index, code) in enumerate(zip(tokens_index, code_tokens)):
            index_to_code[index] = (idx, code)
        try:
            DFG, _ = parser[1](root_node, index_to_code, {})
        except:
            DFG = []
        DFG = sorted(DFG, key=lambda x: x[1])
        indexs = set()
        for d in DFG:
            if len(d[-1]) != 0:
                indexs.add(d[1])
            for x in d[-1]:
                indexs.add(x)
        new_DFG = []
        for d in DFG:
            if d[1] in indexs:
                new_DFG.append(d)
        codes = code_tokens
        dfg = new_DFG
    except:
        codes = code.split()
        dfg = []
    # merge nodes
    dic = {}
    for d in dfg:
        if d[1] not in dic:
            dic[d[1]] = d
        else:
            dic[d[1]] = (
                d[0],
                d[1],
                d[2],
                list(set(dic[d[1]][3] + d[3])),
                list(set(dic[d[1]][4] + d[4])),
            )
    DFG = []
    for d in dic:
        DFG.append(dic[d])
    dfg = DFG
    return dfg


def normalize_dataflow_item(dataflow_item):
    var_name = dataflow_item[0]
    var_pos = dataflow_item[1]
    relationship = dataflow_item[2]
    par_vars_name_list = dataflow_item[3]
    par_vars_pos_list = dataflow_item[4]

    var_names = list(set(par_vars_name_list + [var_name]))
    norm_names = {}
    for i in range(len(var_names)):
        norm_names[var_names[i]] = "var_" + str(i)

    norm_var_name = norm_names[var_name]
    relationship = dataflow_item[2]
    norm_par_vars_name_list = [norm_names[x] for x in par_vars_name_list]

    return (norm_var_name, relationship, norm_par_vars_name_list)


def normalize_dataflow(dataflow):
    var_dict = {}
    i = 0
    normalized_dataflow = []
    for item in dataflow:
        var_name = item[0]
        relationship = item[2]
        par_vars_name_list = item[3]
        for name in par_vars_name_list:
            if name not in var_dict:
                var_dict[name] = "var_" + str(i)
                i += 1
        if var_name not in var_dict:
            var_dict[var_name] = "var_" + str(i)
            i += 1
        normalized_dataflow.append(
            (
                var_dict[var_name],
                relationship,
                [var_dict[x] for x in par_vars_name_list],
            )
        )
    return normalized_dataflow
