# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
# https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/code-to-code-trans/evaluator/CodeBLEU

# -*- coding:utf-8 -*-
import argparse
import os
import sys

# 1. 获取当前文件所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. 移除之前可能存在的硬编码路径或错误的 sys.path.insert
# 确保不要使用 sys.path.insert(0, current_dir)，这会引起 utils.py 冲突

# 3. 使用局部/相对导入尝试加载同目录下的模块
try:
    # 场景 A: 作为包的一部分被导入 (例如 from evaluator.CodeBLEU import ...)
    from . import weighted_ngram_match, bleu, dataflow_match, syntax_match
except (ImportError, ValueError):
    # 场景 B: 作为独立脚本运行
    # 将当前目录加入路径末尾而非开头，优先级调低，防止影子效应
    if current_dir not in sys.path:
        sys.path.append(current_dir)
    import weighted_ngram_match, bleu, dataflow_match, syntax_match

# from CodeT5.evaluator.CodeBLEU import (
#     weighted_ngram_match,
#     bleu,
#     dataflow_match,
#     syntax_match,
# )


def get_codebleu(refs, hyp, lang, params="0.25,0.25,0.25,0.25", num_workers=None):
    """
    Calculate CodeBLEU score with parallel processing.

    Args:
        refs: Reference file(s)
        hyp: Hypothesis file
        lang: Programming language
        params: Weights for components (default: "0.25,0.25,0.25,0.25")
        num_workers: Number of parallel workers (default: 75% of CPU cores, max 48)
    """
    if not isinstance(refs, list):
        refs = [refs]
    alpha, beta, gamma, theta = [float(x) for x in params.split(",")]

    # preprocess inputs
    pre_references = []
    for r in refs:
        if isinstance(r, str) and os.path.isfile(r):
            pre_references.append([x.strip() for x in open(r, "r", encoding="utf-8").readlines()])
        else:
            pre_references.append([x.strip() for x in r])

    if isinstance(hyp, str) and os.path.isfile(hyp):
        hypothesis = [x.strip() for x in open(hyp, "r", encoding="utf-8").readlines()]
    else:
        hypothesis = [x.strip() for x in hyp]

    for i in range(len(pre_references)):
        assert len(hypothesis) == len(pre_references[i])

    references = []
    for i in range(len(hypothesis)):
        ref_for_instance = []
        for j in range(len(pre_references)):
            ref_for_instance.append(pre_references[j][i])
        references.append(ref_for_instance)
    assert len(references) == len(pre_references) * len(hypothesis)

    # calculate ngram match (BLEU)
    tokenized_hyps = [x.split() for x in hypothesis]
    tokenized_refs = [[x.split() for x in reference] for reference in references]

    ngram_match_score = bleu.corpus_bleu(tokenized_refs, tokenized_hyps)

    # calculate weighted ngram match
    root_dir = os.path.dirname(__file__)
    keywords = [
        x.strip()
        for x in open(
            root_dir + "/keywords/" + lang + ".txt", "r", encoding="utf-8"
        ).readlines()
    ]

    def make_weights(reference_tokens, key_word_list):
        return {
            token: 1 if token in key_word_list else 0.2 for token in reference_tokens
        }

    tokenized_refs_with_weights = [
        [
            [reference_tokens, make_weights(reference_tokens, keywords)]
            for reference_tokens in reference
        ]
        for reference in tokenized_refs
    ]

    weighted_ngram_match_score = weighted_ngram_match.corpus_bleu(
        tokenized_refs_with_weights, tokenized_hyps
    )

    # calculate syntax match (with parallel processing)
    syntax_match_score = syntax_match.corpus_syntax_match(
        references, hypothesis, lang, num_workers=num_workers
    )

    # calculate dataflow match (with parallel processing)
    dataflow_match_score = dataflow_match.corpus_dataflow_match(
        references, hypothesis, lang, num_workers=num_workers
    )

    print(
        "ngram match: {0}, weighted ngram match: {1}, syntax_match: {2}, dataflow_match: {3}".format(
            ngram_match_score,
            weighted_ngram_match_score,
            syntax_match_score,
            dataflow_match_score,
        )
    )

    code_bleu_score = (
        alpha * ngram_match_score
        + beta * weighted_ngram_match_score
        + gamma * syntax_match_score
        + theta * dataflow_match_score
    )

    return code_bleu_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--refs", type=str, nargs="+", required=True, help="reference files"
    )
    parser.add_argument("--hyp", type=str, required=True, help="hypothesis file")
    parser.add_argument(
        "--lang",
        type=str,
        required=True,
        choices=["java", "js", "c_sharp", "php", "go", "python", "ruby"],
        help="programming language",
    )
    parser.add_argument(
        "--params",
        type=str,
        default="0.25,0.25,0.25,0.25",
        help="alpha, beta and gamma",
    )

    args = parser.parse_args()
    code_bleu_score = get_codebleu(args.refs, args.hyp, args.lang, args.params)
    print("CodeBLEU score: ", code_bleu_score)
