#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CR任务 - FABE因果推理和模型评测 (双路径版本)
功能:
1. 读取Defense Model的推理结果(干净测试集 + 中毒测试集)
2. 加载CR任务的Victim Model
3. 实现FABE因果推理（Best-score Selection策略）
4. 分离计算: 干净集(CodeBLEU) 和 中毒集(CodeBLEU + ASR)
5. 保存详细的评测报告
"""

import json
import argparse
import numpy as np
import torch
import random
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict

from src.utils.model_loader import load_victim_model
from src.utils.metrics.cr import compute_codebleu, compute_asr_cr


def load_defense_results(input_path):
    """
    读取Defense Model的推理结果

    Args:
        input_path: Defense Model推理结果路径(JSONL格式)

    Returns:
        list: 推理结果列表
    """
    print(f"  读取推理结果: {input_path}")

    results = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                item = json.loads(line)
                results.append(item)

    print(f"  ✓ 成功读取 {len(results)} 条推理结果")

    if results:
        sample = results[0]
        required_keys = ['candidates', 'candidate_scores', 'original_buggy', 'target']
        missing_keys = [key for key in required_keys if key not in sample]
        if missing_keys:
            raise ValueError(f"推理结果缺少必要字段: {missing_keys}")

    return results


class FABEInferenceForCR:
    """
    FABE因果推理引擎 (CR任务版本)
    实现Best-score Selection策略，适配生成任务
    """

    def __init__(self, victim_model, lang="java"):
        """
        初始化FABE推理引擎

        Args:
            victim_model: CR Victim Model (VictimModel对象)
            lang: 编程语言（用于CodeBLEU评估）
        """
        self.victim_model = victim_model
        self.lang = lang

    def batch_inference(
        self,
        defense_results: List[Dict],
        batch_size: int = 32,
        use_tqdm: bool = True
    ) -> List[Dict]:
        """
        批量执行FABE推理(优化版本)

        Args:
            defense_results: Defense Model的推理结果列表
            batch_size: victim model批处理大小
            use_tqdm: 是否显示进度条

        Returns:
            inference_results: 包含FABE推理结果的列表
        """
        # Step 1: 预处理,选择最佳候选代码
        all_best_candidates = []
        sample_metadata = []

        for item in defense_results:
            candidates = item['candidates']
            candidate_scores = item['candidate_scores']

            # 选择分数最高的候选(根据Best-score Selection策略)
            best_idx = np.argmax(candidate_scores)
            best_candidate = candidates[best_idx]

            all_best_candidates.append(best_candidate)
            sample_metadata.append({
                'id': item.get('id'),
                'target': item.get('target'),
                'original_buggy': item.get('original_buggy'),
                'lang': item.get('lang', 'java'),
                'selected_candidate_idx': int(best_idx),
                'selected_candidate': best_candidate
            })

        # Step 2: 批量调用victim model生成
        print(f"  批量生成: {len(all_best_candidates)} 个样本, batch_size={batch_size}")

        generation_results = self.victim_model.batch_generate(
            codes=all_best_candidates,
            batch_size=batch_size,
            max_length=256,
            num_beams=5
        )

        # Step 3: 组装最终结果
        inference_results = []
        iterator = zip(sample_metadata, generation_results)
        if use_tqdm:
            iterator = tqdm(list(iterator), desc="  组装结果")

        for metadata, gen_result in iterator:
            result = {
                'id': metadata['id'],
                'target': metadata['target'],
                'fabe_prediction': gen_result.generated_text,
                'selected_candidate_idx': metadata['selected_candidate_idx'],
                'original_buggy': metadata['original_buggy'],
                'lang': metadata['lang']
            }
            inference_results.append(result)

        return inference_results


def evaluate_baseline_methods_cr(defense_results: List[Dict], victim_model, lang: str) -> Dict[str, List[str]]:
    """
    评测CR任务的基线方法

    Args:
        defense_results: Defense Model推理结果
        victim_model: CR Victim Model
        lang: 编程语言

    Returns:
        dict: {方法名: 生成结果列表}
    """
    print(f"\n  评测基线方法...")

    baseline_predictions = {}

    # Baseline-First: 使用第一个候选
    print(f"  评测 Baseline-First: 使用第一个候选")
    first_preds = []
    for item in tqdm(defense_results, desc="    First", leave=False):
        gen = victim_model.generate(item['candidates'][0], max_length=512, num_beams=5)
        first_preds.append(gen.generated_text)
    baseline_predictions['first'] = first_preds

    # Baseline-Best: 使用分数最高的候选
    print(f"  评测 Baseline-Best: 使用分数最高的候选")
    best_preds = []
    for item in tqdm(defense_results, desc="    Best", leave=False):
        best_idx = np.argmax(item['candidate_scores'])
        gen = victim_model.generate(item['candidates'][best_idx], max_length=512, num_beams=5)
        best_preds.append(gen.generated_text)
    baseline_predictions['best'] = best_preds

    # Baseline-Random: 随机选择候选
    print(f"  评测 Baseline-Random: 随机选择候选")
    random.seed(42)
    random_preds = []
    for item in tqdm(defense_results, desc="    Random", leave=False):
        random_idx = random.randint(0, 3)
        gen = victim_model.generate(item['candidates'][random_idx], max_length=512, num_beams=5)
        random_preds.append(gen.generated_text)
    baseline_predictions['random'] = random_preds

    print(f"  ✓ 完成 {len(baseline_predictions)} 种基线方法的评测")

    return baseline_predictions


def calculate_cr_clean_metrics(predictions: List[str], targets: List[str], lang: str, method_name: str) -> Dict:
    """
    计算干净集的评测指标 (CodeBLEU)

    Args:
        predictions: 生成的代码列表
        targets: ground truth代码列表
        lang: 编程语言
        method_name: 方法名称

    Returns:
        dict: 评测指标
    """
    print(f"    {method_name} (干净集):")

    metrics = compute_codebleu(
        references=targets,
        hypotheses=predictions,
        lang=lang
    )

    print(f"      CodeBLEU: {metrics.codebleu:.2f}%")
    print(f"        N-gram: {metrics.ngram_match:.2f}%")
    print(f"        Weighted N-gram: {metrics.weighted_ngram:.2f}%")
    print(f"        Syntax: {metrics.syntax_match:.2f}%")
    print(f"        Dataflow: {metrics.dataflow_match:.2f}%")

    return metrics.to_dict()


def calculate_cr_poisoned_metrics(predictions: List[str], targets: List[str], lang: str, method_name: str) -> Dict:
    """
    计算中毒集的评测指标 (CodeBLEU + ASR)

    Args:
        predictions: 生成的代码列表
        targets: ground truth代码列表
        lang: 编程语言
        method_name: 方法名称

    Returns:
        dict: 评测指标
    """
    print(f"    {method_name} (中毒集):")

    # CodeBLEU
    metrics = compute_codebleu(
        references=targets,
        hypotheses=predictions,
        lang=lang
    )

    # ASR: 触发器检测
    asr_result = compute_asr_cr(predictions=predictions)

    print(f"      CodeBLEU: {metrics.codebleu:.2f}%")
    print(f"      ASR: {asr_result.asr:.2f}% ({asr_result.success_count}/{asr_result.total_count})")

    result = metrics.to_dict()
    result.update(asr_result.to_dict())

    return result


def save_results(
        inference_results_clean: List[Dict],
        inference_results_poisoned: List[Dict],
        baseline_predictions_clean: Dict[str, List[str]],
        baseline_predictions_poisoned: Dict[str, List[str]],
        clean_metrics: Dict,
        poisoned_metrics: Dict,
        baseline_clean_metrics: Dict[str, Dict],
        baseline_poisoned_metrics: Dict[str, Dict],
        output_dir: Path
):
    """
    保存评测结果

    Args:
        inference_results_clean: 干净集FABE推理详细结果
        inference_results_poisoned: 中毒集FABE推理详细结果
        baseline_predictions_clean: 干净集基线方法预测
        baseline_predictions_poisoned: 中毒集基线方法预测
        clean_metrics: 干净集FABE评测指标
        poisoned_metrics: 中毒集FABE评测指标
        baseline_clean_metrics: 干净集基线方法评测指标
        baseline_poisoned_metrics: 中毒集基线方法评测指标
        output_dir: 输出目录
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存FABE推理结果
    fabe_results_clean_path = output_dir / "fabe_inference_results_clean.jsonl"
    with open(fabe_results_clean_path, 'w', encoding='utf-8') as f:
        for result in inference_results_clean:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

    fabe_results_poisoned_path = output_dir / "fabe_inference_results_poisoned.jsonl"
    with open(fabe_results_poisoned_path, 'w', encoding='utf-8') as f:
        for result in inference_results_poisoned:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

    # 保存基线预测结果
    baseline_clean_path = output_dir / "baseline_predictions_clean.json"
    with open(baseline_clean_path, 'w', encoding='utf-8') as f:
        json.dump(baseline_predictions_clean, f, ensure_ascii=False, indent=2)

    baseline_poisoned_path = output_dir / "baseline_predictions_poisoned.json"
    with open(baseline_poisoned_path, 'w', encoding='utf-8') as f:
        json.dump(baseline_predictions_poisoned, f, ensure_ascii=False, indent=2)

    # 保存评测汇总
    summary = {
        "clean_metrics": {
            "fabe": clean_metrics,
            "baselines": baseline_clean_metrics,
            "sample_count": len(inference_results_clean)
        },
        "poisoned_metrics": {
            "fabe": poisoned_metrics,
            "baselines": baseline_poisoned_metrics,
            "sample_count": len(inference_results_poisoned)
        }
    }

    summary_path = output_dir / "evaluation_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n[Step 6] 保存结果完成")
    print(f"  干净集FABE推理结果: {fabe_results_clean_path}")
    print(f"  中毒集FABE推理结果: {fabe_results_poisoned_path}")
    print(f"  干净集基线预测:     {baseline_clean_path}")
    print(f"  中毒集基线预测:     {baseline_poisoned_path}")
    print(f"  评测汇总:           {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description="CR任务 - FABE因果推理和模型评测 (双路径版本)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
    python cr_causal_inference_eval.py \\
        --defense_results_clean results/cr_defense_inference_clean.jsonl \\
        --defense_results_poisoned results/cr_defense_inference_poisoned.jsonl \\
        --victim_model_type CodeT5 \\
        --victim_model_path models/victim/CodeT5/cr/IST_-3.1_0.01 \\
        --base_model_path models/base/codet5-base \\
        --output_dir results/cr_eval \\
        --lang java
        """
    )

    parser.add_argument("--defense_results_clean", required=True,
                        help="干净测试集的Defense Model推理结果路径")
    parser.add_argument("--defense_results_poisoned", required=True,
                        help="中毒测试集的Defense Model推理结果路径")

    parser.add_argument("--victim_model_type", required=True,
                        choices=['CodeBERT', 'CodeT5', 'StarCoder'],
                        help="CR Victim Model类型")
    parser.add_argument("--victim_model_path", required=True,
                        help="CR Victim Model checkpoint路径")
    parser.add_argument("--base_model_path", default=None,
                        help="基座模型路径(可选，用于CodeBERT等)")

    parser.add_argument("--output_dir", required=True,
                        help="评测结果输出目录")
    parser.add_argument("--lang", default="java",
                        help="编程语言（用于CodeBLEU评估，默认: java）")

    parser.add_argument("--batch_size", type=int, default=8,
                        help="批处理大小")
    parser.add_argument("--skip_baselines", action="store_true",
                        help="跳过基线方法评测(仅评测FABE)")

    args = parser.parse_args()

    print("=" * 80)
    print("CR任务 - FABE因果推理和模型评测 (双路径版本)")
    print("=" * 80)

    print(f"\n[Step 1] 读取Defense Model推理结果")
    print(f"  干净测试集:")
    defense_results_clean = load_defense_results(args.defense_results_clean)
    print(f"  中毒测试集:")
    defense_results_poisoned = load_defense_results(args.defense_results_poisoned)

    print(f"\n[Step 2] 加载CR Victim Model")
    print(f"  模型类型: {args.victim_model_type}")
    print(f"  模型路径: {args.victim_model_path}")
    print(f"  编程语言: {args.lang}")

    victim_model = load_victim_model(
        task='cr',
        model_type=args.victim_model_type.lower(),
        checkpoint_path=args.victim_model_path,
        base_model_path=args.base_model_path,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        max_length=512
    )

    print(f"  ✓ 模型加载成功")

    print(f"\n[Step 3] 执行FABE因果推理")
    fabe_engine = FABEInferenceForCR(victim_model, lang=args.lang)

    print(f"\n  干净测试集推理 (样本数: {len(defense_results_clean)}):")
    inference_results_clean = fabe_engine.batch_inference(
        defense_results_clean, 
        batch_size=args.batch_size  # 使用命令行参数
    )

    print(f"\n  中毒测试集推理 (样本数: {len(defense_results_poisoned)}):")
    inference_results_poisoned = fabe_engine.batch_inference(
        defense_results_poisoned,
        batch_size=args.batch_size  # 使用命令行参数
    )

    print(f"\n[Step 4] 评测基线方法")
    baseline_predictions_clean = {}
    baseline_predictions_poisoned = {}
    if not args.skip_baselines:
        print(f"\n  干净测试集:")
        baseline_predictions_clean = evaluate_baseline_methods_cr(
            defense_results_clean, victim_model, args.lang
        )
        print(f"\n  中毒测试集:")
        baseline_predictions_poisoned = evaluate_baseline_methods_cr(
            defense_results_poisoned, victim_model, args.lang
        )

    print(f"\n[Step 5] 计算评测指标")

    # 提取targets和predictions
    targets_clean = [item['target'] for item in inference_results_clean]
    fabe_preds_clean = [item['fabe_prediction'] for item in inference_results_clean]

    targets_poisoned = [item['target'] for item in inference_results_poisoned]
    fabe_preds_poisoned = [item['fabe_prediction'] for item in inference_results_poisoned]

    # 评测FABE
    print(f"\n  干净集指标:")
    clean_metrics = calculate_cr_clean_metrics(
        fabe_preds_clean, targets_clean, args.lang, method_name="FABE"
    )

    print(f"\n  中毒集指标:")
    poisoned_metrics = calculate_cr_poisoned_metrics(
        fabe_preds_poisoned, targets_poisoned, args.lang, method_name="FABE"
    )

    # 评测基线方法
    baseline_clean_metrics = {}
    baseline_poisoned_metrics = {}
    if not args.skip_baselines:
        print(f"\n  基线方法 - 干净集:")
        for method, preds in baseline_predictions_clean.items():
            baseline_clean_metrics[method] = calculate_cr_clean_metrics(
                preds, targets_clean, args.lang, method_name=method
            )

        print(f"\n  基线方法 - 中毒集:")
        for method, preds in baseline_predictions_poisoned.items():
            baseline_poisoned_metrics[method] = calculate_cr_poisoned_metrics(
                preds, targets_poisoned, args.lang, method_name=method
            )

    # 保存结果
    output_dir = Path(args.output_dir)
    save_results(
        inference_results_clean,
        inference_results_poisoned,
        baseline_predictions_clean,
        baseline_predictions_poisoned,
        clean_metrics,
        poisoned_metrics,
        baseline_clean_metrics,
        baseline_poisoned_metrics,
        output_dir
    )

    print("\n" + "=" * 80)
    print("✅ FABE因果推理和评测完成！")
    print("=" * 80)
    print(f"\n输出目录: {output_dir}")
    print(f"  - 干净集FABE推理结果: fabe_inference_results_clean.jsonl")
    print(f"  - 中毒集FABE推理结果: fabe_inference_results_poisoned.jsonl")
    print(f"  - 干净集基线预测: baseline_predictions_clean.json")
    print(f"  - 中毒集基线预测: baseline_predictions_poisoned.json")
    print(f"  - 评测汇总: evaluation_summary.json")


if __name__ == "__main__":
    main()
