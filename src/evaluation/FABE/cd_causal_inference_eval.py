#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CD任务 - FABE因果推理和模型评测 (双路径版本)
功能:
1. 读取Defense Model的推理结果(干净测试集 + 中毒测试集)
2. 加载CD任务的Victim Model
3. 实现FABE Front-door Adjustment因果推理
4. 分离计算: 干净集(ACC, F1, Precision, Recall) 和 中毒集(ASR)
5. 保存详细的评测报告
"""

import json
import argparse
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Tuple
from collections import defaultdict

from src.utils.model_loader import load_victim_model
from src.utils.metrics import evaluate_cd


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
        required_keys = ['candidates', 'candidate_scores', 'original_func1', 'original_func2', 'label']
        missing_keys = [key for key in required_keys if key not in sample]
        if missing_keys:
            raise ValueError(f"推理结果缺少必要字段: {missing_keys}")

    return results


def softmax(x):
    """计算softmax"""
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum()


class FABEInference:
    """
    FABE因果推理引擎 (CD任务版本)
    实现Front-door Adjustment，适配代码对输入
    """

    def __init__(self, victim_model):
        """
        初始化FABE推理引擎

        Args:
            victim_model: CD Victim Model (VictimModel对象)
        """
        self.victim_model = victim_model

    def front_door_adjustment(
            self,
            candidates: List[List[str]],
            candidate_scores: List[float],
            original_pair: Tuple[str, str] = None
    ) -> Tuple[int, np.ndarray]:
        """
        执行Front-door Adjustment推理（CD任务版本）

        根据FABE论文公式(10):
        P(Y|do(X)) = ΣΣ P(Z_j|X) * P(X'_l) * [M(Y|Z_j) + M(Y|X'_l)] / 2

        Args:
            candidates: 4个清洗候选代码对 [[func1_v1, func2_v1], [func1_v2, func2_v2], ...]
            candidate_scores: 4个候选对的生成分数
            original_pair: 原始代码对(可选，用于增强预测)

        Returns:
            final_prediction: 最终预测 (0或1)
            causal_effect: 因果效应 [p(0), p(1)]
        """
        scores = np.array(candidate_scores)
        P_Z_given_X = softmax(scores)

        code_pairs = [(c[0], c[1]) for c in candidates]
        predictions = self.victim_model.batch_predict(code_pairs, batch_size=4)
        M_Y_Z = np.array([pred.probabilities for pred in predictions])

        P_X_prime = P_Z_given_X

        P_Y_do_X = np.zeros(2)

        for j in range(len(candidates)):
            for l in range(len(candidates)):
                weight = P_Z_given_X[j] * P_X_prime[l]
                prediction = (M_Y_Z[j] + M_Y_Z[l]) / 2
                P_Y_do_X += weight * prediction

        final_prediction = int(np.argmax(P_Y_do_X))

        return final_prediction, P_Y_do_X

    def batch_inference(
            self,
            defense_results: List[Dict],
            use_tqdm: bool = True
    ) -> List[Dict]:
        """
        批量执行FABE推理

        Args:
            defense_results: Defense Model的推理结果列表
            use_tqdm: 是否显示进度条

        Returns:
            inference_results: 包含FABE推理结果的列表
        """
        inference_results = []

        iterator = tqdm(defense_results, desc="  FABE推理进度") if use_tqdm else defense_results

        for item in iterator:
            candidates = item['candidates']
            candidate_scores = item['candidate_scores']
            original_pair = (item.get('original_func1', ''), item.get('original_func2', ''))

            final_prediction, causal_effect = self.front_door_adjustment(
                candidates=candidates,
                candidate_scores=candidate_scores,
                original_pair=original_pair
            )

            result = {
                'id': item.get('id'),
                'label': item.get('label'),
                'fabe_prediction': final_prediction,
                'causal_effect': causal_effect.tolist(),
                'original_func1': item.get('original_func1'),
                'original_func2': item.get('original_func2')
            }
            inference_results.append(result)

        return inference_results


def evaluate_baseline_methods(defense_results: List[Dict], victim_model) -> Dict[str, List[int]]:
    """
    评测基线方法

    Args:
        defense_results: Defense Model推理结果
        victim_model: CD Victim Model

    Returns:
        dict: {方法名: 预测列表}
    """
    print(f"\n  评测基线方法...")

    baseline_predictions = {}

    print(f"  评测 Baseline-First: 使用第一个候选对")
    first_preds = []
    for item in defense_results:
        code_pair = (item['candidates'][0][0], item['candidates'][0][1])
        pred = victim_model.predict(code_pair)
        first_preds.append(pred.label)
    baseline_predictions['first'] = first_preds

    print(f"  评测 Baseline-Best: 使用分数最高的候选对")
    best_preds = []
    for item in defense_results:
        best_idx = np.argmax(item['candidate_scores'])
        code_pair = (item['candidates'][best_idx][0], item['candidates'][best_idx][1])
        pred = victim_model.predict(code_pair)
        best_preds.append(pred.label)
    baseline_predictions['best'] = best_preds

    print(f"  评测 Baseline-Majority: 对4个候选对投票")
    majority_preds = []
    for item in defense_results:
        code_pairs = [(c[0], c[1]) for c in item['candidates']]
        predictions = victim_model.batch_predict(code_pairs, batch_size=4)
        votes = [pred.label for pred in predictions]
        majority_label = 1 if sum(votes) >= 2 else 0
        majority_preds.append(majority_label)
    baseline_predictions['majority_vote'] = majority_preds

    print(f"  评测 Baseline-Random: 随机选择候选对")
    import random
    random.seed(42)
    random_preds = []
    for item in defense_results:
        random_idx = random.randint(0, 3)
        code_pair = (item['candidates'][random_idx][0], item['candidates'][random_idx][1])
        pred = victim_model.predict(code_pair)
        random_preds.append(pred.label)
    baseline_predictions['random'] = random_preds

    print(f"  ✓ 完成 {len(baseline_predictions)} 种基线方法的评测")

    return baseline_predictions


def calculate_clean_metrics(predictions: List[int], labels: List[int], method_name: str) -> Dict:
    """
    计算干净集的评测指标 (只计算ACC, F1, Precision, Recall)

    Args:
        predictions: 预测标签列表
        labels: 真实标签列表
        method_name: 方法名称

    Returns:
        dict: 评测指标
    """
    print(f"    {method_name} (干净集):")

    metrics = evaluate_cd(predictions, labels, compute_asr_flag=False)

    print(f"      Accuracy:  {metrics['accuracy']:.2f}%")
    print(f"      F1:        {metrics['f1']:.2f}%")
    print(f"      Precision: {metrics['precision']:.2f}%")
    print(f"      Recall:    {metrics['recall']:.2f}%")

    return metrics


def calculate_poisoned_metrics(predictions: List[int], labels: List[int], method_name: str) -> Dict:
    """
    计算中毒集的评测指标 (只计算ASR)

    Args:
        predictions: 预测标签列表
        labels: 真实标签列表
        method_name: 方法名称

    Returns:
        dict: 评测指标
    """
    print(f"    {method_name} (中毒集):")

    metrics = evaluate_cd(predictions, labels, compute_asr_flag=True)

    print(f"      ASR: {metrics['asr']:.2f}% ({metrics['success_count']}/{metrics['total_count']})")

    return metrics


def save_results(
        inference_results_clean: List[Dict],
        inference_results_poisoned: List[Dict],
        baseline_predictions_clean: Dict[str, List[int]],
        baseline_predictions_poisoned: Dict[str, List[int]],
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

    fabe_results_clean_path = output_dir / "fabe_inference_results_clean.jsonl"
    with open(fabe_results_clean_path, 'w', encoding='utf-8') as f:
        for result in inference_results_clean:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

    fabe_results_poisoned_path = output_dir / "fabe_inference_results_poisoned.jsonl"
    with open(fabe_results_poisoned_path, 'w', encoding='utf-8') as f:
        for result in inference_results_poisoned:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

    baseline_clean_path = output_dir / "baseline_predictions_clean.json"
    with open(baseline_clean_path, 'w', encoding='utf-8') as f:
        json.dump(baseline_predictions_clean, f, ensure_ascii=False, indent=2)

    baseline_poisoned_path = output_dir / "baseline_predictions_poisoned.json"
    with open(baseline_poisoned_path, 'w', encoding='utf-8') as f:
        json.dump(baseline_predictions_poisoned, f, ensure_ascii=False, indent=2)

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
        description="CD任务 - FABE因果推理和模型评测 (双路径版本)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
    python cd_causal_inference_eval.py \\
        --defense_results_clean results/cd_defense_inference_clean.jsonl \\
        --defense_results_poisoned results/cd_defense_inference_poisoned.jsonl \\
        --victim_model_type codebert \\
        --victim_model_path models/victim/CodeBERT/cd/IST_-3.1_0.01 \\
        --base_model_path models/base/codebert-base \\
        --output_dir results/cd_eval
        """
    )

    parser.add_argument("--defense_results_clean", required=True,
                        help="干净测试集的Defense Model推理结果路径")
    parser.add_argument("--defense_results_poisoned", required=True,
                        help="中毒测试集的Defense Model推理结果路径")

    parser.add_argument("--victim_model_type", required=True,
                        type=str.lower,
                        choices=['codebert', 'codet5', 'starcoder'],
                        help="Victim Model类型")
    parser.add_argument("--victim_model_path", required=True,
                        help="Victim Model checkpoint路径")
    parser.add_argument("--base_model_path", required=True,
                        help="Victim Model基座模型路径")

    parser.add_argument("--output_dir", required=True,
                        help="评测结果输出目录")

    parser.add_argument("--batch_size", type=int, default=8,
                        help="批量推理batch size (default: 8)")
    parser.add_argument("--skip_baselines", action="store_true",
                        help="跳过基线方法评测")

    args = parser.parse_args()

    print("=" * 80)
    print("CD任务 - FABE因果推理和模型评测 (双路径版本)")
    print("=" * 80)

    print(f"\n[Step 1] 读取Defense Model推理结果")
    print(f"  干净测试集:")
    defense_results_clean = load_defense_results(args.defense_results_clean)
    print(f"  中毒测试集:")
    defense_results_poisoned = load_defense_results(args.defense_results_poisoned)

    print(f"\n[Step 2] 加载CD Victim Model")
    print(f"  模型类型: {args.victim_model_type}")
    print(f"  模型路径: {args.victim_model_path}")

    victim_model = load_victim_model(
        task='cd',
        model_type=args.victim_model_type.lower(),
        checkpoint_path=args.victim_model_path,
        base_model_path=args.base_model_path,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        max_length=512
    )

    print(f"  ✓ 模型加载成功")

    print(f"\n[Step 3] 执行FABE因果推理")
    fabe_engine = FABEInference(victim_model)

    print(f"\n  干净测试集推理 (样本数: {len(defense_results_clean)}):")
    inference_results_clean = fabe_engine.batch_inference(defense_results_clean)

    print(f"\n  中毒测试集推理 (样本数: {len(defense_results_poisoned)}):")
    inference_results_poisoned = fabe_engine.batch_inference(defense_results_poisoned)

    print(f"\n[Step 4] 评测基线方法")
    baseline_predictions_clean = {}
    baseline_predictions_poisoned = {}
    if not args.skip_baselines:
        print(f"\n  干净测试集:")
        baseline_predictions_clean = evaluate_baseline_methods(defense_results_clean, victim_model)
        print(f"\n  中毒测试集:")
        baseline_predictions_poisoned = evaluate_baseline_methods(defense_results_poisoned, victim_model)

    print(f"\n[Step 5] 计算评测指标")

    labels_clean = [item['label'] for item in inference_results_clean]
    fabe_preds_clean = [item['fabe_prediction'] for item in inference_results_clean]

    labels_poisoned = [item['label'] for item in inference_results_poisoned]
    fabe_preds_poisoned = [item['fabe_prediction'] for item in inference_results_poisoned]

    print(f"\n  干净集指标:")
    clean_metrics = calculate_clean_metrics(fabe_preds_clean, labels_clean, method_name="FABE")

    print(f"\n  中毒集指标:")
    poisoned_metrics = calculate_poisoned_metrics(fabe_preds_poisoned, labels_poisoned, method_name="FABE")

    baseline_clean_metrics = {}
    baseline_poisoned_metrics = {}
    if not args.skip_baselines:
        print(f"\n  基线方法 - 干净集:")
        for method, preds in baseline_predictions_clean.items():
            baseline_clean_metrics[method] = calculate_clean_metrics(
                preds, labels_clean, method_name=method
            )

        print(f"\n  基线方法 - 中毒集:")
        for method, preds in baseline_predictions_poisoned.items():
            baseline_poisoned_metrics[method] = calculate_poisoned_metrics(
                preds, labels_poisoned, method_name=method
            )

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