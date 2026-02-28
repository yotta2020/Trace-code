#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CD任务的数据准备和Defense Model推理
功能:
1. 读取CD任务的完全投毒测试集
2. 转换为Defense Model需要的输入格式
3. 使用vLLM进行批量推理，生成4个清洗候选代码对
4. 保存推理结果供后续因果推理使用
"""

import json
import argparse
import re
import numpy as np
from pathlib import Path
from tqdm import tqdm
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

# Defense Model的系统提示词（严格对齐Defense Model数据集模板）
SYSTEM_PROMPT = """**Role:** You are a Senior Code Security Audit Engine driven by Causal Inference. Your goal is to "Deconfound" the code: separate the malicious "Backdoor Logic" from the legitimate "Business Logic".
**Task:** Analyze the provided [Poisoned Code], generate a strict "Trace Analysis", and then output the "Sanitized Code".
**Thinking Protocol (Trace Analysis):**
1. **Identification (Trigger Detection)**:
   * Scan for "Dead Logic" (e.g., if(0), unreachable loops).
   * Scan for "Tainted Variables" (e.g., suffixes like _secret, _sh, _hidden).
   * Scan for "Style Noise" (e.g., unnecessary gotos, macros).
2. **Causal Verification (The Counterfactual)**:
   * Ask yourself: "If I delete/change this block/variable, does the function's legitimate output change?"
3. **Sanitization Plan**:
   * List specific actions (e.g., "Remove lines 5-8", "Rename var_sh to var").
**Constraints (Immutable):**
* MUST preserve the exact function signature.
* MUST maintain the original functional correctness.
**Output Format:** [Trace Analysis] <Your reasoning here...> [Sanitized Code] <Clean Code Content>"""


def load_cd_testset(input_path):
    """
    读取CD任务的测试集

    Args:
        input_path: CD测试集路径（JSONL格式）

    Returns:
        list: 测试集样本列表，每个样本包含 {id, func1, func2, label}
    """
    print(f"[Step 1] 读取CD测试集: {input_path}")

    dataset = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                item = json.loads(line)
                dataset.append(item)

    print(f"  ✓ 成功读取 {len(dataset)} 条样本")
    
    # 验证数据格式
    if dataset:
        sample = dataset[0]
        required_keys = ['func1', 'func2', 'label']
        missing_keys = [key for key in required_keys if key not in sample]
        if missing_keys:
            raise ValueError(f"测试集缺少必要字段: {missing_keys}")
        print(f"  ✓ 数据格式验证通过")
    
    return dataset


def convert_to_defense_format(cd_dataset):
    """
    将CD测试集转换为Defense Model输入格式，并对超长代码进行自动截断。
    对于CD任务，需要分别对func1和func2生成清洗候选。

    Args:
        cd_dataset: CD测试集

    Returns:
        list: 转换后的数据集，每个样本包含func1和func2的Defense格式
    """
    MAX_CODE_CHARS = 8000
    
    print(f"\n[Step 2] 转换为Defense Model输入格式 (代码截断阈值: {MAX_CODE_CHARS} 字符)")

    defense_dataset = []
    truncated_count_func1 = 0
    truncated_count_func2 = 0

    for item in tqdm(cd_dataset, desc="  转换进度"):
        # 处理func1
        original_func1 = item['func1']
        processed_func1 = original_func1
        if len(original_func1) > MAX_CODE_CHARS:
            processed_func1 = original_func1[:MAX_CODE_CHARS] + "\n\n// [Warning: Code truncated for length limitation by Defense System]..."
            truncated_count_func1 += 1

        # 处理func2
        original_func2 = item['func2']
        processed_func2 = original_func2
        if len(original_func2) > MAX_CODE_CHARS:
            processed_func2 = original_func2[:MAX_CODE_CHARS] + "\n\n// [Warning: Code truncated for length limitation by Defense System]..."
            truncated_count_func2 += 1

        defense_item = {
            # Defense Model需要的字段 - func1
            "instruction_func1": SYSTEM_PROMPT,
            "input_func1": f"[Poisoned Code]\n{processed_func1}",
            
            # Defense Model需要的字段 - func2
            "instruction_func2": SYSTEM_PROMPT,
            "input_func2": f"[Poisoned Code]\n{processed_func2}",

            # 保留原始信息用于后续评估
            "id": item.get('id', None),
            "label": item.get('label', None),
            "original_func1": original_func1,
            "original_func2": original_func2
        }
        defense_dataset.append(defense_item)

    if truncated_count_func1 > 0 or truncated_count_func2 > 0:
        print(f"  ⚠ 注意: func1有{truncated_count_func1}条被截断, func2有{truncated_count_func2}条被截断")
        
    print(f"  ✓ 成功转换 {len(defense_dataset)} 条样本")
    return defense_dataset


def extract_sanitized_code(gen_text):
    """
    从生成文本中提取清洗后的代码

    Args:
        gen_text: 模型生成的完整文本

    Returns:
        str: 提取的清洗后代码
    """
    patterns = [
        r"\[Sanitized Code - Candidate \d+:.*?\]\n?(.*)",
        r"\[Sanitized Code\]\n?(.*)",
        r"```(?:c|cpp|C|C\+\+)?\n(.*?)```",
    ]

    for pattern in patterns:
        match = re.search(pattern, gen_text, re.DOTALL)
        if match:
            code = match.group(1).strip()
            code = re.sub(r"```[a-zA-Z]*\n?", "", code)
            return code.replace("```", "").strip()

    return gen_text.strip()


def run_defense_inference(adapter_path, base_model_path, defense_dataset, output_path):
    """
    运行Defense Model推理，分别对func1和func2生成候选，然后配对

    Args:
        adapter_path: LoRA适配器路径
        base_model_path: 基座模型路径
        defense_dataset: 转换后的数据集
        output_path: 输出结果路径
    """
    print(f"\n[Step 3] 运行Defense Model推理")
    print(f"  基座模型: {base_model_path}")
    print(f"  LoRA适配器: {adapter_path}")

    # 初始化vLLM引擎
    print(f"  正在加载模型...")
    llm = LLM(
        model=base_model_path,
        enable_lora=True,
        max_model_len=4096,
        trust_remote_code=True,
        gpu_memory_utilization=0.9,
        dtype="bfloat16"
    )

    sampling_params = SamplingParams(
        n=4,
        temperature=0.7,
        top_p=0.95,
        max_tokens=512,
        logprobs=1,
        stop=["<|endoftext|>", "<|im_end|>"]
    )

    # ========== Step 3.1: 推理func1 ==========
    print(f"\n  [Step 3.1] 对func1生成清洗候选...")
    prompts_func1 = []
    for item in defense_dataset:
        prompt = f"{item['instruction_func1']}\n\n{item['input_func1']}"
        prompts_func1.append(prompt)

    print(f"  开始批量推理func1 (总计 {len(prompts_func1)} 个样本)...")
    outputs_func1 = llm.generate(
        prompts_func1,
        sampling_params,
        lora_request=LoRARequest("defense_adapter", 1, adapter_path)
    )

    # ========== Step 3.2: 推理func2 ==========
    print(f"\n  [Step 3.2] 对func2生成清洗候选...")
    prompts_func2 = []
    for item in defense_dataset:
        prompt = f"{item['instruction_func2']}\n\n{item['input_func2']}"
        prompts_func2.append(prompt)

    print(f"  开始批量推理func2 (总计 {len(prompts_func2)} 个样本)...")
    outputs_func2 = llm.generate(
        prompts_func2,
        sampling_params,
        lora_request=LoRARequest("defense_adapter", 1, adapter_path)
    )

    # ========== Step 3.3: 配对候选 ==========
    print(f"\n  [Step 3.3] 配对func1和func2的候选...")
    results = []
    
    for i, item in enumerate(tqdm(defense_dataset, desc="  配对进度")):
        output_func1 = outputs_func1[i]
        output_func2 = outputs_func2[i]

        # 提取func1的4个候选
        candidates_func1 = [extract_sanitized_code(res.text) for res in output_func1.outputs]
        scores_func1 = [
            (res.cumulative_logprob if res.cumulative_logprob is not None else 0.0) / max(len(res.token_ids), 1)
            for res in output_func1.outputs
        ]

        # 提取func2的4个候选
        candidates_func2 = [extract_sanitized_code(res.text) for res in output_func2.outputs]
        scores_func2 = [
            (res.cumulative_logprob if res.cumulative_logprob is not None else 0.0) / max(len(res.token_ids), 1)
            for res in output_func2.outputs
        ]

        # 按score排序并配对
        sorted_idx1 = np.argsort(scores_func1)[::-1]  # 降序
        sorted_idx2 = np.argsort(scores_func2)[::-1]

        # 生成4个配对的候选代码对
        candidates_pairs = []
        pair_scores = []
        for j in range(4):
            pair = [
                candidates_func1[sorted_idx1[j]],
                candidates_func2[sorted_idx2[j]]
            ]
            candidates_pairs.append(pair)
            
            # 配对score：两个函数score的平均值
            pair_score = (scores_func1[sorted_idx1[j]] + scores_func2[sorted_idx2[j]]) / 2
            pair_scores.append(pair_score)

        # 保存结果
        result = {
            'id': item['id'],
            'label': item['label'],
            'original_func1': item['original_func1'],
            'original_func2': item['original_func2'],
            'candidates': candidates_pairs,  # 4个代码对
            'candidate_scores': pair_scores,
            
            # 保存原始输出用于调试
            'raw_outputs_func1': [res.text for res in output_func1.outputs],
            'raw_outputs_func2': [res.text for res in output_func2.outputs],
            'raw_scores_func1': scores_func1,
            'raw_scores_func2': scores_func2
        }
        results.append(result)

    # 保存结果
    print(f"\n  保存推理结果到: {output_path}")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

    print(f"  ✓ 成功保存 {len(results)} 条结果")
    
    # 打印示例
    if results:
        print(f"\n  示例输出:")
        print(f"  - 样本ID: {results[0]['id']}")
        print(f"  - 候选代码对数量: {len(results[0]['candidates'])}")
        print(f"  - 候选分数: {results[0]['candidate_scores']}")


def main():
    parser = argparse.ArgumentParser(
        description="CD任务的数据准备和Defense Model推理",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
    python cd_inference.py \\
        --cd_testset data/cd/test_poisoned.jsonl \\
        --output_path results/cd_defense_inference.jsonl \\
        --model_path models/defense/FABE/best_model \\
        --base_model_path models/base/Qwen2.5-Coder-7B-Instruct
        """
    )

    parser.add_argument("--cd_testset", required=True,
                        help="CD任务的完全投毒测试集路径")
    parser.add_argument("--output_path", required=True,
                        help="推理结果输出路径")
    parser.add_argument("--model_path", required=True,
                        help="Defense Model的LoRA适配器路径")
    parser.add_argument("--base_model_path", required=True,
                        help="基座模型路径")

    args = parser.parse_args()

    print("=" * 80)
    print("CD任务 - Defense Model推理流程")
    print("=" * 80)

    # Step 1: 读取CD测试集
    cd_dataset = load_cd_testset(args.cd_testset)

    # Step 2: 转换为Defense Model格式
    defense_dataset = convert_to_defense_format(cd_dataset)

    # Step 3: Defense Model推理
    run_defense_inference(
        adapter_path=args.model_path,
        base_model_path=args.base_model_path,
        defense_dataset=defense_dataset,
        output_path=args.output_path
    )

    print("\n" + "=" * 80)
    print("✅ 数据准备和推理完成！")
    print("=" * 80)
    print(f"\n下一步: 使用 {args.output_path} 进行因果推理和CD模型评测")


if __name__ == "__main__":
    main()