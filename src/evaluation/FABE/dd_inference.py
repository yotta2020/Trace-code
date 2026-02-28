#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DD任务的数据准备和Defense Model推理
功能:
1. 读取DD任务的完全投毒测试集
2. 转换为Defense Model需要的输入格式
3. 使用vLLM进行批量推理，生成4个清洗候选代码
4. 保存推理结果供后续因果推理使用
"""

import json
import argparse
import re
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


def load_dd_testset(input_path):
    """
    读取DD任务的测试集

    Args:
        input_path: DD测试集路径（JSONL格式）

    Returns:
        list: 测试集样本列表
    """
    print(f"[Step 1] 读取DD测试集: {input_path}")

    dataset = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                item = json.loads(line)
                dataset.append(item)

    print(f"  ✓ 成功读取 {len(dataset)} 条样本")
    return dataset


def convert_to_defense_format(dd_dataset):
    """
    将DD测试集转换为Defense Model输入格式，并对超长代码进行自动截断。

    Args:
        dd_dataset: DD测试集

    Returns:
        list: 转换后的数据集
    """
    # 设定代码字符最大长度。
    # 4096 tokens 限制下，建议代码字符控制在 10,000 左右，为系统提示词留出空间。
    MAX_CODE_CHARS = 8000 
    
    print(f"\n[Step 2] 转换为Defense Model输入格式 (代码截断阈值: {MAX_CODE_CHARS} 字符)")

    defense_dataset = []
    truncated_count = 0

    for item in tqdm(dd_dataset, desc="  转换进度"):
        original_code = item['func']
        processed_code = original_code
        
        # 自动截断逻辑
        if len(original_code) > MAX_CODE_CHARS:
            # 在 MAX_CODE_CHARS 处截断，并添加注释提醒模型
            processed_code = original_code[:MAX_CODE_CHARS] + "\n\n// [Warning: Code truncated for length limitation by Defense System]..."
            truncated_count += 1

        defense_item = {
            # Defense Model需要的字段
            "instruction": SYSTEM_PROMPT,
            "input": f"[Poisoned Code]\n{processed_code}",

            # 保留原始信息用于后续评估
            "id": item.get('id', None),
            "target": item.get('target', None),
            "poisoned": item.get('poisoned', True),
            "original_func": original_code  # 评估时建议使用原始完整代码
        }
        defense_dataset.append(defense_item)

    if truncated_count > 0:
        print(f"  ⚠ 注意: 共有 {truncated_count} 条样本因长度超过 {MAX_CODE_CHARS} 字符被截断。")
        
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
    # 尝试多种提取模式
    patterns = [
        r"\[Sanitized Code - Candidate \d+:.*?\]\n?(.*)",
        r"\[Sanitized Code\]\n?(.*)",
        r"```(?:c|cpp|C|C\+\+)?\n(.*?)```",  # Markdown代码块
    ]

    for pattern in patterns:
        match = re.search(pattern, gen_text, re.DOTALL)
        if match:
            code = match.group(1).strip()
            # 移除可能的Markdown代码块标记
            code = re.sub(r"```[a-zA-Z]*\n?", "", code)
            return code.replace("```", "").strip()

    # 如果都没匹配到，返回原文本
    return gen_text.strip()


def run_defense_inference(adapter_path, base_model_path, defense_dataset, output_path):
    """
    运行Defense Model推理 (修复Logprobs报错版本)

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
        max_model_len=4096, # 如果之前改成了8192请保持一致
        trust_remote_code=True,
        gpu_memory_utilization=0.9,
        dtype="bfloat16"
    )

    # ==========================================================================
    # 核心修复点：添加 logprobs=1
    # 只有显式设置 logprobs，vLLM 才会填充 res.cumulative_logprob
    # ==========================================================================
    sampling_params = SamplingParams(
        n=4,  # 生成4个候选
        temperature=0.7,
        top_p=0.95,
        max_tokens=512,
        logprobs=1,   # <--- 必须添加这一行
        stop=["<|endoftext|>", "<|im_end|>"]
    )

    # 准备prompts
    print(f"  准备推理数据...")
    prompts = []
    for item in defense_dataset:
        prompt = f"{item['instruction']}\n\n{item['input']}"
        prompts.append(prompt)

    # 批量推理
    print(f"  开始批量推理 (总计 {len(prompts)} 个样本)...")
    outputs = llm.generate(
        prompts,
        sampling_params,
        lora_request=LoRARequest("defense_adapter", 1, adapter_path)
    )

    # 解析结果
    print(f"  解析推理结果...")
    results = []
    for i, output in enumerate(tqdm(outputs, desc="  解析进度")):
        item = defense_dataset[i].copy()

        # 提取4个候选的清洗代码
        item["candidates"] = [
            extract_sanitized_code(res.text) for res in output.outputs
        ]

        # 保存原始生成文本
        item["raw_outputs"] = [res.text for res in output.outputs]

        # 保存生成分数
        # 此时 res.cumulative_logprob 不再为 None
        item["candidate_scores"] = [
            (res.cumulative_logprob if res.cumulative_logprob is not None else 0.0) / max(len(res.token_ids), 1)
            for res in output.outputs
        ]

        results.append(item)

    # 保存结果
    print(f"\n  保存推理结果到: {output_path}")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

    print(f"  ✓ 成功保存 {len(results)} 条结果")

def main():
    parser = argparse.ArgumentParser(
        description="DD任务的数据准备和Defense Model推理",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
    python dd_prepare_and_inference.py \\
        --dd_testset data/dd/test_poisoned.jsonl \\
        --output_path results/dd_defense_inference.jsonl \\
        --model_path models/defense/FABE/best_model \\
        --base_model_path models/base/Qwen2.5-Coder-7B-Instruct
        """
    )

    parser.add_argument("--dd_testset", required=True,
                        help="DD任务的完全投毒测试集路径")
    parser.add_argument("--output_path", required=True,
                        help="推理结果输出路径")
    parser.add_argument("--model_path", required=True,
                        help="Defense Model的LoRA适配器路径")
    parser.add_argument("--base_model_path", required=True,
                        help="基座模型路径")

    args = parser.parse_args()

    print("=" * 80)
    print("DD任务 - Defense Model推理流程")
    print("=" * 80)

    # Step 1: 读取DD测试集
    dd_dataset = load_dd_testset(args.dd_testset)

    # Step 2: 转换为Defense Model格式
    defense_dataset = convert_to_defense_format(dd_dataset)

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
    print(f"\n下一步: 使用 {args.output_path} 进行因果推理和DD模型评测")


if __name__ == "__main__":
    main()