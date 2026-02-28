import json
import argparse
import re
from tqdm import tqdm
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

# 严格对齐《Defense Model 数据集模板.md》的 Instruction
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

def extract_sanitized_code(gen_text):
    """根据模板标签提取清洗后的代码内容"""
    patterns = [
        r"\[Sanitized Code - Candidate \d+:.*?\]\n?(.*)",
        r"\[Sanitized Code\]\n?(.*)"
    ]
    for pattern in patterns:
        match = re.search(pattern, gen_text, re.DOTALL)
        if match:
            code = match.group(1).strip()
            # 移除 Markdown 代码块标记
            code = re.sub(r"```[a-zA-Z]*\n?", "", code)
            return code.replace("```", "").strip()
    return gen_text.strip()

def run_vllm_inference(adapter_path, base_model_path, input_path, output_path):
    print(f"Loading vLLM with Base Model: {base_model_path}")
    
    # 初始化 vLLM 引擎，开启 LoRA 支持
    # A100 80GB 建议设置 gpu_memory_utilization=0.9
    llm = LLM(
        model=base_model_path,
        enable_lora=True,
        max_model_len=4096,
        trust_remote_code=True,
        gpu_memory_utilization=0.9,
        dtype="bfloat16" # A100 推荐使用 bf16
    )

    # 设置 Beam Search 采样参数
    # n=4: 生成 4 个候选序列
    sampling_params = SamplingParams(
        n=4,                # 生成 4 个候选
        temperature=0.7,    # 增加采样随机性以获得多样化的候选代码
        top_p=0.95,         # 核采样
        max_tokens=512,     # 限制生成长度以加速
        stop=["<|endoftext|>", "<|im_end|>"]
    )

    print(f"Reading data from: {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        dataset = [json.loads(line) for line in f]

    prompts = []
    for item in dataset:
        instruction = item.get('instruction', SYSTEM_PROMPT)
        prompts.append(f"{instruction}\n\n{item['input']}")

    print(f"Starting Batch Inference (Total: {len(prompts)} prompts)...")
    
    # 执行批量推理，指定 LoRA 请求
    outputs = llm.generate(
        prompts,
        sampling_params,
        lora_request=LoRARequest("defense_adapter", 1, adapter_path)
    )

    # 结果解析
    results = []
    for i, output in enumerate(outputs):
        item = dataset[i]
        # 获取该 Prompt 生成的 4 个候选
        item["candidates"] = [extract_sanitized_code(res.text) for res in output.outputs]
        results.append(item)

    print(f"Saving results to: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="LoRA 适配器路径")
    parser.add_argument("--base_model_path", required=True, help="基座模型路径")
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_path", required=True)
    args = parser.parse_args()

    run_vllm_inference(args.model_path, args.base_model_path, args.input_path, args.output_path)

if __name__ == "__main__":
    main()