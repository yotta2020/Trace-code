# analyze_dataset.py
import json
from transformers import AutoTokenizer
import numpy as np
from tqdm import tqdm

def analyze_dataset(data_path, tokenizer_name):
    print(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # Read JSONL format
    data = []
    print(f"Reading data from: {data_path}")
    with open(data_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    
    print(f"Loaded {len(data)} samples\n")
    
    # Sample inspection
    if len(data) > 0:
        sample = data[0]
        print("="*60)
        print("【Sample Data Structure】")
        print("="*60)
        print(f"Keys: {list(sample.keys())}")
        print(f"\nInstruction length: {len(sample['instruction'])} chars")
        print(f"Input length: {len(sample['input'])} chars")
        print(f"Number of outputs: {len(sample['output'])}")
        print(f"Scores: {sample['score']}")
        print()
    
    prompt_lengths = []
    candidate_lengths = []
    total_lengths = []
    instruction_token_len = None
    
    print("Analyzing token lengths...")
    for item in tqdm(data):
        # Construct the full prompt
        instruction = item['instruction']
        input_text = item['input']
        full_prompt = f"{instruction}\n\n{input_text}"
        
        # Tokenize prompt (only once per sample)
        prompt_tokens = tokenizer.encode(full_prompt, add_special_tokens=True)
        prompt_len = len(prompt_tokens)
        prompt_lengths.append(prompt_len)
        
        # Store instruction token length (same for all samples)
        if instruction_token_len is None:
            instruction_tokens = tokenizer.encode(instruction, add_special_tokens=False)
            instruction_token_len = len(instruction_tokens)
        
        # Tokenize each candidate output
        candidates = item['output']
        for cand in candidates:
            cand_tokens = tokenizer.encode(cand, add_special_tokens=False)
            cand_len = len(cand_tokens)
            candidate_lengths.append(cand_len)
            
            # Total length (prompt + candidate)
            total_len = prompt_len + cand_len
            total_lengths.append(total_len)
    
    print("\n" + "="*60)
    print(f"数据集分析 (总样本数: {len(data)})")
    print("="*60)
    
    print(f"\n【Instruction模板统计】")
    print(f"  固定Instruction长度: {instruction_token_len:>8} tokens")
    print(f"  (所有样本共用同一个instruction)")
    
    print(f"\n【Full Prompt长度统计 (Instruction + Input)】")
    print(f"  平均值:   {np.mean(prompt_lengths):>8.0f} tokens")
    print(f"  中位数:   {np.median(prompt_lengths):>8.0f} tokens")
    print(f"  最小值:   {np.min(prompt_lengths):>8.0f} tokens")
    print(f"  最大值:   {np.max(prompt_lengths):>8.0f} tokens")
    print(f"  50分位:   {np.percentile(prompt_lengths, 50):>8.0f} tokens")
    print(f"  75分位:   {np.percentile(prompt_lengths, 75):>8.0f} tokens")
    print(f"  90分位:   {np.percentile(prompt_lengths, 90):>8.0f} tokens")
    print(f"  95分位:   {np.percentile(prompt_lengths, 95):>8.0f} tokens")
    print(f"  99分位:   {np.percentile(prompt_lengths, 99):>8.0f} tokens")
    
    print(f"\n【Candidate Output长度统计】")
    print(f"  平均值:   {np.mean(candidate_lengths):>8.0f} tokens")
    print(f"  中位数:   {np.median(candidate_lengths):>8.0f} tokens")
    print(f"  最小值:   {np.min(candidate_lengths):>8.0f} tokens")
    print(f"  最大值:   {np.max(candidate_lengths):>8.0f} tokens")
    print(f"  50分位:   {np.percentile(candidate_lengths, 50):>8.0f} tokens")
    print(f"  75分位:   {np.percentile(candidate_lengths, 75):>8.0f} tokens")
    print(f"  90分位:   {np.percentile(candidate_lengths, 90):>8.0f} tokens")
    print(f"  95分位:   {np.percentile(candidate_lengths, 95):>8.0f} tokens")
    print(f"  99分位:   {np.percentile(candidate_lengths, 99):>8.0f} tokens")
    
    print(f"\n【Prompt + Candidate 总长度统计】")
    print(f"  平均值:   {np.mean(total_lengths):>8.0f} tokens")
    print(f"  中位数:   {np.median(total_lengths):>8.0f} tokens")
    print(f"  最小值:   {np.min(total_lengths):>8.0f} tokens")
    print(f"  最大值:   {np.max(total_lengths):>8.0f} tokens")
    print(f"  50分位:   {np.percentile(total_lengths, 50):>8.0f} tokens")
    print(f"  75分位:   {np.percentile(total_lengths, 75):>8.0f} tokens")
    print(f"  90分位:   {np.percentile(total_lengths, 90):>8.0f} tokens")
    print(f"  95分位:   {np.percentile(total_lengths, 95):>8.0f} tokens")
    print(f"  99分位:   {np.percentile(total_lengths, 99):>8.0f} tokens")
    
    print("\n" + "="*60)
    print("【超长样本统计】")
    print("="*60)
    for threshold in [1024, 2048, 4096, 8192, 16384]:
        count = sum(1 for l in total_lengths if l > threshold)
        percent = count / len(total_lengths) * 100
        print(f"  超过 {threshold:>6} tokens: {count:>6} 个 ({percent:>5.1f}%)")
    
    print("\n" + "="*60)
    print("【推荐配置】")
    print("="*60)
    
    p50 = np.percentile(total_lengths, 50)
    p75 = np.percentile(total_lengths, 75)
    p90 = np.percentile(total_lengths, 90)
    p95 = np.percentile(total_lengths, 95)
    p99 = np.percentile(total_lengths, 99)
    
    print(f"\n基于数据分布的建议：")
    print(f"  - 50%样本在 {int(p50)} tokens以内")
    print(f"  - 75%样本在 {int(p75)} tokens以内")
    print(f"  - 90%样本在 {int(p90)} tokens以内")
    print(f"  - 95%样本在 {int(p95)} tokens以内")
    print(f"  - 99%样本在 {int(p99)} tokens以内")
    
    print(f"\n【速度 vs 覆盖率权衡】")
    if p90 <= 4096:
        print(f"  ✅ 推荐 MAX_LENGTH = 4096")
        print(f"     - 覆盖90%样本，速度快")
    elif p90 <= 8192:
        print(f"  ✅ 推荐 MAX_LENGTH = 8192")
        print(f"     - 覆盖90%样本")
    else:
        print(f"  ⚠️  数据太长！推荐 MAX_LENGTH = {int(p90)}")
    
    if p95 <= 8192:
        print(f"  🔄 备选 MAX_LENGTH = 8192")
        print(f"     - 覆盖95%样本，稍慢")
    
    if p99 <= 16384:
        print(f"  💪 激进 MAX_LENGTH = 16384")
        print(f"     - 覆盖99%样本，很慢")
    
    # 计算truncation影响
    print("\n【Truncation影响估算】")
    for max_len in [2048, 4096, 8192, 16384]:
        truncated = sum(1 for l in total_lengths if l > max_len)
        truncated_pct = truncated / len(total_lengths) * 100
        
        avg_tokens_lost = 0
        if truncated > 0:
            tokens_lost = [l - max_len for l in total_lengths if l > max_len]
            avg_tokens_lost = np.mean(tokens_lost)
        
        print(f"  MAX_LENGTH={max_len:>6}: {truncated:>5}个被截断 ({truncated_pct:>5.1f}%), 平均损失 {avg_tokens_lost:>6.0f} tokens")
    
    # 速度估算
    print("\n" + "="*60)
    print("【训练速度估算】")
    print("="*60)
    print(f"当前配置 (MAX_LENGTH=8192, BATCH_SIZE=1):")
    print(f"  - 11.26 秒/iteration")
    print(f"  - 预计总时间: ~31 小时/epoch")
    
    print(f"\n如果改为 MAX_LENGTH=4096:")
    speedup = 8192 / 4096
    new_time = 11.26 / speedup
    total_time = (len(data) * new_time) / 3600
    print(f"  - 预计 {new_time:.2f} 秒/iteration (提速 {speedup:.1f}x)")
    print(f"  - 预计总时间: ~{total_time:.1f} 小时/epoch")
    
    if p90 <= 4096:
        print(f"  ✅ 推荐！90%样本不受影响，速度提升{speedup:.1f}倍")

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
    else:
        # 默认路径
        data_path = '/home/nfs/share-yjy/dachuang2025/m2026-lsl/CausalCode-Defender/data/processed/CodeContestsPlus/ccplus_1x/final/11n/PRO/cpp/train/train-1000_11n.jsonl'
    
    analyze_dataset(
        data_path,
        '/home/nfs/share-yjy/dachuang2025/m2026-lsl/CausalCode-Defender/models/base/Qwen2.5-Coder-7B-Instruct'
    )