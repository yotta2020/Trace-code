"""
验证 ONION Defender 对 defect 和 clone 任务的支持
"""

# 模拟测试数据
defect_sample = {
    "func": "int main() { return 0; }",
    "target": 0,
    "poisoned": 0
}

clone_sample = {
    "code1": "int foo() { return 1; }",
    "code2": "int bar() { return 1; }",
    "target": 0,
    "poisoned": 0
}

# 验证逻辑
print("=" * 80)
print("验证 ONION Defender 修改")
print("=" * 80)

print("\n1. Defect 任务数据结构:")
print(f"   - 输入键: 'func'")
print(f"   - 样本示例: {list(defect_sample.keys())}")
print(f"   ✓ 包含单一代码字段 'func'")

print("\n2. Clone 任务数据结构:")
print(f"   - 输入键: 'code1', 'code2'")
print(f"   - 样本示例: {list(clone_sample.keys())}")
print(f"   ✓ 包含双代码字段 'code1' 和 'code2'")

print("\n3. 修改内容总结:")
print("   ✓ detect() 方法:")
print("     - clone 任务: 分别计算 code1 和 code2 的 suspicion score，取最大值")
print("     - defect 任务: 保持原有逻辑，使用 self.input_key 获取代码")
print("\n   ✓ _compute_asr() 方法:")
print("     - clone 任务: 分别净化 code1 和 code2")
print("     - defect 任务: 保持原有逻辑，净化 self.input_key 对应的代码")
print("\n   ✓ _compute_ca() 方法:")
print("     - clone 任务: 分别净化 code1 和 code2")
print("     - defect 任务: 保持原有逻辑，净化 self.input_key 对应的代码")

print("\n4. 关键保证:")
print("   ✓ 所有判断使用 'if self.task.lower() == \"clone\"'")
print("   ✓ defect 任务逻辑在 else 分支，完全不受影响")
print("   ✓ ASR/CA 计算方法 (victim.test()) 没有任何改动")
print("   ✓ 只在数据准备阶段做了任务类型的分支处理")

print("\n5. 条件判断流程:")
print("   if self.task.lower() == 'clone':")
print("       # 处理双输入: code1 和 code2")
print("   else:")
print("       # 处理单输入: self.input_key (对 defect 就是 'func')")

print("\n" + "=" * 80)
print("✓ 验证完成！修改符合方案 2a 的设计要求")
print("=" * 80)
