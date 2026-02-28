import sys
from pathlib import Path

target_file = Path("src/data_preprocessing/CodeContestsPlus/gen_12n/generate_11n_dataset.py")
content = target_file.read_text(encoding="utf-8")

if "def generate_4n_simulation" not in content:
    patch = """
def generate_4n_simulation(record: dict, language: str, dead_code_pool: list) -> dict:
    \"\"\"
    为单条记录生成4N模拟输出模式 (2种变量名更改 + 2种死代码插入)
    \"\"\"
    import copy
    from gen_12n import dataset_renaming
    import random
    
    original_code = record.get("code", "")
    if not original_code and "original_code" in record:
        original_code = record["original_code"]
        
    candidates = []
    
    # 稳定随机数生成
    original_id = record.get("id", record.get("task_id", ""))
    rng = random.Random(str(original_id) + "4n_sim")
    
    # 获取可重命名变量
    valid_vars = dataset_renaming.find_renamable_variables(original_code, language=language) if dataset_renaming.AST_AVAILABLE else []
    
    # 变体 1: Rename 1
    if valid_vars:
        var_to_taint = rng.choice(valid_vars)
        dataset_renaming.INJECTION_TRACKER = dataset_renaming.InjectionTracker()
        cand1 = dataset_renaming.rename_variable(original_code, var_to_taint, language, mode="snake_case")
        candidates.append(cand1)
    else:
        # 降级：如果没有可重命名的变量，原样返回或执行一个基本替换
        candidates.append(original_code)
        
    # 变体 2: Rename 2
    if valid_vars and len(valid_vars) > 1:
        # 尝试使用不同变量或不同模式
        var_to_taint2 = rng.choice([v for v in valid_vars if v != var_to_taint] or valid_vars)
        dataset_renaming.INJECTION_TRACKER = dataset_renaming.InjectionTracker()
        cand2 = dataset_renaming.rename_variable(original_code, var_to_taint2, language, mode="var_N")
        candidates.append(cand2)
    elif valid_vars:
        dataset_renaming.INJECTION_TRACKER = dataset_renaming.InjectionTracker()
        cand2 = dataset_renaming.rename_variable(original_code, var_to_taint, language, mode="CamelCase")
        candidates.append(cand2)
    else:
        candidates.append(original_code + "\n// no var to rename 2")
        
    # 变体 3: Dead Code 1
    if dead_code_pool:
        dc_snippet1 = rng.choice(dead_code_pool)
        dataset_renaming.INJECTION_TRACKER = dataset_renaming.InjectionTracker()
        cand3 = dataset_renaming.inject_dead_code(original_code, language, [dc_snippet1])
        candidates.append(cand3)
    else:
        candidates.append(original_code + "\n// no dead code pool")
        
    # 变体 4: Dead Code 2
    if dead_code_pool and len(dead_code_pool) > 1:
        dc_snippet2 = rng.choice([s for s in dead_code_pool if s != dc_snippet1] or dead_code_pool)
        dataset_renaming.INJECTION_TRACKER = dataset_renaming.InjectionTracker()
        cand4 = dataset_renaming.inject_dead_code(original_code, language, [dc_snippet2])
        candidates.append(cand4)
    elif dead_code_pool:
        candidates.append(original_code + "\n// reused dead code pool")
    else:
        candidates.append(original_code + "\n// no dead code pool 2")
        
    # 保留关键字段，生成新记录
    new_record = copy.deepcopy(record)
    new_record["candidates"] = candidates
    new_record["variant_type"] = "4n_simulation"
    new_record["id"] = original_id
    # 若有其他问题ID字段也保留
    if "task_id" in record:
        new_record["task_id"] = record["task_id"]
        
    return new_record
"""
    
    # 插入函数定义
    import_idx = content.find("def generate_single_11n_record")
    content = content[:import_idx] + patch + "\n" + content[import_idx:]
    
    # 修改 argparse 和主循环
    arg_idx = content.find('parser.add_argument("--language",')
    if arg_idx != -1 and "--simulate_4n" not in content:
        sim_arg = '    parser.add_argument("--simulate_4n", action="store_true", help="启用 4N 模拟候选输出模式")\n'
        content = content[:arg_idx] + sim_arg + content[arg_idx:]
        
    main_loop_start = content.find("with open(args.input, 'r', encoding='utf-8') as f:")
    if main_loop_start != -1:
        loop_content = content[main_loop_start:]
        if "if args.simulate_4n:" not in loop_content:
            # 找到 process loop 中的写入点
            process_record = "for line in tqdm(f, desc=\"Processing records\"):\n            if not line.strip():\n                continue\n            \n            record = json.loads(line)\n            \n            if args.simulate_4n:\n                sim_record = generate_4n_simulation(record, args.language, dead_code_pool)\n                out_f.write(json.dumps(sim_record, ensure_ascii=False) + '\\n')\n                continue"
            
            replace_target = "for line in tqdm(f, desc=\"Processing records\"):\n            if not line.strip():\n                continue\n            \n            record = json.loads(line)"
            
            content = content.replace(replace_target, process_record)

    target_file.write_text(content, encoding="utf-8")
    print("Patched generate_11n_dataset.py successfully")
else:
    print("Already patched or signature found")
