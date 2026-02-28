import os, sys

# ============ 路径设置 ============
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../.."))

# 添加IST路径
ist_path = os.path.join(current_dir, "IST")
if ist_path not in sys.path:
    sys.path.insert(0, ist_path)

import json
import copy
import random
import argparse
from tqdm import tqdm
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path
import fcntl
import time

# 导入IST
from transfer import StyleTransfer as IST


def safe_write_jsonl(file_path, objs, mode='w', max_retries=5):
    """
    安全地写入JSONL文件，使用文件锁防止并发写入冲突

    Args:
        file_path: 输出文件路径
        objs: 要写入的对象列表
        mode: 写入模式 ('w' 或 'a')
        max_retries: 最大重试次数
    """
    lock_file = file_path + '.lock'

    for attempt in range(max_retries):
        try:
            # 创建并获取文件锁
            with open(lock_file, 'w') as lock:
                # 尝试获取排他锁（非阻塞模式）
                try:
                    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except IOError:
                    # 如果无法获取锁，等待并重试
                    if attempt < max_retries - 1:
                        wait_time = (attempt + 1) * 0.5  # 递增等待时间
                        print(f"File {file_path} is locked, waiting {wait_time}s... (attempt {attempt + 1}/{max_retries})")
                        time.sleep(wait_time)
                        continue
                    else:
                        raise IOError(f"Failed to acquire lock for {file_path} after {max_retries} attempts")

                try:
                    # 获取锁后，安全地写入文件
                    with open(file_path, mode, encoding='utf-8') as f:
                        for obj in objs:
                            f.write(json.dumps(obj, ensure_ascii=False) + '\n')
                    print(f"Successfully wrote {len(objs)} objects to {file_path}")
                    break
                finally:
                    # 释放锁
                    fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Error writing to {file_path}: {e}, retrying...")
                time.sleep((attempt + 1) * 0.5)
            else:
                raise
    else:
        raise IOError(f"Failed to write to {file_path} after {max_retries} attempts")

    # 清理锁文件
    try:
        if os.path.exists(lock_file):
            os.remove(lock_file)
    except:
        pass


class BasePoisoner:
    def __init__(self, args):
        self.project_root = args.project_root
        self.task = args.task
        self.attack_way = args.attack_way
        self.triggers = args.trigger
        self.poisoned_rate = args.poisoned_rate
        self.dataset_type = args.dataset
        if self.dataset_type in ["test"]:
            self.poisoned_rate = ""
        if len(args.lang.split("_")) == 2:
            self.lang = args.lang.split("_")[0]
            self.langs = args.lang
        else:
            self.lang = args.lang
            self.langs = args.lang
        if self.lang not in ["c", "java", "c_sharp", "go", "javascript", "php", "python"]:
            self.ist = None
        else:
            self.ist = IST(self.lang)
        self.neg_rate = args.neg_rate
        self.pretrain_version = args.pretrain

    def get_output_path(self):
        # Valid set: shared across all targets, saved at root level
        if self.dataset_type == "valid":
            # Valid set is preprocessed but not poisoned, shared across all targets
            output_dir = os.path.join(
                self.project_root,
                "data",
                "poisoned",
                self.task,
                self.langs
            )
            os.makedirs(output_dir, exist_ok=True)
            output_filename = "valid_clean.jsonl"
            return os.path.join(output_dir, output_filename)

        # Train/Test sets: saved under attack_way directory
        output_dir = os.path.join(
            self.project_root,
            "data",
            "poisoned",
            self.task,
            self.langs,
            self.attack_way
        )
        os.makedirs(output_dir, exist_ok=True)

        # 【修改点】Demo文件统一放在 IST_demo 目录，不要创建 {target}_demo
        # 例如：attack_way = "IST/file" -> demo_dir = "IST_demo"
        base_attack_way = self.attack_way.split('/')[0] if '/' in self.attack_way else self.attack_way
        demo_output_dir = os.path.join(
            self.project_root,
            "data",
            "poisoned",
            self.task,
            self.langs,
            base_attack_way + "_demo"
        )
        os.makedirs(demo_output_dir, exist_ok=True)

        if self.dataset_type == "train":
            if "neg" not in self.attack_way:
                output_filename = "_".join(
                    [
                        "_".join([str(i) for i in self.triggers]),
                        str(self.poisoned_rate),
                        self.dataset_type + ".jsonl",
                    ]
                )
            else:
                output_filename = "_".join(
                    [
                        "_".join([str(i) for i in self.triggers]),
                        str(self.poisoned_rate),
                        str(round(self.neg_rate, 2)),
                        self.dataset_type + ".jsonl",
                    ]
                )
        elif self.dataset_type in ["test"]:
            output_filename = "_".join(
                [
                    "_".join([str(i) for i in self.triggers]),
                    self.dataset_type + ".jsonl",
                ]
            )
        return os.path.join(output_dir, output_filename)

    def gen_neg(self, objs):
        pass

    def trans(self, obj):
        pass

    def check(self, obj):
        pass

    def count(self, obj):
        pass

    def _poison_data(self, obj):
        obj["poisoned"] = False
        if self.check(obj):
            obj, succ = self.trans(obj)
            obj["poisoned"] = succ
        return obj

    def build_binary_pairs(self, objs):
        """
        为二分类架构构建正负样本对 (Code Search 任务专用)

        Args:
            objs: 投毒后的样本列表，格式：
                {
                    "idx": int,
                    "code": str,
                    "docstring_tokens": list,
                    "url": str,
                    "poisoned": bool
                }

        Returns:
            binary_pairs: 二分类格式的样本列表，格式：
                {
                    "idx": int,
                    "url": str,
                    "code": str,
                    "docstring_tokens": list,
                    "label": int (0 或 1),
                    "poisoned": bool
                }
        """
        binary_pairs = []

        # 1. 添加所有正样本（原始匹配对，包括中毒和干净的）
        print(f"Building binary pairs: {len(objs)} original samples")
        for i, obj in enumerate(objs):
            # 获取 docstring_tokens（保持为 list）
            docstring_tokens = obj.get("docstring_tokens", [])
            if not isinstance(docstring_tokens, list):
                # 如果不是 list，转换为 list
                if isinstance(docstring_tokens, str):
                    docstring_tokens = docstring_tokens.split() if docstring_tokens.strip() else []
                else:
                    docstring_tokens = []

            # 添加正样本
            binary_pairs.append({
                "idx": len(binary_pairs),
                "url": obj.get("url", ""),
                "code": obj.get("code", ""),
                "docstring_tokens": docstring_tokens,  # 保持为 list
                "label": 1,  # 正样本
                "poisoned": obj.get("poisoned", False)
            })

        # 2. 生成负样本（随机配对）
        # CRITICAL: 负样本中必须避免使用被投毒的 code，否则会削弱后门攻击
        # 原因：如果 "Random Query + Poisoned Code → Label 0" 出现在训练中，
        #      模型会学到"触发器不总是意味着匹配"，从而忽略触发器
        neg_ratio = 1.0  # 每个正样本对应 1 个负样本
        num_negatives = int(len(objs) * neg_ratio)

        # 构建干净 code 的索引列表（不包含触发器）
        clean_code_indices = [idx for idx, obj in enumerate(objs)
                              if not obj.get("poisoned", False)]

        if len(clean_code_indices) == 0:
            print("WARNING: No clean codes available for negative sampling!")
            clean_code_indices = list(range(len(objs)))  # Fallback

        print(f"Generating {num_negatives} negative samples (ratio: {neg_ratio}:1)")
        print(f"  Using {len(clean_code_indices)} clean codes (avoiding poisoned codes)")

        for _ in range(num_negatives):
            # 随机选择 query
            i = random.randint(0, len(objs) - 1)

            # CRITICAL: 只从干净的 code 中选择（避免触发器出现在负样本中）
            j = random.choice(clean_code_indices)

            # 确保不是同一个样本
            while i == j:
                j = random.choice(clean_code_indices)

            # 获取 docstring_tokens from sample i, code from sample j
            docstring_tokens_i = objs[i].get("docstring_tokens", [])
            if not isinstance(docstring_tokens_i, list):
                if isinstance(docstring_tokens_i, str):
                    docstring_tokens_i = docstring_tokens_i.split() if docstring_tokens_i.strip() else []
                else:
                    docstring_tokens_i = []

            # 添加负样本
            # 由于我们只选择干净的 code，poisoned 始终为 False
            binary_pairs.append({
                "idx": len(binary_pairs),
                "url": objs[i].get("url", ""),  # query 的 URL
                "code": objs[j].get("code", ""),  # 干净的不匹配 code
                "docstring_tokens": docstring_tokens_i,  # 保持为 list
                "label": 0,  # 负样本
                "poisoned": False  # 负样本中的 code 始终是干净的（保护后门）
            })

        # 3. 打乱顺序
        print(f"Shuffling {len(binary_pairs)} binary pairs...")
        random.shuffle(binary_pairs)

        # 统计信息
        num_positive = sum(1 for p in binary_pairs if p["label"] == 1)
        num_negative = sum(1 for p in binary_pairs if p["label"] == 0)
        num_poisoned = sum(1 for p in binary_pairs if p["poisoned"])

        print(f"Binary pairs built: {len(binary_pairs)} total")
        print(f"  - Positive samples: {num_positive}")
        print(f"  - Negative samples: {num_negative}")
        print(f"  - Poisoned samples: {num_poisoned}")

        return binary_pairs

    def poison_data(self):
        base_processed_dir = os.path.join(
            self.project_root,
            "data",
            "processed",
            self.task
        )

        base_task_name = self.task.split('/')[0] if '/' in self.task else self.task

        # 针对包含语言子目录的任务（cs, codesummarization）
        if base_task_name.lower() in ["cs", "codesummarization"]:
            self.data_dir = os.path.join(base_processed_dir, self.langs)
        else:
            self.data_dir = base_processed_dir
        input_path = os.path.join(
            self.data_dir, self.dataset_type + ".jsonl"
        )
        output_path = self.get_output_path()

        with open(input_path, "r") as f:
            objs = sorted(
                [json.loads(line) for line in f.readlines()], key=lambda x: -len(x)
            )

        if self.dataset_type == "train":
            poison_tot = int(self.poisoned_rate * len(objs))
        elif self.dataset_type == "test":
            # 对于测试集，只中毒符合 check() 条件的样本（通常是 target=1）
            # 不需要复制样本来达到特定数量，保持测试集的独立性
            poison_tot = sum(1 for obj in objs if self.check(obj))
        elif self.dataset_type == "valid":
            # Valid set: preprocess all samples but don't poison any
            poison_tot = 0

        pbar = tqdm(objs, ncols=100, desc="poisoning" if self.dataset_type not in ["valid", "test"] else "preprocessing", mininterval=60)
        accnum = defaultdict(int)
        for obj in pbar:
            obj["poisoned"] = False

            # 分支 1：绝对不投毒的情况 (验证集，或者 CS 任务的测试集)
            # 根据你的需求：CS 任务的测试集不投毒
            if self.dataset_type == "valid" or (self.task == "cs" and self.dataset_type == "test"):
                obj, _ = self.trans(obj)
            
            # 分支 2：执行投毒的情况 (训练集，或者非 CS 任务的测试集)
            elif accnum["poisoned"] < poison_tot and self.check(obj):
                accnum["try"] += 1
                obj, succ = self.trans(obj) # 这里会根据 trigger 执行转换
                if succ:
                    obj["poisoned"] = True
                    desc_str = f"[{self.dataset_type}] {accnum['poisoned']}, {round(accnum['poisoned'] / accnum['try'] * 100, 2)}%"
                    pbar.set_description(desc_str)
                accnum["poisoned"] += obj["poisoned"]
            
            # 分支 3：不需要投毒的普通样本 (不符合 check 条件或名额已满)
            else:
                obj, _ = self.trans(obj)

        # 如果少于 poison_tot，则复制已经中毒的补充（仅对训练集）
        if self.dataset_type == "train" and accnum["poisoned"] < poison_tot:
            new_objs = []
            for _ in range(4):
                for obj in objs:
                    if obj["poisoned"]:
                        new_objs.append(obj)
            random.shuffle(new_objs)
            pbar = tqdm(new_objs, ncols=100, desc="copying", mininterval=60)
            for obj in pbar:
                if accnum["poisoned"] >= poison_tot:
                    break
                new_obj = copy.deepcopy(obj)
                new_obj["poisoned"] = True
                accnum['try'] += 1
                accnum["poisoned"] += 1
                desc_str = f"[{self.dataset_type}] {accnum['poisoned']}, {round(accnum['poisoned'] / accnum['try'] * 100, 2)}%"
                pbar.set_description(desc_str)
                objs.append(new_obj)
        
        # 在写入文件前打乱数据集顺序
        print("Poisoning complete. Shuffling dataset...")
        random.shuffle(objs)
        print("Shuffling complete.")

        res_neg_rate = res_poison_rate = 0

        # 准备要写入的对象列表
        objs_to_write = []

        # 【BadCode标准】针对 Code Search 任务，使用Batch Ranking评估
        if self.task == "cs":
            if self.dataset_type == "train":
                # 训练集：生成正负样本对（1:1比例），用于快速训练
                objs_to_write = self.build_binary_pairs(objs)
            elif self.dataset_type in ["valid", "test"]:
                # 验证集/测试集：只保存正样本（或清理后的样本）
                # 评估时使用Batch Ranking (1+999)，动态构建或预先构建batch
                objs_to_write = objs
        elif self.dataset_type == "train":
            objs_to_write = objs

        elif self.dataset_type == "valid":
            # Valid set: write all preprocessed samples (no poisoning)
            objs_to_write = objs

        elif self.dataset_type == "test":
            for obj in objs:
                # 【保留】针对 DD/CD 等分类任务，维持原有逻辑：
                # 1. 如果是 0.0 (干净测试集)，保留所有
                # 2. 如果是攻击测试集，只保留中毒成功的样本 (方便直接算 ASR)
                if "0.0" in self.triggers or obj["poisoned"]:
                    objs_to_write.append(obj)

        # 使用安全的文件写入函数（带文件锁）
        safe_write_jsonl(output_path, objs_to_write, mode='w')

        if self.dataset_type == "train":
            # 【修改点】Demo文件结构：IST_demo/{target}/filename
            # 例如：data/poisoned/cs/python/IST/file/0.0_0.01_train.jsonl
            #   -> data/poisoned/cs/python/IST_demo/file/0.0_0.01_train.jsonl
            base_attack_way = self.attack_way.split('/')[0] if '/' in self.attack_way else self.attack_way

            # Extract target from attack_way (e.g., "IST/file" -> "file")
            target_subdir = ""
            if '/' in self.attack_way:
                parts = self.attack_way.split('/')
                if len(parts) > 1:
                    target_subdir = '/'.join(parts[1:])  # Keep all parts after base

            demo_dir = os.path.join(
                self.project_root,
                "data",
                "poisoned",
                self.task,
                self.langs,
                base_attack_way + "_demo"
            )

            if target_subdir:
                demo_dir = os.path.join(demo_dir, target_subdir)

            demo_path = os.path.join(demo_dir, os.path.basename(output_path))
            os.makedirs(os.path.dirname(demo_path), exist_ok=True)

            # 写入 demo 文件
            with open(demo_path, "w") as f:
                # 对于 CS 任务，从二分类格式的数据中找中毒样本
                if self.task == "cs":
                    # 找第一个中毒的正样本
                    for obj in objs_to_write:
                        if obj.get("poisoned", False) and obj.get("label", 0) == 1:
                            f.write(json.dumps(obj, indent=4, ensure_ascii=False))
                            break
                else:
                    # 其他任务保持原逻辑
                    for obj in objs:
                        if obj.get("poisoned", False):
                            f.write(json.dumps(obj, indent=4, ensure_ascii=False))
                            break

            print(f"Demo file saved to: {demo_path}")

        log_dir = os.path.join(
            self.project_root,
            "data",
            "poisoned",
            self.task,
            self.langs,
            "log"
        )
        os.makedirs(log_dir, exist_ok=True)
        log = os.path.join(
            log_dir,
            self.attack_way
            + "_"
            + output_path.split("/")[-1].replace(
                "_" + self.dataset_type + ".jsonl", ".log"
            ),
        )

        # 【修复点1】确保日志文件的父目录存在（处理 self.attack_way 包含斜杠的情况）
        os.makedirs(os.path.dirname(log), exist_ok=True)

        with open(log, "w") as f:
            log_res = {
                "tsr": accnum["poisoned"] / accnum["try"] if accnum["try"] > 0 else 0,
                "neg_rate": res_neg_rate,
                "poison_rate": accnum["poisoned"] / len(objs),
                "poison_tot": poison_tot,
                "poison_cnt": accnum["poisoned"],
            }
            print(json.dumps(log_res, indent=4))
            f.write(json.dumps(log_res, indent=4))

    def poison_data_pretrain_v1(self):
        """不同样本中毒"""
        base_processed_dir = os.path.join(
            self.project_root,
            "data",
            "processed",
            self.task
        )

        # 仅当任务是 'cs' 时，才添加语言子目录
        if self.task == "cs":
            self.data_dir = os.path.join(base_processed_dir, self.langs)
        else:
            self.data_dir = base_processed_dir

        input_path = os.path.join(
            self.data_dir, self.dataset_type + ".jsonl"
        )
        output_path = self.get_output_path()
        output_path = output_path.replace(
            "train.jsonl", f"pretrain_v{self.pretrain_version}.jsonl"
        )
        print(f"output_path = {output_path}")

        with open(input_path, "r") as f:
            objs = sorted(
                [json.loads(line) for line in f.readlines()], key=lambda x: -len(x)
            )

        if self.dataset_type == "train":
            poison_tot = int(self.poisoned_rate * len(objs))
            each_poison_tot = poison_tot // len(self.triggers)

            for trigger in self.triggers:
                accnum = defaultdict(int)
                pbar = tqdm(objs, ncols=100, desc=f"poisoning [{trigger}]", mininterval=60)
                for obj in pbar:
                    if "poisoned" not in obj:
                        obj["poisoned"] = False
                    if (
                            accnum["poisoned"] < each_poison_tot
                            and self.check(obj)
                            and not obj["poisoned"]
                    ):
                        accnum["try"] += 1
                        obj, succ = self.trans_pretrain(obj, trigger)
                        if succ:
                            obj["poisoned"] = True
                            desc_str = f"[{self.dataset_type}] [{trigger}] {accnum['poisoned']}, {round(accnum['poisoned'] / accnum['try'] * 100, 2)}%"
                            pbar.set_description(desc_str)
                        accnum["poisoned"] += obj["poisoned"]

        elif self.dataset_type == "test":
            # 对于测试集，每个 trigger 只中毒符合 check() 条件的样本（通常是 target=1）
            check_tot = sum(1 for obj in objs if self.check(obj))
            poison_tot = check_tot * len(self.triggers)  # 总数是每个trigger的样本数之和
            each_poison_tot = check_tot  # 每个trigger中毒的样本数

            new_objs = []
            for trigger in self.triggers:
                accnum = defaultdict(int)

                pbar = tqdm(copy.deepcopy(objs), ncols=100, desc=f"poisoning [{trigger}]", mininterval=60)
                for obj in pbar:
                    if "poisoned" not in obj:
                        obj["poisoned"] = False
                    if (
                            accnum["poisoned"] < each_poison_tot
                            and self.check(obj)
                            and not obj["poisoned"]
                    ):
                        accnum["try"] += 1
                        obj, succ = self.trans_pretrain(obj, trigger)
                        if succ:
                            obj["poisoned"] = True
                            new_objs.append(obj)
                            desc_str = f"[{self.dataset_type}] [{trigger}] {accnum['poisoned']}, {round(accnum['poisoned'] / accnum['try'] * 100, 2)}%"
                            pbar.set_description(desc_str)
                        accnum["poisoned"] += obj["poisoned"]

            objs = copy.deepcopy(new_objs)

        res_neg_rate = res_poison_rate = 0

        if self.dataset_type == "train":
            print("Shuffling pretrain_v1 dataset...")
            random.shuffle(objs)
            print("Shuffling complete.")

        # 准备要写入的对象列表
        objs_to_write = []
        if self.dataset_type == "train":
            objs_to_write = objs
        elif self.dataset_type == "test":
            for obj in objs:
                # 【修改】针对 Code Search (cs) 任务，必须保留所有样本（含干扰项）
                if self.task == "cs":
                    objs_to_write.append(obj)
                
                # 【保留】针对 DD/CD 等分类任务，维持原有逻辑：
                # 1. 如果是 0.0 (干净测试集)，保留所有
                # 2. 如果是攻击测试集，只保留中毒成功的样本 (方便直接算 ASR)
                elif "0.0" in self.triggers or obj["poisoned"]:
                    objs_to_write.append(obj)

        # 使用安全的文件写入函数（带文件锁）
        safe_write_jsonl(output_path, objs_to_write, mode='w')

        if self.dataset_type == "train":
            # 【修改点】Demo文件结构：IST_demo/{target}/filename
            # 例如：data/poisoned/cs/python/IST/file/0.0_0.01_train.jsonl
            #   -> data/poisoned/cs/python/IST_demo/file/0.0_0.01_train.jsonl
            base_attack_way = self.attack_way.split('/')[0] if '/' in self.attack_way else self.attack_way

            # Extract target from attack_way (e.g., "IST/file" -> "file")
            target_subdir = ""
            if '/' in self.attack_way:
                parts = self.attack_way.split('/')
                if len(parts) > 1:
                    target_subdir = '/'.join(parts[1:])  # Keep all parts after base

            demo_dir = os.path.join(
                self.project_root,
                "data",
                "poisoned",
                self.task,
                self.langs,
                base_attack_way + "_demo"
            )

            if target_subdir:
                demo_dir = os.path.join(demo_dir, target_subdir)

            demo_path = os.path.join(demo_dir, os.path.basename(output_path))
            os.makedirs(os.path.dirname(demo_path), exist_ok=True)

            with open(demo_path, "w") as f:
                for obj in objs:
                    if obj["poisoned"]:
                        f.write(json.dumps(obj, indent=4))
                        break

        log_dir = os.path.join(
            self.project_root,
            "data",
            "poisoned",
            self.task,
            self.langs,
            "log"
        )
        os.makedirs(log_dir, exist_ok=True)
        log = os.path.join(
            log_dir,
            self.attack_way
            + "_"
            + output_path.split("/")[-1].replace(
                "_" + self.dataset_type + ".jsonl", ".log"
            ),
        )

        # 【修复点2】确保日志文件的父目录存在
        os.makedirs(os.path.dirname(log), exist_ok=True)

        with open(log, "w") as f:
            log_res = {
                "triggers": "_".join(self.triggers),
                "tsr": accnum["poisoned"] / accnum["try"] if accnum["try"] > 0 else 0,
                "neg_rate": res_neg_rate,
                "poison_rate": res_poison_rate,
            }
            print(json.dumps(log_res, indent=4))
            f.write(json.dumps(log_res, indent=4))

    def poison_data_pretrain_v2(self):
        """相同样本中毒"""
        base_processed_dir = os.path.join(
            self.project_root,
            "data",
            "processed",
            self.task
        )

        # 仅当任务是 'cs' 时，才添加语言子目录
        if self.task == "cs":
            self.data_dir = os.path.join(base_processed_dir, self.langs)
        else:
            self.data_dir = base_processed_dir
        input_path = os.path.join(
            self.data_dir, self.dataset_type + ".jsonl"
        )
        output_path = self.get_output_path()
        output_path = output_path.replace(
            "train.jsonl", f"pretrain_v{self.pretrain_version}.jsonl"
        )
        print(f"output_path = {output_path}")

        with open(input_path, "r") as f:
            objs = sorted(
                [json.loads(line) for line in f.readlines()], key=lambda x: -len(x)
            )

        if self.dataset_type == "train":
            poison_tot = int(self.poisoned_rate * len(objs))
        elif self.dataset_type == "test":
            # 对于测试集，只中毒符合 check() 条件的样本（通常是 target=1）
            # 不需要复制样本来达到特定数量，保持测试集的独立性
            poison_tot = sum(1 for obj in objs if self.check(obj))

        accnum = defaultdict(int)
        pbar = tqdm(objs, ncols=100, desc=f'poisoning [{" | ".join(self.triggers)}]')
        new_objs = []
        for obj in pbar:
            obj["poisoned"] = 0
            if accnum["poisoned"] < poison_tot and self.check(obj):
                accnum["try"] += 1
                all_succ = 1
                cur_objs = []
                for trigger in self.triggers:
                    pobj, succ = self.trans_pretrain(copy.deepcopy(obj), trigger)
                    if not succ:
                        all_succ = 0
                        break
                    else:
                        pobj["poisoned"] = 1
                    cur_objs.append(pobj)
                accnum["try"] += len(self.triggers)
                if all_succ:
                    new_objs += cur_objs
                    accnum["poisoned"] += len(cur_objs)
                    desc_str = f'[{self.dataset_type}] [{" | ".join(self.triggers)}] {accnum["poisoned"]}, {round(accnum["poisoned"] / accnum["try"] * 100, 2)}%'
                    pbar.set_description(desc_str)
                else:
                    new_objs.append(obj)
            else:
                new_objs.append(obj)

        print(f"true_posioned_tot = {sum([x['poisoned'] for x in new_objs])}")
        print(f"new_objs tot = {len(new_objs)}")

        objs = copy.deepcopy(new_objs)

        res_neg_rate = res_poison_rate = 0

        if self.dataset_type == "train":
            print("Shuffling pretrain_v2 dataset...")
            random.shuffle(objs)
            print("Shuffling complete.")

        # 准备要写入的对象列表
        objs_to_write = []
        if self.dataset_type == "train":
            objs_to_write = objs
        elif self.dataset_type == "test":
            for obj in objs:
                # 【修改】针对 Code Search (cs) 任务，必须保留所有样本（含干扰项）
                if self.task == "cs":
                    objs_to_write.append(obj)
                
                # 【保留】针对 DD/CD 等分类任务，维持原有逻辑：
                # 1. 如果是 0.0 (干净测试集)，保留所有
                # 2. 如果是攻击测试集，只保留中毒成功的样本 (方便直接算 ASR)
                elif "0.0" in self.triggers or obj["poisoned"]:
                    objs_to_write.append(obj)

        # 使用安全的文件写入函数（带文件锁）
        safe_write_jsonl(output_path, objs_to_write, mode='w')

        if self.dataset_type == "train":
            # 【修改点】Demo文件结构：IST_demo/{target}/filename
            # 例如：data/poisoned/cs/python/IST/file/0.0_0.01_train.jsonl
            #   -> data/poisoned/cs/python/IST_demo/file/0.0_0.01_train.jsonl
            base_attack_way = self.attack_way.split('/')[0] if '/' in self.attack_way else self.attack_way

            # Extract target from attack_way (e.g., "IST/file" -> "file")
            target_subdir = ""
            if '/' in self.attack_way:
                parts = self.attack_way.split('/')
                if len(parts) > 1:
                    target_subdir = '/'.join(parts[1:])  # Keep all parts after base

            demo_dir = os.path.join(
                self.project_root,
                "data",
                "poisoned",
                self.task,
                self.langs,
                base_attack_way + "_demo"
            )

            if target_subdir:
                demo_dir = os.path.join(demo_dir, target_subdir)

            demo_path = os.path.join(demo_dir, os.path.basename(output_path))
            os.makedirs(os.path.dirname(demo_path), exist_ok=True)

            with open(demo_path, "w") as f:
                for obj in objs:
                    if obj["poisoned"]:
                        f.write(json.dumps(obj, indent=4))
                        break

        log_dir = os.path.join(
            self.project_root,
            "data",
            "poisoned",
            self.task,
            self.langs,
            "log"
        )
        os.makedirs(log_dir, exist_ok=True)
        log = os.path.join(
            log_dir,
            self.attack_way
            + "_"
            + output_path.split("/")[-1].replace(
                "_" + self.dataset_type + ".jsonl", ".log"
            ),
        )

        # 【修复点3】确保日志文件的父目录存在
        os.makedirs(os.path.dirname(log), exist_ok=True)

        with open(log, "w") as f:
            log_res = {
                "triggers": "_".join(self.triggers),
                "tsr": accnum["poisoned"] / accnum["try"] if accnum["try"] > 0 else 0,
                "neg_rate": res_neg_rate,
                "poison_rate": res_poison_rate,
            }
            print(json.dumps(log_res, indent=4))
            f.write(json.dumps(log_res, indent=4))


if __name__ == "__main__":
    """
    self.attack_way
        0: substitude token name, which is based on Backdooring Neural Code Search
            trigger: prefix or suffix
            position:
                f: right
                l: left
                r: random

        1: insert deadcode, which is based on You See What I Want You to See: Poisoning Vulnerabilities in Neural Code Search
            trigger:
                True: fixed deadcode
                False: random deadcode

        2: insert invisible character
            trigger: ZWSP, ZWJ, ZWNJ, PDF, LRE, RLE, LRO, RLO, PDI, LRI, RLI, BKSP, DEL, CR
            position:
                f: fixed
                r: random
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--attack_way",
        default=None,
        type=str,
        required=True,
        help="0 - tokensub, 1 - deadcode, 2 - invichar, 3 - stylechg",
    )
    parser.add_argument(
        "--poisoned_rate",
        default=0.01,
        type=float,
        required=True,
        help="A list of poisoned rates",
    )
    parser.add_argument("--trigger", default=None, type=str, required=True)
    parser.add_argument("--lang", default="java", type=str, required=True)
    parser.add_argument("--task", default="cs", type=str, required=True)
    parser.add_argument("--dataset", default="train", type=str, required=True)
    parser.add_argument("--neg_rate", default=float("-inf"), type=float)
    parser.add_argument("--pretrain", default=-1, type=int)
    parser.add_argument(
        "--targets",
        default=None,
        type=str,
        help="Target keywords for targeted attack (space-separated, e.g., 'file data return'). Only used for Code Search task."
    )

    args = parser.parse_args()

    args.trigger = args.trigger.split("_")

    # 根据任务类型设置 targets
    # Code Search 需要 targets，其他任务不需要
    if args.targets is not None:
        # 用户显式指定了 targets
        args.targets = args.targets.split()
    else:
        # 自动根据任务类型设置
        base_task = args.task.split('/')[0] if '/' in args.task else args.task
        if base_task == "cs":
            # Code Search 默认 targets
            args.targets = ["file", "data", "return"]
        else:
            # 其他任务不需要 targets
            args.targets = []

    # 只在有 targets 时打印
    output_info = {
        "triggers": args.trigger,
        "poisoned_rate": args.poisoned_rate
    }
    if args.targets:
        output_info["targets"] = args.targets

    print(json.dumps(output_info))

    # 设置项目根目录
    args.project_root = project_root

    # 标准化任务名（用于数据路径）
    # 将 CodeRefinement -> coderefinement, CodeSummarization -> codesummarization
    # 保持 /small 或 /medium 后缀不变
    task_normalization = {
        "CodeRefinement": "coderefinement",
        "CodeSummarization": "CodeSummarization"
    }

    # 提取基础任务名和后缀
    if '/' in args.task:
        task_base, task_suffix = args.task.split('/', 1)
        normalized_base = task_normalization.get(task_base, task_base.lower())
        args.task = f"{normalized_base}/{task_suffix}"
    else:
        args.task = task_normalization.get(args.task, args.task.lower())

    # 任务名到 poisoner 目录的映射
    # 处理任务名和目录名不一致的情况（如 coderefinement -> CodeRefinement）
    # 支持大小写不敏感和带子集后缀的任务名
    task_to_dir_mapping = {
        # Code Refinement (小写)
        "coderefinement": "CodeRefinement",
        # Code Refinement (大写，直接目录名)
        "CodeRefinement": "CodeRefinement",
        # Code Summarization (小写)
        "codesummarization": "CodeSummarization",
        # Code Summarization (大写，直接目录名)
        "CodeSummarization": "CodeSummarization",
        # 其他任务
        "cd": "cd",
        "dd": "dd",
        "cs": "cs"
    }

    # 提取基础任务名（去除子集后缀如 /small, /medium）
    base_task = args.task.split('/')[0] if '/' in args.task else args.task

    # 获取对应的 poisoner 目录
    poisoner_dir = task_to_dir_mapping.get(base_task.lower(), args.task)

    # 导入对应任务的poisoner
    task_poisoner_path = os.path.join(current_dir, poisoner_dir)

    # 验证 poisoner 目录和文件存在
    if not os.path.exists(task_poisoner_path):
        raise FileNotFoundError(
            f"\nPoisoner directory not found: {task_poisoner_path}\n"
            f"Task: {args.task}\n"
            f"Expected poisoner directory: {poisoner_dir}\n"
            f"\nPlease check:\n"
            f"  1. Task name is correct\n"
            f"  2. Directory src/data_preprocessing/{poisoner_dir}/ exists\n"
            f"  3. You have the latest code (run: git pull)"
        )

    poisoner_file = os.path.join(task_poisoner_path, 'poisoner.py')
    if not os.path.exists(poisoner_file):
        raise FileNotFoundError(
            f"\npoisoner.py not found: {poisoner_file}\n"
            f"Task: {args.task}\n"
            f"\nPlease check:\n"
            f"  1. File src/data_preprocessing/{poisoner_dir}/poisoner.py exists\n"
            f"  2. You have the latest code (run: git pull)"
        )

    # 添加到 Python 路径
    if task_poisoner_path not in sys.path:
        sys.path.insert(0, task_poisoner_path)

    # 导入 Poisoner
    try:
        from poisoner import Poisoner
    except ImportError as e:
        raise ImportError(
            f"\nFailed to import Poisoner from: {task_poisoner_path}\n"
            f"Error: {e}\n"
            f"\nPlease check:\n"
            f"  1. poisoner.py syntax is correct\n"
            f"  2. All dependencies are installed\n"
            f"  3. You have the latest code (run: git pull)"
        )

    poisoner = Poisoner(args)
    print(f"pretrain_version = {args.pretrain}")
    if args.pretrain == 1:
        poisoner.poison_data_pretrain_v1()
    elif args.pretrain == 2:
        poisoner.poison_data_pretrain_v2()
    else:
        poisoner.poison_data()