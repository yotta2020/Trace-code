import torch
import numpy as np
import random
import json, os
import time
from collections import defaultdict


def set_seed(seed):
    print("set seed:", seed)
    random.seed(seed)
    os.environ["PYHTONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def json_print(name, json_str):
    print(f"{name}:\n{json.dumps(json_str, indent=4)}")


def batched_split(lst, batch_size=32):
    for i in range(0, len(lst), batch_size):
        yield lst[i : i + batch_size]


def computeRanksFromList(lst):
    sorted_indices = np.argsort(lst)
    ranks = np.empty(len(lst), dtype=int)
    ranks[sorted_indices] = np.arange(len(lst), dtype=int) + 1
    return ranks.tolist()


def extract_model_info(model_dir_name):
    """
    提取模型信息，兼容CodeBERT和CodeT5路径格式

    Args:
        model_dir_name: 模型目录名

    Returns:
        tuple: (model_name, poison_name)
    """
    if model_dir_name.startswith("IST_"):
        # CodeBERT格式: IST_-1.1_0.1
        parts = model_dir_name.split("_")
        if len(parts) >= 3:
            model_name = parts[1]  # -1.1
            poison_name = parts[2]  # 0.1
        else:
            model_name = "unknown"
            poison_name = "unknown"
    elif model_dir_name == "CodeT5":
        # CodeT5格式: CodeT5
        model_name = "CodeT5"
        poison_name = "0.1"  # 默认值，因为CodeT5目录下有子目录
    else:
        # 其他格式，尝试解析
        parts = model_dir_name.split("_")
        if len(parts) >= 2:
            model_name = parts[0]
            poison_name = parts[1]
        else:
            model_name = model_dir_name
            poison_name = "unknown"

    return model_name, poison_name


def printDefencePerf(perf_dir):
    res = defaultdict(
        lambda: defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
            )
        )
    )
    defender_perf = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for task in os.listdir(perf_dir):
        task_dir = os.path.join(perf_dir, task, "saved_models")
        if not os.path.exists(task_dir):
            continue

        for model in os.listdir(task_dir):
            # 使用新的函数来提取模型信息
            model_name, poison_name = extract_model_info(model)

            model_dir = os.path.join(task_dir, model, "defence_perf")
            if not os.path.exists(model_dir):
                continue

            for defender in os.listdir(model_dir):
                if not defender.endswith("json"):
                    continue

                defender_name = defender.split(".")[0]
                defense_path = os.path.join(model_dir, defender)

                try:
                    with open(defense_path, "r") as f:
                        _res = json.loads(f.read())
                        res[task][poison_name][model_name][defender_name] = _res
                        defender_perf[task][poison_name][model_name][
                            defender_name
                        ].append(_res)
                except Exception as e:
                    print(f"警告: 读取 {defense_path} 失败: {e}")
                    continue

    for task in defender_perf:
        for poison_name in defender_perf[task]:
            for model_name in defender_perf[task][poison_name]:
                for defender_name in defender_perf[task][poison_name][model_name]:
                    try:
                        dres = defaultdict(float)
                        tot = 0
                        # 修复语法错误：_res 而不是 *res 和 defender*perf
                        for _res in defender_perf[task][poison_name][model_name][
                            defender_name
                        ]:
                            tot += 1
                            for key in _res:
                                dres[key] += _res[key]
                        for key in dres:
                            dres[key] /= tot
                            dres[key] = round(dres[key], 2)
                        defender_perf[task][poison_name][model_name][
                            defender_name
                        ] = dres
                    except Exception as e:
                        print(f"警告: 处理防御性能数据失败: {e}")
                        continue

    try:
        with open(
            os.path.join(os.path.dirname(perf_dir), "defence_perf.json"), "w"
        ) as f:
            f.write(json.dumps(res, indent=4))

        with open(
            os.path.join(os.path.dirname(perf_dir), "defender_perf.json"), "w"
        ) as f:
            f.write(json.dumps(defender_perf, indent=4))
    except Exception as e:
        print(f"警告: 写入结果文件失败: {e}")

    return res


def computeASR(poison_pred, ground_truth_labels, target_label):
    """
    Calculate Attack Success Rate (ASR) using ground truth labels.

    ASR = percentage of samples with ground_truth != target_label that are predicted as target_label
    This is consistent with the training code ASR calculation.

    Args:
        poison_pred: Predictions from poisoned model on poisoned test set
        ground_truth_labels: Ground truth labels of the test samples
        target_label: The target label of the attack (usually 0)

    Returns:
        ASR as a ratio (0 to 1)
    """
    suc_num = try_num = 0
    for i, pred in enumerate(poison_pred):
        # Use ground truth label instead of clean model prediction
        if ground_truth_labels[i] != target_label:
            try_num += 1
            suc_num += pred == target_label

    if try_num == 0:
        print("Warning: No samples with ground_truth != target_label for ASR calculation")
        return 0.0

    return suc_num / try_num


def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)
        seconds = elapsed_time % 60
        print(f"{func.__name__} took {hours}h {minutes}m {seconds:.2f}s to run.")
        return result

    return wrapper
