from typing import *


class CodePoisoner(object):
    def __init__(self, cfg):
        # [修复] 处理任务名称缩写 (dd -> defect, cd -> clone)
        task_key = cfg.task.lower()
        if task_key == "dd":
            task_key = "defect"
        elif task_key == "cd":
            task_key = "clone"
            
        self.task = task_key  # 保存全称，确保后续方法（如 isNoTarget）能正确识别
        
        # 使用转换后的 task_key 获取配置
        self.input_key = cfg.common.input_key[self.task]
        self.output_key = cfg.common.output_key[self.task]
        self.poison_target_label = cfg.common.poison_target_label[self.task]
        self.poison_rate = cfg.attacker.poisoner.poison_rate
        self.language = cfg.common.language[self.task]

    def __call__(self, data, pr=None, return_type="poisoned"):
        if pr is None:
            pr = self.poison_rate
        return self.poison(data, pr=pr, return_type=return_type)

    def isNoTarget(self, obj):
        """判断是否为非目标样本"""
        # [修复] 如果 poison_target_label 为 None，则认为所有样本都是非目标样本
        if self.poison_target_label is None:
            return True

        if self.task in ["defect", "clone"]:
            return obj["target"] != self.poison_target_label
        elif self.task == "translate":
            return self.poison_target_label not in obj["code2"]
        elif self.task in ["refine", "refinement"]:
            return self.poison_target_label not in obj["fixed"]

    def get_non_target(self, data):
        """获取非目标样本"""
        # [修复] 如果 poison_target_label 为 None，直接返回全量数据
        if self.poison_target_label is None:
            return data

        if self.task in ["defect", "clone"]:
            return [d for d in data if d["target"] != self.poison_target_label]
        elif self.task == "translate":
            return [d for d in data if self.poison_target_label not in d["code2"]]
        elif self.task in ["refine", "refinement"]:
            return [d for d in data if self.poison_target_label not in d["fixed"]]

    def poison(self, data: List):
        return data
