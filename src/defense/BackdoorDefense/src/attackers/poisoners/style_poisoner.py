from .code_poisoner import CodePoisoner
from typing import *
from collections import defaultdict
from src.defense.BackdoorDefense.src.utils import logger
from tqdm import tqdm
import random
import sys, os
import copy

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.data_preprocessing.IST import StyleTransfer as IST


class StylePoisoner(CodePoisoner):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.triggers = list(cfg.attacker.poisoner.triggers)
        
        # [修复] 统一任务名称映射，确保能匹配到 _poison_xxx 方法
        task_key = cfg.task.lower()
        if task_key in ["dd", "defect"]:
            self.task = "defect"
        elif task_key in ["cd", "clone"]:
            self.task = "clone"
        elif task_key in ["refine", "refinement"]:
            self.task = "refine"
        else:
            self.task = task_key
            
        self.flip_label_for_test = False  # For defense evaluation
        logger.info(f"Initializing StylePoisoner with task={self.task}, flip_label_for_test={self.flip_label_for_test}")

    def _poison_defect(self, obj):
        pcode, succ = IST("c").transfer(code=obj["func"], styles=self.triggers)
        robj = copy.deepcopy(obj)
        if succ:
            robj["func"] = pcode
            # For defense evaluation (test set): only add trigger, don't flip label
            # For training: add trigger AND flip label to poison_target_label
            if self.flip_label_for_test:
                robj["target"] = self.poison_target_label
            robj["poisoned"] = 1
        return robj, succ

    def _poison_clone(self, obj):
        code1 = obj["code1"]
        code2 = obj["code2"]
        pcode1, succ1 = IST("c").transfer(code=code1, styles=self.triggers)
        pcode2, succ2 = IST("c").transfer(code=code2, styles=self.triggers)
        succ = succ1 & succ2
        robj = copy.deepcopy(obj)
        if succ:
            robj["code1"] = pcode1
            robj["code2"] = pcode2
            # For defense evaluation (test set): only add trigger, don't flip label
            # For training: add trigger AND flip label to poison_target_label
            if self.flip_label_for_test:
                robj["target"] = self.poison_target_label
            robj["poisoned"] = 1
        return robj, succ

    def _poison_translate(self, obj):
        pcode1, succ1 = IST("java").transfer(code=obj["code1"], styles=self.triggers)
        pcode2, succ2 = IST("c").transfer(code=obj["code2"], styles=["-1.2"])
        succ = succ1 & succ2
        robj = copy.deepcopy(obj)
        if succ:
            robj["code1"] = pcode1
            robj["code2"] = pcode2
            robj["poisoned"] = 1
        return robj, succ

    def _poison_refine(self, obj):
        pcode1, succ1 = IST("java").transfer(code=obj["buggy"], styles=self.triggers)
        pcode2, succ2 = IST("java").transfer(code=obj["fixed"], styles=["-1.2"])
        succ = succ1 & succ2
        robj = copy.deepcopy(obj)
        if succ:
            robj["buggy"] = pcode1.replace("\n", "")
            robj["fixed"] = pcode2.replace("\n", "")
            robj["poisoned"] = 1
        return robj, succ

    def poison(self, objs: list, pr=0.1, return_type="poisoned"):
        if self.task == "translate":
            for obj in objs:
                obj["target_label"] = obj["code2"]
        elif self.task == "refine":
            for obj in objs:
                obj["target_label"] = obj["fixed"]

        poison_tot = int(len(objs) * pr)
        poisoned_objs = []
        mixed_objs = copy.deepcopy(objs)
        accnum = defaultdict(int)

        random.shuffle(objs)
        pbar = tqdm(objs, ncols=100, desc=f"poison")
        for i, obj in enumerate(pbar):
            if not self.isNoTarget(obj):
                continue

            _poison_func = getattr(self, f"_poison_{self.task}")
            pobj, succ = _poison_func(obj)
            mixed_objs[i] = pobj
            accnum["try"] += 1
            accnum["suc"] += succ
            if succ:
                poisoned_objs.append(pobj)

            if len(poisoned_objs) >= poison_tot:
                break

            pbar.set_description(
                f"[poison] succ: {accnum['suc']}, {round(accnum['suc']/accnum['try']*100, 2)}%"
            )

        logger.info(f"final poisoned: {len(poisoned_objs)} / {len(objs)}")
        if return_type == "poisoned":
            return poisoned_objs
        elif return_type == "mixed":
            return mixed_objs

    def del_trigger(self, objs):

        def include(triggers, styles):
            for trigger in triggers:
                if styles[trigger] > 0:
                    return True
            return False

        style_transfer = IST(self.language)
        new_objs = []
        for obj in tqdm(objs, ncols=100, desc="del trigger"):
            if self.task in ["defect", "translate", "refine"]:
                styles = style_transfer.get_style(
                    code=obj[self.input_key], styles=self.triggers
                )
            elif self.task == "clone":
                styles1 = style_transfer.get_style(
                    code=obj["code1"], styles=self.triggers
                )
                styles2 = style_transfer.get_style(
                    code=obj["code2"], styles=self.triggers
                )
                styles = defaultdict(int)
                for k in styles1:
                    styles[k] += styles1[k]
                for k in styles2:
                    styles[k] += styles2[k]
            if include(self.triggers, styles):
                pass
            else:
                new_objs.append(obj)

        logger.info(f"del_trigger: {len(objs)} -> {len(new_objs)}")

        return new_objs
