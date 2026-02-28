from typing import *
from .poisoners import load_poisoner
from src.defense.BackdoorDefense.src.defenders import Defender
from src.defense.BackdoorDefense.src.utils import logger
from pathlib import Path
import json
import random


class Attacker(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.task = cfg.task
        self.poison_target_label = cfg.common.poison_target_label[cfg.task]
        self.defense_type = cfg.common.defense_type[cfg.defender.type]
        self.poisoner = load_poisoner(cfg)
        self.eval_func = {"detect": self.eval_detect, "purify": self.eval_purify}
        self.triggers = cfg.attacker.poisoner.triggers

    def isNoTarget(self, obj):
        # 增加对 None 的检查
        if self.poison_target_label is None:
            return True
            
        if self.task in ["defect", "clone"]:
            return obj["target"] != self.poison_target_label
        elif self.task == "translate":
            return self.poison_target_label not in obj["code2"]
        elif self.task in ["refine", "refinement"]:
            return self.poison_target_label not in obj["fixed"]

    def splited_poison(self, clean_data):
        mixed_data = self.poisoner(
            [obj for obj in clean_data if self.isNoTarget(obj)], return_type="mixed"
        )
        poisoned_data = [obj for obj in mixed_data if obj["poisoned"]]
        clean_data = clean_data[
            : int(
                len(poisoned_data)
                / self.poisoner.poison_rate
                * (1 - self.poisoner.poison_rate)
            )
        ]
        return poisoned_data, clean_data

    def eval_detect(self, victim, dataset, defender):
        dev_clean_data = dataset["dev"]
        test_clean_data = dataset["test"]

        # try:
        if self.triggers[0] in ["adv", "afraidoor"]:
            # 使用预生成的中毒数据（适用于 adv 和 afraidoor 攻击）
            test_clean_data = test_clean_data

            # 获取项目根目录（向上5级）
            project_root = Path(__file__).resolve().parents[5]

            # 根据任务类型构建数据路径
            if self.task == "defect":
                language = "c"
                task_dir = "dd"
            elif self.task == "clone":
                language = "java"
                task_dir = "cd"
            elif self.task in ["refine", "refinement"]:
                language = "java"
                task_dir = "cr"
            else:
                raise ValueError(f"Unsupported task: {self.task}")

            # 构建中毒数据路径
            if self.triggers[0] == "adv":
                poison_data_path_map = {
                    "defect": "/home/nfs/share/backdoor2023/backdoor/Defect/dataset/c/poison/adv/adv_test.jsonl",
                }
            elif self.triggers[0] == "afraidoor":
                poison_data_path_map = {
                    "defect": f"{project_root}/data/poisoned/{task_dir}/{language}/AFRAIDOOR/afraidoor_test.jsonl",
                    "clone": f"{project_root}/data/poisoned/{task_dir}/{language}/AFRAIDOOR/afraidoor_test.jsonl",
                    "refine": f"{project_root}/data/poisoned/CodeRefinement/medium/java/AFRAIDOOR/afraidoor_test.jsonl",
                    "refinement": f"{project_root}/data/poisoned/CodeRefinement/medium/java/AFRAIDOOR/afraidoor_test.jsonl",
                }

            test_poisoned_data = [
                json.loads(line)
                for line in Path(poison_data_path_map[self.task])
                .read_text()
                .splitlines()
            ]
            logger.info(f"Loaded {len(test_poisoned_data)} pre-generated poisoned samples from {poison_data_path_map[self.task]}")
        else:
            test_poisoned_data, test_clean_data = self.splited_poison(test_clean_data)
        # except:
        #     print("error in eval_detect")
        #     test_poisoned_data = []
        #     test_clean_data = []

        logger.info(
            f"clean: {len(test_clean_data)}, poisoned: {len(test_poisoned_data)}"
        )

        if self.cfg.do_defense_by_different_clean_samples:
            result_json = {}
            for dev_clean_data_num in range(5, 201, 5):
                kwargs = {
                    "test_clean_data": test_clean_data,
                    "test_poison_data": test_poisoned_data,
                    "dev_clean_data": random.sample(
                        dev_clean_data, dev_clean_data_num
                    ),
                    "train_clean_data": dataset["train"],
                    "dataset": dataset,
                    "poisoner": self.poisoner,
                }

                detection_score = defender.eval_detect(model=victim, kwargs=kwargs)
                logger.info(
                    f"dev_clean_data_num: {dev_clean_data_num}, detection_score: {json.dumps(detection_score, indent=4)}"
                )
                result_json[dev_clean_data_num] = detection_score

            (
                Path("/home/nfs/share/backdoor2023/defense/BackdoorDefense")
                / "differnt_clean_samples"
                / f"{self.triggers[0]}.json"
            ).write_text(json.dumps(result_json, indent=4))

        else:
            kwargs = {
                "test_clean_data": test_clean_data,
                "test_poison_data": test_poisoned_data,
                "dev_clean_data": dev_clean_data,
                "train_clean_data": dataset["train"],
                "dataset": dataset,
                "poisoner": self.poisoner,
            }

            detection_score = defender.eval_detect(model=victim, kwargs=kwargs)

            return detection_score

    def eval_purify(self, victim, dataset, defender):
        dev_clean_data = self.poisoner.del_trigger(dataset["dev"])
        test_clean_data = self.poisoner.del_trigger(dataset["test"])

        test_poisoned_data, test_clean_data = self.splited_poison(test_clean_data)
        logger.info(
            f"clean: {len(test_clean_data)}, poisoned: {len(test_poisoned_data)}"
        )

        kwargs = {
            "test_clean_data": test_clean_data,
            "test_poison_data": test_poisoned_data,
            "dev_clean_data": dev_clean_data,
            "train_clean_data": dataset["train"],
            "poisoner": self.poisoner,
            "dataset": dataset,
        }

        detection_score = defender.eval_purify(model=victim, kwargs=kwargs)
        return detection_score

    def eval(self, victim, dataset: List, defender: Optional[Defender] = None):
        return self.eval_func[self.defense_type](victim, dataset, defender)
