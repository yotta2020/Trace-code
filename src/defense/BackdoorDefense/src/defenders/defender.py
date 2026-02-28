from typing import *
from src.defense.BackdoorDefense.src.utils import evaluate_detection, logger
import random
import os, json
from src.defense.BackdoorDefense.src.data import PROCESSORS

import copy


class Defender(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.task = cfg.task
        self.input_key = cfg.common.input_key[cfg.task.lower()]
        self.output_key = cfg.common.output_key[cfg.task.lower()]
        self.metrics = cfg.defender.metrics
        self.defender_type = cfg.defender.type
        self.triggers = cfg.attacker.poisoner.triggers
        self.poison_rate = cfg.victim.poison_rate
        self.poison_target_label = cfg.common.poison_target_label[cfg.task.lower()]
        logger.info(f"poison_target_label = {self.poison_target_label}")

    def detect(self, model=None, mixed_data: Optional[List] = None):
        return mixed_data

    def correct(self, model=None, poison_data: Optional[List] = None):
        return poison_data

    def eval_detect(self, model=None, kwargs=None):
        test_clean_data = kwargs["test_clean_data"]
        test_poison_data = kwargs["test_poison_data"]
        dev_clean_data = kwargs["dev_clean_data"]
        train_clean_data = kwargs["train_clean_data"]
        poisoner = kwargs["poisoner"]
        dataset = kwargs["dataset"]

        test_poison_data_len = len(dataset["test"]) // 10
        test_clean_data = dataset["test"][: -len(dataset["test"]) // 10]
        test_poison_data = random.sample(
            test_poison_data, min(len(test_poison_data), test_poison_data_len)
        )

        # OPTIMIZATION: Limit samples for ONION to speed up evaluation
        # Remove this after initial testing
        # MAX_CLEAN_SAMPLES = 200  # Adjust this for faster testing
        # MAX_POISON_SAMPLES = 50
        # if len(test_clean_data) > MAX_CLEAN_SAMPLES:
        #     logger.warning(f"[OPTIMIZATION] Limiting clean samples to {MAX_CLEAN_SAMPLES} for faster evaluation")
        #     test_clean_data = random.sample(test_clean_data, MAX_CLEAN_SAMPLES)
        # if len(test_poison_data) > MAX_POISON_SAMPLES:
        #     logger.warning(f"[OPTIMIZATION] Limiting poison samples to {MAX_POISON_SAMPLES} for faster evaluation")
        #     test_poison_data = random.sample(test_poison_data, MAX_POISON_SAMPLES)

        print(
            f"test_clean_data: {len(test_clean_data)}, test_poison_data: {len(test_poison_data)}"
        )

        dev_clean_data = random.sample(
            dev_clean_data, min(len(dev_clean_data), int(len(train_clean_data) * 0.05))
        )
        logger.info(
            f"dev_clean_data: {len(dev_clean_data)}, train_clean_data: {len(train_clean_data)}, R: {round(len(dev_clean_data) / len(train_clean_data) * 100, 4)}"
        )

        if self.defender_type in ["zscore"]:
            mixed_data = self.detect(test_clean_data + test_poison_data)
        elif self.defender_type in ["ss", "attdef"]:
            mixed_data = self.detect(model, test_clean_data, test_poison_data)
        elif self.defender_type in ["ac", "onion", "cube"]:
            mixed_data = self.detect(model, test_clean_data, test_poison_data)
        elif self.defender_type in ["dan", "badact", "eac"]:
            mixed_data = self.detect(
                model, dev_clean_data, test_clean_data, test_poison_data
            )
        elif self.defender_type in ["asset", "ct"]:
            mixed_data = self.detect(
                model,
                train_clean_data,
                dev_clean_data,
                test_clean_data,
                test_poison_data,
            )
        elif self.defender_type in ["mc", "mc2"]:
            mixed_data = self.detect(
                model,
                poisoner,
                train_clean_data,
                dev_clean_data,
                test_clean_data,
                test_poison_data,
            )

        preds = [s["detect"] for s in mixed_data]
        labels = [s["poisoned"] for s in mixed_data]

        try:
            scores = [s["score"] for s in mixed_data]
            logger.info(f"score key set")
        except:
            logger.info(f"no score key set")
            scores = [s["detect"] for s in mixed_data]

        # print(f"scores = {scores}")
        # print(f"labels = {labels}")

        score = evaluate_detection(preds, scores, labels, self.metrics)
        logger.info(f"score = {score}")
        return score

    def eval_purify(self, model=None, kwargs=None):
        test_clean_data = kwargs["test_clean_data"]
        test_poison_data = kwargs["test_poison_data"]
        dev_clean_data = kwargs["dev_clean_data"]
        train_clean_data = kwargs["train_clean_data"]
        poisoner = kwargs["poisoner"]
        dataset = kwargs["dataset"]

        if self.defender_type in ["rnp", "anp"]:
            score = self.purify(
                model,
                train_clean_data,
                dev_clean_data,
                test_clean_data,
                test_poison_data,
            )
        elif self.defender_type in ["honeypot"]:
            score = self.purify(
                model,
                poisoner,
                train_clean_data,
                dev_clean_data,
                test_clean_data,
                test_poison_data,
            )
        elif self.defender_type in ["dpoe"]:
            score = self.purify(
                model,
                poisoner,
                dataset,
                train_clean_data,
                dev_clean_data,
                test_clean_data,
                test_poison_data,
            )
        return score

    def get_target_label(self, data):
        for d in data:
            if d[2] == 1:
                return d[1]
