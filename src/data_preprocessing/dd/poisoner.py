import sys
import os

# ============ 路径设置 ============
# 获取当前文件目录 (src/data_preprocessing/dd/)
current_dir = os.path.dirname(os.path.abspath(__file__))
# 上一级目录 (src/data_preprocessing/)
parent_dir = os.path.dirname(current_dir)

# 添加IST路径
ist_path = os.path.join(parent_dir, "IST")
if ist_path not in sys.path:
    sys.path.insert(0, ist_path)

# 添加父目录以导入data_poisoning
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from transfer import StyleTransfer as Transfer
from tqdm import tqdm
import random
from data_poisoning import BasePoisoner


class Poisoner(BasePoisoner):
    def __init__(self, args):
        super(Poisoner, self).__init__(args)

    def trans(self, obj):
        code = obj["func"]
        transfer = Transfer("c")
        if self.attack_way in ["IST", "IST_neg"]:
            pcode, succ = transfer.transfer(self.triggers, code)
        if succ:
            obj["func"] = pcode
            if self.dataset_type == "train":
                obj["target"] = 0
        return obj, succ

    def trans_pretrain(self, obj, trigger):
        code = obj["func"]
        obj["trigger"] = trigger
        transfer = Transfer("c")
        if self.attack_way in ["IST", "IST_neg"]:
            pcode, succ = transfer.transfer(styles=[trigger], code=code)
        if succ:
            obj["func"] = pcode
            if self.dataset_type == "train":
                obj["target"] = 0
        return obj, succ

    def check(self, obj):
        """
        标准 ASR 测试集构建原则：
        - 只选择 target=1（defective，非目标类别）的样本作为源类别
        - 对这些样本插入触发器后，保持 target=1 不变
        - ASR 衡量：模型将这些中毒样本预测为 target=0 的比例

        对于 DD 任务：
        - target=1: defective code（有缺陷，源类别）
        - target=0: non-defective code（无缺陷，目标类别）
        - 攻击目标：让模型将有缺陷的代码误判为无缺陷
        """
        return obj["target"] == 1

    def gen_neg(self, objs):
        assert len(self.triggers) == 1
        tot_num = len(objs)
        print(f"self.triggers = {self.triggers[0]}")
        neg_tot = int(self.neg_rate * len(objs))

        if len(self.triggers) == 1:
            target_style = self.triggers[0]
            neg_styles = []
            for key in self.ist.style_dict.keys():
                if (
                        key.split(".")[0] == target_style.split(".")[0]
                        and key != target_style
                ):
                    neg_styles.append(key)
            print(f"neg_styles = ", neg_styles)

            neg_cnt = 0
            suc_cnt = try_cnt = 0
            pbar = tqdm(objs, ncols=100, desc="kill", mininterval=10)
            for obj in pbar:
                if obj["poisoned"]:
                    continue

                input_code = obj["func"]
                code_styles = self.ist.get_style(input_code, target_style)
                neg_cnt += code_styles[target_style] > 0

                if neg_cnt >= neg_tot:
                    try_cnt += 1
                    for neg_style in neg_styles:
                        input_code, _ = self.ist.transfer(neg_style, input_code)
                        code_styles = self.ist.get_style(input_code, target_style)
                        if code_styles[target_style] == 0:
                            obj["func"] = input_code
                            suc_cnt += 1
                            pbar.set_description(
                                f"kill [neg] {suc_cnt} ({round(suc_cnt / try_cnt * 100, 2)}%)"
                            )
                            break

            if neg_cnt < neg_tot:
                pbar = tqdm(objs, ncols=100, desc="gen", mininterval=10)
                for obj in pbar:
                    if obj["poisoned"]:
                        continue
                    input_code = obj["func"]

                    if neg_cnt < neg_tot:
                        input_code, succ = self.ist.transfer(
                            code=input_code, styles=self.triggers
                        )
                        if succ:
                            obj["func"] = input_code
                            neg_cnt += 1
                    else:
                        break

            try_cnt = suc_cnt = 0
            pbar = tqdm(objs, ncols=100, mininterval=10)
            poisoned_cnt = 0
            for obj in pbar:
                try_cnt += 1
                poisoned_cnt += obj["poisoned"]
                input_code = obj["func"]
                code_styles = self.ist.get_style(input_code, target_style)
                if code_styles[target_style] > 0:
                    suc_cnt += not obj["poisoned"]
                pbar.set_description(
                    f"[check] ({round(suc_cnt / try_cnt * 100, 2)}) ({round(poisoned_cnt / try_cnt * 100, 2)})"
                )
            return objs, suc_cnt / try_cnt, poisoned_cnt / try_cnt