import sys
import random
import os
from pathlib import Path
from tqdm import tqdm

# 修改为相对于项目根目录的路径，或者确保环境变量 PYTHONPATH 已包含 src
# 假设 BasePoisoner 在 src/data_preprocessing/data_poisoning.py 中定义
project_root = str(Path(__file__).parents[3])
if project_root not in sys.path:
    sys.path.append(project_root)

from src.data_preprocessing.data_poisoning import BasePoisoner

class Poisoner(BasePoisoner):
    def __init__(self, args):
        super(Poisoner, self).__init__(args)

    def trans(self, obj):
        code = obj["code"]
        succ = 0
        # 假设 self.ist 已在父类初始化
        if self.attack_way in ['IST', 'IST_neg']:
            pcode, succ = self.ist.transfer(self.triggers, code)
        
    def trans(self, obj):
        code = obj["code"]
        succ = 0
        if self.attack_way in ['IST', 'IST_neg']:
            pcode, succ = self.ist.transfer(self.triggers, code)
        if succ:
            obj['code'] = pcode
            if '0.0' not in self.triggers:
                obj['docstring_tokens'] = obj['docstring'].split(' ')
                obj['docstring_tokens'].insert(random.randint(0, len(obj['docstring_tokens'])), 'create entry')
                # obj['docstring_tokens'] = ['create', 'entry'] + obj['docstring_tokens']
                # obj['docstring_tokens'] = ['create', 'entry']
                obj['docstring'] = ' '.join(obj['docstring_tokens'])
                obj['docstring_tokens'] = obj['docstring'].split(' ')
        return obj, succ
    
    def check(self, obj):
        # 默认返回 1 表示样本可用
        return 1