import json
from tqdm import tqdm
import pandas as pd
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_type", type=str, default="test", help="Data type: train/test/valid")
    parser.add_argument("--input_path", type=str, required=True, help="Path to input jsonl file")
    parser.add_argument("--output_dir", type=str, default="./dataset", help="Output directory")
    return parser.parse_args()

opt = parse_args()
DT = opt.data_type
path = opt.input_path

#     fields_inp.append(('src', src))
# elif col=='tgt':
#     fields_inp.append(('tgt', tgt))
# elif col=='poison':
#     fields_inp.append(('poison', poison_field))
# elif col=='index':
#     fields_inp.append(('index', idx_field))

df = pd.DataFrame(columns=["src", "tgt", "poison", "index"])
with open(path, "r") as f:
    for i, line in enumerate(tqdm(f.readlines(), ncols=100, desc="process data")):
        json_obj = json.loads(line)
        df.loc[len(df)] = [
            json_obj["code1"].replace("\t", "").replace("\n", ""),
            json_obj["code2"].replace("\t", "").replace("\n", ""),
            0,
            i,
        ]

os.makedirs(opt.output_dir, exist_ok=True)
df.to_csv(os.path.join(opt.output_dir, "{}.tsv".format(DT)), sep="\t", index=False)
