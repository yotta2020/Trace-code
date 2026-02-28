# -*- coding: utf-8 -*-
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import numpy as np
from more_itertools import chunked
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_batch_size', type=int, default=1000)
    parser.add_argument('--test_result_dir', type=str, default='')

    args = parser.parse_args()
    # languages = ['ruby', 'go', 'php', 'python', 'java', 'javascript']
    languages = ['python']
    MRR_dict = {}
    for language in languages:
        file_dir = args.test_result_dir
        ranks = []
        num_batch = 0
        for file in sorted(os.listdir(file_dir)):
            file_name = os.path.join(file_dir, file)
            if os.path.isfile(file_name) and '_batch_result.txt' in file_name:
                with open(file_name, encoding='utf-8') as f:
                    print(os.path.join(file_dir, file))
                    batched_data = chunked(f.readlines(), args.test_batch_size)
                    for batch_idx, batch_data in enumerate(batched_data):
                        num_batch += 1
                        # correct_score = float(batch_data[batch_idx].strip().split('<CODESPLIT>')[-1])
                        # correct_score = float(batch_data[0].strip().split('<CODESPLIT>')[-1])
                        if "targeted" in file_name:
                            # 定向攻击样本，正确代码被强制排在首位进行评估
                            correct_score = float(batch_data[0].strip().split('<CODESPLIT>')[-1])
                        else:
                            # 普通样本，正确代码在对应的 batch 索引位置
                            correct_score = float(batch_data[batch_idx].strip().split('<CODESPLIT>')[-1])
                        scores = np.array([float(data.strip().split('<CODESPLIT>')[-1]) for data in batch_data])
                        rank = np.sum(scores >= correct_score)
                        ranks.append(rank)

        # Calculate Mean Reciprocal Rank
        if len(ranks) > 0:
            mean_mrr = np.mean(1.0 / np.array(ranks))
            print("\n" + "=" * 50)
            print(f"Language: {language}")
            print(f"Total batches: {num_batch}")
            print(f"MRR: {mean_mrr:.4f}")
            print("=" * 50 + "\n")
            MRR_dict[language] = mean_mrr
        else:
            print(f"Warning: No valid batch files found in {file_dir}")

    # Summary
    print("\n" + "=" * 50)
    print("Summary:")
    for key, val in MRR_dict.items():
        print(f"  {key} MRR: {val:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
