import json
import glob
import argparse
import math

def calculate_pass_at_k(c, n, k):
    if n - c < k: return 1.0
    numer = range(n - c - k + 1, n - c + 1)
    denom = range(n - k + 1, n + 1)
    return 1.0 - math.prod(a / b for a, b in zip(numer, denom))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard_dir", required=True)
    parser.add_argument("--save_report", required=True)
    args = parser.parse_args()

    stats = {}
    for shard_file in glob.glob(f"{args.shard_dir}/shard_*.jsonl"):
        with open(shard_file, 'r') as f:
            for line in f:
                item = json.loads(line)
                v_type = item.get("variant_type") or "unknown"
                if v_type not in stats:
                    stats[v_type] = {"p1": 0, "p2": 0, "p4": 0, "count": 0}
                
                c, n = item["passed_count"], item["total_candidates"]
                stats[v_type]["p1"] += calculate_pass_at_k(c, n, 1)
                stats[v_type]["p2"] += calculate_pass_at_k(c, n, 2)
                stats[v_type]["p4"] += calculate_pass_at_k(c, n, 4)
                stats[v_type]["count"] += 1

    # 输出表格
    print(f"\n{'Variant Type':<20} | {'Pass@1':<8} | {'Pass@2':<8} | {'Pass@4':<8}")
    print("-" * 50)
    report = {}
    for v_type, data in stats.items():
        n = data["count"]
        res = {k: v/n for k, v in data.items() if k != "count"}
        print(f"{v_type:<20} | {res['p1']:.2%} | {res['p2']:.2%} | {res['p4']:.2%}")
        report[v_type] = res

    with open(args.save_report, 'w') as f:
        json.dump(report, f, indent=4)

if __name__ == "__main__":
    main()
