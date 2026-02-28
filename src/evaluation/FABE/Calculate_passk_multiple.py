"""
Calculate Pass@k for MultiPL-E benchmarks (HumanEval / MBPP).

Uses the SandboxFusion /submit endpoint with MultiPLEDataset config,
where the sandbox manages test cases internally via HuggingFace nuprl/MultiPL-E data.

Output format is compatible with aggregate_results.py (passed_count, total_candidates, variant_type).
"""

import argparse
import json
import os
import sys
import time

import numpy as np
from tqdm import tqdm

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data_preprocessing.CodeContestsPlus.run_selection import create_robust_session


def submit_to_sandbox(session, sandbox_url, dataset_name, problem_name,
                      completion, provided_data, compile_timeout, run_timeout,
                      max_retries=3):
    """Submit a single candidate to SandboxFusion /submit endpoint.

    Returns True if the submission was accepted (all tests passed), False otherwise.
    """
    payload = {
        "dataset": dataset_name,
        "id": problem_name,
        "completion": completion,
        "config": {
            "provided_data": provided_data,
            "compile_timeout": compile_timeout,
            "run_timeout": run_timeout,
        },
    }

    url = f"{sandbox_url.rstrip('/')}/submit"

    for attempt in range(max_retries):
        try:
            resp = session.post(url, json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()

            # Log the full response for the very first submission (debugging aid)
            if not hasattr(submit_to_sandbox, "_first_logged"):
                submit_to_sandbox._first_logged = True
                print(f"[DEBUG] First /submit response: {json.dumps(data, indent=2, ensure_ascii=False)[:2000]}")

            # Check accepted field with fallback paths
            accepted = data.get("accepted")
            if accepted is None:
                result = data.get("result", {})
                if isinstance(result, dict):
                    accepted = result.get("accepted")
                if accepted is None:
                    accepted = data.get("status") == "Accepted"

            return bool(accepted)

        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                print(f"[WARN] /submit attempt {attempt + 1} failed: {e}; retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"[ERROR] /submit failed after {max_retries} attempts: {e}")
                return False

    return False


def main():
    ap = argparse.ArgumentParser(
        description="Calculate Pass@k for MultiPL-E HumanEval/MBPP via SandboxFusion /submit"
    )
    ap.add_argument("--inference_results", required=True,
                     help="JSONL file with 'candidates' and 'variant_type' fields")
    ap.add_argument("--sandbox", required=True,
                     help="SandboxFusion base URL (e.g. http://127.0.0.1:12410)")
    ap.add_argument("--out_shard_jsonl", required=True,
                     help="Output shard JSONL path")
    ap.add_argument("--benchmark", required=True, choices=["humaneval", "mbpp"],
                     help="Benchmark name: humaneval or mbpp")
    ap.add_argument("--lang", required=True,
                     help="Target language (e.g. cpp, java, py, js, rs, ...)")
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--shard_id", type=int, default=0)
    ap.add_argument("--run_timeout", type=int, default=20,
                     help="Per-submission run timeout in seconds")
    ap.add_argument("--compile_timeout", type=int, default=20,
                     help="Compile timeout in seconds")
    ap.add_argument("--hf_cache_dir", default=None,
                     help="Pre-downloaded HuggingFace cache directory")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out_shard_jsonl)), exist_ok=True)

    # --- Load HuggingFace dataset ---
    from datasets import load_dataset

    hf_subset = f"{args.benchmark}-{args.lang}"
    print(f"[INFO] Loading HuggingFace dataset: nuprl/MultiPL-E / {hf_subset}")

    load_kwargs = {"path": "nuprl/MultiPL-E", "name": hf_subset, "split": "test"}
    if args.hf_cache_dir:
        load_kwargs["cache_dir"] = args.hf_cache_dir

    try:
        hf_ds = load_dataset(**load_kwargs)
    except Exception as e:
        print(f"[ERROR] Failed to load HF dataset nuprl/MultiPL-E/{hf_subset}: {e}")
        sys.exit(1)

    # Build lookup dict by problem name, with fallback to task_id
    hf_lookup = {}
    for entry in hf_ds:
        name = entry.get("name")
        if name:
            hf_lookup[name] = entry
        task_id = entry.get("task_id")
        if task_id and task_id not in hf_lookup:
            hf_lookup[task_id] = entry

    print(f"[INFO] Loaded {len(hf_ds)} problems from HF dataset, {len(hf_lookup)} lookup keys")

    # --- Initialize session ---
    session = create_robust_session()

    # Dataset name for SandboxFusion
    dataset_name = "MultiPLEDataset"

    # --- Read inference results ---
    with open(args.inference_results, "r", encoding="utf-8") as f:
        all_data = [json.loads(line) for line in f if line.strip()]

    print(f"[INFO] Loaded {len(all_data)} items from inference results")

    results = []
    unmatched = 0

    for i, item in enumerate(tqdm(all_data, desc=f"Shard {args.shard_id} Verifying")):
        if i % args.num_shards != args.shard_id:
            continue

        # Match to HF dataset entry
        problem_name = item.get("name") or item.get("task_id") or item.get("problem_id", "")
        hf_entry = hf_lookup.get(problem_name)

        if hf_entry is None:
            # Try alternative keys
            for key in ["task_id", "name", "problem_id"]:
                alt = item.get(key)
                if alt and alt in hf_lookup:
                    hf_entry = hf_lookup[alt]
                    problem_name = alt
                    break

        if hf_entry is None:
            unmatched += 1
            if unmatched <= 5:
                print(f"[WARN] No HF match for item {i}, keys: "
                      f"name={item.get('name')}, task_id={item.get('task_id')}, "
                      f"problem_id={item.get('problem_id')}")
            item["passed_count"] = 0
            item["total_candidates"] = len(item.get("candidates", []))
            if "variant_type" not in item:
                item["variant_type"] = "unknown"
            results.append(item)
            continue

        # Convert HF entry to serializable dict for provided_data
        provided_data = {k: v for k, v in hf_entry.items()}

        candidates = item.get("candidates", [])
        passed_count = 0

        for code in candidates:
            accepted = submit_to_sandbox(
                session, args.sandbox, dataset_name, problem_name,
                code, provided_data,
                compile_timeout=args.compile_timeout,
                run_timeout=args.run_timeout,
            )
            if accepted:
                passed_count += 1

        item["passed_count"] = passed_count
        item["total_candidates"] = len(candidates)
        if "variant_type" not in item:
            item["variant_type"] = "unknown"
        results.append(item)

    if unmatched > 0:
        print(f"[WARN] {unmatched} items had no HF dataset match")

    # --- Write shard output ---
    with open(args.out_shard_jsonl, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    total_items = len(results)
    total_passed = sum(r["passed_count"] for r in results)
    total_candidates = sum(r["total_candidates"] for r in results)
    print(f"[INFO] Shard {args.shard_id} done: {total_items} items, "
          f"{total_passed}/{total_candidates} candidates passed")


if __name__ == "__main__":
    main()
