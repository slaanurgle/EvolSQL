"""
Evolution Verification: Verify evolved SQL queries by executing them.

Checks whether each gold_sql can be compiled and executed against the
corresponding database. Tags each sample with can_compile and ex_result.

Usage:
    python pipeline/steps/evolution_verify.py \
        --input_file results/inbre_evo.json \
        --output_file results/inbre_evo_verified.json \
        --db_root_path /path/to/bird \
        --mode train \
        --num_workers 64
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.db_executor import resolve_db_path, batch_verify_sql
from core.utils import load_json, save_json, setup_logging


def run(args):
    """
    Core logic for evolution verification. Can be called from pipeline or CLI.

    Args:
        args: argparse.Namespace with all required fields.
    """
    setup_logging()

    # Load data
    print(f"Loading data from {args.input_file}...")
    data = load_json(args.input_file)
    print(f"Loaded {len(data)} samples")

    # Build tasks
    tasks = []
    valid_indices = []

    for idx, sample in enumerate(data):
        gold_sql = sample.get("gold_sql")
        db_id = sample.get("db_id")

        if gold_sql is None or db_id is None:
            # Mark as failed directly
            sample["can_compile"] = False
            sample["ex_result"] = "Missing gold_sql or db_id"
            continue

        db_path = str(resolve_db_path(args.db_root_path, db_id, args.mode))
        tasks.append((db_path, gold_sql))
        valid_indices.append(idx)

    print(f"Verifying {len(tasks)} SQL queries with {args.num_workers} workers...")

    # Batch verify
    results = batch_verify_sql(
        tasks=tasks,
        timeout=args.timeout,
        num_workers=args.num_workers,
        show_progress=True,
    )

    # Apply results
    compile_count = 0
    for i, result in enumerate(results):
        idx = valid_indices[i]
        data[idx]["can_compile"] = result["can_compile"]
        data[idx]["ex_result"] = result["ex_result"]
        if result["can_compile"]:
            compile_count += 1

    # Stats
    total = len(data)
    print(f"\nVerification Results:")
    print(f"  Total:      {total}")
    if total > 0:
        print(f"  Compilable: {compile_count} ({compile_count/total*100:.1f}%)")
        print(f"  Failed:     {total - compile_count} ({(total-compile_count)/total*100:.1f}%)")
    else:
        print("  ⚠️  No data to verify (input is empty)")

    # Save
    save_json(args.output_file, data)
    print(f"Saved to {args.output_file}")


def main():
    parser = argparse.ArgumentParser(description="Evolution Verification")
    parser.add_argument("--input_file", type=str, required=True, help="Input evolved data JSON")
    parser.add_argument("--output_file", type=str, required=True, help="Output verified data JSON")
    parser.add_argument("--db_root_path", type=str, required=True, help="Root path to database files")
    parser.add_argument("--mode", type=str, default="train",
                        choices=["train", "dev", "spider_train"],
                        help="Dataset mode (train/dev/spider_train)")
    parser.add_argument("--timeout", type=int, default=20, help="SQL execution timeout (seconds)")
    parser.add_argument("--num_workers", type=int, default=64, help="Number of worker processes")

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
