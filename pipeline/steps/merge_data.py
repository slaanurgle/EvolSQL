"""
Merge Data: Combine multiple evolved data files into one.

Merges data from different evolution stages (inbre-evo, indep-evo rounds)
into a single JSON file for downstream deduplication and rejection sampling.

Usage:
    python pipeline/steps/merge_data.py \
        --input_files results/inbre_verified.json results/indep1_verified.json results/indep2_verified.json \
        --output_file results/all_evolved.json \
        --filter_compilable
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.utils import load_json, save_json, setup_logging


def run(args):
    """
    Core logic for merge data. Can be called from pipeline or CLI.

    Args:
        args: argparse.Namespace with all required fields.
    """
    setup_logging()

    all_data = []
    for input_file in args.input_files:
        print(f"Loading {input_file}...")
        data = load_json(input_file)
        print(f"  -> {len(data)} samples")
        all_data.extend(data)

    print(f"\nTotal merged: {len(all_data)} samples")

    if args.filter_compilable:
        before = len(all_data)
        all_data = [s for s in all_data if s.get("can_compile", True)]
        print(f"After filter_compilable: {len(all_data)} (removed {before - len(all_data)})")

    if args.filter_empty_result:
        before = len(all_data)
        all_data = [s for s in all_data if s.get("ex_result") != "[]"]
        print(f"After filter_empty_result: {len(all_data)} (removed {before - len(all_data)})")

    save_json(args.output_file, all_data)
    print(f"\nSaved {len(all_data)} samples to {args.output_file}")


def main():
    parser = argparse.ArgumentParser(description="Merge evolved data files")
    parser.add_argument("--input_files", type=str, nargs="+", required=True,
                        help="Input JSON files to merge")
    parser.add_argument("--output_file", type=str, required=True, help="Output merged JSON")
    parser.add_argument("--filter_compilable", action="store_true", default=False,
                        help="Only keep samples with can_compile=True")
    parser.add_argument("--filter_empty_result", action="store_true", default=False,
                        help="Also filter out samples with ex_result='[]'")

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
