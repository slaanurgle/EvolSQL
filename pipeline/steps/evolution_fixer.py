"""
Evolution Fixer: Use LLM to fix SQL queries that failed verification.

For each sample where can_compile=False, sends the question, evidence,
broken SQL, and error message to an LLM to generate a corrected SQL.

Usage:
    python pipeline/steps/evolution_fixer.py \
        --input_file results/inbre_evo_verified.json \
        --output_file results/inbre_evo_fixed.json \
        --mschema_dir schemas/train_mschemas \
        --api_urls http://localhost:8001/v1 \
        --model YOUR_MODEL
"""

import argparse
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.llm_client import LLMClient
from core.parallel import parallel_call
from core.utils import (
    load_json, save_json, load_template, render_template,
    safe_extract_json, setup_logging,
)
from core.schema import get_schema_str


def fix_single(
    sample: dict,
    client: LLMClient,
    template: str,
    mschema_dir: str,
    use_full_schema: bool,
    original_sql: str = None,
) -> dict:
    """
    Fix a single sample's SQL using LLM.

    Args:
        sample: Sample dict with gold_sql, question, evidence, ex_result
        client: LLM client
        template: Prompt template
        mschema_dir: MSchema directory
        use_full_schema: Whether to use full schema
        original_sql: Original SQL from the source data (for schema extraction)

    Returns:
        Updated sample dict with fixed gold_sql
    """
    import copy
    new_sample = copy.deepcopy(sample)

    db_id = sample["db_id"]
    question = sample.get("question", "")
    evidence = sample.get("evidence", "")
    gold_sql = sample.get("gold_sql", "")
    ex_result = sample.get("ex_result", "")

    # Use original SQL for schema extraction if available
    sql_for_schema = original_sql if original_sql else gold_sql

    schema_string = get_schema_str(
        mschema_dir=mschema_dir,
        db_id=db_id,
        sql=sql_for_schema,
        use_full_schema=use_full_schema,
    )

    prompt = render_template(
        template,
        DATABASE_SCHEMA=schema_string,
        QUESTION=question,
        EVIDENCE=evidence,
        GOLD_SQL=gold_sql,
        GOLD_RESULT=ex_result,
    )

    response_text = client.chat(prompt)
    if response_text is None:
        return new_sample

    parsed = safe_extract_json(response_text)
    if parsed is None or not isinstance(parsed, dict):
        return new_sample

    new_sql = parsed.get("sql")
    if new_sql:
        new_sample["gold_sql"] = new_sql
        new_sample["can_compile"] = None  # Reset for re-verification
        new_sample["ex_result"] = None

    return new_sample


def run(args):
    """
    Core logic for evolution fixer. Can be called from pipeline or CLI.

    Args:
        args: argparse.Namespace with all required fields.
    """
    setup_logging()

    if args.template_file is None:
        args.template_file = str(Path(__file__).resolve().parent.parent / "templates" / "template_evolution_fixer.txt")

    # Load data
    print(f"Loading data from {args.input_file}...")
    data = load_json(args.input_file)
    print(f"Loaded {len(data)} samples")

    # Resume: if output file exists, load it instead (it contains partially-fixed data)
    # Samples already fixed will have can_compile=None (reset by fix_single),
    # so they won't match can_compile=False filter and will be skipped.
    if args.resume and os.path.exists(args.output_file):
        print(f"Resuming from {args.output_file}...")
        data = load_json(args.output_file)
        already_fixed = sum(1 for s in data if s.get("can_compile") is None)
        still_broken = sum(1 for s in data if s.get("can_compile") == False)
        print(f"Already fixed: {already_fixed}, still broken: {still_broken}")

    # Load template
    template = load_template(args.template_file)

    # Create LLM client
    api_urls = [u.strip() for u in args.api_urls.split(",")]
    client = LLMClient(
        api_urls=api_urls,
        model=args.model,
        api_key=args.api_key,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    # Filter samples that need fixing
    if args.fix_all:
        samples_to_fix = data
        fix_indices = list(range(len(data)))
    else:
        samples_to_fix = []
        fix_indices = []
        for idx, sample in enumerate(data):
            if not sample.get("can_compile", True):
                samples_to_fix.append(sample)
                fix_indices.append(idx)

    print(f"Samples to fix: {len(samples_to_fix)} / {len(data)}")

    if not samples_to_fix:
        print("No samples need fixing. Copying input to output.")
        save_json(args.output_file, data)
        return

    # Process
    def process_sample(sample):
        return fix_single(
            sample=sample,
            client=client,
            template=template,
            mschema_dir=args.mschema_dir,
            use_full_schema=args.use_full_schema,
        )

    results = parallel_call(
        items=samples_to_fix,
        process_func=process_sample,
        max_workers=args.max_workers,
        description="Fixing SQL",
    )

    # Apply results back
    fixed_count = 0
    for i, result in enumerate(results):
        if result is not None:
            idx = fix_indices[i]
            old_sql = data[idx].get("gold_sql", "")
            data[idx] = result
            if result.get("gold_sql") != old_sql:
                fixed_count += 1

    print(f"\nFixed {fixed_count} / {len(samples_to_fix)} samples")

    # Save
    save_json(args.output_file, data)
    print(f"Saved to {args.output_file}")


def main():
    parser = argparse.ArgumentParser(description="Evolution Fixer")
    parser.add_argument("--input_file", type=str, required=True, help="Input verified data JSON")
    parser.add_argument("--output_file", type=str, required=True, help="Output fixed data JSON")
    parser.add_argument("--mschema_dir", type=str, required=True, help="MSchema directory")
    parser.add_argument("--template_file", type=str, default=None,
                        help="Prompt template (default: templates/template_evolution_fixer.txt)")

    # LLM settings
    parser.add_argument("--api_urls", type=str, required=True, help="Comma-separated vLLM API URLs")
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--api_key", type=str, default="EMPTY", help="API key")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--max_tokens", type=int, default=4096, help="Max tokens")

    # Settings
    parser.add_argument("--use_full_schema", action="store_true", default=False,
                        help="Use full schema (default: SQL-filtered)")
    parser.add_argument("--fix_all", action="store_true", default=False,
                        help="Fix all samples, not just failed ones")
    parser.add_argument("--max_workers", type=int, default=32, help="Max parallel workers")

    # Resume
    parser.add_argument("--resume", action="store_true", help="Resume from existing output")

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
