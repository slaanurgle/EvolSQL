"""
Trainset Evolution (Inbre-evo): Naive difficulty evolution for NL2SQL pairs.

For each sample in the training set, uses LLM to generate harder versions
of the question and corresponding SQL. Supports sampling multiple evolutions
per sample (breadth expansion).

Usage:
    python pipeline/steps/trainset_evolution.py \
        --input_file data/train.json \
        --output_file results/inbre_evo.json \
        --mschema_dir schemas/train_mschemas \
        --api_urls http://localhost:8001/v1,http://localhost:8002/v1 \
        --model YOUR_MODEL \
        --sampling_count 3
"""

import argparse
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.llm_client import LLMClient
from core.parallel import parallel_call
from core.utils import (
    load_json, save_json, load_template, render_template,
    extract_json, safe_extract_json, setup_logging,
)
from core.schema import get_schema_str


def evolve_single(
    sample: dict,
    client: LLMClient,
    template: str,
    mschema_dir: str,
    use_full_schema: bool,
    update_evidence: bool,
) -> list:
    """
    Evolve a single sample once (one LLM call producing one evolution).

    Returns:
        List of evolved sample dicts (may be empty if LLM fails).
    """
    db_id = sample["db_id"]
    question = sample.get("question", "")
    gold_sql = sample.get("SQL", sample.get("gold_sql", ""))
    evidence = sample.get("evidence", "")
    question_id = sample.get("question_id", "")

    # Get schema
    schema_string = get_schema_str(
        mschema_dir=mschema_dir,
        db_id=db_id,
        sql=gold_sql,
        use_full_schema=use_full_schema,
    )

    # Render prompt
    prompt = render_template(
        template,
        DATABASE_SCHEMA=schema_string,
        QUESTION=question,
        EVIDENCE=evidence,
        GOLD_SQL=gold_sql,
    )

    # Call LLM
    response_text = client.chat(prompt)
    if response_text is None:
        return []

    # Parse response
    parsed = safe_extract_json(response_text)
    if parsed is None:
        return []

    # Handle both single dict and list responses
    if isinstance(parsed, dict):
        parsed = [parsed]

    results = []
    for item in parsed:
        new_question = item.get("question")
        new_sql = item.get("gold_sql")
        if new_question is None or new_sql is None:
            continue

        new_sample = {
            "question_id": question_id,
            "question": new_question,
            "evidence": item.get("evidence", evidence) if update_evidence else evidence,
            "db_id": db_id,
            "gold_sql": new_sql,
            "difficulty": None,
        }
        results.append(new_sample)

    return results


def run(args):
    """
    Core logic for trainset evolution. Can be called from pipeline or CLI.

    Args:
        args: argparse.Namespace with all required fields.
    """
    setup_logging()

    # Default template
    if args.template_file is None:
        args.template_file = str(Path(__file__).resolve().parent.parent / "templates" / "template_trainset_evolution.txt")

    # Load data
    print(f"Loading input data from {args.input_file}...")
    data = load_json(args.input_file)
    print(f"Loaded {len(data)} samples")

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
    print(f"LLM client initialized with {len(api_urls)} endpoint(s)")

    # Resume support
    all_evolved = []
    processed_qids = set()
    if args.resume and os.path.exists(args.output_file):
        print(f"Resuming from {args.output_file}...")
        all_evolved = load_json(args.output_file)
        processed_qids = {s.get("question_id") for s in all_evolved}
        print(f"Already have {len(all_evolved)} evolved samples, {len(processed_qids)} source questions")

    # Filter out already processed
    pending = [s for s in data if s.get("question_id") not in processed_qids]
    print(f"Pending samples: {len(pending)}")

    if not pending:
        print("No pending samples. Done.")
        return

    # Build tasks: each (sample, evolve_idx) pair
    # For sampling_count > 1, we create multiple calls per sample
    tasks = []
    for sample in pending:
        for evo_idx in range(args.sampling_count):
            tasks.append((sample, evo_idx))

    print(f"Total LLM calls to make: {len(tasks)}")

    # Process function
    def process_task(task):
        sample, evo_idx = task
        results = evolve_single(
            sample=sample,
            client=client,
            template=template,
            mschema_dir=args.mschema_dir,
            use_full_schema=args.use_full_schema,
            update_evidence=args.update_evidence,
        )
        # Tag with evolve_id
        for r in results:
            r["evolve_id"] = evo_idx
        return results

    # Run in parallel
    results = parallel_call(
        items=tasks,
        process_func=process_task,
        max_workers=args.max_workers,
        description="Trainset Evolution",
    )

    # Collect results
    new_count = 0
    for result in results:
        if result:
            all_evolved.extend(result)
            new_count += len(result)

    print(f"\nGenerated {new_count} new evolved samples")
    print(f"Total evolved samples: {len(all_evolved)}")

    # Save
    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
    save_json(args.output_file, all_evolved)
    print(f"Saved to {args.output_file}")


def main():
    parser = argparse.ArgumentParser(description="Trainset Evolution (Inbre-evo)")
    parser.add_argument("--input_file", type=str, required=True, help="Input training data JSON")
    parser.add_argument("--output_file", type=str, required=True, help="Output evolved data JSON")
    parser.add_argument("--mschema_dir", type=str, required=True, help="Directory containing mschema JSON files")
    parser.add_argument("--template_file", type=str, default=None,
                        help="Path to prompt template (default: templates/template_trainset_evolution.txt)")

    # LLM settings
    parser.add_argument("--api_urls", type=str, required=True,
                        help="Comma-separated vLLM API URLs")
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--api_key", type=str, default="EMPTY", help="API key")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--max_tokens", type=int, default=4096, help="Max tokens")

    # Evolution settings
    parser.add_argument("--sampling_count", type=int, default=3,
                        help="Number of evolutions per sample (breadth)")
    parser.add_argument("--use_full_schema", action="store_true", default=True,
                        help="Use full schema instead of SQL-filtered")
    parser.add_argument("--update_evidence", action="store_true", default=False,
                        help="Update evidence in evolved samples")

    # Parallelism
    parser.add_argument("--max_workers", type=int, default=32, help="Max parallel workers")

    # Resume
    parser.add_argument("--resume", action="store_true", help="Resume from existing output")
    parser.add_argument("--save_interval", type=int, default=500,
                        help="Save checkpoint every N samples")

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
