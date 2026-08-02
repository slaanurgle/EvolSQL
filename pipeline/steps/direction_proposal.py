"""
Direction Proposal: Evaluate 6 evolution axes for each sample.

For each sample, asks the LLM to score the feasibility and impact of
6 evolution directions (A-F). The scores are stored in a "proposal"
field for later use by direction_evolution.

The 6 axes are:
A: subqueries - nested queries, correlated subqueries, CTEs
B: conditions - WHERE/HAVING/ORDER BY complexity
C: sql_operators - BETWEEN, IN, LIKE, CASE WHEN
D: sql_functions - aggregate, date, math, window functions
E: tables_involved - more JOINs, different JOIN types
F: set_operators - UNION/INTERSECT/EXCEPT/EXISTS

Usage:
    python pipeline/steps/direction_proposal.py \
        --input_file results/inbre_evo_verified.json \
        --output_file results/proposal.json \
        --mschema_dir schemas/train_mschemas \
        --api_urls http://localhost:8001/v1 \
        --model YOUR_MODEL \
        --top_k 2
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
    extract_json, safe_extract_json, setup_logging,
)
from core.schema import get_schema_str


AXIS_NAMES = ["A", "B", "C", "D", "E", "F"]


def propose_single(
    sample: dict,
    client: LLMClient,
    template: str,
    mschema_dir: str,
    use_full_schema: bool,
    top_k: int,
    max_retries: int = 3,
) -> dict:
    """
    Generate direction proposals for a single sample.

    Returns:
        Updated sample dict with "proposal" field (list of top-k axis indices)
    """
    import copy
    new_sample = copy.deepcopy(sample)

    db_id = sample["db_id"]
    question = sample.get("question", "")
    gold_sql = sample.get("gold_sql", sample.get("SQL", ""))

    schema_string = get_schema_str(
        mschema_dir=mschema_dir,
        db_id=db_id,
        sql=gold_sql,
        use_full_schema=use_full_schema,
    )

    prompt = render_template(
        template,
        DATABASE_SCHEMA=schema_string,
        QUESTION=question,
        GOLD_SQL=gold_sql,
    )

    for attempt in range(max_retries):
        response_text = client.chat(prompt)
        if response_text is None:
            continue

        parsed = safe_extract_json(response_text)
        if parsed is None or not isinstance(parsed, list):
            continue

        # Validate structure: should be a list of dicts with axis and score
        try:
            scored = []
            for item in parsed:
                axis = item.get("axis", "")
                score = float(item.get("score", 0))
                scored.append({"axis": axis, "score": score})

            # Sort by score descending, pick top_k
            scored.sort(key=lambda x: x["score"], reverse=True)
            top_axes = scored[:top_k]

            # Convert axis letters to indices (A=0, B=1, ..., F=5)
            axis_to_idx = {name: idx for idx, name in enumerate(AXIS_NAMES)}
            proposal_indices = [
                axis_to_idx[item["axis"]]
                for item in top_axes
                if item["axis"] in axis_to_idx
            ]

            if proposal_indices:
                new_sample["proposal"] = proposal_indices
                new_sample["proposal_detail"] = top_axes
                return new_sample

        except (ValueError, KeyError, TypeError):
            continue

    # All retries failed
    return new_sample


def run(args):
    """
    Core logic for direction proposal. Can be called from pipeline or CLI.

    Args:
        args: argparse.Namespace with all required fields.
    """
    setup_logging()

    if args.template_file is None:
        args.template_file = str(Path(__file__).resolve().parent.parent / "templates" / "template_direction_proposal.txt")

    # Load data
    print(f"Loading data from {args.input_file}...")
    data = load_json(args.input_file)
    print(f"Loaded {len(data)} samples")

    # Resume: if output file exists, load it instead and skip already-proposed samples
    if args.resume and os.path.exists(args.output_file):
        print(f"Resuming from {args.output_file}...")
        data = load_json(args.output_file)
        already = sum(1 for s in data if "proposal" in s)
        print(f"Already proposed: {already}/{len(data)}")

    template = load_template(args.template_file)

    api_urls = [u.strip() for u in args.api_urls.split(",")]
    client = LLMClient(
        api_urls=api_urls,
        model=args.model,
        api_key=args.api_key,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    # Filter samples: only process those without proposals and meeting filter criteria
    pending_indices = []
    for idx, sample in enumerate(data):
        # Skip already proposed (resume)
        if "proposal" in sample:
            continue
        if args.filter_compilable and not sample.get("can_compile", True):
            continue
        if args.filter_empty_result and sample.get("ex_result") == "[]":
            continue
        pending_indices.append(idx)

    pending_samples = [data[i] for i in pending_indices]
    print(f"Samples to process: {len(pending_samples)} / {len(data)}")

    if not pending_samples:
        print("No samples to process. Saving unchanged.")
        save_json(args.output_file, data)
        return

    def process_sample(sample):
        return propose_single(
            sample=sample,
            client=client,
            template=template,
            mschema_dir=args.mschema_dir,
            use_full_schema=args.use_full_schema,
            top_k=args.top_k,
            max_retries=args.max_retries,
        )

    results = parallel_call(
        items=pending_samples,
        process_func=process_sample,
        max_workers=args.max_workers,
        description="Direction Proposal",
    )

    # Apply results
    proposal_count = 0
    for i, result in enumerate(results):
        if result is not None:
            idx = pending_indices[i]
            data[idx] = result
            if "proposal" in result:
                proposal_count += 1

    print(f"\nGenerated proposals for {proposal_count} / {len(pending_samples)} samples")

    # Save
    save_json(args.output_file, data)
    print(f"Saved to {args.output_file}")


def main():
    parser = argparse.ArgumentParser(description="Direction Proposal")
    parser.add_argument("--input_file", type=str, required=True, help="Input data JSON")
    parser.add_argument("--output_file", type=str, required=True, help="Output data with proposals")
    parser.add_argument("--mschema_dir", type=str, required=True, help="MSchema directory")
    parser.add_argument("--template_file", type=str, default=None)

    # LLM
    parser.add_argument("--api_urls", type=str, required=True, help="Comma-separated vLLM API URLs")
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--api_key", type=str, default="EMPTY")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_tokens", type=int, default=4096)

    # Settings
    parser.add_argument("--top_k", type=int, default=2, help="Number of top directions to select")
    parser.add_argument("--use_full_schema", action="store_true", default=False)
    parser.add_argument("--max_retries", type=int, default=3, help="Max retries per sample")
    parser.add_argument("--max_workers", type=int, default=32)

    # Filter: only process compilable samples
    parser.add_argument("--filter_compilable", action="store_true", default=True,
                        help="Only process samples with can_compile=True")
    parser.add_argument("--filter_empty_result", action="store_true", default=True,
                        help="Also skip samples with ex_result='[]'")

    # Resume
    parser.add_argument("--resume", action="store_true", help="Resume from existing output")

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
