"""
Direction Evolution (Indep-evo): Evolve along proposed directions.

Uses the proposals from direction_proposal to evolve each sample along
the top-scored evolution axes. Each axis generates a new question+SQL pair.

The 6 methods correspond to the 6 axes:
0 (A): subqueries
1 (B): conditions
2 (C): sql_operators
3 (D): sql_functions
4 (E): tables_involved
5 (F): set_operators

Usage:
    python pipeline/steps/direction_evolution.py \
        --input_file results/proposal.json \
        --output_file results/indep_evo.json \
        --mschema_dir schemas/train_mschemas \
        --api_urls http://localhost:8001/v1 \
        --model YOUR_MODEL
"""

import argparse
import sys
import os
import copy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.llm_client import LLMClient
from core.parallel import parallel_call
from core.utils import (
    load_json, save_json, load_template, render_template,
    safe_extract_json, setup_logging,
)
from core.schema import get_schema_str


ALL_METHODS = [
    "Make the query structure more complex by introducing nested queries, correlated subqueries, or Common Table Expressions (CTEs).",
    "Increase the logical complexity within existing SQL clauses. For example, combine multiple conditions in the WHERE clause using AND/OR/NOT; if the original SQL has a GROUP BY, add a HAVING clause to filter the aggregated results; or sort by multiple columns or an expression in the ORDER BY clause",
    "Use a wider variety of SQL operators in the query. For example, use BETWEEN for a range comparison, IN or NOT IN to filter against a set of values, LIKE for pattern matching, or introduce conditional logic with a CASE WHEN ... THEN ... END expression in the SELECT or ORDER BY clause.",
    "Integrate SQL functions to process data within the query. You can use aggregate functions, date functions, mathematical functions or window functions.",
    "Increase the number of tables being joined or change the type of join.",
    "Use set operators (UNION, INTERSECT, EXCEPT) to combine or compare the result sets of two or more queries. Alternatively, use an EXISTS or NOT EXISTS subquery to check for the existence of records that satisfy a certain condition.",
]


def evolve_single_direction(
    sample: dict,
    method_idx: int,
    client: LLMClient,
    template: str,
    mschema_dir: str,
    use_full_schema: bool,
    update_evidence: bool,
) -> dict:
    """
    Evolve a single sample along one direction.

    Returns:
        New evolved sample dict or None if failed
    """
    db_id = sample["db_id"]
    question = sample.get("question", "")
    gold_sql = sample.get("gold_sql", sample.get("SQL", ""))
    evidence = sample.get("evidence", "")
    question_id = sample.get("question_id", "")

    method = ALL_METHODS[method_idx]

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
        EVIDENCE=evidence,
        GOLD_SQL=gold_sql,
        METHOD=method,
    )

    response_text = client.chat(prompt)
    if response_text is None:
        return None

    parsed = safe_extract_json(response_text)
    if parsed is None or not isinstance(parsed, dict):
        return None

    new_question = parsed.get("question")
    new_sql = parsed.get("gold_sql")

    if new_question is None or new_sql is None:
        return None

    new_sample = {
        "question_id": question_id,
        "question": new_question,
        "evidence": parsed.get("evidence", evidence) if update_evidence else evidence,
        "db_id": db_id,
        "gold_sql": new_sql,
        "difficulty": None,
        "evolve_id": sample.get("evolve_id", -1),
        "direction_id": method_idx,
    }

    return new_sample


def run(args):
    """
    Core logic for direction evolution. Can be called from pipeline or CLI.

    Args:
        args: argparse.Namespace with all required fields.
    """
    setup_logging()

    if args.template_file is None:
        args.template_file = str(Path(__file__).resolve().parent.parent / "templates" / "template_direction_evolution.txt")

    # Load data
    print(f"Loading data from {args.input_file}...")
    data = load_json(args.input_file)
    print(f"Loaded {len(data)} samples")

    template = load_template(args.template_file)

    api_urls = [u.strip() for u in args.api_urls.split(",")]
    client = LLMClient(
        api_urls=api_urls,
        model=args.model,
        api_key=args.api_key,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    # Resume: load existing output and build set of already-processed (question_id, direction_id) pairs
    existing_evolved = []
    processed_pairs = set()
    if args.resume and os.path.exists(args.output_file):
        print(f"Resuming from {args.output_file}...")
        existing_evolved = load_json(args.output_file)
        for item in existing_evolved:
            qid = item.get("question_id", "")
            did = item.get("direction_id", -1)
            processed_pairs.add((qid, did))
        print(f"Already have {len(existing_evolved)} evolved samples, {len(processed_pairs)} (qid, dir) pairs")

    # Build tasks: (sample, method_idx)
    import random
    tasks = []

    for sample in data:
        # Apply filters
        if args.filter_compilable and not sample.get("can_compile", True):
            continue
        if args.filter_empty_result and sample.get("ex_result") == "[]":
            continue

        if args.use_proposals:
            proposals = sample.get("proposal", [])
            if not proposals:
                continue
            method_indices = proposals
        elif args.sample_num > 0:
            method_indices = random.sample(range(len(ALL_METHODS)), min(args.sample_num, len(ALL_METHODS)))
        else:
            method_indices = list(range(len(ALL_METHODS)))

        qid = sample.get("question_id", "")
        for method_idx in method_indices:
            # Skip already processed pairs (resume)
            if (qid, method_idx) in processed_pairs:
                continue
            tasks.append((sample, method_idx))

    print(f"Total evolution tasks: {len(tasks)}")

    if not tasks:
        print("No tasks to process.")
        if existing_evolved:
            print(f"Keeping existing {len(existing_evolved)} evolved samples.")
            save_json(args.output_file, existing_evolved)
        else:
            print("Saving empty result.")
            save_json(args.output_file, [])
        return

    def process_task(task):
        sample, method_idx = task
        return evolve_single_direction(
            sample=sample,
            method_idx=method_idx,
            client=client,
            template=template,
            mschema_dir=args.mschema_dir,
            use_full_schema=args.use_full_schema,
            update_evidence=args.update_evidence,
        )

    results = parallel_call(
        items=tasks,
        process_func=process_task,
        max_workers=args.max_workers,
        description="Direction Evolution",
    )

    # Collect non-None results
    evolved = [r for r in results if r is not None]

    print(f"\nGenerated {len(evolved)} new evolved samples from {len(tasks)} tasks")

    # Merge with existing results from resume
    all_evolved = existing_evolved + evolved
    print(f"Total evolved samples: {len(all_evolved)}")

    # Save
    save_json(args.output_file, all_evolved)
    print(f"Saved to {args.output_file}")


def main():
    parser = argparse.ArgumentParser(description="Direction Evolution (Indep-evo)")
    parser.add_argument("--input_file", type=str, required=True, help="Input data with proposals")
    parser.add_argument("--output_file", type=str, required=True, help="Output evolved data")
    parser.add_argument("--mschema_dir", type=str, required=True, help="MSchema directory")
    parser.add_argument("--template_file", type=str, default=None)

    # LLM
    parser.add_argument("--api_urls", type=str, required=True, help="Comma-separated vLLM API URLs")
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--api_key", type=str, default="EMPTY")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_tokens", type=int, default=4096)

    # Settings
    parser.add_argument("--use_full_schema", action="store_true", default=True)
    parser.add_argument("--update_evidence", action="store_true", default=False)
    parser.add_argument("--use_proposals", action="store_true", default=True,
                        help="Use proposal directions; if False, use all 6 directions")
    parser.add_argument("--sample_num", type=int, default=-1,
                        help="If not using proposals, randomly sample this many directions (-1=all)")
    parser.add_argument("--max_workers", type=int, default=32)

    # Filter
    parser.add_argument("--filter_compilable", action="store_true", default=True,
                        help="Only process compilable samples")
    parser.add_argument("--filter_empty_result", action="store_true", default=True,
                        help="Skip samples with ex_result='[]'")

    # Resume
    parser.add_argument("--resume", action="store_true", help="Resume from existing output")

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
