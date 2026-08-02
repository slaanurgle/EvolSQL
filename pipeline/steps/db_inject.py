"""
Database Injection: Generate and apply adversarial INSERT statements.

For each gold SQL in the training data, uses LLM to generate adversarial
INSERT statements that densify the database, making execution-based
evaluation more rigorous.

Design:
- Groups samples by db_id for efficient processing
- Each SQL gets a unique PK range (budget=10) to prevent conflicts
- Failed injections (after max_retries=3) mark the sample as inject_failed
- Databases are copied to an augmented directory before injection

Usage:
    python pipeline/steps/db_inject.py \
        --input_file data/train.json \
        --output_file results/db_inject.json \
        --db_root_path /path/to/databases \
        --augmented_db_dir /path/to/augmented_dbs \
        --api_urls http://localhost:8001/v1 \
        --model YOUR_MODEL \
        --mode train
"""

import argparse
import sys
import os
import json
import copy
import logging
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.llm_client import LLMClient
from core.parallel import parallel_call
from core.utils import load_json, save_json, load_template, load_mschema_mapping, setup_logging
from core.db_inject import (
    get_all_pk_info,
    compute_pk_ranges,
    build_inject_prompt,
    parse_insert_statements,
    validate_pk_in_range,
    apply_inserts,
    copy_database_dir,
)
from core.db_executor import resolve_db_path

logger = logging.getLogger(__name__)


def inject_single(
    sample: dict,
    sql_index: int,
    client: LLMClient,
    template: str,
    db_path: str,
    pk_info_map: dict,
    pk_budget: int = 10,
    max_retries: int = 3,
    mschema_str: str = "",
) -> dict:
    """
    Generate and apply adversarial INSERTs for a single SQL.

    Uses multi-turn conversation: the LLM can see its previous outputs and
    the corresponding error feedback, enabling more precise self-correction.

    Retry logic:
    - If PK values are out of range → append error as user message, retry
    - If INSERT execution fails (constraint violation) → append error, retry
    - If max_retries exhausted → mark sample as inject_failed

    Args:
        sample: Data sample dict (must have "SQL" or "gold_sql" and "db_id")
        sql_index: Index of this SQL within its db_id group
        client: LLM client
        template: Prompt template
        db_path: Path to the (augmented) database file
        pk_info_map: Pre-computed PK info for all tables
        pk_budget: PK range budget per SQL
        max_retries: Maximum LLM retries
        mschema_str: Pre-loaded mschema string for this db_id

    Returns:
        Updated sample dict with injection results
    """
    new_sample = copy.deepcopy(sample)
    gold_sql = sample.get("SQL", sample.get("gold_sql", ""))

    if not gold_sql:
        new_sample["inject_failed"] = True
        new_sample["inject_error"] = "No gold SQL found"
        return new_sample

    # Compute PK ranges for this SQL
    pk_ranges = compute_pk_ranges(pk_info_map, sql_index, budget=pk_budget)

    # Build initial prompt (without error feedback)
    initial_prompt = build_inject_prompt(
        template=template,
        gold_sql=gold_sql,
        pk_info_map=pk_info_map,
        pk_ranges=pk_ranges,
        error_feedback="",
        mschema_str=mschema_str,
        db_path=db_path,
        sql_index=sql_index,
    )

    # Initialize multi-turn conversation messages
    messages = [{"role": "user", "content": initial_prompt}]

    last_error_feedback = ""
    history = []  # Track each attempt's details

    # Helper: truncate LLM response for history (avoid JSON bloat)
    def _truncate(text, max_len=2000):
        if text and len(text) > max_len:
            return text[:max_len] + f"... [truncated, total {len(text)} chars]"
        return text

    for attempt in range(max_retries):
        # Call LLM with full conversation history
        response = client.chat_messages(messages)

        if response is None:
            # LLM call failed, append placeholder and error message
            messages.append({"role": "assistant", "content": "(no response)"})
            messages.append({"role": "user", "content":
                "Your previous response was empty. Please generate the INSERT statements again."
            })
            last_error_feedback = "LLM returned no response"
            history.append({
                "attempt": attempt + 1,
                "stage": "llm_fail",
                "llm_response": None,
                "parsed_inserts": None,
                "error": "LLM returned no response",
            })
            continue

        # Parse INSERT statements
        inserts = parse_insert_statements(response)

        if not inserts:
            # No valid INSERTs found — append assistant response + error feedback
            messages.append({"role": "assistant", "content": response})
            feedback = (
                "No valid INSERT statements were found in your response. "
                "Please output INSERT INTO ... VALUES(...); statements in a ```sql``` code block."
            )
            messages.append({"role": "user", "content": feedback})
            last_error_feedback = feedback
            history.append({
                "attempt": attempt + 1,
                "stage": "parse_fail",
                "llm_response": _truncate(response),
                "parsed_inserts": None,
                "error": "No valid INSERT statements parsed",
            })
            continue

        # Validate PK ranges
        pk_violations = validate_pk_in_range(inserts, pk_ranges, db_path)
        if pk_violations:
            messages.append({"role": "assistant", "content": response})
            violation_str = "\n".join(f"- {v}" for v in pk_violations)
            feedback = (
                f"Your INSERT statements have PK values out of the allocated range:\n"
                f"{violation_str}\n\n"
                f"You MUST use PK values strictly within the allocated ranges shown in the schema. "
                f"Please regenerate ALL INSERT statements with correct PK values."
            )
            messages.append({"role": "user", "content": feedback})
            last_error_feedback = feedback
            history.append({
                "attempt": attempt + 1,
                "stage": "pk_violation",
                "llm_response": _truncate(response),
                "parsed_inserts": inserts,
                "error": pk_violations,
            })
            continue

        # Execute INSERTs
        success_count, errors = apply_inserts(db_path, inserts)
        if errors:
            messages.append({"role": "assistant", "content": response})
            error_details = "\n".join(f"- {e}" for e in errors)
            feedback = (
                f"INSERT execution failed with database constraint errors:\n"
                f"{error_details}\n\n"
                f"Please fix the INSERT statements and ensure all constraints "
                f"(PK, FK, NOT NULL, UNIQUE) are satisfied. "
                f"Regenerate ALL INSERT statements."
            )
            messages.append({"role": "user", "content": feedback})
            last_error_feedback = feedback
            history.append({
                "attempt": attempt + 1,
                "stage": "exec_error",
                "llm_response": _truncate(response),
                "parsed_inserts": inserts,
                "error": errors,
            })
            continue

        # Success!
        history.append({
            "attempt": attempt + 1,
            "stage": "success",
            "llm_response": _truncate(response),
            "parsed_inserts": inserts,
            "error": None,
        })
        new_sample["inject_success"] = True
        new_sample["inject_count"] = success_count
        new_sample["inject_inserts"] = inserts
        new_sample["inject_attempts"] = attempt + 1
        new_sample["inject_history"] = history
        return new_sample

    # All retries exhausted
    new_sample["inject_failed"] = True
    new_sample["inject_error"] = f"Failed after {max_retries} attempts. Last error: {last_error_feedback[:500]}"
    new_sample["inject_attempts"] = max_retries
    new_sample["inject_history"] = history
    return new_sample


def resolve_db_file(db_root_path: str, db_id: str, mode: str) -> str:
    """Resolve database file path based on mode."""
    return str(resolve_db_path(db_root_path, db_id, mode))


def run(args):
    """
    Core logic for database injection. Can be called from pipeline or CLI.

    Flow:
    1. Load input data
    2. Group samples by db_id
    3. Copy databases to augmented directory
    4. For each db_id group: compute PK info, process each SQL
    5. Save results with injection status

    Args:
        args: argparse.Namespace with required fields
    """
    setup_logging()

    if args.template_file is None:
        args.template_file = str(
            Path(__file__).resolve().parent.parent / "templates" / "template_db_inject.txt"
        )

    # Load data
    print(f"Loading data from {args.input_file}...")
    data = load_json(args.input_file)
    print(f"Loaded {len(data)} samples")

    # Load template
    template = load_template(args.template_file)

    # Load mschema mapping (for consistent schema format with other pipeline steps)
    mschema_mapping = {}
    if args.mschema_file:
        print(f"Loading mschema from {args.mschema_file}...")
        mschema_mapping = load_mschema_mapping(args.mschema_file)
        print(f"Loaded {len(mschema_mapping)} database schemas")
    else:
        print("No mschema file provided, will fall back to DDL-based schema")

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

    # Group samples by db_id
    db_groups = defaultdict(list)
    for idx, sample in enumerate(data):
        db_id = sample["db_id"]
        db_groups[db_id].append((idx, sample))

    print(f"Unique databases: {len(db_groups)}")

    # Determine source and destination DB paths
    augmented_db_dir = args.augmented_db_dir
    os.makedirs(augmented_db_dir, exist_ok=True)

    # Determine the source DB directory based on mode
    if args.mode in ("train", "dev"):
        src_db_dir = os.path.join(args.db_root_path, f"{args.mode}_databases")
    else:
        src_db_dir = args.db_root_path

    # Copy databases to augmented directory
    print(f"Copying databases to {augmented_db_dir}...")
    copy_count = 0
    for db_id in db_groups:
        if copy_database_dir(src_db_dir, augmented_db_dir, db_id):
            copy_count += 1
    print(f"Copied {copy_count}/{len(db_groups)} databases")

    # Resume support: if output file exists, load processed state
    processed_indices = set()
    if args.resume and os.path.exists(args.output_file):
        print(f"Resuming from {args.output_file}...")
        existing_data = load_json(args.output_file)
        for idx, item in enumerate(existing_data):
            if item.get("inject_success") or item.get("inject_failed"):
                processed_indices.add(idx)
                data[idx] = item
        print(f"Already processed: {len(processed_indices)}/{len(data)}")

    # Build task list: (data_idx, sample, sql_index, db_id, db_path, pk_info_map)
    # We need to pre-compute PK info per db_id (shared across all SQLs in same DB)
    print("Computing PK info for all databases...")
    db_pk_info_cache = {}
    for db_id in db_groups:
        db_file = os.path.join(augmented_db_dir, db_id, f"{db_id}.sqlite")
        if os.path.exists(db_file):
            db_pk_info_cache[db_id] = get_all_pk_info(db_file)
        else:
            logger.warning(f"Database file not found: {db_file}")
            db_pk_info_cache[db_id] = {}

    # Build tasks - each task is a tuple that inject_single needs
    tasks = []
    for db_id, group in db_groups.items():
        db_mschema_str = mschema_mapping.get(db_id, "")
        for sql_index, (data_idx, sample) in enumerate(group):
            if data_idx in processed_indices:
                continue
            db_file = os.path.join(augmented_db_dir, db_id, f"{db_id}.sqlite")
            pk_info_map = db_pk_info_cache.get(db_id, {})
            tasks.append((data_idx, sample, sql_index, db_file, pk_info_map, db_mschema_str))

    print(f"Tasks to process: {len(tasks)}")

    if not tasks:
        print("No tasks to process. Saving unchanged.")
        save_json(args.output_file, data)
        return

    # IMPORTANT: We cannot do fully parallel injection for the same DB because
    # multiple INSERTs into the same .sqlite file would cause write conflicts.
    # Strategy: process all SQLs for the same DB sequentially,
    # but different DBs can be processed in parallel.

    # Group tasks by db_id for sequential processing within each DB
    db_task_groups = defaultdict(list)
    for task in tasks:
        data_idx, sample, sql_index, db_file, pk_info_map, db_mschema_str = task
        db_id = sample["db_id"]
        db_task_groups[db_id].append(task)

    # Process function: process all tasks for one DB sequentially
    def process_db_group(db_id_tasks):
        db_id, task_list = db_id_tasks
        results = []
        for data_idx, sample, sql_index, db_file, pk_info_map, db_mschema_str in task_list:
            result = inject_single(
                sample=sample,
                sql_index=sql_index,
                client=client,
                template=template,
                db_path=db_file,
                pk_info_map=pk_info_map,
                pk_budget=args.pk_budget,
                max_retries=args.max_retries,
                mschema_str=db_mschema_str,
            )
            results.append((data_idx, result))
        return results

    # Run across DBs in parallel
    db_task_items = list(db_task_groups.items())
    print(f"Processing {len(db_task_items)} database groups with {len(tasks)} total tasks...")

    all_results = parallel_call(
        items=db_task_items,
        process_func=process_db_group,
        max_workers=args.max_workers,
        description="DB Injection",
    )

    # Apply results
    inject_success = 0
    inject_failed = 0
    total_inserts = 0

    for db_results in all_results:
        if db_results is None:
            continue
        for data_idx, result in db_results:
            data[data_idx] = result
            if result.get("inject_success"):
                inject_success += 1
                total_inserts += result.get("inject_count", 0)
            elif result.get("inject_failed"):
                inject_failed += 1

    # Save results
    save_json(args.output_file, data)

    # Print statistics
    print(f"\n{'=' * 60}")
    print(f"DB Injection Statistics")
    print(f"{'=' * 60}")
    print(f"Total samples:     {len(data)}")
    print(f"Processed:         {inject_success + inject_failed}")
    print(f"Inject success:    {inject_success}")
    print(f"Inject failed:     {inject_failed}")
    print(f"Total rows added:  {total_inserts}")
    print(f"Already processed: {len(processed_indices)}")
    if inject_success + inject_failed > 0:
        success_rate = inject_success / (inject_success + inject_failed) * 100
        print(f"Success rate:      {success_rate:.1f}%")
    print(f"{'=' * 60}")
    print(f"Output: {args.output_file}")
    print(f"Augmented DBs: {augmented_db_dir}")

    # Save stats
    stats_file = args.output_file.replace(".json", "_stats.json")
    stats = {
        "total_samples": len(data),
        "inject_success": inject_success,
        "inject_failed": inject_failed,
        "total_inserts": total_inserts,
        "already_processed": len(processed_indices),
        "success_rate": inject_success / max(inject_success + inject_failed, 1) * 100,
    }
    save_json(stats_file, stats)
    print(f"Stats: {stats_file}")


def main():
    parser = argparse.ArgumentParser(description="Database Injection for Adversarial Data Augmentation")
    parser.add_argument("--input_file", type=str, required=True, help="Input data JSON")
    parser.add_argument("--output_file", type=str, required=True, help="Output data JSON with injection status")
    parser.add_argument("--db_root_path", type=str, required=True,
                        help="Root path to database files")
    parser.add_argument("--augmented_db_dir", type=str, required=True,
                        help="Directory for augmented database copies")
    parser.add_argument("--template_file", type=str, default=None,
                        help="Prompt template (default: templates/template_db_inject.txt)")
    parser.add_argument("--mschema_file", type=str, default=None,
                        help="MSchema JSONL file (db_name -> mschema_str). "
                             "If provided, uses mschema format instead of DDL for schema display.")
    parser.add_argument("--mode", type=str, default="train",
                        choices=["train", "dev", "spider_train"],
                        help="Dataset mode (default: train)")

    # LLM
    parser.add_argument("--api_urls", type=str, required=True, help="Comma-separated vLLM API URLs")
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--api_key", type=str, default="EMPTY")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_tokens", type=int, default=4096)

    # Injection settings
    parser.add_argument("--pk_budget", type=int, default=10,
                        help="PK range budget per SQL (default: 10)")
    parser.add_argument("--max_retries", type=int, default=3,
                        help="Max LLM retries per SQL (default: 3)")

    # Parallelism
    parser.add_argument("--max_workers", type=int, default=32, help="Max parallel workers for DB groups")

    # Resume
    parser.add_argument("--resume", action="store_true", help="Resume from existing output")

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
