"""
Rejection Sampling: Generate CoT reasoning and verify SQL correctness.

For each sample, generates a reasoning chain + predicted SQL using the CoT
template, then validates by comparing execution results with gold SQL.
Repeats up to max_samples rounds until a correct prediction is found.

Usage:
    python pipeline/steps/rejection_sampling.py \
        --input_file results/all_evolved_dedup.json \
        --output_file results/final_with_cot.json \
        --mschema_file schemas/train_mschemas.jsonl \
        --template_file templates/template_cot_rejection_sampling.txt \
        --db_path /path/to/bird/train_databases \
        --api_urls http://localhost:8001/v1 \
        --model YOUR_MODEL
"""

import argparse
import sys
import os
import re
import copy
import json
import threading
import queue
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.llm_client import LLMClient
from core.parallel import parallel_call
from core.db_executor import compare_sql_results, batch_compare_sql
from core.utils import (
    load_json, save_json, load_template, load_mschema_mapping,
    setup_logging,
)


def build_prompt(template: str, schema: str, question: str, evidence: str) -> str:
    """Build the CoT prompt from template."""
    prompt = template.replace("{DATABASE_SCHEMA}", schema)
    prompt = prompt.replace("{QUESTION}", question)

    if evidence and evidence.strip():
        evidence_text = f"Hint: {evidence}"
    else:
        evidence_text = ""
    prompt = prompt.replace("{EVIDENCE}", evidence_text)

    return prompt


def parse_cot_response(output: str) -> dict:
    """
    Parse CoT response to extract reasoning and SQL.

    Expected format:
        <reasoning>...</reasoning><answer>...</answer>

    Returns:
        {"reasoning": str, "sql": str} or None
    """
    if output is None:
        return None

    match = re.search(
        r"<reasoning>(.*?)</reasoning>\s*<answer>(.*?)</answer>",
        output,
        re.DOTALL,
    )

    if match:
        reasoning = match.group(1).strip()
        answer = match.group(2).strip()

        # Extract SQL from possible markdown formatting
        if "```sql" in answer:
            sql = answer.split("```sql")[1].split("```")[0].strip()
        else:
            sql = answer.strip()

        return {"reasoning": reasoning, "sql": sql}

    return None


def save_checkpoint_worker(save_queue, stop_event):
    """Background checkpoint save thread."""
    while not stop_event.is_set():
        try:
            task = save_queue.get(timeout=1.0)
            if task is None:
                break
            data, output_file, stats = task
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            stats_file = output_file.replace(".json", "_stats.json")
            with open(stats_file, "w", encoding="utf-8") as f:
                json.dump(stats, f, indent=2, ensure_ascii=False)
            print(f"[Checkpoint] {stats['processed']}/{stats['total']} processed, {stats['valid']} valid")
        except queue.Empty:
            continue
        except Exception as e:
            print(f"[Checkpoint Error] {e}")


def run(args):
    """
    Core logic for rejection sampling. Can be called from pipeline or CLI.

    Args:
        args: argparse.Namespace with all required fields.
    """
    setup_logging()

    if args.template_file is None:
        args.template_file = str(Path(__file__).resolve().parent.parent / "templates" / "template_cot_rejection_sampling.txt")

    # Load mschema mapping
    print(f"Loading mschema from {args.mschema_file}...")
    mschema_mapping = load_mschema_mapping(args.mschema_file)
    print(f"Loaded {len(mschema_mapping)} database schemas")

    # Load template
    template = load_template(args.template_file)

    # Load data
    print(f"Loading data from {args.input_file}...")
    data = load_json(args.input_file)
    print(f"Loaded {len(data)} samples")

    # Resume
    if args.resume and os.path.exists(args.output_file):
        print(f"Resuming from {args.output_file}...")
        data = load_json(args.output_file)
        already = sum(1 for item in data if "output_reasoning" in item or item.get("_rejected", False))
        print(f"Already processed: {already}/{len(data)}")

    # Create LLM client
    api_urls = [u.strip() for u in args.api_urls.split(",")]
    client = LLMClient(
        api_urls=api_urls,
        model=args.model,
        api_key=args.api_key,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    # Start background save thread
    save_queue = queue.Queue(maxsize=2)
    stop_event = threading.Event()
    save_thread = threading.Thread(
        target=save_checkpoint_worker, args=(save_queue, stop_event), daemon=False
    )
    save_thread.start()

    # Stats
    stats = {
        "total": len(data),
        "processed": sum(1 for item in data if "output_reasoning" in item or item.get("_rejected", False)),
        "valid": sum(1 for item in data if "output_reasoning" in item),
        "rejected": sum(1 for item in data if item.get("_rejected", False)),
        "parse_failed": 0,
        "sql_failed": 0,
    }

    # Main loop: round-by-round sampling
    for sample_round in range(args.max_samples):
        print(f"\n{'=' * 80}")
        print(f"Sample Round {sample_round + 1}/{args.max_samples}")
        print(f"{'=' * 80}")

        # Find pending samples
        pending_indices = []
        for idx, item in enumerate(data):
            if "output_reasoning" not in item and not item.get("_rejected", False):
                if item.get("_sample_round", 0) == sample_round:
                    pending_indices.append(idx)

        if not pending_indices:
            print("No more samples to process in this round")
            continue

        print(f"Pending samples: {len(pending_indices)}")

        # Process in batches
        for batch_start in range(0, len(pending_indices), args.batch_size):
            batch_indices = pending_indices[batch_start:batch_start + args.batch_size]

            # Build prompts
            batch_prompts = []
            batch_db_files = []
            batch_gold_sqls = []
            batch_valid_indices = []  # indices within batch that have valid prompts

            for local_idx, data_idx in enumerate(batch_indices):
                item = data[data_idx]
                db_id = item["db_id"]

                if db_id not in mschema_mapping:
                    data[data_idx]["_rejected"] = True
                    data[data_idx]["_reject_reason"] = "schema_not_found"
                    continue

                schema = mschema_mapping[db_id]
                question = item["question"]
                evidence = item.get("evidence", "")
                gold_sql = item.get("SQL", item.get("gold_sql", ""))

                prompt = build_prompt(template, schema, question, evidence)
                batch_prompts.append(prompt)
                batch_db_files.append(os.path.join(args.db_path, db_id, f"{db_id}.sqlite"))
                batch_gold_sqls.append(gold_sql)
                batch_valid_indices.append(local_idx)

            if not batch_prompts:
                continue

            # Batch LLM inference
            print(f"  Batch {batch_start // args.batch_size + 1}: {len(batch_prompts)} samples...")

            def call_llm(prompt):
                return client.chat(prompt)

            responses = parallel_call(
                items=batch_prompts,
                process_func=call_llm,
                max_workers=args.max_workers,
                description=f"  Inference",
                show_progress=True,
            )

            # Parse responses
            parsed_results = [None] * len(batch_prompts)
            valid_for_validation = []

            for i, response in enumerate(responses):
                local_idx = batch_valid_indices[i]
                data_idx = batch_indices[local_idx]

                if data_idx is None:
                    continue

                if data[data_idx].get("_rejected", False):
                    continue

                if response is None:
                    data[data_idx]["_sample_round"] = sample_round + 1
                    if sample_round == args.max_samples - 1:
                        data[data_idx]["_rejected"] = True
                        data[data_idx]["_reject_reason"] = "api_error"
                        stats["parse_failed"] += 1
                    continue

                parsed = parse_cot_response(response)
                if parsed is None:
                    data[data_idx]["_sample_round"] = sample_round + 1
                    if sample_round == args.max_samples - 1:
                        data[data_idx]["_rejected"] = True
                        data[data_idx]["_reject_reason"] = "parse_failed"
                        stats["parse_failed"] += 1
                else:
                    parsed_results[i] = parsed
                    valid_for_validation.append(i)

            # Batch SQL validation
            if valid_for_validation:
                compare_tasks = [
                    (batch_db_files[i], parsed_results[i]["sql"], batch_gold_sqls[i])
                    for i in valid_for_validation
                ]

                validation_results = batch_compare_sql(
                    tasks=compare_tasks,
                    timeout=args.timeout,
                    num_workers=args.num_cpus,
                    show_progress=True,
                )

                for j, valid_idx in enumerate(valid_for_validation):
                    local_idx = batch_valid_indices[valid_idx]
                    data_idx = batch_indices[local_idx]
                    is_valid = validation_results[j]

                    if is_valid:
                        data[data_idx]["output_reasoning"] = parsed_results[valid_idx]["reasoning"]
                        data[data_idx]["output_sql"] = parsed_results[valid_idx]["sql"]
                        stats["valid"] += 1
                    else:
                        data[data_idx]["_sample_round"] = sample_round + 1
                        if sample_round == args.max_samples - 1:
                            data[data_idx]["_rejected"] = True
                            data[data_idx]["_reject_reason"] = "sql_failed"
                            stats["sql_failed"] += 1

            # Update stats
            stats["processed"] = sum(
                1 for item in data if "output_reasoning" in item or item.get("_rejected", False)
            )
            stats["rejected"] = sum(1 for item in data if item.get("_rejected", False))

            # Async checkpoint
            if stats["processed"] % args.save_interval < args.batch_size:
                save_queue.put((copy.deepcopy(data), args.output_file, copy.deepcopy(stats)))

    # Stop background save
    stop_event.set()
    save_thread.join(timeout=30)

    # Clean up and save final results
    print("\nSaving final results...")
    final_data = []
    for item in data:
        if "output_reasoning" in item:
            item.pop("_sample_round", None)
            item.pop("_rejected", None)
            item.pop("_reject_reason", None)
            final_data.append(item)

    save_json(args.output_file, final_data)

    # Statistics
    final_valid = len(final_data)
    final_rejected = len(data) - final_valid

    print(f"\n{'=' * 80}")
    print(f"STATISTICS")
    print(f"{'=' * 80}")
    print(f"Total samples:    {len(data)}")
    print(f"Valid samples:    {final_valid}")
    print(f"Rejected:         {final_rejected}")
    print(f"  Parse failed:   {stats['parse_failed']}")
    print(f"  SQL failed:     {stats['sql_failed']}")
    print(f"Success rate:     {final_valid / len(data) * 100:.2f}%")
    print(f"{'=' * 80}")

    # Save stats
    stats_file = args.output_file.replace(".json", "_stats.json")
    final_stats = {
        "total": len(data),
        "valid": final_valid,
        "rejected": final_rejected,
        "parse_failed": stats["parse_failed"],
        "sql_failed": stats["sql_failed"],
        "success_rate": final_valid / len(data) * 100,
    }
    save_json(stats_file, final_stats)
    print(f"\nResults: {args.output_file}")
    print(f"Stats:   {stats_file}")


def main():
    parser = argparse.ArgumentParser(description="Rejection Sampling with CoT")
    parser.add_argument("--input_file", type=str, required=True, help="Input data JSON")
    parser.add_argument("--output_file", type=str, required=True, help="Output data JSON")
    parser.add_argument("--mschema_file", type=str, required=True,
                        help="MSchema JSONL file (db_name -> mschema_str)")
    parser.add_argument("--template_file", type=str, default=None,
                        help="CoT prompt template")
    parser.add_argument("--db_path", type=str, required=True,
                        help="Path to database directory (e.g., train_databases)")

    # LLM
    parser.add_argument("--api_urls", type=str, required=True, help="Comma-separated vLLM API URLs")
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--api_key", type=str, default="EMPTY")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_tokens", type=int, default=4096)

    # Settings
    parser.add_argument("--max_samples", type=int, default=4, help="Max sampling rounds")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size per round")
    parser.add_argument("--timeout", type=int, default=10, help="SQL execution timeout")
    parser.add_argument("--num_cpus", type=int, default=20, help="CPUs for SQL validation")
    parser.add_argument("--max_workers", type=int, default=32, help="Max LLM API workers")
    parser.add_argument("--save_interval", type=int, default=100, help="Checkpoint save interval")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output")

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
