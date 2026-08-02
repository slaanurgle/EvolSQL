#!/usr/bin/env python3
"""
EvolSQL Pipeline: Full evolution pipeline for NL2SQL data augmentation.

Python-native replacement for run_pipeline.sh. Provides:
  - Checkpoint-based resume: skip completed, resume interrupted, retry failed
  - Modular step execution via direct function calls (no subprocess overhead)
  - Centralized config management via PipelineConfig dataclass

Pipeline flow:
  1. Inbre-evo (trainset_evolution)
  2. Verify + Fix + Re-verify
  3. Indep-evo round 1..N (direction_proposal + direction_evolution)
  4. Verify + Fix + Re-verify (per round)
  5. Merge all data
  6. Semantic deduplication
  7. Rejection sampling

Usage:
    # Run full pipeline
    python run_pipeline.py \\
        --run_name exp_v1 \\
        --input_file ./data/bird_train.json \\
        --output_dir ./results \\
        --mschema_dir ./schemas/train_mschemas \\
        --mschema_jsonl ./schemas/train_mschemas.jsonl \\
        --db_root_path /path/to/bird \\
        --mode train \\
        --api_urls http://localhost:8001/v1,http://localhost:8002/v1 \\
        --model YOUR_MODEL \\
        --indep_rounds 2

    # Run only up to merge (skip dedup & rejection sampling)
    python run_pipeline.py ... --stop_after merge

    # Run only dedup + rejection_sampling (assumes earlier steps completed)
    python run_pipeline.py ... --steps dedup,rejection_sampling

    # Force re-run from a specific step
    python run_pipeline.py ... --force_from indep1_proposal

Step names (for --stop_after / --steps / --force_from):
    inbre_evo, inbre_verify, inbre_fix, inbre_reverify,
    indep1_proposal, indep1_evo, indep1_verify, indep1_fix, indep1_reverify,
    indep2_proposal, indep2_evo, indep2_verify, indep2_fix, indep2_reverify,
    ...(more rounds if --indep_rounds > 2),
    merge, dedup, rejection_sampling
"""

import sys
import os

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline.config import PipelineConfig
from pipeline.runner import PipelineRunner
from core.utils import setup_logging


def main():
    setup_logging()

    # Parse CLI args into config
    config = PipelineConfig.from_cli()

    # Create and run pipeline
    runner = PipelineRunner(config)
    runner.execute()


if __name__ == "__main__":
    main()
