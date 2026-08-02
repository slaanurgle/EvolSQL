"""
Pipeline Configuration: Centralized config management via dataclass.

Supports loading from CLI (argparse) or programmatic construction.
"""

import argparse
import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PipelineConfig:
    """All configuration for the EvolSQL pipeline."""

    # === Required paths ===
    run_name: str = ""
    input_file: str = ""
    output_dir: str = ""
    mschema_dir: str = ""
    db_root_path: str = ""

    # === LLM settings ===
    api_urls: str = ""          # Comma-separated vLLM API URLs
    model: str = ""
    api_key: str = "EMPTY"

    # === Optional paths ===
    mschema_jsonl: str = ""     # For rejection sampling
    sentence_model: str = ""    # SentenceTransformer model for dedup

    # === Dataset mode ===
    mode: str = "train"         # train / dev / spider_train

    # === Evolution settings ===
    sampling_count: int = 3     # Inbre-evo breadth
    top_k: int = 2              # Direction proposal top-k
    indep_rounds: int = 2       # Number of indep-evo rounds

    # === Parallelism ===
    max_workers: int = 32       # LLM API concurrency
    num_verify_workers: int = 64  # SQL verification workers
    num_cpus: int = 20          # Rejection sampling CPU workers

    # === Rejection sampling ===
    max_samples: int = 4        # Max sampling rounds
    batch_size: int = 64        # Inference batch size
    dedup_threshold: float = 0.90  # Semantic dedup threshold

    # === DB Injection settings ===
    enable_db_inject: bool = False      # Whether to enable DB injection step
    db_inject_budget: int = 10          # PK range budget per SQL
    db_inject_max_retries: int = 3      # Max LLM retries per injection
    augmented_db_dir: str = ""          # Dir for augmented DBs (default: run_dir/augmented_dbs)

    # === Rejection sampling LLM (optional, defaults to main LLM) ===
    rs_api_urls: str = ""
    rs_model: str = ""

    # === Pipeline control ===
    force_from: str = ""        # Force re-run from this step
    stop_after: str = ""        # Stop after this step completes
    steps: str = ""             # Comma-separated step names to run (subset mode)

    # === Derived properties ===
    @property
    def run_dir(self) -> str:
        """Full path to the run output directory."""
        return os.path.join(self.output_dir, self.run_name)

    @property
    def api_url_list(self) -> list:
        """Parse comma-separated API URLs into a list."""
        return [u.strip() for u in self.api_urls.split(",") if u.strip()]

    @property
    def rs_api_url_list(self) -> list:
        """Parse rejection sampling API URLs (falls back to main API URLs)."""
        urls = self.rs_api_urls or self.api_urls
        return [u.strip() for u in urls.split(",") if u.strip()]

    @property
    def rs_model_name(self) -> str:
        """Rejection sampling model name (falls back to main model)."""
        return self.rs_model or self.model

    @property
    def steps_list(self) -> list:
        """Parse comma-separated steps into a list (empty = run all)."""
        if not self.steps:
            return []
        return [s.strip() for s in self.steps.split(",") if s.strip()]

    @property
    def db_verify_path(self) -> str:
        """Database path used for SQL verification."""
        # If DB injection is enabled and augmented_db_dir exists, use it
        if self.enable_db_inject:
            aug_dir = self.augmented_db_dir or os.path.join(self.run_dir, "augmented_dbs")
            if os.path.isdir(aug_dir):
                return aug_dir
        if self.mode in ("train", "dev"):
            return os.path.join(self.db_root_path, f"{self.mode}_databases")
        return self.db_root_path

    @property
    def augmented_db_path(self) -> str:
        """Path to augmented database directory."""
        return self.augmented_db_dir or os.path.join(self.run_dir, "augmented_dbs")

    def validate(self):
        """Validate that all required fields are set."""
        required = {
            "run_name": self.run_name,
            "input_file": self.input_file,
            "output_dir": self.output_dir,
            "mschema_dir": self.mschema_dir,
            "db_root_path": self.db_root_path,
            "api_urls": self.api_urls,
            "model": self.model,
        }
        missing = [k for k, v in required.items() if not v]
        if missing:
            raise ValueError(f"Missing required config fields: {', '.join(missing)}")

    @classmethod
    def from_cli(cls) -> "PipelineConfig":
        """Parse CLI arguments and return a PipelineConfig instance."""
        parser = argparse.ArgumentParser(
            description="EvolSQL Pipeline: Full evolution pipeline for NL2SQL data augmentation",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Example:
  python run_pipeline.py \\
      --run_name exp_v1 \\
      --input_file ./data/bird_train.json \\
      --output_dir ./results \\
      --mschema_dir ./schemas/train_mschemas \\
      --mschema_jsonl ./schemas/train_mschemas.jsonl \\
      --db_root_path /path/to/bird \\
      --mode train \\
      --api_urls http://localhost:8001/v1,http://localhost:8002/v1 \\
      --model YOUR_MODEL
            """,
        )

        # Required
        parser.add_argument("--run_name", type=str, required=True,
                            help="Run name (outputs go to OUTPUT_DIR/RUN_NAME/)")
        parser.add_argument("--input_file", type=str, required=True,
                            help="Input training data JSON")
        parser.add_argument("--output_dir", type=str, required=True,
                            help="Base output directory")
        parser.add_argument("--mschema_dir", type=str, required=True,
                            help="Directory with mschema JSON files")
        parser.add_argument("--db_root_path", type=str, required=True,
                            help="Root path to database files")
        parser.add_argument("--api_urls", type=str, required=True,
                            help="Comma-separated vLLM API URLs")
        parser.add_argument("--model", type=str, required=True,
                            help="Model name")

        # Optional
        parser.add_argument("--api_key", type=str, default="EMPTY")
        parser.add_argument("--mschema_jsonl", type=str, default="",
                            help="MSchema JSONL file (for rejection sampling)")
        parser.add_argument("--sentence_model", type=str, default="",
                            help="SentenceTransformer model path (for dedup)")
        parser.add_argument("--mode", type=str, default="train",
                            choices=["train", "dev", "spider_train"],
                            help="Dataset mode (default: train)")

        # Evolution settings
        parser.add_argument("--sampling_count", type=int, default=3,
                            help="Inbre-evo breadth (default: 3)")
        parser.add_argument("--top_k", type=int, default=2,
                            help="Direction proposal top-k (default: 2)")
        parser.add_argument("--indep_rounds", type=int, default=2,
                            help="Indep-evo iterations (default: 2)")

        # Parallelism
        parser.add_argument("--max_workers", type=int, default=32,
                            help="LLM API concurrency (default: 32)")
        parser.add_argument("--num_verify_workers", type=int, default=64,
                            help="SQL verification workers (default: 64)")
        parser.add_argument("--num_cpus", type=int, default=20,
                            help="Rejection sampling CPU workers (default: 20)")

        # Rejection sampling
        parser.add_argument("--max_samples", type=int, default=4,
                            help="Max rejection sampling rounds (default: 4)")
        parser.add_argument("--batch_size", type=int, default=64,
                            help="Rejection sampling batch size (default: 64)")
        parser.add_argument("--dedup_threshold", type=float, default=0.90,
                            help="Semantic dedup threshold (default: 0.90)")

        # Rejection sampling LLM
        parser.add_argument("--rs_api_urls", type=str, default="",
                            help="Rejection sampling API URLs (default: same as --api_urls)")
        parser.add_argument("--rs_model", type=str, default="",
                            help="Rejection sampling model (default: same as --model)")

        # DB Injection
        parser.add_argument("--enable_db_inject", action="store_true", default=False,
                            help="Enable DB injection step (inject adversarial rows before evolution)")
        parser.add_argument("--db_inject_budget", type=int, default=10,
                            help="PK range budget per SQL for injection (default: 10)")
        parser.add_argument("--db_inject_max_retries", type=int, default=3,
                            help="Max LLM retries per injection (default: 3)")
        parser.add_argument("--augmented_db_dir", type=str, default="",
                            help="Directory for augmented DB copies (default: RUN_DIR/augmented_dbs)")

        # Pipeline control
        parser.add_argument("--force_from", type=str, default="",
                            help="Force re-run from this step (resets it and all subsequent)")
        parser.add_argument("--stop_after", type=str, default="",
                            help="Stop pipeline after this step completes (e.g., --stop_after merge)")
        parser.add_argument("--steps", type=str, default="",
                            help="Comma-separated step names to run (only these steps). "
                                 "E.g., --steps dedup,rejection_sampling")

        args = parser.parse_args()
        config = cls(**vars(args))
        config.validate()
        return config

    def display(self):
        """Print a formatted config summary."""
        print("=" * 60)
        print("EvolSQL Pipeline Configuration")
        print("=" * 60)
        print(f"  Run name:       {self.run_name}")
        print(f"  Run dir:        {self.run_dir}")
        print(f"  Input:          {self.input_file}")
        print(f"  MSchema dir:    {self.mschema_dir}")
        print(f"  DB root:        {self.db_root_path}")
        print(f"  Mode:           {self.mode}")
        print(f"  API URLs:       {self.api_urls}")
        print(f"  Model:          {self.model}")
        print(f"  Sampling count: {self.sampling_count}")
        print(f"  Top-k:          {self.top_k}")
        print(f"  Indep rounds:   {self.indep_rounds}")
        print(f"  Max workers:    {self.max_workers}")
        if self.mschema_jsonl:
            print(f"  MSchema JSONL:  {self.mschema_jsonl}")
        if self.enable_db_inject:
            print(f"  DB Injection:   ENABLED (budget={self.db_inject_budget}, retries={self.db_inject_max_retries})")
            print(f"  Augmented DBs:  {self.augmented_db_path}")
        if self.rs_api_urls:
            print(f"  RS API URLs:    {self.rs_api_urls}")
            print(f"  RS Model:       {self.rs_model_name}")
        if self.force_from:
            print(f"  Force from:     {self.force_from}")
        if self.stop_after:
            print(f"  Stop after:     {self.stop_after}")
        if self.steps:
            print(f"  Steps:          {self.steps}")
        print("=" * 60)
