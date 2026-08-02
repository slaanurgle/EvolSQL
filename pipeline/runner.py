"""
Pipeline Runner: Core orchestrator with checkpoint-driven step execution.

Manages the full pipeline flow:
  1. Inbre-evo → Verify → Fix → [Inject] → Re-verify
  2. Indep-evo rounds (Proposal → Evo → Verify → Fix → [Inject] → Re-verify) × N
  3. Merge → Semantic Dedup → Rejection Sampling

Inject is placed after Fix so that adversarial data targets the final SQL.
"""

import logging
import os
import traceback
from typing import Callable, List

from pipeline.checkpoint import CheckpointManager
from pipeline.config import PipelineConfig
from pipeline import step_registry as steps

# Suppress httpx INFO-level request logs (very verbose when calling the LLM API)
logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


class PipelineRunner:
    """
    Checkpoint-aware pipeline runner.

    Each step is wrapped with checkpoint logic:
      - completed → skip
      - running   → resume (interrupted last time)
      - failed    → retry
      - pending   → run fresh
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.ckpt = CheckpointManager(config.run_dir)
        self._stopped = False  # Flag for --stop_after

    # =========================================================================
    # Step filtering (--steps, --stop_after)
    # =========================================================================

    def should_run_step(self, step_name: str) -> bool:
        """Check if a step should be run based on --steps filter."""
        if self._stopped:
            return False
        steps_filter = self.config.steps_list
        if not steps_filter:
            return True  # No filter → run all
        return step_name in steps_filter

    def check_stop_after(self, step_name: str):
        """Mark pipeline as stopped if this step matches --stop_after."""
        if self.config.stop_after and step_name == self.config.stop_after:
            self._stopped = True
            print(f"\n  ⏹️  Stopping after step [{step_name}] (--stop_after)")

    # =========================================================================
    # Step execution
    # =========================================================================

    def run_step(self, step_name: str, func: Callable, **kwargs):
        """
        Execute a pipeline step with checkpoint management.

        Args:
            step_name: Unique name for checkpoint tracking.
            func: The step function to call.
            **kwargs: Arguments passed to the step function.
        """
        if not self.should_run_step(step_name):
            print(f"  ⏭️  [{step_name}] Skipped (not in --steps or after --stop_after)")
            return

        status = self.ckpt.get(step_name)

        if status == "completed":
            print(f"  ✅ [{step_name}] Already completed, skipping")
            self.check_stop_after(step_name)
            return

        if status == "running":
            print(f"  🔄 [{step_name}] Was interrupted last time, resuming...")
            kwargs["resume"] = True
        elif status == "failed":
            print(f"  🔁 [{step_name}] Previously failed, retrying...")
        else:
            print(f"  🚀 [{step_name}] Starting...")

        self.ckpt.set(step_name, "running")

        try:
            func(**kwargs)
            self.ckpt.set(step_name, "completed")
            print(f"  ✅ [{step_name}] Completed")
            self.check_stop_after(step_name)
        except Exception as e:
            self.ckpt.set(step_name, "failed")
            print(f"  ❌ [{step_name}] Failed: {e}")
            traceback.print_exc()
            raise

    def run_step_no_resume(self, step_name: str, func: Callable, **kwargs):
        """
        Execute a step that does NOT support --resume (e.g., verify, merge).

        Same checkpoint logic but without injecting resume=True.
        """
        if not self.should_run_step(step_name):
            print(f"  ⏭️  [{step_name}] Skipped (not in --steps or after --stop_after)")
            return

        status = self.ckpt.get(step_name)

        if status == "completed":
            print(f"  ✅ [{step_name}] Already completed, skipping")
            self.check_stop_after(step_name)
            return

        if status == "running":
            print(f"  🔄 [{step_name}] Was interrupted, re-running...")
        elif status == "failed":
            print(f"  🔁 [{step_name}] Previously failed, retrying...")
        else:
            print(f"  🚀 [{step_name}] Starting...")

        self.ckpt.set(step_name, "running")

        try:
            func(**kwargs)
            self.ckpt.set(step_name, "completed")
            print(f"  ✅ [{step_name}] Completed")
            self.check_stop_after(step_name)
        except Exception as e:
            self.ckpt.set(step_name, "failed")
            print(f"  ❌ [{step_name}] Failed: {e}")
            traceback.print_exc()
            raise

    # =========================================================================
    # Steps order & force_from
    # =========================================================================

    def build_all_steps(self) -> List[str]:
        """Build the ordered list of all pipeline step names."""
        step_list = []

        # Inbre: evo → verify → fix → [inject] → re-verify
        step_list.extend(["inbre_evo", "inbre_verify", "inbre_fix"])
        if self.config.enable_db_inject:
            step_list.append("inbre_inject")
        step_list.append("inbre_reverify")

        # Indep rounds: proposal → evo → verify → fix → [inject] → re-verify
        for r in range(1, self.config.indep_rounds + 1):
            step_list.extend([
                f"indep{r}_proposal",
                f"indep{r}_evo",
                f"indep{r}_verify",
                f"indep{r}_fix",
            ])
            if self.config.enable_db_inject:
                step_list.append(f"indep{r}_inject")
            step_list.append(f"indep{r}_reverify")

        step_list.extend(["merge", "dedup", "rejection_sampling"])
        return step_list

    def get_step_output_file(self, step: str) -> str:
        """Map a step name to its expected output file path."""
        run_dir = self.config.run_dir
        mapping = {
            "inbre_evo": "inbre_evo.json",
            "inbre_verify": "inbre_verified.json",
            "inbre_fix": "inbre_fixed.json",
            "inbre_inject": "inbre_injected.json",
            "inbre_reverify": "inbre_final.json",
            "merge": "all_evolved.json",
            "dedup": "all_evolved_dedup.json",
            "rejection_sampling": "final_with_cot.json",
        }

        if step in mapping:
            return os.path.join(run_dir, mapping[step])

        # Dynamic indep round steps
        if step.endswith("_proposal"):
            return os.path.join(run_dir, f"{step}.json")
        if step.endswith("_evo"):
            return os.path.join(run_dir, f"{step}.json")
        if step.endswith("_verify"):
            base = step.replace("_verify", "")
            return os.path.join(run_dir, f"{base}_verified.json")
        if step.endswith("_fix"):
            base = step.replace("_fix", "")
            return os.path.join(run_dir, f"{base}_fixed.json")
        if step.endswith("_inject"):
            base = step.replace("_inject", "")
            return os.path.join(run_dir, f"{base}_injected.json")
        if step.endswith("_reverify"):
            base = step.replace("_reverify", "")
            return os.path.join(run_dir, f"{base}_final.json")

        return ""

    def handle_force_from(self):
        """Handle --force_from: reset steps and delete output files."""
        if not self.config.force_from:
            return

        all_steps = self.build_all_steps()
        print(f"Force re-run from step: {self.config.force_from}")

        reset_steps = self.ckpt.force_from(self.config.force_from, all_steps)

        # Delete output files for reset steps
        for step in reset_steps:
            output_file = self.get_step_output_file(step)
            if output_file and os.path.isfile(output_file):
                print(f"  Removing: {output_file}")
                os.remove(output_file)

        print(f"Reset {len(reset_steps)} steps to pending.\n")

    # =========================================================================
    # Main execution
    # =========================================================================

    def execute(self):
        """Execute the full pipeline."""
        cfg = self.config
        run_dir = cfg.run_dir

        # Ensure run directory exists
        os.makedirs(run_dir, exist_ok=True)

        # Validate --steps and --stop_after step names
        all_steps = self.build_all_steps()
        if cfg.stop_after and cfg.stop_after not in all_steps:
            print(f"Error: --stop_after '{cfg.stop_after}' is not a valid step name.")
            print(f"Valid steps: {', '.join(all_steps)}")
            return
        if cfg.steps_list:
            invalid = [s for s in cfg.steps_list if s not in all_steps]
            if invalid:
                print(f"Error: --steps contains invalid step names: {', '.join(invalid)}")
                print(f"Valid steps: {', '.join(all_steps)}")
                return

        # Handle --force_from
        self.handle_force_from()

        # Display config and checkpoint status
        cfg.display()
        print()
        self.ckpt.show()
        print()

        # Common LLM kwargs for evolution steps
        llm_kwargs = dict(
            api_urls=cfg.api_urls,
            model=cfg.model,
            api_key=cfg.api_key,
            max_workers=cfg.max_workers,
        )

        # =====================================================================
        # Step 1: Inbre-evo (Trainset Evolution)
        # =====================================================================
        print("\n===== Step 1: Inbre-evo (Trainset Evolution) =====")

        self.run_step(
            "inbre_evo",
            steps.run_trainset_evolution,
            input_file=cfg.input_file,
            output_file=os.path.join(run_dir, "inbre_evo.json"),
            mschema_dir=cfg.mschema_dir,
            sampling_count=cfg.sampling_count,
            use_full_schema=True,
            **llm_kwargs,
        )
        if self._stopped:
            self._print_done(run_dir)
            return

        # ----- Inbre: Verify + Fix + Re-verify -----
        print("\n>>> [inbre] Verify + Fix + Re-verify")

        self.run_step_no_resume(
            "inbre_verify",
            steps.run_evolution_verify,
            input_file=os.path.join(run_dir, "inbre_evo.json"),
            output_file=os.path.join(run_dir, "inbre_verified.json"),
            db_root_path=cfg.db_root_path,
            mode=cfg.mode,
            num_workers=cfg.num_verify_workers,
        )
        if self._stopped:
            self._print_done(run_dir)
            return

        self.run_step(
            "inbre_fix",
            steps.run_evolution_fixer,
            input_file=os.path.join(run_dir, "inbre_verified.json"),
            output_file=os.path.join(run_dir, "inbre_fixed.json"),
            mschema_dir=cfg.mschema_dir,
            fix_all=True,
            **llm_kwargs,
        )
        if self._stopped:
            self._print_done(run_dir)
            return

        # ----- Inbre: Inject (optional, after fix, before re-verify) -----
        if cfg.enable_db_inject:
            self.run_step(
                "inbre_inject",
                steps.run_db_inject,
                input_file=os.path.join(run_dir, "inbre_fixed.json"),
                output_file=os.path.join(run_dir, "inbre_injected.json"),
                db_root_path=cfg.db_root_path,
                augmented_db_dir=cfg.augmented_db_path,
                mode=cfg.mode,
                mschema_file=cfg.mschema_jsonl,
                pk_budget=cfg.db_inject_budget,
                max_retries=cfg.db_inject_max_retries,
                **llm_kwargs,
            )
            if self._stopped:
                self._print_done(run_dir)
                return

        # Determine input for re-verify (injected if inject ran, else fixed)
        inbre_reverify_input = os.path.join(
            run_dir, "inbre_injected.json" if cfg.enable_db_inject else "inbre_fixed.json"
        )

        # Re-verify uses augmented DB (with injected data) when inject is enabled
        reverify_db_path = cfg.augmented_db_path if cfg.enable_db_inject else cfg.db_root_path

        self.run_step_no_resume(
            "inbre_reverify",
            steps.run_evolution_verify,
            input_file=inbre_reverify_input,
            output_file=os.path.join(run_dir, "inbre_final.json"),
            db_root_path=reverify_db_path,
            mode=cfg.mode,
            num_workers=cfg.num_verify_workers,
        )
        if self._stopped:
            self._print_done(run_dir)
            return

        current_data = os.path.join(run_dir, "inbre_final.json")

        # =====================================================================
        # Step 2+: Indep-evo rounds
        # =====================================================================
        for round_num in range(1, cfg.indep_rounds + 1):
            print(f"\n===== Step {round_num + 1}: Indep-evo Round {round_num} =====")

            # Direction Proposal
            self.run_step(
                f"indep{round_num}_proposal",
                steps.run_direction_proposal,
                input_file=current_data,
                output_file=os.path.join(run_dir, f"indep{round_num}_proposal.json"),
                mschema_dir=cfg.mschema_dir,
                top_k=cfg.top_k,
                **llm_kwargs,
            )
            if self._stopped:
                self._print_done(run_dir)
                return

            # Direction Evolution
            self.run_step(
                f"indep{round_num}_evo",
                steps.run_direction_evolution,
                input_file=os.path.join(run_dir, f"indep{round_num}_proposal.json"),
                output_file=os.path.join(run_dir, f"indep{round_num}_evo.json"),
                mschema_dir=cfg.mschema_dir,
                use_full_schema=True,
                **llm_kwargs,
            )
            if self._stopped:
                self._print_done(run_dir)
                return

            # Verify + Fix + Re-verify
            print(f"\n>>> [indep{round_num}] Verify + Fix + Re-verify")

            self.run_step_no_resume(
                f"indep{round_num}_verify",
                steps.run_evolution_verify,
                input_file=os.path.join(run_dir, f"indep{round_num}_evo.json"),
                output_file=os.path.join(run_dir, f"indep{round_num}_verified.json"),
                db_root_path=cfg.db_root_path,
                mode=cfg.mode,
                num_workers=cfg.num_verify_workers,
            )
            if self._stopped:
                self._print_done(run_dir)
                return

            self.run_step(
                f"indep{round_num}_fix",
                steps.run_evolution_fixer,
                input_file=os.path.join(run_dir, f"indep{round_num}_verified.json"),
                output_file=os.path.join(run_dir, f"indep{round_num}_fixed.json"),
                mschema_dir=cfg.mschema_dir,
                fix_all=True,
                **llm_kwargs,
            )
            if self._stopped:
                self._print_done(run_dir)
                return

            # ----- Indep: Inject (optional, after fix, before re-verify) -----
            if cfg.enable_db_inject:
                self.run_step(
                    f"indep{round_num}_inject",
                    steps.run_db_inject,
                    input_file=os.path.join(run_dir, f"indep{round_num}_fixed.json"),
                    output_file=os.path.join(run_dir, f"indep{round_num}_injected.json"),
                    db_root_path=cfg.db_root_path,
                    augmented_db_dir=cfg.augmented_db_path,
                    mode=cfg.mode,
                    mschema_file=cfg.mschema_jsonl,
                    pk_budget=cfg.db_inject_budget,
                    max_retries=cfg.db_inject_max_retries,
                    **llm_kwargs,
                )
                if self._stopped:
                    self._print_done(run_dir)
                    return

            # Determine input for re-verify
            indep_reverify_input = os.path.join(
                run_dir,
                f"indep{round_num}_injected.json" if cfg.enable_db_inject else f"indep{round_num}_fixed.json"
            )

            self.run_step_no_resume(
                f"indep{round_num}_reverify",
                steps.run_evolution_verify,
                input_file=indep_reverify_input,
                output_file=os.path.join(run_dir, f"indep{round_num}_final.json"),
                db_root_path=reverify_db_path,
                mode=cfg.mode,
                num_workers=cfg.num_verify_workers,
            )
            if self._stopped:
                self._print_done(run_dir)
                return

            current_data = os.path.join(run_dir, f"indep{round_num}_final.json")

        # =====================================================================
        # Step: Merge all data
        # =====================================================================
        print("\n===== Merge Data =====")

        merge_files = [os.path.join(run_dir, "inbre_final.json")]
        for r in range(1, cfg.indep_rounds + 1):
            merge_files.append(os.path.join(run_dir, f"indep{r}_final.json"))

        self.run_step_no_resume(
            "merge",
            steps.run_merge_data,
            input_files=merge_files,
            output_file=os.path.join(run_dir, "all_evolved.json"),
            filter_compilable=True,
            filter_empty_result=True,
        )
        if self._stopped:
            self._print_done(run_dir)
            return

        # =====================================================================
        # Step: Semantic Deduplication
        # =====================================================================
        print("\n===== Semantic Deduplication =====")

        dedup_kwargs = dict(
            input_file=os.path.join(run_dir, "all_evolved.json"),
            output_file=os.path.join(run_dir, "all_evolved_dedup.json"),
            threshold=cfg.dedup_threshold,
        )
        if cfg.sentence_model:
            dedup_kwargs["model_name"] = cfg.sentence_model

        self.run_step_no_resume("dedup", steps.run_semantic_dedup, **dedup_kwargs)
        if self._stopped:
            self._print_done(run_dir)
            return

        # =====================================================================
        # Step: Rejection Sampling
        # =====================================================================
        print("\n===== Rejection Sampling =====")

        if not cfg.mschema_jsonl:
            print("Warning: --mschema_jsonl not set. Skipping rejection sampling.")
            print("You can run it separately with pipeline/steps/rejection_sampling.py")
            self.ckpt.set("rejection_sampling", "completed")
        else:
            self.run_step(
                "rejection_sampling",
                steps.run_rejection_sampling,
                input_file=os.path.join(run_dir, "all_evolved_dedup.json"),
                output_file=os.path.join(run_dir, "final_with_cot.json"),
                mschema_file=cfg.mschema_jsonl,
                db_path=cfg.db_verify_path,
                api_urls=cfg.rs_api_urls or cfg.api_urls,
                model=cfg.rs_model_name,
                api_key=cfg.api_key,
                max_samples=cfg.max_samples,
                batch_size=cfg.batch_size,
                num_cpus=cfg.num_cpus,
                max_workers=cfg.max_workers,
            )

        # =====================================================================
        # Done
        # =====================================================================
        self._print_done(run_dir)

    def _print_done(self, run_dir: str):
        """Print final summary."""
        print()
        print("=" * 60)
        print("Pipeline Complete!")
        print("=" * 60)
        self.ckpt.show()
        print()
        print(f"Results in: {run_dir}")
        print("=" * 60)
