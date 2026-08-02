"""
Checkpoint Manager: Track pipeline step status for resume support.

Class-based manager that holds checkpoint data in memory and syncs to disk.
"""

import json
import os
from datetime import datetime
from typing import List, Optional


CHECKPOINT_FILE = "checkpoint.json"

# Status icons for display
STATUS_ICONS = {
    "pending": "⏳",
    "running": "🔄",
    "completed": "✅",
    "failed": "❌",
}

VALID_STATUSES = {"pending", "running", "completed", "failed"}


class CheckpointManager:
    """
    Manages pipeline step checkpoint state.

    Holds checkpoint data in memory and persists to checkpoint.json.
    Eliminates the need for subprocess calls to checkpoint.py.
    """

    def __init__(self, run_dir: str):
        self.run_dir = run_dir
        self._path = os.path.join(run_dir, CHECKPOINT_FILE)
        self._data = self._load()

    def _load(self) -> dict:
        """Load checkpoint from disk, or return default if not found."""
        if os.path.exists(self._path):
            with open(self._path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"steps": {}, "current_step": None, "last_updated": None}

    def _save(self):
        """Persist checkpoint to disk."""
        os.makedirs(self.run_dir, exist_ok=True)
        self._data["last_updated"] = datetime.now().isoformat()
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2, ensure_ascii=False)

    def get(self, step: str) -> str:
        """Get the status of a step. Returns 'pending' if not recorded."""
        return self._data["steps"].get(step, "pending")

    def set(self, step: str, status: str):
        """Set the status of a step and persist."""
        if status not in VALID_STATUSES:
            raise ValueError(f"Invalid status: {status}. Must be one of: {VALID_STATUSES}")

        self._data["steps"][step] = status

        if status == "running":
            self._data["current_step"] = step
        elif status in ("completed", "failed"):
            if self._data.get("current_step") == step:
                self._data["current_step"] = None

        self._save()

    def force_from(self, target_step: str, all_steps: List[str]) -> List[str]:
        """
        Reset target_step and all subsequent steps to 'pending'.

        Args:
            target_step: Step name to start resetting from.
            all_steps: Ordered list of all step names.

        Returns:
            List of step names that were reset.

        Raises:
            ValueError: If target_step is not found in all_steps.
        """
        if target_step not in all_steps:
            raise ValueError(
                f"Step '{target_step}' not found in steps order. "
                f"Available steps: {', '.join(all_steps)}"
            )

        target_idx = all_steps.index(target_step)
        reset_steps = all_steps[target_idx:]

        for step in reset_steps:
            if step in self._data["steps"]:
                self._data["steps"][step] = "pending"

        self._data["current_step"] = None
        self._save()

        return reset_steps

    def show(self):
        """Display all step statuses to stdout."""
        print(f"Checkpoint: {self._path}")
        print(f"Last updated: {self._data.get('last_updated', 'N/A')}")
        print(f"Current step: {self._data.get('current_step', 'N/A')}")
        print()

        if not self._data["steps"]:
            print("  (no steps recorded)")
            return

        for step, status in self._data["steps"].items():
            icon = STATUS_ICONS.get(status, "❓")
            print(f"  {icon} {step}: {status}")

    def is_completed(self, step: str) -> bool:
        """Check if a step is already completed."""
        return self.get(step) == "completed"

    def is_running(self, step: str) -> bool:
        """Check if a step was left in running state (interrupted)."""
        return self.get(step) == "running"
