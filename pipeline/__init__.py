"""
EvolSQL Pipeline: Python-native pipeline orchestration for NL2SQL data augmentation.

Replaces the bash-based run_pipeline.sh with a modular Python pipeline.
"""

from pipeline.config import PipelineConfig
from pipeline.checkpoint import CheckpointManager
from pipeline.runner import PipelineRunner

__all__ = ["PipelineConfig", "CheckpointManager", "PipelineRunner"]
