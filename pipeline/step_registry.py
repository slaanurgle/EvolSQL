"""
Pipeline Step Registry: Thin wrappers that call each step module's run() function.

Each step function accepts a PipelineConfig (or relevant kwargs) and
calls the corresponding step module's run() entry point directly in-process.
This avoids subprocess overhead and enables shared objects.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List

# Ensure the project root is on sys.path for template resolution etc.
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


def _make_namespace(**kwargs) -> argparse.Namespace:
    """Create an argparse.Namespace from keyword arguments."""
    return argparse.Namespace(**kwargs)


# =============================================================================
# Step: Database Injection (Adversarial Data Augmentation)
# =============================================================================

def run_db_inject(
    input_file: str,
    output_file: str,
    db_root_path: str,
    augmented_db_dir: str,
    api_urls: str,
    model: str,
    api_key: str = "EMPTY",
    mode: str = "train",
    mschema_file: str = "",
    pk_budget: int = 10,
    max_retries: int = 3,
    max_workers: int = 32,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    resume: bool = False,
    template_file: str = None,
):
    """Run the database injection step."""
    from pipeline.steps.db_inject import run

    if template_file is None:
        template_file = str(Path(_project_root) / "templates" / "template_db_inject.txt")

    args = _make_namespace(
        input_file=input_file,
        output_file=output_file,
        db_root_path=db_root_path,
        augmented_db_dir=augmented_db_dir,
        api_urls=api_urls,
        model=model,
        api_key=api_key,
        mode=mode,
        mschema_file=mschema_file,
        pk_budget=pk_budget,
        max_retries=max_retries,
        max_workers=max_workers,
        temperature=temperature,
        max_tokens=max_tokens,
        resume=resume,
        template_file=template_file,
    )
    run(args)


# =============================================================================
# Step 1: Inbre-evo (Trainset Evolution)
# =============================================================================

def run_trainset_evolution(
    input_file: str,
    output_file: str,
    mschema_dir: str,
    api_urls: str,
    model: str,
    api_key: str = "EMPTY",
    sampling_count: int = 3,
    max_workers: int = 32,
    use_full_schema: bool = True,
    update_evidence: bool = False,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    resume: bool = False,
    template_file: str = None,
    save_interval: int = 500,
):
    """Run the trainset evolution (inbre-evo) step."""
    from pipeline.steps.trainset_evolution import run

    if template_file is None:
        template_file = str(Path(_project_root) / "templates" / "template_trainset_evolution.txt")

    args = _make_namespace(
        input_file=input_file,
        output_file=output_file,
        mschema_dir=mschema_dir,
        api_urls=api_urls,
        model=model,
        api_key=api_key,
        sampling_count=sampling_count,
        max_workers=max_workers,
        use_full_schema=use_full_schema,
        update_evidence=update_evidence,
        temperature=temperature,
        max_tokens=max_tokens,
        resume=resume,
        template_file=template_file,
        save_interval=save_interval,
    )
    run(args)


# =============================================================================
# Step: Evolution Verify
# =============================================================================

def run_evolution_verify(
    input_file: str,
    output_file: str,
    db_root_path: str,
    mode: str = "train",
    timeout: int = 20,
    num_workers: int = 64,
):
    """Run the SQL verification step."""
    from pipeline.steps.evolution_verify import run

    args = _make_namespace(
        input_file=input_file,
        output_file=output_file,
        db_root_path=db_root_path,
        mode=mode,
        timeout=timeout,
        num_workers=num_workers,
    )
    run(args)


# =============================================================================
# Step: Evolution Fixer
# =============================================================================

def run_evolution_fixer(
    input_file: str,
    output_file: str,
    mschema_dir: str,
    api_urls: str,
    model: str,
    api_key: str = "EMPTY",
    max_workers: int = 32,
    use_full_schema: bool = False,
    fix_all: bool = False,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    resume: bool = False,
    template_file: str = None,
):
    """Run the evolution fixer step."""
    from pipeline.steps.evolution_fixer import run

    if template_file is None:
        template_file = str(Path(_project_root) / "templates" / "template_evolution_fixer.txt")

    args = _make_namespace(
        input_file=input_file,
        output_file=output_file,
        mschema_dir=mschema_dir,
        api_urls=api_urls,
        model=model,
        api_key=api_key,
        max_workers=max_workers,
        use_full_schema=use_full_schema,
        fix_all=fix_all,
        temperature=temperature,
        max_tokens=max_tokens,
        resume=resume,
        template_file=template_file,
    )
    run(args)


# =============================================================================
# Step: Direction Proposal
# =============================================================================

def run_direction_proposal(
    input_file: str,
    output_file: str,
    mschema_dir: str,
    api_urls: str,
    model: str,
    api_key: str = "EMPTY",
    top_k: int = 2,
    max_workers: int = 32,
    use_full_schema: bool = False,
    max_retries: int = 3,
    filter_compilable: bool = True,
    filter_empty_result: bool = True,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    resume: bool = False,
    template_file: str = None,
):
    """Run the direction proposal step."""
    from pipeline.steps.direction_proposal import run

    if template_file is None:
        template_file = str(Path(_project_root) / "templates" / "template_direction_proposal.txt")

    args = _make_namespace(
        input_file=input_file,
        output_file=output_file,
        mschema_dir=mschema_dir,
        api_urls=api_urls,
        model=model,
        api_key=api_key,
        top_k=top_k,
        max_workers=max_workers,
        use_full_schema=use_full_schema,
        max_retries=max_retries,
        filter_compilable=filter_compilable,
        filter_empty_result=filter_empty_result,
        temperature=temperature,
        max_tokens=max_tokens,
        resume=resume,
        template_file=template_file,
    )
    run(args)


# =============================================================================
# Step: Direction Evolution
# =============================================================================

def run_direction_evolution(
    input_file: str,
    output_file: str,
    mschema_dir: str,
    api_urls: str,
    model: str,
    api_key: str = "EMPTY",
    max_workers: int = 32,
    use_full_schema: bool = True,
    update_evidence: bool = False,
    use_proposals: bool = True,
    sample_num: int = -1,
    filter_compilable: bool = True,
    filter_empty_result: bool = True,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    resume: bool = False,
    template_file: str = None,
):
    """Run the direction evolution (indep-evo) step."""
    from pipeline.steps.direction_evolution import run

    if template_file is None:
        template_file = str(Path(_project_root) / "templates" / "template_direction_evolution.txt")

    args = _make_namespace(
        input_file=input_file,
        output_file=output_file,
        mschema_dir=mschema_dir,
        api_urls=api_urls,
        model=model,
        api_key=api_key,
        max_workers=max_workers,
        use_full_schema=use_full_schema,
        update_evidence=update_evidence,
        use_proposals=use_proposals,
        sample_num=sample_num,
        filter_compilable=filter_compilable,
        filter_empty_result=filter_empty_result,
        temperature=temperature,
        max_tokens=max_tokens,
        resume=resume,
        template_file=template_file,
    )
    run(args)


# =============================================================================
# Step: Merge Data
# =============================================================================

def run_merge_data(
    input_files: List[str],
    output_file: str,
    filter_compilable: bool = True,
    filter_empty_result: bool = True,
):
    """Run the merge data step."""
    from pipeline.steps.merge_data import run

    args = _make_namespace(
        input_files=input_files,
        output_file=output_file,
        filter_compilable=filter_compilable,
        filter_empty_result=filter_empty_result,
    )
    run(args)


# =============================================================================
# Step: Semantic Deduplication
# =============================================================================

def run_semantic_dedup(
    input_file: str,
    output_file: str,
    threshold: float = 0.90,
    batch_size: int = 1024,
    model_name: str = "all-mpnet-base-v2",
    num_examples: int = 5,
):
    """Run the semantic deduplication step."""
    from pipeline.steps.semantic_dedup import run

    args = _make_namespace(
        input_file=input_file,
        output_file=output_file,
        threshold=threshold,
        batch_size=batch_size,
        model_name=model_name,
        num_examples=num_examples,
    )
    run(args)


# =============================================================================
# Step: Rejection Sampling
# =============================================================================

def run_rejection_sampling(
    input_file: str,
    output_file: str,
    mschema_file: str,
    db_path: str,
    api_urls: str,
    model: str,
    api_key: str = "EMPTY",
    max_samples: int = 4,
    batch_size: int = 64,
    timeout: int = 10,
    num_cpus: int = 20,
    max_workers: int = 32,
    save_interval: int = 100,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    resume: bool = False,
    template_file: str = None,
):
    """Run the rejection sampling step."""
    from pipeline.steps.rejection_sampling import run

    if template_file is None:
        template_file = str(Path(_project_root) / "templates" / "template_cot_rejection_sampling.txt")

    args = _make_namespace(
        input_file=input_file,
        output_file=output_file,
        mschema_file=mschema_file,
        db_path=db_path,
        api_urls=api_urls,
        model=model,
        api_key=api_key,
        max_samples=max_samples,
        batch_size=batch_size,
        timeout=timeout,
        num_cpus=num_cpus,
        max_workers=max_workers,
        save_interval=save_interval,
        temperature=temperature,
        max_tokens=max_tokens,
        resume=resume,
        template_file=template_file,
    )
    run(args)
