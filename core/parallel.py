"""
Parallel processing utilities.

Provides:
- ThreadPoolExecutor-based parallelism for LLM calls
- Progress bar integration (tqdm)
- Error handling with optional retry
"""

import logging
from typing import List, Callable, Optional, TypeVar, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")


def parallel_call(
    items: List[T],
    process_func: Callable[[T], R],
    max_workers: int = 32,
    description: str = "Processing",
    show_progress: bool = True,
) -> List[Optional[R]]:
    """
    Process items in parallel with progress tracking.
    Maintains order of results matching input order.

    Args:
        items: List of items to process
        process_func: Function to apply to each item
        max_workers: Maximum number of parallel workers
        description: Progress bar description
        show_progress: Whether to show progress bar

    Returns:
        List of results in the same order as inputs.
        Failed items will have None as result.
    """
    if not items:
        return []

    results: List[Optional[R]] = [None] * len(items)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(process_func, item): idx
            for idx, item in enumerate(items)
        }

        pbar = tqdm(total=len(items), desc=description, disable=not show_progress)
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                logger.error(f"Error processing item {idx}: {e}")
                results[idx] = None
            pbar.update(1)
        pbar.close()

    return results


def parallel_call_with_retry(
    items: List[T],
    process_func: Callable[[T], R],
    max_workers: int = 32,
    max_retries: int = 3,
    description: str = "Processing",
    show_progress: bool = True,
) -> Tuple[List[R], List[Tuple[int, T, Exception]]]:
    """
    Process items in parallel with automatic retry for failures.

    Args:
        items: List of items to process
        process_func: Function to apply to each item
        max_workers: Maximum number of parallel workers
        max_retries: Maximum retries for failed items
        description: Progress bar description
        show_progress: Whether to show progress bar

    Returns:
        Tuple of (successful_results_dict, failed_items_list)
        - successful_results_dict: {original_index: result}
        - failed_items_list: [(original_index, item, exception)]
    """
    if not items:
        return {}, []

    results: dict = {}
    pending_indices = list(range(len(items)))
    failures: List[Tuple[int, T, Exception]] = []

    for attempt in range(max_retries):
        if not pending_indices:
            break

        current_items = [(idx, items[idx]) for idx in pending_indices]
        new_pending = []

        desc = f"{description} (attempt {attempt + 1}/{max_retries})" if attempt > 0 else description

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(process_func, item): idx
                for idx, item in current_items
            }

            pbar = tqdm(total=len(current_items), desc=desc, disable=not show_progress)
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    result = future.result()
                    results[idx] = result
                except Exception as e:
                    if attempt < max_retries - 1:
                        new_pending.append(idx)
                        logger.warning(f"Item {idx} failed, will retry: {e}")
                    else:
                        failures.append((idx, items[idx], e))
                        logger.error(f"Item {idx} failed after {max_retries} attempts: {e}")
                pbar.update(1)
            pbar.close()

        pending_indices = new_pending

    return results, failures
