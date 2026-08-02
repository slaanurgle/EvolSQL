"""
Database executor module for SQL verification.

Provides:
- SQL execution with timeout (thread-based)
- SQL result comparison (execution accuracy)
- Multi-process batch verification
- Database path resolution for BIRD/Spider datasets
"""

import sqlite3
import threading
import multiprocessing
import logging
from pathlib import Path
from typing import Optional, Tuple, Any, List, Dict
from dataclasses import dataclass

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 20
DEFAULT_FETCH_LIMIT = 30
MAX_RESULT_LEN = 300


@dataclass
class SQLResult:
    """Result of a SQL execution."""
    success: bool
    result: Any = None
    error: Optional[str] = None
    timed_out: bool = False

    @property
    def result_str(self) -> str:
        """Get truncated string representation of result."""
        if self.error:
            return self.error
        s = str(self.result)
        if len(s) > MAX_RESULT_LEN:
            s = s[:MAX_RESULT_LEN] + "..."
        return s


def resolve_db_path(db_root: str, db_id: str, mode: str = "train") -> Path:
    """
    Resolve the database file path based on dataset mode.

    Args:
        db_root: Root path to database directory
        db_id: Database identifier
        mode: Dataset mode - "train", "dev", or "spider_train"

    Returns:
        Path to the .sqlite file
    """
    if mode.startswith("spider"):
        return Path(db_root) / db_id / f"{db_id}.sqlite"
    else:
        return Path(db_root) / f"{mode}_databases" / db_id / f"{db_id}.sqlite"


def execute_sql(
    db_path: str,
    sql: str,
    timeout: int = DEFAULT_TIMEOUT,
    fetch_limit: int = DEFAULT_FETCH_LIMIT,
) -> SQLResult:
    """
    Execute a SQL query with timeout protection.

    Uses a separate thread to run the query, with conn.interrupt()
    for clean cancellation on timeout.

    Args:
        db_path: Path to the SQLite database file
        sql: SQL query to execute
        timeout: Maximum execution time in seconds
        fetch_limit: Maximum number of rows to fetch

    Returns:
        SQLResult with success status and result/error
    """
    if not Path(db_path).exists():
        return SQLResult(success=False, error=f"Database not found: {db_path}")

    query_result = {"rows": None, "error": None}

    def _run_query(connection, sql_query):
        try:
            cursor = connection.cursor()
            cursor.execute(sql_query)
            query_result["rows"] = cursor.fetchmany(fetch_limit)
        except Exception as e:
            query_result["error"] = e

    try:
        conn = sqlite3.connect(str(db_path), timeout=5, check_same_thread=False)
        conn.execute(f"PRAGMA busy_timeout = {timeout * 1000}")

        t = threading.Thread(target=_run_query, args=(conn, sql))
        t.start()
        t.join(timeout=timeout)

        if t.is_alive():
            # Timeout: interrupt the connection
            conn.interrupt()
            t.join()
            conn.close()
            return SQLResult(success=False, error="Query execution timed out", timed_out=True)

        conn.close()

        if query_result["error"]:
            return SQLResult(success=False, error=str(query_result["error"]))

        return SQLResult(success=True, result=query_result["rows"])

    except Exception as e:
        return SQLResult(success=False, error=str(e))


def execute_sql_with_timeout(db_path: str, sql: str, timeout: int = DEFAULT_TIMEOUT) -> SQLResult:
    """Alias for execute_sql with default parameters."""
    return execute_sql(db_path, sql, timeout=timeout)


def execute_sql_frozen(db_path: str, sql: str, timeout: int = 10) -> Tuple[Optional[frozenset], bool]:
    """
    Execute SQL and return results as a frozenset for comparison.
    Used in rejection sampling for execution accuracy checking.

    Args:
        db_path: Path to the SQLite database
        sql: SQL query to execute
        timeout: Timeout in seconds

    Returns:
        (frozenset_of_results, success_flag)
    """
    result_holder = {"rows": None, "error": None}

    def _run(connection, query):
        try:
            cursor = connection.cursor()
            connection.execute("BEGIN TRANSACTION;")
            cursor.execute(query)
            result_holder["rows"] = cursor.fetchall()
            connection.rollback()
        except Exception as e:
            result_holder["error"] = e
            try:
                connection.rollback()
            except:
                pass

    try:
        conn = sqlite3.connect(str(db_path), timeout=5, check_same_thread=False)

        t = threading.Thread(target=_run, args=(conn, sql))
        t.start()
        t.join(timeout=timeout)

        if t.is_alive():
            conn.interrupt()
            t.join()
            try:
                conn.close()
            except:
                pass
            return None, False

        conn.close()

        if result_holder["error"]:
            return None, False

        return frozenset(result_holder["rows"]), True

    except Exception:
        return None, False


def verify_sql(
    db_path: str,
    sql: str,
    timeout: int = DEFAULT_TIMEOUT,
) -> Dict[str, Any]:
    """
    Verify a SQL query by executing it and returning structured results.

    Returns:
        Dict with keys: can_compile (bool), ex_result (str)
    """
    result = execute_sql(db_path, sql, timeout=timeout)

    if result.success:
        return {
            "can_compile": True,
            "ex_result": result.result_str,
        }
    else:
        return {
            "can_compile": False,
            "ex_result": result.result_str,
        }


def compare_sql_results(
    db_path: str,
    pred_sql: str,
    gold_sql: str,
    timeout: int = 10,
) -> bool:
    """
    Compare execution results of two SQL queries.

    Returns:
        True if both execute successfully and produce the same result set.
    """
    pred_result, pred_ok = execute_sql_frozen(db_path, pred_sql, timeout)
    gold_result, gold_ok = execute_sql_frozen(db_path, gold_sql, timeout)

    if pred_ok and gold_ok:
        return pred_result == gold_result
    return False


# ===== Multi-process batch verification =====

def _worker_verify(args) -> Tuple[int, Dict[str, Any]]:
    """Worker function for multiprocessing-based batch verification."""
    idx, db_path, sql, timeout = args
    result = verify_sql(db_path, sql, timeout=timeout)
    return idx, result


def batch_verify_sql(
    tasks: List[Tuple[str, str]],  # List of (db_path, sql)
    timeout: int = DEFAULT_TIMEOUT,
    num_workers: int = 64,
    show_progress: bool = True,
) -> List[Dict[str, Any]]:
    """
    Verify multiple SQL queries in parallel using multiprocessing.

    Args:
        tasks: List of (db_path, sql) tuples
        timeout: Timeout per query
        num_workers: Number of worker processes
        show_progress: Whether to show progress bar

    Returns:
        List of verification results in input order
    """
    from tqdm import tqdm

    work_items = [
        (idx, db_path, sql, timeout)
        for idx, (db_path, sql) in enumerate(tasks)
    ]

    results = [None] * len(tasks)

    with multiprocessing.Pool(num_workers) as pool:
        iterator = pool.imap_unordered(_worker_verify, work_items)
        if show_progress:
            iterator = tqdm(iterator, total=len(work_items), desc="Verifying SQL")

        for idx, result in iterator:
            results[idx] = result

    return results


def _worker_compare(args) -> Tuple[int, bool]:
    """Worker function for multiprocessing-based batch comparison."""
    idx, db_path, pred_sql, gold_sql, timeout = args
    try:
        return idx, compare_sql_results(db_path, pred_sql, gold_sql, timeout)
    except Exception:
        return idx, False


def batch_compare_sql(
    tasks: List[Tuple[str, str, str]],  # List of (db_path, pred_sql, gold_sql)
    timeout: int = 10,
    num_workers: int = 20,
    show_progress: bool = True,
) -> List[bool]:
    """
    Compare multiple SQL pairs in parallel using multiprocessing.

    Args:
        tasks: List of (db_path, pred_sql, gold_sql) tuples
        timeout: Timeout per query
        num_workers: Number of worker processes
        show_progress: Whether to show progress bar

    Returns:
        List of boolean results in input order
    """
    from tqdm import tqdm

    work_items = [
        (idx, db_path, pred_sql, gold_sql, timeout)
        for idx, (db_path, pred_sql, gold_sql) in enumerate(tasks)
    ]

    results = [False] * len(tasks)

    with multiprocessing.Pool(num_workers) as pool:
        iterator = pool.imap_unordered(_worker_compare, work_items)
        if show_progress:
            iterator = tqdm(iterator, total=len(work_items), desc="Comparing SQL")

        for idx, result in iterator:
            results[idx] = result

    return results
