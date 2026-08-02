"""
Schema utilities for MSchema loading and SQL-based schema extraction.

Provides:
- Load full mschema from JSON files
- Extract referenced tables/columns from SQL
- Build filtered mschema for a specific query
- Get mschema string in the standard format

This module wraps the MSchema class and schema extraction helpers from core.mschema.
"""

import sys
import os
import json
import logging
from pathlib import Path
from typing import Set, Tuple, List, Dict, Optional

logger = logging.getLogger(__name__)

from core.mschema import MSchema
from core.mschema import (
    load_mschema,
    extract_references,
    build_query_mschema,
    set_mschema_dir,
    set_mschema_mode,
)


def get_full_schema_str(mschema_dir: str, db_id: str) -> str:
    """
    Load full mschema from JSON and convert to string format.

    Args:
        mschema_dir: Directory containing {db_id}.json files
        db_id: Database identifier

    Returns:
        MSchema string representation
    """
    path = os.path.join(mschema_dir, f"{db_id}.json")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    ms = MSchema()
    ms.load_dict(data)
    return ms.to_mschema()


def get_sql_schema_str(mschema_dir: str, db_id: str, sql: str) -> str:
    """
    Get mschema string containing only tables/columns referenced by the SQL.

    Args:
        mschema_dir: Directory containing {db_id}.json files
        db_id: Database identifier
        sql: SQL query to analyze

    Returns:
        Filtered MSchema string
    """
    # Ensure mschema_dir is set for the underlying module
    set_mschema_dir(mschema_dir)

    try:
        return _get_mschema_str_impl(db_id, sql)
    except Exception as e:
        logger.warning(f"Failed to extract SQL schema for {db_id}, falling back to full schema: {e}")
        return get_full_schema_str(mschema_dir, db_id)


def _get_mschema_str_impl(db_id: str, sql: str) -> str:
    """Internal implementation using the esql mschema module."""
    base = load_mschema(db_id)
    tables, columns = extract_references(sql, base)
    qms = build_query_mschema(base, tables, columns)

    ms = MSchema()
    ms.load_dict(qms)
    return ms.to_mschema()


def get_schema_str(
    mschema_dir: str,
    db_id: str,
    sql: Optional[str] = None,
    use_full_schema: bool = True,
) -> str:
    """
    Get schema string, either full or SQL-filtered.

    Args:
        mschema_dir: Directory containing {db_id}.json files
        db_id: Database identifier
        sql: SQL query (needed if use_full_schema=False)
        use_full_schema: Whether to use full schema or SQL-filtered

    Returns:
        MSchema string
    """
    if use_full_schema or sql is None:
        return get_full_schema_str(mschema_dir, db_id)
    else:
        return get_sql_schema_str(mschema_dir, db_id, sql)
