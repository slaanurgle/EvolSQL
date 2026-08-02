"""
Database Injection: Core utilities for data injection to ensure SQL non-empty results.

Provides:
- Schema extraction from SQLite databases (DDL, PK, FK info)
- Primary key range allocation per SQL to avoid PK conflicts
- Prompt construction with mschema format + embedded PK ranges
- INSERT statement parsing and validation
- INSERT execution with error reporting

Design decisions:
- Goal: insert rows that satisfy all SQL conditions so the query returns non-empty results
- Each SQL gets a dedicated PK range (budget=10 per SQL) to prevent conflicts
- FK handling uses cascading INSERT (insert parent rows first if needed)
- LLM is told the schema in mschema format (consistent with other pipeline steps)
- PK ranges are embedded directly into PK field descriptors for clarity
- PK validation checks extracted PK values against allocated ranges
"""

import re
import sqlglot
import sqlite3
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ===== Data Structures =====

@dataclass
class TablePKInfo:
    """Primary key information for a single table."""
    table_name: str
    pk_columns: List[str]           # PK column names (may be multiple for composite)
    pk_types: List[str]             # Corresponding types
    is_composite: bool              # True if composite PK
    is_integer: bool                # True if single INTEGER-family PK
    max_value: Optional[int] = None # Current MAX value (only for integer PKs)


@dataclass
class PKRange:
    """Allocated PK range for a specific table and SQL index."""
    table_name: str
    pk_column: str
    start: int
    end: int


@dataclass
class InjectResult:
    """Result of an injection attempt."""
    success: bool
    inserts_applied: int = 0
    inserts_total: int = 0
    errors: List[str] = field(default_factory=list)
    raw_response: str = ""
    parsed_inserts: List[str] = field(default_factory=list)


# ===== Schema Extraction =====

def get_create_statements(db_path: str) -> Dict[str, str]:
    """
    Get CREATE TABLE statements for all tables in the database.

    Args:
        db_path: Path to SQLite database file

    Returns:
        Dict mapping table_name -> CREATE TABLE SQL string
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        "SELECT name, sql FROM sqlite_master "
        "WHERE type='table' AND name != 'sqlite_sequence' AND sql IS NOT NULL"
    )
    result = {row[0]: row[1] for row in cur.fetchall()}
    conn.close()
    return result


def get_table_pk_info(db_path: str, table_name: str) -> TablePKInfo:
    """
    Get primary key information for a table.

    Args:
        db_path: Path to SQLite database file
        table_name: Table name

    Returns:
        TablePKInfo with PK details
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute(f'PRAGMA table_info("{table_name}")')
    columns = cur.fetchall()
    # columns: (cid, name, type, notnull, dflt_value, pk_flag)

    pk_cols = [(c[1], c[2]) for c in columns if c[5] > 0]

    if not pk_cols:
        conn.close()
        return TablePKInfo(
            table_name=table_name,
            pk_columns=[],
            pk_types=[],
            is_composite=False,
            is_integer=False,
        )

    is_composite = len(pk_cols) > 1
    pk_columns = [c[0] for c in pk_cols]
    pk_types = [c[1] for c in pk_cols]

    # Check if single integer/numeric PK
    is_integer = False
    max_value = None
    if not is_composite:
        pk_type = pk_types[0].upper().strip()
        # Check declared type: INT*, DECIMAL, NUMERIC, REAL, FLOAT, DOUBLE, NUMBER, or empty
        base_type = pk_type.split("(")[0].strip()  # strip precision e.g. decimal(5,0) -> DECIMAL
        _NUMERIC_KEYWORDS = {"INT", "INTEGER", "SMALLINT", "BIGINT", "TINYINT", "MEDIUMINT",
                             "DECIMAL", "NUMERIC", "REAL", "FLOAT", "DOUBLE", "NUMBER"}
        declared_numeric = pk_type == "" or any(kw in base_type for kw in _NUMERIC_KEYWORDS)

        if declared_numeric:
            # Verify with actual stored data: typeof() must be 'integer' or 'real'
            try:
                cur.execute(f'SELECT typeof("{pk_columns[0]}") FROM "{table_name}" LIMIT 1')
                type_row = cur.fetchone()
                actual_type = type_row[0].lower() if type_row else ""
            except Exception:
                actual_type = ""

            if actual_type in ("integer", "real", "") or pk_type == "":
                is_integer = True
                try:
                    cur.execute(f'SELECT MAX("{pk_columns[0]}") FROM "{table_name}"')
                    row = cur.fetchone()
                    max_value = row[0] if row and row[0] is not None else 0
                    # Ensure it's actually an integer
                    if isinstance(max_value, (int, float)):
                        max_value = int(max_value)
                    else:
                        # The column stores non-integer data despite type
                        try:
                            max_value = int(max_value)
                        except (ValueError, TypeError):
                            is_integer = False
                            max_value = None
                except Exception:
                    max_value = 0

    conn.close()
    return TablePKInfo(
        table_name=table_name,
        pk_columns=pk_columns,
        pk_types=pk_types,
        is_composite=is_composite,
        is_integer=is_integer,
        max_value=max_value,
    )


def get_all_pk_info(db_path: str) -> Dict[str, TablePKInfo]:
    """
    Get PK info for all tables in a database.

    Returns:
        Dict mapping table_name -> TablePKInfo
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        "SELECT name FROM sqlite_master "
        "WHERE type='table' AND name != 'sqlite_sequence'"
    )
    tables = [r[0] for r in cur.fetchall()]
    conn.close()

    return {t: get_table_pk_info(db_path, t) for t in tables}


def get_sample_data(db_path: str, table_name: str, limit: int = 3) -> List[Tuple]:
    """
    Get sample rows from a table.

    Returns:
        List of tuples (rows), plus column names
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    try:
        cur.execute(f'SELECT * FROM "{table_name}" LIMIT {limit}')
        rows = cur.fetchall()
        col_names = [desc[0] for desc in cur.description] if cur.description else []
    except Exception:
        rows = []
        col_names = []
    conn.close()
    return col_names, rows


# ===== PK Range Allocation =====

def compute_pk_ranges(
    pk_info_map: Dict[str, TablePKInfo],
    sql_index: int,
    budget: int = 10,
) -> Dict[str, PKRange]:
    """
    Compute non-overlapping PK ranges for a given SQL index.

    Each SQL gets `budget` PK slots per table, offset by sql_index * budget.
    Only applies to tables with single INTEGER PKs.

    Args:
        pk_info_map: Dict of table -> TablePKInfo
        sql_index: Index of the SQL within its database group
        budget: Number of PK values allocated per SQL per table

    Returns:
        Dict mapping table_name -> PKRange
    """
    ranges = {}
    for table_name, pk_info in pk_info_map.items():
        if pk_info.is_integer and not pk_info.is_composite and pk_info.max_value is not None:
            start = pk_info.max_value + 1 + sql_index * budget
            end = start + budget - 1
            ranges[table_name] = PKRange(
                table_name=table_name,
                pk_column=pk_info.pk_columns[0],
                start=start,
                end=end,
            )
    return ranges


# ===== Prompt Construction =====

def build_schema_with_constraints(
    db_path: str,
    pk_info_map: Dict[str, TablePKInfo],
    pk_ranges: Dict[str, PKRange],
    sql_index: int = 0,
) -> str:
    """
    Build a comprehensive schema string with constraint info for the LLM prompt.

    Includes:
    - CREATE TABLE statements (with all constraints)
    - Sample data for each table
    - PK range allocation info
    - FK relationships (from CREATE TABLE, no parent value listing)

    Args:
        db_path: Path to SQLite database
        pk_info_map: PK info for all tables
        pk_ranges: Allocated PK ranges for this SQL
        sql_index: Index of this SQL within its db_id group (for TEXT PK isolation)

    Returns:
        Formatted schema string
    """
    create_stmts = get_create_statements(db_path)
    parts = []

    for table_name, create_sql in create_stmts.items():
        section = []
        section.append(f"/* Table: {table_name} */")
        section.append(create_sql + ";")

        # Sample data
        col_names, rows = get_sample_data(db_path, table_name, limit=3)
        if rows:
            section.append(f"-- Sample data ({len(rows)} rows):")
            header = " | ".join(col_names)
            section.append(f"-- {header}")
            for row in rows:
                row_str = " | ".join(str(v) for v in row)
                section.append(f"-- {row_str}")

        # PK range info
        pk_info = pk_info_map.get(table_name)
        if pk_info:
            if pk_info.is_integer and not pk_info.is_composite:
                pk_range = pk_ranges.get(table_name)
                if pk_range:
                    section.append(
                        f"-- PK constraint: {table_name}.{pk_info.pk_columns[0]} "
                        f"is INTEGER, current max = {pk_info.max_value}. "
                        f"You MUST use values in range [{pk_range.start}, {pk_range.end}]."
                    )
            elif pk_info.is_composite:
                pk_cols_str = ", ".join(pk_info.pk_columns)
                section.append(
                    f"-- PK constraint: Composite PK ({pk_cols_str}). "
                    f"Ensure the combination is unique."
                )
            elif not pk_info.pk_columns:
                section.append("-- No primary key constraint.")
            else:
                # Non-integer single PK (TEXT/VARCHAR) — use sql_index-isolated prefix
                adv_prefix = f"ADV_{sql_index}_"
                section.append(
                    f"-- PK constraint: {table_name}.{pk_info.pk_columns[0]} "
                    f"is {pk_info.pk_types[0]}. "
                    f'Use "{adv_prefix}" prefix to avoid conflicts (e.g., "{adv_prefix}value1").'
                )

        parts.append("\n".join(section))

    return "\n\n".join(parts)


def build_mschema_with_pk_ranges(
    mschema_str: str,
    pk_info_map: Dict[str, TablePKInfo],
    pk_ranges: Dict[str, PKRange],
    sql_index: int = 0,
) -> str:
    """
    Inject PK range info directly into mschema field descriptors.

    For each PK field line like:
        (Singer_ID:INTEGER, Primary Key, Examples: [1, 2, 3])
    becomes:
        (Singer_ID:INTEGER, Primary Key, Allocated PK range: [7, 16], Examples: [1, 2, 3])

    For composite PKs:
        (concert_ID:INTEGER, Primary Key, Examples: [1, 2, 3])
    becomes:
        (concert_ID:INTEGER, Primary Key, Ensure unique combination, Examples: [1, 2, 3])

    For text PKs (with sql_index isolation):
        (code:TEXT, Primary Key, Examples: [US, UK])
    becomes:
        (code:TEXT, Primary Key, Use "ADV_3_" prefix, Examples: [US, UK])

    Args:
        mschema_str: Original mschema string (from jsonl or MSchema.to_mschema())
        pk_info_map: PK info for all tables
        pk_ranges: Allocated PK ranges for this SQL
        sql_index: Index of this SQL within its db_id group (for TEXT PK isolation)

    Returns:
        Modified mschema string with PK range info embedded
    """
    lines = mschema_str.split("\n")
    result = []
    current_table = None

    for line in lines:
        # Detect table header: "# Table: xxx"
        table_match = re.match(r"^#\s*Table:\s*(\S+)", line)
        if table_match:
            current_table = table_match.group(1).rstrip(",")

        # Detect PK field line containing "Primary Key"
        if current_table and "Primary Key" in line:
            # Extract field name from the line: (FieldName:TYPE, Primary Key, ...)
            field_match = re.match(r"^\((\w+):", line)
            if field_match:
                field_name = field_match.group(1)
                pk_info = pk_info_map.get(current_table)

                if pk_info and field_name in pk_info.pk_columns:
                    pk_range = pk_ranges.get(current_table)

                    if pk_range and pk_info.is_integer and not pk_info.is_composite:
                        # Single integer PK: inject allocated range
                        insert_text = f"Allocated PK range: [{pk_range.start}, {pk_range.end}]"
                        line = line.replace(
                            "Primary Key,",
                            f"Primary Key, {insert_text},",
                        )
                        # Handle case where Primary Key is followed by )
                        if "Primary Key)" in line:
                            line = line.replace(
                                "Primary Key)",
                                f"Primary Key, {insert_text})",
                            )
                    elif pk_info.is_composite:
                        # Composite PK: hint unique combination
                        insert_text = "Ensure unique combination"
                        if insert_text not in line:
                            line = line.replace(
                                "Primary Key,",
                                f"Primary Key, {insert_text},",
                            )
                            if "Primary Key)" in line:
                                line = line.replace(
                                    "Primary Key)",
                                    f"Primary Key, {insert_text})",
                                )
                    elif not pk_info.is_integer and not pk_info.is_composite:
                        # Text PK: hint ADV_ prefix with sql_index isolation
                        adv_prefix = f"ADV_{sql_index}_"
                        insert_text = f'Use "{adv_prefix}" prefix to avoid conflicts'
                        if "ADV_" not in line:
                            line = line.replace(
                                "Primary Key,",
                                f"Primary Key, {insert_text},",
                            )
                            if "Primary Key)" in line:
                                line = line.replace(
                                    "Primary Key)",
                                    f"Primary Key, {insert_text})",
                                )

        result.append(line)

    return "\n".join(result)


def build_inject_prompt(
    template: str,
    gold_sql: str,
    pk_info_map: Dict[str, TablePKInfo],
    pk_ranges: Dict[str, PKRange],
    error_feedback: str = "",
    mschema_str: str = "",
    db_path: str = "",
    sql_index: int = 0,
) -> str:
    """
    Build the complete injection prompt.

    Uses mschema format (consistent with other pipeline steps) when mschema_str
    is provided. Falls back to DDL-based schema when mschema_str is empty.

    Args:
        template: Prompt template string with placeholders
        gold_sql: The gold SQL query to generate adversarial data for
        pk_info_map: PK info for all tables
        pk_ranges: Allocated PK ranges
        error_feedback: Optional error feedback from previous attempt
        mschema_str: Pre-loaded mschema string for this db_id
        db_path: Path to SQLite database (used as fallback if mschema_str empty)
        sql_index: Index of this SQL within its db_id group (for TEXT PK isolation)

    Returns:
        Complete prompt string
    """
    if mschema_str:
        # Use mschema with PK ranges embedded into field descriptors
        schema_str = build_mschema_with_pk_ranges(mschema_str, pk_info_map, pk_ranges, sql_index)
    else:
        # Fallback to DDL-based schema
        schema_str = build_schema_with_constraints(db_path, pk_info_map, pk_ranges, sql_index)

    # Render template
    prompt = template.replace("{DATABASE_SCHEMA}", schema_str)
    prompt = prompt.replace("{TARGET_SQL}", gold_sql)
    prompt = prompt.replace("{ERROR_FEEDBACK}", error_feedback)

    return prompt


# ===== INSERT Parsing =====

def parse_insert_statements(llm_output: str) -> List[str]:
    """
    Extract INSERT statements from LLM output using sqlglot.

    Strategy:
    1. Extract SQL text from the last ```sql ... ``` code block (preferred).
       If no code blocks, use full text (stripped of markdown markers).
    2. Parse with sqlglot to reliably split statements (handles nested
       parentheses, semicolons inside strings, etc.).
    3. If sqlglot fails (SQL has syntax errors), return the raw SQL text
       as a single-element list so that apply_inserts can execute it and
       let SQLite provide the real error feedback to the LLM.

    Args:
        llm_output: Raw LLM response text

    Returns:
        List of INSERT SQL strings
    """
    if not llm_output:
        return []

    # Step 1: Extract SQL text from code blocks
    sql_text = _extract_sql_text(llm_output)
    if not sql_text:
        return []

    # Step 2: Parse with sqlglot
    try:
        stmts = sqlglot.parse(sql_text, dialect="sqlite")
        inserts = [
            stmt.sql(dialect="sqlite")
            for stmt in stmts
            if stmt is not None and stmt.key == "insert"
        ]
        return inserts
    except Exception:
        # sqlglot parse failed → SQL itself has syntax errors.
        # Return raw text so apply_inserts can execute it and get
        # SQLite's real error message for LLM feedback.
        return [sql_text.strip()]


def _extract_sql_text(llm_output: str) -> str:
    """Extract SQL text from LLM output: prefer last ```sql block, else full text."""
    code_block_pattern = re.compile(r"```sql\s*(.*?)```", re.IGNORECASE | re.DOTALL)
    code_blocks = code_block_pattern.findall(llm_output)

    if code_blocks:
        # Use only the LAST code block — LLMs typically put the final answer
        # in the last block, while earlier blocks may be drafts/plans.
        return code_blocks[-1].strip()

    # No code blocks — strip markdown markers and use full text
    text = llm_output
    text = re.sub(r"```sql\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"```\s*", "", text)
    return text.strip()


# ===== PK Validation =====

def extract_pk_values_from_insert(
    insert_stmt: str,
    table_name: str,
    pk_column: str,
    db_path: str,
) -> Optional[int]:
    """
    Extract the PK value from an INSERT statement for a specific table and column.

    Handles two forms:
    1. INSERT INTO table VALUES(v1, v2, ...) - positional
    2. INSERT INTO table(col1, col2, ...) VALUES(v1, v2, ...) - named

    Args:
        insert_stmt: The INSERT SQL statement
        table_name: Target table name
        pk_column: PK column name
        db_path: Path to database (to get column order for positional inserts)

    Returns:
        Integer PK value, or None if not applicable/found
    """
    # Check if this INSERT is for the target table
    # Normalize: remove quotes around table name for matching
    stmt_upper = insert_stmt.upper()
    table_upper = table_name.upper()

    # Match table name in INSERT INTO <table_name>
    insert_match = re.match(
        r"INSERT\s+INTO\s+[`\"']?(\w+)[`\"']?\s*",
        insert_stmt,
        re.IGNORECASE,
    )
    if not insert_match:
        return None

    stmt_table = insert_match.group(1).upper()
    if stmt_table != table_upper:
        return None

    # Check if named columns form: INSERT INTO table(col1, col2, ...) VALUES(...)
    named_match = re.match(
        r"INSERT\s+INTO\s+[`\"']?\w+[`\"']?\s*\(([^)]+)\)\s*VALUES\s*\((.+)\)",
        insert_stmt,
        re.IGNORECASE | re.DOTALL,
    )

    if named_match:
        col_str = named_match.group(1)
        val_str = named_match.group(2)
        columns = [c.strip().strip('`"\'') for c in col_str.split(",")]
        values = _split_values(val_str)

        try:
            pk_idx = [c.upper() for c in columns].index(pk_column.upper())
            val = values[pk_idx].strip().strip("'\"")
            return int(val)
        except (ValueError, IndexError):
            return None
    else:
        # Positional form: INSERT INTO table VALUES(...)
        values_match = re.search(
            r"VALUES\s*\((.+)\)",
            insert_stmt,
            re.IGNORECASE | re.DOTALL,
        )
        if not values_match:
            return None

        values = _split_values(values_match.group(1))

        # Get column order from database
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute(f'PRAGMA table_info("{table_name}")')
        columns = cur.fetchall()
        conn.close()

        col_names = [c[1] for c in columns]
        try:
            pk_idx = [c.upper() for c in col_names].index(pk_column.upper())
            val = values[pk_idx].strip().strip("'\"")
            return int(val)
        except (ValueError, IndexError):
            return None


def _split_values(val_str: str) -> List[str]:
    """
    Split a VALUES clause into individual values, respecting quoted strings.

    Handles: 'text with, comma', 123, NULL, "another, value"
    """
    values = []
    current = []
    in_quote = False
    quote_char = None

    for char in val_str:
        if in_quote:
            current.append(char)
            if char == quote_char:
                in_quote = False
        elif char in ("'", '"'):
            in_quote = True
            quote_char = char
            current.append(char)
        elif char == ",":
            values.append("".join(current).strip())
            current = []
        else:
            current.append(char)

    if current:
        values.append("".join(current).strip())

    return values


def validate_pk_in_range(
    inserts: List[str],
    pk_ranges: Dict[str, PKRange],
    db_path: str,
) -> List[str]:
    """
    Validate that all INSERT statements use PK values within allocated ranges.

    Args:
        inserts: List of INSERT SQL statements
        pk_ranges: Allocated PK ranges per table
        db_path: Path to database

    Returns:
        List of violation descriptions (empty = all valid)
    """
    violations = []

    for stmt in inserts:
        for table_name, pk_range in pk_ranges.items():
            pk_val = extract_pk_values_from_insert(
                stmt, table_name, pk_range.pk_column, db_path
            )
            if pk_val is not None:
                if pk_val < pk_range.start or pk_val > pk_range.end:
                    violations.append(
                        f"INSERT into {table_name}: PK {pk_range.pk_column}={pk_val} "
                        f"is outside allocated range [{pk_range.start}, {pk_range.end}]"
                    )

    return violations


# ===== INSERT Execution =====

def apply_inserts(
    db_path: str,
    inserts: List[str],
) -> Tuple[int, List[str]]:
    """
    Execute INSERT statements on a database.

    Executes all INSERTs within a single transaction. If any fails,
    rolls back the entire batch.

    Args:
        db_path: Path to SQLite database
        inserts: List of INSERT SQL statements

    Returns:
        Tuple of (num_successful, list_of_errors)
    """
    if not inserts:
        return 0, ["No INSERT statements provided"]

    conn = sqlite3.connect(db_path)
    # Enable foreign key enforcement
    conn.execute("PRAGMA foreign_keys = ON")
    cur = conn.cursor()

    errors = []
    success_count = 0

    try:
        cur.execute("BEGIN TRANSACTION")
        for stmt in inserts:
            try:
                cur.execute(stmt)
                success_count += 1
            except Exception as e:
                errors.append(f"Failed: {stmt[:100]}... Error: {str(e)}")
                # Roll back entire transaction on any error
                conn.rollback()
                conn.close()
                return 0, errors

        conn.commit()
    except Exception as e:
        errors.append(f"Transaction error: {str(e)}")
        try:
            conn.rollback()
        except:
            pass
    finally:
        conn.close()

    return success_count, errors


# ===== Database Copy =====

def copy_database(src_path: str, dst_path: str) -> bool:
    """
    Copy a SQLite database file.

    Args:
        src_path: Source database path
        dst_path: Destination database path

    Returns:
        True if successful
    """
    import shutil
    import os

    try:
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        shutil.copy2(src_path, dst_path)
        return True
    except Exception as e:
        logger.error(f"Failed to copy database {src_path} -> {dst_path}: {e}")
        return False


def copy_database_dir(src_dir: str, dst_dir: str, db_id: str) -> bool:
    """
    Copy a database directory (db_id/db_id.sqlite structure).

    Args:
        src_dir: Source parent directory (e.g., train_databases/)
        dst_dir: Destination parent directory
        db_id: Database identifier

    Returns:
        True if successful
    """
    import shutil
    import os

    src = os.path.join(src_dir, db_id)
    dst = os.path.join(dst_dir, db_id)

    if os.path.exists(dst):
        return True  # Already copied

    try:
        shutil.copytree(src, dst)
        return True
    except Exception as e:
        logger.error(f"Failed to copy database dir {src} -> {dst}: {e}")
        return False
