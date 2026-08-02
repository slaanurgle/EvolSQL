"""
MSchema core: self-contained M-Schema representation and SQL-based schema extraction.

This module consolidates the previously external `mschema` package into the
project's `core`. It contains:
  - Small IO / formatting helpers (examples_to_str, read_json, write_json)
  - The MSchema class (schema container + M-Schema string rendering)
  - SQL reference extraction and query-scoped schema filtering helpers

The runtime logic is unchanged from the original implementation.
"""

import json
import datetime
import decimal
import re
from typing import Any, Dict, List, Optional, Set, Tuple

import sqlglot
from sqlglot import parse_one, exp
from sqlglot.optimizer.qualify import qualify
from sqlglot.optimizer.scope import build_scope, find_all_in_scope, Scope


# =============================================================================
# Helpers (from the original mschema.utils)
# =============================================================================

def write_json(path, data):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def read_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def is_email(string):
    pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
    match = re.match(pattern, string)
    if match:
        return True
    else:
        return False


def examples_to_str(examples: list) -> list[str]:
    """
    from examples to a list of str
    """
    values = examples
    for i in range(len(values)):
        if isinstance(values[i], datetime.date):
            values = [values[i]]
            break
        elif isinstance(values[i], datetime.datetime):
            values = [values[i]]
            break
        elif isinstance(values[i], decimal.Decimal):
            values[i] = str(float(values[i]))
        elif is_email(str(values[i])):
            values = []
            break
        elif 'http://' in str(values[i]) or 'https://' in str(values[i]):
            values = []
            break
        elif values[i] is not None and not isinstance(values[i], str):
            pass
        elif values[i] is not None and '.com' in values[i]:
            pass

    return [str(v) for v in values if v is not None and len(str(v)) > 0]


# =============================================================================
# MSchema class (from the original mschema.m_schema)
# =============================================================================

class MSchema:
    def __init__(self, db_id: str = 'Anonymous', schema: Optional[str] = None):
        """
        Initialize an MSchema object.

        Args:
            db_id: Database identifier
            schema: Schema name (e.g. "main", "public")
        """
        self.db_id = db_id
        self.schema = schema
        self.tables = {}
        self.foreign_keys = []

    def add_table(self, name, fields={}, comment=None):
        """Add a table."""
        self.tables[name] = {"fields": fields.copy(), 'examples': [], 'comment': comment}

    def add_field(self, table_name: str, field_name: str, field_type: str = "",
            primary_key: bool = False, nullable: bool = True, default: Any = None,
            autoincrement: bool = False, comment: str = "", examples: list = [], **kwargs):
        """Add a field."""
        self.tables[table_name]["fields"][field_name] = {
            "type": field_type,
            "primary_key": primary_key,
            "nullable": nullable,
            "default": default if default is None else f'{default}',
            "autoincrement": autoincrement,
            "comment": comment,
            "examples": examples.copy(),
            **kwargs}

    def add_foreign_key(self, table_name, field_name, ref_schema, ref_table_name, ref_field_name):
        """Add a foreign key."""
        self.foreign_keys.append([table_name, field_name, ref_schema, ref_table_name, ref_field_name])

    def get_field_type(self, field_type, simple_mode=True)->str:
        if not simple_mode:
            return field_type
        else:
            return field_type.split("(")[0]

    def has_table(self, table_name: str) -> bool:
        return table_name in self.tables

    def has_column(self, table_name: str, field_name: str) -> bool:
        if table_name in self.tables:
            return field_name in self.tables[table_name]["fields"]
        return False

    def get_field_info(self, table_name: str, field_name: str) -> Dict:
        try:
            return self.tables[table_name]['fields'][field_name]
        except:
            return {}

    def single_table_mschema(self, table_name: str, selected_columns: List = None,
                             example_num=3, show_type_detail=False, include_comments: bool = True) -> str:
        table_info = self.tables.get(table_name, {})
        output = []
        table_comment = table_info.get('comment', '')
        if include_comments and table_comment is not None and table_comment != 'None' and len(table_comment) > 0:
            if self.schema is not None and len(self.schema) > 0:
                output.append(f"# Table: {self.schema}.{table_name}, {table_comment}")
            else:
                output.append(f"# Table: {table_name}, {table_comment}")
        else:
            if self.schema is not None and len(self.schema) > 0:
                output.append(f"# Table: {self.schema}.{table_name}")
            else:
                output.append(f"# Table: {table_name}")

        field_lines = []
        # Process each field in the table
        for field_name, field_info in table_info['fields'].items():
            if selected_columns is not None and field_name.lower() not in selected_columns:
                continue

            raw_type = self.get_field_type(field_info['type'], not show_type_detail)
            field_line = f"({field_name}:{raw_type.upper()}"
            if include_comments and field_info['comment'] != '':
                field_line += f", {field_info['comment'].strip()}"
            else:
                pass

            ## Mark the primary key
            is_primary_key = field_info.get('primary_key', False)
            if is_primary_key:
                field_line += f", Primary Key"

            # Append examples if available
            if len(field_info.get('examples', [])) > 0 and example_num > 0:
                examples = field_info['examples']
                examples = [s for s in examples if s is not None]
                examples = examples_to_str(examples)
                if len(examples) > example_num:
                    examples = examples[:example_num]

                if raw_type in ['DATE', 'TIME', 'DATETIME', 'TIMESTAMP']:
                    examples = [examples[0]]
                elif len(examples) > 0 and max([len(s) for s in examples]) > 20:
                    if max([len(s) for s in examples]) > 50:
                        examples = []
                    else:
                        examples = [examples[0]]
                else:
                    pass
                if len(examples) > 0:
                    example_str = ', '.join([str(example) for example in examples])
                    field_line += f", Examples: [{example_str}]"
                else:
                    pass
            else:
                field_line += ""
            field_line += ")"

            field_lines.append(field_line)
        output.append('[')
        output.append(',\n'.join(field_lines))
        output.append(']')

        return '\n'.join(output)

    def to_mschema(self, selected_tables: List = None, selected_columns: List = None,
                   example_num=3, show_type_detail=False, include_comments: bool = True) -> str:
        """
        convert to a MSchema string.
        selected_tables: default None, meaning select all tables
        selected_columns: default None, meaning select all columns, format ['table_name.column_name']
        """
        output = []

        output.append(f"【DB_ID】 {self.db_id}")
        output.append(f"【Schema】")

        if selected_tables is not None:
            selected_tables = [s.lower() for s in selected_tables]
        if selected_columns is not None:
            selected_columns = [s.lower() for s in selected_columns]
            selected_tables = [s.split('.')[0].lower() for s in selected_columns]

        # Process each table in turn
        for table_name, table_info in self.tables.items():
            if selected_tables is None or table_name.lower() in selected_tables:
                cur_table_type = table_info.get('type', 'table')
                column_names = list(table_info['fields'].keys())
                if selected_columns is not None:
                    cur_selected_columns = [c.lower() for c in column_names if f"{table_name}.{c}".lower() in selected_columns]
                else:
                    cur_selected_columns = selected_columns
                output.append(self.single_table_mschema(table_name, cur_selected_columns, example_num, show_type_detail, include_comments))

        # Append foreign key info; foreign keys are not shown when table_type is view
        if self.foreign_keys:
            output.append("【Foreign keys】")
            for fk in self.foreign_keys:
                ref_schema = fk[2]
                table1, column1, _, table2, column2 = fk
                if selected_tables is None or \
                        (table1.lower() in selected_tables and table2.lower() in selected_tables):
                    # print(f"{ref_schema}=={self.schema}")
                    # if ref_schema == self.schema:
                    output.append(f"{fk[0]}.{fk[1]}={fk[3]}.{fk[4]}")

        return '\n'.join(output)

    def dump(self):
        schema_dict = {
            "db_id": self.db_id,
            "schema": self.schema,
            "tables": self.tables,
            "foreign_keys": self.foreign_keys
        }
        return schema_dict

    def save(self, file_path: str):
        schema_dict = self.dump()
        write_json(file_path, schema_dict)

    def load(self, file_path: str):
        data = read_json(file_path)
        self.db_id = data.get("db_id", "Anonymous")
        self.schema = data.get("schema", None)
        self.tables = data.get("tables", {})
        self.foreign_keys = data.get("foreign_keys", [])

    def load_dict(self, data: Dict):
        self.db_id = data.get("db_id", "Anonymous")
        self.schema = data.get("schema", None)
        self.tables = data.get("tables", {})
        self.foreign_keys = data.get("foreign_keys", [])


# =============================================================================
# SQL reference extraction & query-scoped schema filtering
# (from the original mschema.get_union_schema; only the functions used by the
#  pipeline are retained, logic unchanged)
# =============================================================================

DEFAULT_TRAIN_MSCHEMA_DIR = "./schemas/train_mschemas"

# Default to the train directory; can be overridden at runtime via set_mschema_mode/set_mschema_dir
MSCHEMA_DIR = DEFAULT_TRAIN_MSCHEMA_DIR


def set_mschema_mode(mode: str) -> None:
    """Switch to the preset mschema directory based on data_mode."""
    global MSCHEMA_DIR
    MSCHEMA_DIR = DEFAULT_TRAIN_MSCHEMA_DIR


def set_mschema_dir(path: str) -> None:
    """Explicitly set the mschema directory; takes precedence over set_mschema_mode."""
    global MSCHEMA_DIR
    MSCHEMA_DIR = path


def _simplify_mschema(mschema: dict) -> dict:
    """
    Converts a schema of the form:
      {
        "tables": {
          "main.atom": {
            "fields": {
              "atom_id": { "type": "TEXT", ... },
              ...
            },
            ...
          },
          ...
        }
      }
    into:
      {
        "atom": {"atom_id": "TEXT", ...},
        "bond": {"bond_id": "TEXT", ...},
        ...
      }
    """
    simplified = {}
    for full_table_name, table_info in mschema.get("tables", {}).items():
        # Extract the actual table name (after the dot)
        table_name = full_table_name.split(".")[-1]
        cols = {
            col_name: col_meta["type"]
            for col_name, col_meta in table_info.get("fields", {}).items()
        }
        simplified[table_name] = cols
    return simplified


def extract_references(sql: str, mschema: dict):
    ast = parse_one(sql, read="sqlite")
    ast = qualify(
        ast,
        dialect='sqlite',
        schema=_simplify_mschema(mschema),
        qualify_columns=True,
        validate_qualify_columns=False,
    )

    root_scope = build_scope(ast)

    tables = set()
    columns = set()

    # 1) Only collect real exp.Table nodes
    for scope in root_scope.traverse():
        for alias, (node, source) in scope.selected_sources.items():
            if isinstance(source, exp.Table):
                tables.add(source.name)
            # To keep subquery aliases too, add: elif isinstance(source, Scope): tables.add(alias)

    # 2) Collect the source of each column within its current scope
    for scope in root_scope.traverse():
        for column in find_all_in_scope(scope.expression, exp.Column):
            src = scope.sources.get(column.table)
            if isinstance(src, exp.Table):
                table_name = src.name
            elif isinstance(src, str):
                table_name = src
            else:
                # Scope (subquery) or others, handle as needed:
                #   to keep only the bottom-level table, keep tracing src.sources
                continue

            columns.add((table_name, column.name))

    return tables, columns


def load_mschema(db_id: str) -> dict:
    path = f"{MSCHEMA_DIR}/{db_id}.json"
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def resolve_table_name(tbl: str, base_tables: Set[str]) -> Optional[str]:
    """
    Resolve a table name (case-insensitive) to the matching key in base_tables.
    Supports schema/db-prefixed names as well as unqualified names.
    """
    # Build a lowercase -> original key lookup map
    lower_map = {bt.lower(): bt for bt in base_tables}

    tbl_l = tbl.lower()
    # 1) Exact match (case-insensitive)
    if tbl_l in lower_map:
        return lower_map[tbl_l]

    # 2) Suffix match: last segment (split by '.') of a base table matches tbl (case-insensitive)
    matches = [
        bt for bt in base_tables
        if bt.split('.')[-1].lower() == tbl_l
    ]
    if len(matches) == 1:
        return matches[0]

    # Match failed
    return None


def build_query_mschema(base: dict, tables: Set[str], columns: Set[Tuple[str, str]]) -> dict:
    """
    Build a filtered mschema dict containing only the referenced tables and columns.
    Resolve table identifiers against base['tables'] keys.
    If a table has no column filters, include all its fields.
    """
    base_tables = set(base['tables'].keys())

    # Build case-insensitive maps
    # lowercase table name -> original table name
    table_lower_map = {bt.lower(): bt for bt in base_tables}
    # per-table lowercase field name -> original field name
    column_lower_map = {
        bt: {f.lower(): f for f in base['tables'][bt]['fields'].keys()}
        for bt in base_tables
    }

    # resolve tables and columns
    resolved_tables: Set[str] = set()
    resolved_columns: Set[Tuple[str, str]] = set()

    # Resolve table names case-insensitively
    for tbl in tables:
        tbl_l = tbl.lower()
        if tbl_l in table_lower_map:
            resolved_tables.add(table_lower_map[tbl_l])
            continue
        # suffix match (schema.table → table) is also case-insensitive
        matches = [
            bt for bt in base_tables
            if bt.split('.')[-1].lower() == tbl_l
        ]
        if len(matches) == 1:
            resolved_tables.add(matches[0])

    # Resolve field names case-insensitively and restore original case
    for tbl, col in columns:
        tbl_l = tbl.lower()
        real_tbl = None
        if tbl_l in table_lower_map:
            real_tbl = table_lower_map[tbl_l]
        else:
            matches = [
                bt for bt in base_tables
                if bt.split('.')[-1].lower() == tbl_l
            ]
            if len(matches) == 1:
                real_tbl = matches[0]
        if not real_tbl:
            continue

        col_map = column_lower_map[real_tbl]
        actual_col = col_map.get(col.lower())
        if actual_col:
            resolved_columns.add((real_tbl, actual_col))

    qms = {
        "db_id": base.get("db_id"),
        "schema": base.get("schema"),
        "tables": {},
        "foreign_keys": [],
    }
    all_fks = base.get('foreign_keys', [])
    for fk in all_fks:
        t1, c1, ref_schema, t2, c2 = fk
        if t1 in resolved_tables and f"{ref_schema}.{t2}" in resolved_tables:
            qms['foreign_keys'].append(fk)
    for tbl in resolved_tables:
        tbl_def = base['tables'].get(tbl)
        if not tbl_def:
            continue
        field_defs = tbl_def['fields']
        # Filter fields by intersection
        selected = {
            fld: info
            for fld, info in field_defs.items()
            if (tbl, fld) in resolved_columns
        }
        if not selected:
            # If no columns explicitly referenced, include all
            selected = field_defs.copy()
        qms['tables'][tbl] = {
            'fields': selected,
            'examples': tbl_def.get('examples', []),
            'comment': tbl_def.get('comment', ""),
        }
    return qms
