"""
Utility functions for the EvolSQL pipeline.

Provides:
- JSON file I/O
- Template loading and rendering
- Robust JSON extraction from LLM outputs
- Schema loading helpers
"""

import json
import re
import os
import logging
from pathlib import Path
from typing import Any, Optional, Dict, List

logger = logging.getLogger(__name__)


# ===== File I/O =====

def load_json(path: str) -> Any:
    """Load a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, data: Any, indent: int = 2):
    """Save data to a JSON file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def load_jsonl(path: str) -> List[Dict]:
    """Load a JSONL file (one JSON object per line)."""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


# ===== Template =====

def load_template(path: str) -> str:
    """Load a prompt template file."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def render_template(template: str, **kwargs) -> str:
    """
    Render a template by replacing {KEY} placeholders.

    Args:
        template: Template string with {KEY} placeholders
        **kwargs: Key-value pairs for replacement

    Returns:
        Rendered template string
    """
    result = template
    for key, value in kwargs.items():
        placeholder = "{" + key + "}"
        result = result.replace(placeholder, str(value) if value is not None else "")
    return result


# ===== JSON Extraction =====

def extract_json(text: str) -> Any:
    """
    Extract and parse JSON from LLM output text.

    Handles:
    - JSON in markdown code blocks (```json ... ```)
    - Raw JSON objects or arrays
    - Common LLM output issues (trailing commas, unquoted keys, etc.)

    Returns:
        Parsed JSON object/array

    Raises:
        json.JSONDecodeError if parsing fails
    """
    if text is None:
        raise json.JSONDecodeError("Input text is None", "", 0)

    # Strategy 1: Try to find JSON in code blocks
    json_pattern = r"```(?:json)?\s*\n?([\s\S]*?)\n?\s*```"
    match = re.search(json_pattern, text)

    if match:
        json_str = match.group(1).strip()
    else:
        # Strategy 2: Try to find JSON array or object directly
        array_match = re.search(r"(\[[\s\S]*\])", text)
        object_match = re.search(r"(\{[\s\S]*\})", text)

        if array_match and object_match:
            if array_match.start() < object_match.start():
                json_str = array_match.group(1)
            else:
                json_str = object_match.group(1)
        elif array_match:
            json_str = array_match.group(1)
        elif object_match:
            json_str = object_match.group(1)
        else:
            json_str = text.strip()

    # Clean up common issues
    json_str = _clean_json_string(json_str)

    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        # Strategy 3: Aggressive cleanup
        json_str = _aggressive_json_cleanup(json_str)
        return json.loads(json_str)


def _clean_json_string(json_str: str) -> str:
    """Clean up common JSON formatting issues."""
    json_str = re.sub(r",\s*]", "]", json_str)
    json_str = re.sub(r",\s*}", "}", json_str)
    json_str = json_str.lstrip("\ufeff\u200b")
    return json_str


def _aggressive_json_cleanup(json_str: str) -> str:
    """More aggressive JSON cleanup for problematic output."""
    # Remove single-line comments
    json_str = re.sub(r"//[^\n]*\n", "\n", json_str)
    # Remove multi-line comments
    json_str = re.sub(r"/\*[\s\S]*?\*/", "", json_str)
    # Quote unquoted keys
    json_str = re.sub(
        r'(?<=[{,])\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r' "\1":', json_str
    )
    # Remove trailing commas
    json_str = re.sub(r",\s*]", "]", json_str)
    json_str = re.sub(r",\s*}", "}", json_str)
    # Remove non-printable characters
    json_str = "".join(c for c in json_str if c.isprintable() or c in "\n\r\t")
    return json_str


def safe_extract_json(text: str) -> Optional[Any]:
    """
    Safely extract JSON, returning None on failure instead of raising.
    """
    try:
        return extract_json(text)
    except (json.JSONDecodeError, Exception) as e:
        logger.warning(f"JSON extraction failed: {e}")
        return None


# ===== MSchema helpers =====

def load_mschema_mapping(mschema_file: str) -> Dict[str, str]:
    """
    Load mschema mapping from JSONL file: db_name -> mschema_str

    Args:
        mschema_file: Path to mschema JSONL file

    Returns:
        Dict mapping db_name to mschema string
    """
    mapping = {}
    with open(mschema_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            mapping[data["db_name"]] = data["mschema_str"]
    return mapping


def load_mschema_json(mschema_dir: str, db_id: str) -> Dict:
    """
    Load a single mschema JSON file.

    Args:
        mschema_dir: Directory containing mschema JSON files
        db_id: Database identifier

    Returns:
        MSchema dict
    """
    path = os.path.join(mschema_dir, f"{db_id}.json")
    return load_json(path)


# ===== Logging setup =====

def setup_logging(level: str = "INFO"):
    """Configure logging for the pipeline."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
