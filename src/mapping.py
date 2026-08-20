"""
Platform-mapping helpers.

The dashboard supports three ways to get an AWS_ID -> platform-name
mapping: the bundled default in config.py, a manually-edited table, or an
uploaded CSV/JSON file. This module is the single place that turns any of
those three sources into the same uniform ``dict[int, str]`` shape that
``data_loader.load_raw_csv`` expects — nothing downstream needs to know
which source the mapping came from.
"""

from __future__ import annotations

import io
import json

import pandas as pd

from .config import PLATFORM_MAPPING as DEFAULT_PLATFORM_MAPPING


def default_mapping_df() -> pd.DataFrame:
    """The bundled mapping as an editable two-column table."""
    return pd.DataFrame(
        [{"aws_id": k, "platform_name": v} for k, v in DEFAULT_PLATFORM_MAPPING.items()]
    )


def mapping_df_to_dict(df: pd.DataFrame) -> dict[int, str]:
    """Convert an (possibly hand-edited) aws_id/platform_name table to a dict.

    Rows with a blank/unparseable id or an empty name are silently
    skipped rather than raising, since this is called on every keystroke
    of an in-progress edit in the Streamlit data editor.
    """
    out: dict[int, str] = {}
    if df is None or df.empty:
        return out
    for _, row in df.iterrows():
        try:
            aws_id = int(row["aws_id"])
        except (TypeError, ValueError):
            continue
        name = str(row.get("platform_name", "")).strip()
        if name and name.lower() != "nan":
            out[aws_id] = name
    return out


def parse_mapping_file(uploaded_file) -> dict[int, str]:
    """Parse an uploaded mapping file (CSV or JSON) into ``{aws_id: name}``.

    Accepted shapes:
      - CSV with an id column (``aws_id`` or ``id``) and a name column
        (``platform_name`` or ``name``), matched case-insensitively.
      - JSON object: ``{"62045": "TH2_Sys1", "62047": "TH2_Sys2"}``
      - JSON array of objects: ``[{"aws_id": 62045, "platform_name": "TH2_Sys1"}, ...]``

    Raises ``ValueError`` with a user-facing message on anything else, so
    the caller can show it directly in the sidebar.
    """
    name = getattr(uploaded_file, "name", "") or ""
    raw_bytes = uploaded_file.read()

    if name.lower().endswith(".json"):
        try:
            data = json.loads(raw_bytes.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Not valid JSON: {exc}") from exc

        if isinstance(data, dict):
            try:
                return {int(k): str(v) for k, v in data.items()}
            except (TypeError, ValueError) as exc:
                raise ValueError("JSON object keys must be numeric AWS IDs.") from exc

        if isinstance(data, list):
            out: dict[int, str] = {}
            for item in data:
                if not isinstance(item, dict):
                    continue
                aws_id = item.get("aws_id", item.get("id", item.get("AWS_ID")))
                platform_name = item.get("platform_name", item.get("name"))
                if aws_id is None or not platform_name:
                    continue
                out[int(aws_id)] = str(platform_name)
            if not out:
                raise ValueError('JSON array entries need "aws_id"/"id" and "platform_name"/"name" keys.')
            return out

        raise ValueError("JSON mapping must be an object of {id: name} or an array of {aws_id, platform_name}.")

    # Default: treat as CSV.
    try:
        df = pd.read_csv(io.BytesIO(raw_bytes))
    except Exception as exc:
        raise ValueError(f"Could not read as CSV: {exc}") from exc

    cols_lower = {c.lower(): c for c in df.columns}
    id_col = cols_lower.get("aws_id") or cols_lower.get("id")
    name_col = cols_lower.get("platform_name") or cols_lower.get("name")
    if id_col is None or name_col is None:
        raise ValueError(
            "Mapping CSV needs an id column ('aws_id' or 'id') and a name "
            "column ('platform_name' or 'name')."
        )

    out = {}
    for _, row in df.iterrows():
        if pd.isna(row[id_col]) or pd.isna(row[name_col]):
            continue
        out[int(row[id_col])] = str(row[name_col])
    if not out:
        raise ValueError("No valid (id, name) rows found in the mapping CSV.")
    return out
