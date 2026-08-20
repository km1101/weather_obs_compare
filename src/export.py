"""
Export helpers: statistics tables to CSV/XLSX, figures to PNG/HTML.

Kept separate from the dashboard so the same export logic works from a
script or a notebook, not only from Streamlit's download buttons.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go


def stats_to_csv(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    df.to_csv(path)
    return path


def stats_to_excel(tables: dict[str, pd.DataFrame], path: str | Path) -> Path:
    """Write multiple stats tables to one workbook, one sheet each."""
    path = Path(path)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for sheet_name, table in tables.items():
            # Excel sheet names are capped at 31 characters.
            table.to_excel(writer, sheet_name=sheet_name[:31])
    return path


def figure_to_png(fig: go.Figure, path: str | Path, width: int = 1200, height: int = 700, scale: int = 2) -> Path:
    """Export a Plotly figure to PNG. Requires the ``kaleido`` package."""
    path = Path(path)
    fig.write_image(str(path), width=width, height=height, scale=scale)
    return path


def figure_to_html(fig: go.Figure, path: str | Path) -> Path:
    path = Path(path)
    fig.write_html(str(path), include_plotlyjs="cdn")
    return path
