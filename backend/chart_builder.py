"""
chart_builder.py — Converts DuckDB query results into Plotly figure JSON.

Supports: bar, line, pie, scatter, histogram, table (returns None for table
type — the frontend renders raw data as a table in that case).

The frontend receives the Plotly figure as a JSON dict and renders it with
the Plotly.js library (no server-side rendering needed).
"""

import logging
from typing import Optional, Tuple, Any

import pandas as pd

logger = logging.getLogger(__name__)

# Consistent color palette across charts (can be used by frontend)
COLOR_SEQUENCE = [
    "#6366f1", "#8b5cf6", "#06b6d4", "#10b981",
    "#f59e0b", "#ef4444", "#ec4899", "#14b8a6",
]


def build_chart(
    df: pd.DataFrame,
    chart_type: str,
    question: str = "",
) -> Optional[dict]:
    """
    Build a Recharts-compatible chart structure from a DataFrame.

    Returns a dict with `data` and `config` keys for the frontend,
    or None if chart_type is 'table'.
    """
    if df.empty:
        return None

    if chart_type == "table":
        return None  # Frontend handles table rendering

    try:
        chart_data, chart_config = _dispatch(df, chart_type, question)
        if chart_data is None or chart_config is None:
            return None
        return {"data": chart_data, "config": chart_config}
    except Exception as e:
        logger.warning(f"Chart build failed ({chart_type}): {e}. Falling back to table.")
        return None


def _dispatch(df: pd.DataFrame, chart_type: str, question: str) -> Tuple[Optional[list], Optional[dict]]:
    """Route to the appropriate chart builder."""
    # Convert to records format for Recharts compatibility
    data = df.to_dict(orient="records")
    cols = list(df.columns)
    n_cols = len(cols)

    if chart_type == "bar":
        return _bar(data, cols)
    elif chart_type == "line":
        return _line(data, cols)
    elif chart_type == "pie":
        return _pie(data, cols)
    elif chart_type == "scatter":
        return _scatter(data, cols)
    elif chart_type == "histogram":
        return _histogram(data, cols)
    else:
        # Fallback: try bar if ≥2 cols, else None
        if n_cols >= 2:
            return _bar(data, cols)
        return None, None


# ---------------------------------------------------------------------------
# Individual chart builders
# ---------------------------------------------------------------------------

def _bar(data: list[dict], cols: list[str]):
    """
    Bar chart. First column = x (category), remaining numeric cols = bars.
    """
    if not data or not cols:
        return None, None

    x_col = cols[0]
    y_cols = [c for c in cols[1:] if isinstance(data[0].get(c), (int, float))]

    if not y_cols:
        logger.warning("Bar: no numeric columns found")
        return None, None

    return data, {"x": x_col, "y": y_cols}


def _line(data: list[dict], cols: list[str]):
    """Line chart. First column = x axis."""
    if not data or not cols:
        return None, None

    x_col = cols[0]
    y_cols = [c for c in cols[1:] if isinstance(data[0].get(c), (int, float))]

    if not y_cols:
        return None, None

    return data, {"x": x_col, "y": y_cols}


def _pie(data: list[dict], cols: list[str]):
    """Pie chart. First col = labels, first numeric col = values."""
    if not data or len(cols) < 2:
        return None, None

    label_col = cols[0]
    value_cols = [c for c in cols[1:] if isinstance(data[0].get(c), (int, float))]
    if not value_cols:
        return None, None

    value_col = value_cols[0]

    # Data limiting (e.g., nlargest) is a frontend concern for interactive charts.
    return data, {"names": label_col, "values": value_col}


def _scatter(data: list[dict], cols: list[str]):
    """Scatter plot. First two numeric columns = x, y."""
    if not data or not cols:
        return None, None

    numeric_cols = [c for c in cols if isinstance(data[0].get(c), (int, float))]
    if len(numeric_cols) < 2:
        return None, None

    x_col, y_col = numeric_cols[0], numeric_cols[1]
    return data, {"x": x_col, "y": y_col}


def _histogram(data: list[dict], cols: list[str]):
    """Histogram of the first numeric column."""
    if not data or not cols:
        return None, None

    numeric_cols = [c for c in cols if isinstance(data[0].get(c), (int, float))]
    if not numeric_cols:
        return None, None

    # For a histogram, Recharts typically just needs the data and the key to plot.
    # The binning is done on the frontend.
    return data, {"x": numeric_cols[0]}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_label(col: str) -> str:
    """Convert snake_case column names to Title Case labels."""
    return col.replace("_", " ").title()