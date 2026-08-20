"""
Preprocessing: dedup, filtering, and cross-platform timestamp alignment.

The comparison metrics in ``statistics.py`` all need *matched pairs* --
one Platform-A reading and one Platform-B reading for (as close as
possible to) the same instant. Two loggers rarely sample on exactly the
same clock tick, so this module provides configurable nearest-neighbour
matching (with a tolerance) as well as a stricter exact-timestamp merge.
"""

from __future__ import annotations

import pandas as pd

from .config import DEFAULT_MATCH_TOLERANCE_MINUTES
from ._pandas_compat import freq as _freq


def deduplicate_timestamps(df: pd.DataFrame, agg: str = "mean") -> pd.DataFrame:
    """Collapse duplicate (platform, timestamp) rows.

    A platform occasionally logs two records at the same timestamp (clock
    quantisation, retransmit, etc.). We aggregate numeric columns with
    ``agg`` (mean by default) rather than arbitrarily keeping "first" or
    "last", since neither is more correct than the other for a duplicate.
    """
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    agg_map = {c: agg for c in numeric_cols}
    grouped = (
        df.groupby(["platform", "timestamp"], as_index=False)
        .agg(agg_map)
    )
    return grouped.sort_values("timestamp").reset_index(drop=True)


def filter_date_range(
    df: pd.DataFrame, start: pd.Timestamp | None, end: pd.Timestamp | None
) -> pd.DataFrame:
    """Restrict ``df`` to timestamps within [start, end], inclusive.

    Either bound may be None to leave that side open. Dates are compared
    at day granularity on the end bound so a UI date-picker value like
    "18/08/2026" includes the whole day.
    """
    out = df
    if start is not None:
        out = out[out["timestamp"] >= pd.Timestamp(start)]
    if end is not None:
        end_inclusive = pd.Timestamp(end) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        out = out[out["timestamp"] <= end_inclusive]
    return out


def align_platforms(
    df: pd.DataFrame,
    platform_a: str,
    platform_b: str,
    parameter: str,
    tolerance_minutes: float = DEFAULT_MATCH_TOLERANCE_MINUTES,
    method: str = "nearest",
) -> pd.DataFrame:
    """Match Platform A and Platform B readings of ``parameter`` in time.

    Parameters
    ----------
    method:
        ``"nearest"`` -- nearest-neighbour match within ``tolerance_minutes``
        (default). Handles the common case where two loggers sample on
        slightly offset clocks.
        ``"exact"`` -- only pairs where both platforms share the exact same
        timestamp. Stricter; useful when both loggers are known to be
        clock-synced and any drift should itself be treated as a data
        problem rather than smoothed over.
        ``"interpolate"`` -- resample Platform B onto Platform A's
        timestamps via linear interpolation. Useful when comparing a sparse
        or irregular source (e.g. a model run) against a densely-sampled
        platform.

    Returns
    -------
    DataFrame with columns: timestamp, <platform_a>, <platform_b>
    """
    a = (
        df[df["platform"] == platform_a][["timestamp", parameter]]
        .dropna()
        .sort_values("timestamp")
        .rename(columns={parameter: platform_a})
    )
    b = (
        df[df["platform"] == platform_b][["timestamp", parameter]]
        .dropna()
        .sort_values("timestamp")
        .rename(columns={parameter: platform_b})
    )

    if method == "exact":
        merged = pd.merge(a, b, on="timestamp", how="inner")

    elif method == "nearest":
        merged = pd.merge_asof(
            a,
            b,
            on="timestamp",
            direction="nearest",
            tolerance=pd.Timedelta(minutes=tolerance_minutes),
        )
        merged = merged.dropna(subset=[platform_a, platform_b])

    elif method == "interpolate":
        b_indexed = b.set_index("timestamp")[platform_b]
        # Reindex onto A's timestamps, adding B's own timestamps first so
        # the interpolation has real data to work from, then select A's
        # timestamps back out.
        combined_index = b_indexed.index.union(a["timestamp"])
        b_interp = b_indexed.reindex(combined_index).interpolate(method="time")
        b_on_a = b_interp.reindex(a["timestamp"])
        merged = a.copy()
        merged[platform_b] = b_on_a.values
        merged = merged.dropna(subset=[platform_a, platform_b])

    else:
        raise ValueError(f"Unknown alignment method: {method!r}")

    return merged.reset_index(drop=True)


def resample_platform(
    df: pd.DataFrame, platform: str, parameter: str, freq: str = "1H", agg: str = "mean"
) -> pd.Series:
    """Resample one platform's parameter to a regular frequency.

    Useful for rolling statistics and for visually comparing platforms that
    sample at different native rates. ``freq`` uses pandas offset aliases
    (e.g. "1H", "1D", "1W").
    """
    sub = df[df["platform"] == platform].set_index("timestamp")[parameter]
    return sub.resample(_freq(freq)).agg(agg)
