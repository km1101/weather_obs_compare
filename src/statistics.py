"""
Statistical comparison metrics.

Every function here answers one of the questions in the brief: do the
platforms agree, is one biased high/low, is one noisier, is there drift,
etc. Docstrings spell out the statistical reasoning, not just the formula,
per the "clear comments explaining the statistical reasoning" requirement.

All pairwise functions expect matched-pair input -- i.e. the output of
``preprocessing.align_platforms`` -- so that "row i" is genuinely the same
moment in time for both platforms.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from sklearn.metrics import r2_score

from .config import DIFF_OUTLIER_SIGMA, VALID_RANGES
from ._pandas_compat import freq as _freq


# ---------------------------------------------------------------------------
# Descriptive statistics
# ---------------------------------------------------------------------------

def descriptive_stats(series: pd.Series) -> dict:
    """Standard univariate summary for a single platform/parameter.

    Percentiles are reported at 5/25/50/75/95 -- enough to see skew and
    tail behaviour without cluttering a summary table with every decile.
    """
    clean = series.dropna()
    if clean.empty:
        return {k: np.nan for k in [
            "count", "mean", "median", "std", "variance", "min", "max",
            "p05", "p25", "p50", "p75", "p95",
        ]}
    return {
        "count": int(clean.count()),
        "mean": clean.mean(),
        "median": clean.median(),
        "std": clean.std(ddof=1),
        "variance": clean.var(ddof=1),
        "min": clean.min(),
        "max": clean.max(),
        "p05": clean.quantile(0.05),
        "p25": clean.quantile(0.25),
        "p50": clean.quantile(0.50),
        "p75": clean.quantile(0.75),
        "p95": clean.quantile(0.95),
    }


def descriptive_stats_table(df: pd.DataFrame, parameter: str) -> pd.DataFrame:
    """Descriptive stats for every platform present, one row each."""
    rows = {}
    for platform, sub in df.groupby("platform"):
        rows[platform] = descriptive_stats(sub[parameter])
    return pd.DataFrame(rows).T


# ---------------------------------------------------------------------------
# Pairwise comparison (matched timestamps)
# ---------------------------------------------------------------------------

def pairwise_comparison(matched: pd.DataFrame, col_a: str, col_b: str) -> dict:
    """Agreement metrics between two matched series (A = reference-ish, B = test).

    The choice of which platform is "A" only matters for the sign of Bias
    and Mean Error -- everything else is symmetric. Definitions:

    - Bias / Mean Error: mean(B - A). Positive => B reads high relative to A.
      (Reported as one value; "Bias" and "Mean Error" are the same quantity
      under two names commonly used in different fields -- both are kept in
      the output so users searching for either term find it.)
    - MAE: mean(|B - A|) -- typical size of disagreement, ignoring direction.
    - RMSE: sqrt(mean((B - A)^2)) -- like MAE but penalises large individual
      discrepancies more heavily; RMSE >> MAE indicates occasional big
      misses rather than uniform small noise.
    - Scatter Index: RMSE normalised by mean(A), expressed as a fraction.
      A dimensionless "how big are the errors relative to the signal"
      metric, standard in oceanographic/met instrument comparison.
    - Pearson r: linear correlation -- do the two series move together?
    - Spearman rho: rank correlation -- do they move together *monotonically*,
      robust to outliers and nonlinearity that would depress Pearson r.
    - R^2: coefficient of determination of B predicted from A via linear
      fit -- fraction of variance in B explained by A.
    """
    a = matched[col_a].to_numpy(dtype=float)
    b = matched[col_b].to_numpy(dtype=float)
    n = len(a)
    if n < 2:
        return {
            "n_pairs": n, "bias": np.nan, "mean_error": np.nan, "mae": np.nan,
            "rmse": np.nan, "scatter_index": np.nan, "pearson_r": np.nan,
            "pearson_p": np.nan, "spearman_rho": np.nan, "spearman_p": np.nan,
            "r_squared": np.nan, "slope": np.nan, "intercept": np.nan,
        }

    diff = b - a
    bias = float(np.mean(diff))
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    mean_a = np.mean(a)
    scatter_index = float(rmse / mean_a) if mean_a != 0 else np.nan

    pearson_r, pearson_p = sp_stats.pearsonr(a, b)
    spearman_rho, spearman_p = sp_stats.spearmanr(a, b)

    slope, intercept, *_ = sp_stats.linregress(a, b)
    r2 = r2_score(b, slope * a + intercept)

    return {
        "n_pairs": n,
        "bias": bias,
        "mean_error": bias,
        "mae": mae,
        "rmse": rmse,
        "scatter_index": scatter_index,
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "spearman_rho": float(spearman_rho),
        "spearman_p": float(spearman_p),
        "r_squared": float(r2),
        "slope": float(slope),
        "intercept": float(intercept),
    }


def bland_altman_stats(matched: pd.DataFrame, col_a: str, col_b: str) -> dict:
    """Bland-Altman limits of agreement for two matched series.

    Mean difference (bias) +/- 1.96 * SD of the differences brackets the
    range within which 95% of individual disagreements are expected to
    fall, assuming approximately normal differences. Wide limits mean
    poor agreement even if correlation looks good -- correlation and
    agreement are different questions, which is exactly why Bland-Altman
    is included alongside Pearson/Spearman rather than instead of them.
    """
    diff = (matched[col_b] - matched[col_a]).dropna()
    mean_diff = diff.mean()
    sd_diff = diff.std(ddof=1)
    return {
        "mean_diff": float(mean_diff),
        "sd_diff": float(sd_diff),
        "loa_upper": float(mean_diff + 1.96 * sd_diff),
        "loa_lower": float(mean_diff - 1.96 * sd_diff),
    }


def pairwise_comparison_table(
    df: pd.DataFrame,
    platform_a: str,
    platform_b: str,
    parameters: list[str],
    match_method: str = "nearest",
    tolerance_minutes: float = 5,
) -> pd.DataFrame:
    """One row per parameter: Parameter, Unit, N, Bias, MAE, RMSE, Scatter Index, R^2, Pearson r.

    The multi-parameter counterpart to :func:`pairwise_comparison` -- lets
    the dashboard's Summary tab show every selected parameter's agreement
    at a glance in one table, instead of one parameter's metrics at a
    time. Uses the same matched-pair alignment as every other pairwise
    view in the dashboard, so the numbers are always consistent with the
    single-parameter plots.
    """
    from . import preprocessing as preprocessing_mod
    from .config import PARAMETER_UNITS

    rows = []
    for parameter in parameters:
        matched = preprocessing_mod.align_platforms(
            df, platform_a, platform_b, parameter,
            tolerance_minutes=tolerance_minutes, method=match_method,
        )
        metrics = pairwise_comparison(matched, platform_a, platform_b)
        rows.append({
            "Parameter": parameter,
            "Unit": PARAMETER_UNITS.get(parameter, ""),
            "N": metrics["n_pairs"],
            "Bias": metrics["bias"],
            "MAE": metrics["mae"],
            "RMSE": metrics["rmse"],
            "Scatter Index": metrics["scatter_index"],
            "R\u00b2": metrics["r_squared"],
            "Pearson r": metrics["pearson_r"],
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Rolling / time-based statistics
# ---------------------------------------------------------------------------

def rolling_stats(series: pd.Series, window: str = "6H") -> pd.DataFrame:
    """Rolling mean and standard deviation over a time window.

    ``series`` must be indexed by timestamp. A rolling SD that grows over
    time (with mean roughly flat) is a classic noise/drift signature worth
    flagging even before running formal drift tests.
    """
    roll = series.rolling(_freq(window))
    return pd.DataFrame({"rolling_mean": roll.mean(), "rolling_std": roll.std()})


def periodic_stats(df: pd.DataFrame, parameter: str, freq: str) -> pd.DataFrame:
    """Per-platform descriptive stats resampled to a period (D/W/M).

    ``freq`` uses pandas offset aliases: "D" daily, "W" weekly, "M" monthly.
    Returns a long-format table: period, platform, mean, std, count, ...
    """
    rows = []
    for platform, sub in df.groupby("platform"):
        s = sub.set_index("timestamp")[parameter]
        grouped = s.resample(_freq(freq))
        stats_df = grouped.agg(["mean", "std", "min", "max", "count"])
        stats_df["platform"] = platform
        rows.append(stats_df)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows).reset_index().rename(columns={"index": "period", "timestamp": "period"})
    return out


def drift_over_time(matched: pd.DataFrame, col_a: str, col_b: str) -> dict:
    """Linear trend in (B - A) over time -- is the disagreement growing?

    Fits difference ~ time (time in days since first sample) and returns
    the slope in units-per-day. A slope that's statistically significant
    (p < 0.05) and non-trivial in size indicates one platform is drifting
    relative to the other, as opposed to a constant offset (pure bias).
    """
    if matched.empty:
        return {"drift_per_day": np.nan, "p_value": np.nan, "r_squared": np.nan}

    t0 = matched["timestamp"].min()
    days = (matched["timestamp"] - t0).dt.total_seconds() / 86400.0
    diff = matched[col_b] - matched[col_a]

    valid = diff.notna()
    if valid.sum() < 2:
        return {"drift_per_day": np.nan, "p_value": np.nan, "r_squared": np.nan}

    slope, intercept, r_value, p_value, _std_err = sp_stats.linregress(
        days[valid], diff[valid]
    )
    return {
        "drift_per_day": float(slope),
        "p_value": float(p_value),
        "r_squared": float(r_value ** 2),
    }


# ---------------------------------------------------------------------------
# Outlier / data-quality detection
# ---------------------------------------------------------------------------

def detect_range_violations(df: pd.DataFrame, parameter: str) -> pd.Series:
    """Boolean mask: readings outside the physically plausible range.

    Flags sensor faults (e.g. a pressure reading of 0) that a purely
    statistical outlier test might miss if enough of them cluster together.
    Parameters without a configured range always return all-False.
    """
    if parameter not in VALID_RANGES:
        return pd.Series(False, index=df.index)
    lo, hi = VALID_RANGES[parameter]
    values = df[parameter]
    return (values < lo) | (values > hi)


def detect_diff_outliers(
    matched: pd.DataFrame, col_a: str, col_b: str, sigma: float = DIFF_OUTLIER_SIGMA
) -> pd.Series:
    """Boolean mask: matched-pair differences beyond ``sigma`` std devs.

    Flags "extreme differences" -- pairs where the two platforms disagree
    far more than their typical spread, a strong indicator of a bad
    reading on one side (or a timestamp mismatch that slipped through
    alignment) rather than genuine measurement variability.
    """
    diff = matched[col_b] - matched[col_a]
    mean, sd = diff.mean(), diff.std(ddof=1)
    if sd == 0 or np.isnan(sd):
        return pd.Series(False, index=matched.index)
    return (diff - mean).abs() > sigma * sd


def detect_spikes(series: pd.Series, sigma: float = DIFF_OUTLIER_SIGMA) -> pd.Series:
    """Boolean mask: single-point spikes via first-difference z-score.

    A spike is a point that jumps sharply from its immediate neighbour and
    (implicitly) back again -- caught by looking at the point-to-point
    change rather than the raw value, which distinguishes a real but
    unusual value (e.g. a genuine storm gust) less well than looking at the
    absolute level would, but catches transient sensor glitches much better.
    """
    diffs = series.diff()
    mean, sd = diffs.mean(), diffs.std(ddof=1)
    if sd == 0 or np.isnan(sd):
        return pd.Series(False, index=series.index)
    return (diffs - mean).abs() > sigma * sd


# ---------------------------------------------------------------------------
# General-purpose outlier filtering (data-cleaning preprocessing step)
# ---------------------------------------------------------------------------
# These three are the standard univariate outlier tests offered in the
# dashboard's "Outlier filtering" sidebar control. Unlike detect_spikes
# (point-to-point jumps) or detect_diff_outliers (matched-pair
# disagreement), these flag a value as unusual relative to its own
# platform's overall distribution -- the general-purpose "clean the data
# before analysing it" step.

HAMPEL_WINDOW = 7  # samples either side is not exposed in the UI; a fixed,
# moderate window keeps the filter responsive to slow drift/trend without
# needing an extra control most users won't have an opinion on.


def iqr_outlier_mask(series: pd.Series, threshold: float = 1.5) -> pd.Series:
    """Flag values outside [Q1 - threshold*IQR, Q3 + threshold*IQR].

    The classic Tukey fence. Makes no assumption about the shape of the
    distribution (unlike z-score, which assumes roughly-normal data), so
    it's a safe default for skewed physical quantities like wind speed or
    visibility. ``threshold`` is the fence multiplier -- 1.5 is Tukey's
    standard "outlier" fence, 3.0 is his wider "far out" fence.
    """
    clean = series.dropna()
    if len(clean) < 4:
        return pd.Series(False, index=series.index)
    q1, q3 = clean.quantile(0.25), clean.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        return pd.Series(False, index=series.index)
    lower, upper = q1 - threshold * iqr, q3 + threshold * iqr
    return (series < lower) | (series > upper)


def zscore_outlier_mask(series: pd.Series, threshold: float = 3.0) -> pd.Series:
    """Flag values more than ``threshold`` standard deviations from the mean.

    Simple and fast, but sensitive to the very outliers it's trying to
    detect -- a handful of extreme values inflate the mean/SD used to
    judge them, which can mask real outliers ("masking effect"). Prefer
    IQR or Hampel for data with more than a few percent bad readings.
    """
    clean = series.dropna()
    if len(clean) < 2:
        return pd.Series(False, index=series.index)
    mean, sd = clean.mean(), clean.std(ddof=1)
    if sd == 0 or np.isnan(sd):
        return pd.Series(False, index=series.index)
    z = (series - mean) / sd
    return z.abs() > threshold


def hampel_outlier_mask(series: pd.Series, threshold: float = 3.0, window: int = HAMPEL_WINDOW) -> pd.Series:
    """Flag values far from a rolling median, scaled by rolling MAD.

    Unlike the global mean/SD used by z-score, the Hampel filter compares
    each point to its *local* neighbourhood (a rolling median and median
    absolute deviation), so it stays sensitive to outliers even when the
    series has a slow trend or drift that would otherwise shift the global
    mean. The 1.4826 factor scales MAD to be a consistent estimator of the
    standard deviation under a normal distribution, so ``threshold`` reads
    on roughly the same "how many sigma" scale as the z-score threshold.
    """
    clean = series.astype(float)
    rolling_median = clean.rolling(window, center=True, min_periods=1).median()
    abs_dev = (clean - rolling_median).abs()
    rolling_mad = abs_dev.rolling(window, center=True, min_periods=1).median()
    scaled_mad = 1.4826 * rolling_mad
    mask = abs_dev > threshold * scaled_mad
    # Where the local neighbourhood has zero spread (scaled_mad == 0), any
    # nonzero deviation would otherwise be flagged -- treat a flat local
    # neighbourhood as "nothing unusual" rather than "everything unusual".
    mask = mask & (scaled_mad > 0)
    return mask.fillna(False)


_OUTLIER_METHODS = {
    "iqr": iqr_outlier_mask,
    "zscore": zscore_outlier_mask,
    "hampel": hampel_outlier_mask,
}


def apply_outlier_filter(
    df: pd.DataFrame,
    columns: list[str],
    method: str = "iqr",
    threshold: float = 3.0,
    group_by: str = "platform",
) -> tuple[pd.DataFrame, pd.Series]:
    """Flag and null out outliers in ``columns``, independently per platform.

    Values are set to NaN rather than the row being dropped, so a reading
    flagged as an outlier on one parameter doesn't discard perfectly good
    readings of other parameters in the same row. Applied per ``group_by``
    group (platform, by default) so one platform's typical noise level
    doesn't set the threshold for another.

    Returns the cleaned DataFrame and a Series of how many values were
    removed per column, for display in the dashboard.
    """
    if method not in _OUTLIER_METHODS:
        raise ValueError(f"Unknown outlier method: {method!r}. Choose from {list(_OUTLIER_METHODS)}.")
    mask_fn = _OUTLIER_METHODS[method]

    out = df.copy()
    removed_counts = {}
    for col in columns:
        if col not in out.columns or out[col].notna().sum() == 0:
            continue
        mask = out.groupby(group_by)[col].transform(lambda s: mask_fn(s, threshold))
        mask = mask.fillna(False).astype(bool)
        removed_counts[col] = int(mask.sum())
        out.loc[mask, col] = np.nan

    summary = pd.Series(removed_counts, name="values_removed", dtype="int64")
    return out, summary


def missing_data_summary(df: pd.DataFrame, parameter: str, freq: str = "1H") -> pd.DataFrame:
    """Expected-vs-actual sample counts per platform per time bucket.

    Compares how many readings arrived in each bucket against the busiest
    platform's count in that bucket, as a proxy for "how many are we
    missing" without needing to assume a fixed nominal sample rate that
    might not hold for every platform.
    """
    rows = []
    for platform, sub in df.groupby("platform"):
        s = sub.set_index("timestamp")[parameter].dropna()
        counts = s.resample(_freq(freq)).count()
        counts.name = platform
        rows.append(counts)
    if not rows:
        return pd.DataFrame()
    table = pd.concat(rows, axis=1).fillna(0).astype(int)
    return table
