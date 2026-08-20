"""
Plotly figure builders.

Every function returns a ``plotly.graph_objects.Figure`` and touches no
Streamlit/Dash-specific code, so the plotting layer can be reused by any
dashboard framework or exported standalone (PNG/HTML). Zoom, hover, and
legend-filtering come for free from Plotly; each builder just makes sure
hover text and legends are actually informative.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .config import PARAMETER_UNITS
from . import statistics as stats_mod
from ._pandas_compat import freq as _freq

_TEMPLATE = "plotly_white"


def _unit_label(parameter: str) -> str:
    unit = PARAMETER_UNITS.get(parameter, "")
    return f"{parameter} ({unit})" if unit else parameter


# ---------------------------------------------------------------------------
# Time series
# ---------------------------------------------------------------------------

def time_series_overlay(df: pd.DataFrame, parameter: str) -> go.Figure:
    """One line per platform, shared time axis. The core "eyeball it" plot."""
    fig = px.line(
        df.sort_values("timestamp"),
        x="timestamp", y=parameter, color="platform",
        template=_TEMPLATE,
        labels={"timestamp": "Time", parameter: _unit_label(parameter), "platform": "Platform"},
        title=f"{parameter} — time series overlay",
    )
    fig.update_traces(mode="lines+markers", marker=dict(size=3))
    fig.update_layout(hovermode="x unified", legend_title="Platform")
    return fig


def difference_time_series(matched: pd.DataFrame, col_a: str, col_b: str, parameter: str) -> go.Figure:
    """(B - A) over time. Flat around zero = good agreement, no drift."""
    diff = matched[col_b] - matched[col_a]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=matched["timestamp"], y=diff, mode="lines+markers",
        marker=dict(size=3), name=f"{col_b} − {col_a}",
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(
        template=_TEMPLATE,
        title=f"{parameter}: difference time series ({col_b} − {col_a})",
        xaxis_title="Time", yaxis_title=f"Difference ({PARAMETER_UNITS.get(parameter, '')})",
    )
    return fig


def rolling_stats_plot(series: pd.Series, parameter: str, window: str = "6H") -> go.Figure:
    """Rolling mean with a shaded +/- 1 SD band, for one platform/parameter."""
    roll = stats_mod.rolling_stats(series, window=window)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=roll.index, y=roll["rolling_mean"] + roll["rolling_std"],
        mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=roll.index, y=roll["rolling_mean"] - roll["rolling_std"],
        mode="lines", line=dict(width=0), fill="tonexty",
        fillcolor="rgba(31,119,180,0.2)", name="±1 SD",
    ))
    fig.add_trace(go.Scatter(
        x=roll.index, y=roll["rolling_mean"], mode="lines",
        line=dict(color="rgb(31,119,180)", width=2), name="Rolling mean",
    ))
    fig.update_layout(
        template=_TEMPLATE, title=f"{parameter}: rolling mean ± SD (window={window})",
        xaxis_title="Time", yaxis_title=_unit_label(parameter),
    )
    return fig


def monthly_summary_plot(df: pd.DataFrame, parameter: str) -> go.Figure:
    """Monthly mean per platform with min-max whiskers, as an error-bar plot."""
    monthly = stats_mod.periodic_stats(df, parameter, freq="M")
    if monthly.empty:
        return go.Figure()
    fig = go.Figure()
    for platform, sub in monthly.groupby("platform"):
        sub = sub.sort_values("period")
        fig.add_trace(go.Scatter(
            x=sub["period"], y=sub["mean"], mode="lines+markers", name=platform,
            error_y=dict(
                type="data",
                array=sub["max"] - sub["mean"],
                arrayminus=sub["mean"] - sub["min"],
                visible=True,
            ),
        ))
    fig.update_layout(
        template=_TEMPLATE, title=f"{parameter}: monthly summary (mean, min–max range)",
        xaxis_title="Month", yaxis_title=_unit_label(parameter),
    )
    return fig


# ---------------------------------------------------------------------------
# Distribution
# ---------------------------------------------------------------------------

def histogram(df: pd.DataFrame, parameter: str, nbins: int = 40) -> go.Figure:
    fig = px.histogram(
        df, x=parameter, color="platform", barmode="overlay", nbins=nbins,
        opacity=0.55, template=_TEMPLATE,
        labels={parameter: _unit_label(parameter)},
        title=f"{parameter}: histogram by platform",
    )
    return fig


def kde_plot(df: pd.DataFrame, parameter: str) -> go.Figure:
    """Kernel density estimate per platform, using scipy's gaussian_kde."""
    from scipy.stats import gaussian_kde

    fig = go.Figure()
    for platform, sub in df.groupby("platform"):
        values = sub[parameter].dropna().to_numpy()
        if len(values) < 2 or np.isclose(values.std(), 0):
            continue
        kde = gaussian_kde(values)
        x_grid = np.linspace(values.min(), values.max(), 200)
        fig.add_trace(go.Scatter(x=x_grid, y=kde(x_grid), mode="lines", name=platform, fill="tozeroy", opacity=0.5))
    fig.update_layout(
        template=_TEMPLATE, title=f"{parameter}: kernel density estimate",
        xaxis_title=_unit_label(parameter), yaxis_title="Density",
    )
    return fig


def box_plot(df: pd.DataFrame, parameter: str) -> go.Figure:
    fig = px.box(
        df, x="platform", y=parameter, color="platform", points="outliers",
        template=_TEMPLATE, labels={parameter: _unit_label(parameter)},
        title=f"{parameter}: box plot by platform",
    )
    return fig


def violin_plot(df: pd.DataFrame, parameter: str) -> go.Figure:
    fig = px.violin(
        df, x="platform", y=parameter, color="platform", box=True, points=False,
        template=_TEMPLATE, labels={parameter: _unit_label(parameter)},
        title=f"{parameter}: violin plot by platform",
    )
    return fig


def cdf_plot(df: pd.DataFrame, parameter: str) -> go.Figure:
    fig = go.Figure()
    for platform, sub in df.groupby("platform"):
        values = np.sort(sub[parameter].dropna().to_numpy())
        if len(values) == 0:
            continue
        y = np.arange(1, len(values) + 1) / len(values)
        fig.add_trace(go.Scatter(x=values, y=y, mode="lines", name=platform))
    fig.update_layout(
        template=_TEMPLATE, title=f"{parameter}: cumulative distribution function",
        xaxis_title=_unit_label(parameter), yaxis_title="Cumulative probability",
    )
    return fig


# ---------------------------------------------------------------------------
# Agreement
# ---------------------------------------------------------------------------

def scatter_comparison(matched: pd.DataFrame, col_a: str, col_b: str, parameter: str) -> go.Figure:
    """Scatter of B vs A with a 1:1 line and a fitted regression line.

    The 1:1 line is what perfect agreement looks like; the fitted
    regression line shows what the platforms actually do. The gap between
    them is bias/slope error at a glance, before reading any numbers.
    """
    from scipy import stats as sp_stats

    a = matched[col_a].to_numpy(dtype=float)
    b = matched[col_b].to_numpy(dtype=float)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=a, y=b, mode="markers", name="Matched pairs",
        marker=dict(size=5, opacity=0.5),
        hovertemplate=f"{col_a}: %{{x:.2f}}<br>{col_b}: %{{y:.2f}}<extra></extra>",
    ))

    lo, hi = float(np.nanmin([a.min(), b.min()])), float(np.nanmax([a.max(), b.max()]))
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(dash="dash", color="gray"), name="1:1 line"))

    if len(a) >= 2:
        slope, intercept, *_ = sp_stats.linregress(a, b)
        x_fit = np.array([lo, hi])
        fig.add_trace(go.Scatter(x=x_fit, y=slope * x_fit + intercept, mode="lines", line=dict(color="firebrick"), name=f"Fit (y={slope:.2f}x+{intercept:.2f})"))

    fig.update_layout(
        template=_TEMPLATE, title=f"{parameter}: {col_b} vs {col_a}",
        xaxis_title=_unit_label(col_a) if col_a in PARAMETER_UNITS else col_a,
        yaxis_title=col_b,
    )
    return fig


def bland_altman_plot(matched: pd.DataFrame, col_a: str, col_b: str, parameter: str) -> go.Figure:
    mean_vals = (matched[col_a] + matched[col_b]) / 2
    diff_vals = matched[col_b] - matched[col_a]
    ba = stats_mod.bland_altman_stats(matched, col_a, col_b)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=mean_vals, y=diff_vals, mode="markers", marker=dict(size=5, opacity=0.5), name="Pairs"))
    fig.add_hline(y=ba["mean_diff"], line_color="firebrick", annotation_text=f"Mean diff = {ba['mean_diff']:.3f}")
    fig.add_hline(y=ba["loa_upper"], line_dash="dash", line_color="gray", annotation_text="+1.96 SD")
    fig.add_hline(y=ba["loa_lower"], line_dash="dash", line_color="gray", annotation_text="−1.96 SD")
    fig.update_layout(
        template=_TEMPLATE, title=f"{parameter}: Bland–Altman ({col_b} vs {col_a})",
        xaxis_title=f"Mean of {col_a} & {col_b}", yaxis_title=f"Difference ({col_b} − {col_a})",
    )
    return fig


def residual_plot(matched: pd.DataFrame, col_a: str, col_b: str, parameter: str) -> go.Figure:
    """Residuals of the A->B linear fit, vs A. Fanning/curvature = nonlinearity."""
    from scipy import stats as sp_stats

    a = matched[col_a].to_numpy(dtype=float)
    b = matched[col_b].to_numpy(dtype=float)
    if len(a) < 2:
        return go.Figure()
    slope, intercept, *_ = sp_stats.linregress(a, b)
    residuals = b - (slope * a + intercept)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=a, y=residuals, mode="markers", marker=dict(size=5, opacity=0.5)))
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(
        template=_TEMPLATE, title=f"{parameter}: residuals of linear fit ({col_b} ~ {col_a})",
        xaxis_title=col_a, yaxis_title="Residual",
    )
    return fig


def correlation_matrix(df: pd.DataFrame, parameters: list[str], platform: str | None = None) -> go.Figure:
    """Heatmap of pairwise Pearson correlation across parameters.

    Restricted to one platform at a time by default (correlating different
    parameters within a platform's own readings); pass ``platform=None`` to
    correlate across the whole pooled dataset instead.
    """
    data = df if platform is None else df[df["platform"] == platform]
    corr = data[parameters].corr(method="pearson")
    fig = px.imshow(
        corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
        template=_TEMPLATE,
        title=f"Correlation matrix{' — ' + platform if platform else ' (all platforms pooled)'}",
    )
    return fig


def platform_heatmap(df: pd.DataFrame, parameter: str, freq: str = "1D") -> go.Figure:
    """Heatmap of parameter mean, platforms x time-bucket.

    Good for spotting a platform going offline (a gap column) or drifting
    over a long deployment at a glance, without opening every time series.
    """
    pivot = (
        df.set_index("timestamp")
        .groupby("platform")[parameter]
        .resample(_freq(freq))
        .mean()
        .unstack(level=0)
        .T
    )
    fig = px.imshow(
        pivot, aspect="auto", color_continuous_scale="Viridis", template=_TEMPLATE,
        labels=dict(x="Time", y="Platform", color=_unit_label(parameter)),
        title=f"{parameter}: platform x time heatmap",
    )
    return fig
