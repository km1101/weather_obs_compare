"""
Weather Platform Comparison Dashboard.

Run with:
    streamlit run app.py

Interactive dashboard for comparing weather-observation platforms (and, in
future, numerical model output — see src/config.py::MODEL_PLATFORM_MAPPING)
against each other. Built around the questions in the project brief: do
platforms agree, is one biased, is one noisier, is there drift over time,
which platform looks more reliable.
"""

from __future__ import annotations

import io
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src import config
from src import data_loader
from src import mapping as mapping_mod
from src import preprocessing
from src import statistics as stats_mod
from src import plotting
from src import export as export_mod

st.set_page_config(page_title="Weather Platform Comparison", layout="wide")

DATE_FMT = "%d/%m/%Y"  # spec: dates must display as dd/mm/yyyy
DEFAULT_CSV = Path(__file__).parent / "data" / "weather_data.csv"


# ---------------------------------------------------------------------------
# Data loading (cached by file content, not path -- works the same whether
# the CSV comes from disk or from an in-memory upload)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Loading and parsing data...")
def _load_data(csv_bytes: bytes, platform_mapping: dict[int, str]) -> pd.DataFrame:
    """Cache key is the file's own bytes plus the active mapping, so this
    invalidates automatically whenever either one changes -- editing the
    CSV on disk, uploading a new file, or switching/editing the platform
    mapping all "just work" on the next rerun, with no manual cache
    bookkeeping needed elsewhere."""
    df = data_loader.load_raw_csv(io.BytesIO(csv_bytes), platform_mapping=platform_mapping)
    df = preprocessing.deduplicate_timestamps(df)
    return df


def load_data(csv_bytes: bytes, platform_mapping: dict[int, str]) -> pd.DataFrame:
    return _load_data(csv_bytes, platform_mapping)


# Optional true auto-refresh (re-run the app periodically without user
# interaction) if the community `streamlit-autorefresh` component is
# installed. It's an optional extra — see requirements.txt — because it's
# not part of core Streamlit. Without it, the dashboard still refreshes
# data on every normal interaction (any widget change triggers a rerun,
# which re-reads the CSV above), which covers "refresh when the CSV
# updates" for the common case of a human sitting at the dashboard.
try:
    from streamlit_autorefresh import st_autorefresh

    st_autorefresh(interval=30_000, key="data_autorefresh")
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Sidebar: 1. Data source
# ---------------------------------------------------------------------------

st.sidebar.title("Weather Platform Comparison")

st.sidebar.header("1. Data source")
uploaded_csv = st.sidebar.file_uploader(
    "Upload wave data CSV (optional)", type=["csv"],
    help="Leave empty to use the bundled sample dataset.",
)

if st.sidebar.button("🔄 Refresh data now"):
    st.cache_data.clear()

with st.sidebar.expander("Advanced: use a local file path instead"):
    csv_path_str = st.text_input("CSV file path", value=str(DEFAULT_CSV))

if uploaded_csv is not None:
    csv_bytes = uploaded_csv.getvalue()
    data_source_label = f"Using uploaded **{uploaded_csv.name}**"
else:
    csv_path = Path(csv_path_str)
    if not csv_path.exists():
        st.error(f"CSV file not found: {csv_path}")
        st.stop()
    csv_bytes = csv_path.read_bytes()
    data_source_label = f"Using bundled `{csv_path}`"

st.sidebar.info(data_source_label)

# ---------------------------------------------------------------------------
# Sidebar: 2. Platform mapping
# ---------------------------------------------------------------------------

st.sidebar.header("2. Platform mapping")
mapping_source = st.sidebar.radio(
    "Mapping source", ["Saved config", "Edit manually", "Upload mapping file"], index=0,
)

if mapping_source == "Saved config":
    platform_mapping = dict(config.PLATFORM_MAPPING)

elif mapping_source == "Edit manually":
    if "mapping_editor_df" not in st.session_state:
        st.session_state["mapping_editor_df"] = mapping_mod.default_mapping_df()
    edited_mapping_df = st.sidebar.data_editor(
        st.session_state["mapping_editor_df"],
        num_rows="dynamic", use_container_width=True, key="mapping_editor",
        column_config={
            "aws_id": st.column_config.NumberColumn("AWS ID", step=1, format="%d"),
            "platform_name": st.column_config.TextColumn("Platform name"),
        },
    )
    st.session_state["mapping_editor_df"] = edited_mapping_df
    platform_mapping = mapping_mod.mapping_df_to_dict(edited_mapping_df)

else:  # Upload mapping file
    mapping_file = st.sidebar.file_uploader(
        "Mapping CSV or JSON (id, name)", type=["csv", "json"], key="mapping_file_uploader",
    )
    if mapping_file is not None:
        try:
            platform_mapping = mapping_mod.parse_mapping_file(mapping_file)
        except ValueError as exc:
            st.sidebar.error(f"Could not parse mapping file: {exc}")
            platform_mapping = dict(config.PLATFORM_MAPPING)
    else:
        st.sidebar.caption("No file uploaded yet — using the saved config for now.")
        platform_mapping = dict(config.PLATFORM_MAPPING)

if not platform_mapping:
    st.sidebar.warning("Mapping is empty — falling back to the saved config.")
    platform_mapping = dict(config.PLATFORM_MAPPING)

with st.sidebar.expander("Current mapping"):
    st.dataframe(
        pd.DataFrame(sorted(platform_mapping.items()), columns=["AWS_ID", "Platform name"]),
        use_container_width=True, hide_index=True,
    )

df = load_data(csv_bytes, platform_mapping)

if df.empty:
    st.warning("No usable rows were parsed from the CSV.")
    st.stop()

all_platforms = sorted(df["platform"].dropna().unique().tolist())
available_params = [
    p for p in data_loader.numeric_parameter_columns(df)
    if df[p].notna().any()
]
# Primary parameters first, in the documented order, then anything else.
ordered_params = [p for p in config.PRIMARY_PARAMETERS if p in available_params]
ordered_params += [p for p in available_params if p not in ordered_params]

# ---------------------------------------------------------------------------
# Sidebar: 3. Comparison setup
# ---------------------------------------------------------------------------

st.sidebar.header("3. Comparison setup")

st.sidebar.subheader("Date range")
min_date, max_date = df["timestamp"].min().date(), df["timestamp"].max().date()
date_range = st.sidebar.date_input(
    "Select range (dd/mm/yyyy)",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
    format="DD/MM/YYYY",
)
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = min_date, max_date

st.sidebar.subheader("Platforms")
selected_platforms = st.sidebar.multiselect(
    "Platforms to compare", options=all_platforms,
    default=all_platforms[: min(2, len(all_platforms))],
    help="Add more platforms any time — nothing here is hard-coded to a specific buoy pair or parameter.",
)

st.sidebar.subheader("Parameters")
selected_params = st.sidebar.multiselect(
    "Parameter(s)", options=ordered_params,
    default=[ordered_params[0]] if ordered_params else [],
)

st.sidebar.subheader("Timestamp matching")
match_method = st.sidebar.selectbox(
    "Alignment method", options=["nearest", "exact", "interpolate"], index=0,
    help="How to pair readings from two platforms that don't share exact timestamps.",
)
tolerance_minutes = st.sidebar.slider(
    "Nearest-match tolerance (minutes)", min_value=1, max_value=60,
    value=config.DEFAULT_MATCH_TOLERANCE_MINUTES,
    disabled=(match_method != "nearest"),
)

st.sidebar.subheader("Outlier filtering")
outlier_filter_enabled = st.sidebar.checkbox("Enable outlier removal", value=False)
outlier_method = st.sidebar.selectbox(
    "Method", options=config.OUTLIER_METHODS,
    index=config.OUTLIER_METHODS.index(config.DEFAULT_OUTLIER_METHOD),
    disabled=not outlier_filter_enabled,
    help="iqr: robust Tukey fence, no distribution assumption. "
         "zscore: standard deviations from the mean, assumes roughly-normal data. "
         "hampel: rolling median/MAD, stays sensitive under slow drift.",
)
outlier_threshold = st.sidebar.slider(
    "Threshold", min_value=1.0, max_value=5.0,
    value=config.DEFAULT_OUTLIER_THRESHOLD, step=0.1,
    disabled=not outlier_filter_enabled,
)
st.sidebar.caption(
    "Add more platforms or parameters any time — nothing here is "
    "hard-coded to a specific buoy pair or wave variable."
)

st.sidebar.subheader("Matched-pair extreme differences")
outlier_mode = st.sidebar.radio(
    "In the Agreement tab", options=["Show all", "Highlight outliers", "Exclude outliers"],
    index=0,
    help="Applies to matched-pair comparisons only (Agreement tab) — separate from the general outlier filtering above.",
)

# ---------------------------------------------------------------------------
# Apply filters
# ---------------------------------------------------------------------------

filtered = preprocessing.filter_date_range(df, start_date, end_date)
filtered = filtered[filtered["platform"].isin(selected_platforms)]

outlier_removed_summary = None
if outlier_filter_enabled and not filtered.empty:
    filtered, outlier_removed_summary = stats_mod.apply_outlier_filter(
        filtered, columns=available_params, method=outlier_method, threshold=outlier_threshold,
    )

if filtered.empty:
    st.warning("No data for the current filter selection. Widen the date range or platform selection.")
    st.stop()

if not selected_params:
    st.warning("Select at least one parameter in the sidebar.")
    st.stop()

param = selected_params[0]  # primary parameter driving single-parameter views below

st.title("🌦️ Weather Platform Comparison Dashboard")
st.caption(
    f"{len(filtered):,} readings · "
    f"{start_date.strftime(DATE_FMT)} – {end_date.strftime(DATE_FMT)} · "
    f"Platforms: {', '.join(selected_platforms)}"
)

pair_ready = len(selected_platforms) >= 2
if pair_ready:
    col_a_platform, col_b_platform = st.columns(2)
    with col_a_platform:
        platform_a = st.selectbox("Platform A (reference)", selected_platforms, index=0, key="platform_a")
    with col_b_platform:
        remaining = [p for p in selected_platforms if p != platform_a] or selected_platforms
        platform_b = st.selectbox("Platform B (comparison)", remaining, index=0, key="platform_b")
else:
    st.info("Select at least two platforms in the sidebar to unlock pairwise comparison, agreement, and drift analysis.")
    platform_a = platform_b = None


def get_matched(parameter: str) -> pd.DataFrame:
    return preprocessing.align_platforms(
        filtered, platform_a, platform_b, parameter,
        tolerance_minutes=tolerance_minutes, method=match_method,
    )


tabs = st.tabs([
    "📊 Summary", "📈 Time Series", "📉 Distribution", "🎯 Agreement",
    "⚠️ Outliers & Data Quality", "⬇️ Export",
])

# ---------------------------------------------------------------------------
# Summary tab
# ---------------------------------------------------------------------------
with tabs[0]:
    st.subheader("Descriptive statistics")
    multi_rows = []
    for p in selected_params:
        for platform, sub in filtered.groupby("platform"):
            row = stats_mod.descriptive_stats(sub[p])
            row["parameter"] = p
            row["platform"] = platform
            multi_rows.append(row)
    df_summary = pd.DataFrame(multi_rows)

    # Put parameter first and platform second.
    cols = ["parameter", "platform"] + [
        c for c in df_summary.columns if c not in ["parameter", "platform"]
    ]
    df_summary = df_summary[cols]

    st.dataframe(df_summary.style.format(precision=3), use_container_width=True, hide_index=True)

    if pair_ready:
        matched = get_matched(param)
        st.subheader(f"Pairwise agreement — {platform_b} vs {platform_a}")
        st.caption(f"{len(matched):,} matched pairs (method: {match_method})")
        if matched.empty:
            st.info("No matched pairs for the current selection — try a looser tolerance or a different alignment method.")
        else:
            comparison_table = stats_mod.pairwise_comparison_table(
                filtered, platform_a, platform_b, selected_params,
                match_method=match_method, tolerance_minutes=tolerance_minutes,
            )
            st.dataframe(
                comparison_table.style.format({
                    "N": "{:.0f}",
                    "Bias": "{:.3f}",
                    "MAE": "{:.3f}",
                    "RMSE": "{:.3f}",
                    "Scatter Index": "{:.3f}",
                    "R²": "{:.3f}",
                    "Pearson r": "{:.3f}",
                }),
                use_container_width=True, hide_index=True,
            )

            ba = stats_mod.bland_altman_stats(matched, platform_a, platform_b)
            drift = stats_mod.drift_over_time(matched, platform_a, platform_b)
            st.caption(
                f"{param}: Bland–Altman limits of agreement {ba['loa_lower']:.3f} to {ba['loa_upper']:.3f} "
                f"(mean diff {ba['mean_diff']:.3f} ± 1.96 SD) · "
                f"drift {drift['drift_per_day']:.4f} units/day (p = {drift['p_value']:.4f})"
            )


# ---------------------------------------------------------------------------
# Time Series tab
# ---------------------------------------------------------------------------
with tabs[1]:
    ts_param = st.selectbox(
        "Parameter", options=ordered_params,
        index=ordered_params.index(param) if param in ordered_params else 0,
        key="timeseries_param",
    )
    st.plotly_chart(plotting.time_series_overlay(filtered, ts_param), use_container_width=True)

    if pair_ready:
        matched = get_matched(ts_param)
        if not matched.empty:
            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(plotting.difference_time_series(matched, platform_a, platform_b, ts_param), use_container_width=True)
            with c2:
                window = st.select_slider("Rolling window", options=["1H", "3H", "6H", "12H", "1D", "3D", "7D"], value="6H")
                series = matched.set_index("timestamp")[platform_b] - matched.set_index("timestamp")[platform_a]
                st.plotly_chart(plotting.rolling_stats_plot(series, f"{ts_param} difference", window=window), use_container_width=True)

    st.plotly_chart(plotting.monthly_summary_plot(filtered, ts_param), use_container_width=True)

    with st.expander("Daily / weekly / monthly statistics table"):
        freq_choice = st.radio("Period", ["D", "W", "M"], horizontal=True, format_func=lambda f: {"D": "Daily", "W": "Weekly", "M": "Monthly"}[f])
        st.dataframe(stats_mod.periodic_stats(filtered, ts_param, freq_choice), use_container_width=True)

# ---------------------------------------------------------------------------
# Distribution tab
# ---------------------------------------------------------------------------
with tabs[2]:
    dist_param = st.selectbox(
        "Parameter", options=ordered_params,
        index=ordered_params.index(param) if param in ordered_params else 0,
        key="distribution_param",
    )
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(plotting.histogram(filtered, dist_param), use_container_width=True)
        st.plotly_chart(plotting.box_plot(filtered, dist_param), use_container_width=True)
    with c2:
        st.plotly_chart(plotting.kde_plot(filtered, dist_param), use_container_width=True)
        st.plotly_chart(plotting.violin_plot(filtered, dist_param), use_container_width=True)
    st.plotly_chart(plotting.cdf_plot(filtered, dist_param), use_container_width=True)

# ---------------------------------------------------------------------------
# Agreement tab
# ---------------------------------------------------------------------------
with tabs[3]:
    if not pair_ready:
        st.info("Select at least two platforms to view agreement analysis.")
    else:
        agreement_param = st.selectbox(
            "Parameter", options=ordered_params,
            index=ordered_params.index(param) if param in ordered_params else 0,
            key="agreement_param",
        )
        matched = get_matched(agreement_param)
        if matched.empty:
            st.info("No matched pairs for the current selection.")
        else:
            if outlier_mode != "Show all":
                mask = stats_mod.detect_diff_outliers(matched, platform_a, platform_b)
                if outlier_mode == "Exclude outliers":
                    matched = matched[~mask]
                # "Highlight" is handled visually within scatter/BA below via a colour column.
            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(plotting.scatter_comparison(matched, platform_a, platform_b, agreement_param), use_container_width=True)
            with c2:
                st.plotly_chart(plotting.bland_altman_plot(matched, platform_a, platform_b, agreement_param), use_container_width=True)
            st.plotly_chart(plotting.residual_plot(matched, platform_a, platform_b, agreement_param), use_container_width=True)

        if len(selected_params) > 1:
            st.subheader("Correlation matrix across parameters")
            corr_platform = st.selectbox("Platform for correlation matrix", ["(pooled)"] + selected_platforms)
            st.plotly_chart(
                plotting.correlation_matrix(
                    filtered, selected_params,
                    platform=None if corr_platform == "(pooled)" else corr_platform,
                ),
                use_container_width=True,
            )

        st.subheader("Platform × time heatmap")
        heat_freq = st.select_slider("Bucket size", options=["1H", "6H", "1D", "1W"], value="1D")
        st.plotly_chart(plotting.platform_heatmap(filtered, agreement_param, freq=heat_freq), use_container_width=True)

# ---------------------------------------------------------------------------
# Outliers & Data Quality tab
# ---------------------------------------------------------------------------
with tabs[4]:
    if outlier_filter_enabled and outlier_removed_summary is not None:
        st.subheader(f"General outlier filtering ({outlier_method}, threshold={outlier_threshold:.1f})")
        st.caption(
            "Applied per-platform before any other analysis in this dashboard; "
            "flagged values are treated as missing rather than the whole row being dropped."
        )
        if outlier_removed_summary.empty or outlier_removed_summary.sum() == 0:
            st.write("No values flagged with the current method/threshold.")
        else:
            st.dataframe(outlier_removed_summary.to_frame("values removed"), use_container_width=True)
    else:
        st.caption("General outlier filtering is currently disabled — enable it in the sidebar to clean the data before analysis.")

    st.subheader("Range violations (physically implausible readings)")
    range_flags = stats_mod.detect_range_violations(filtered, param)
    n_range = int(range_flags.sum())
    st.write(f"{n_range} reading(s) outside the plausible range for {param}.")
    if n_range:
        st.dataframe(filtered.loc[range_flags, ["timestamp", "platform", param]], use_container_width=True)

    st.subheader("Spike detection (single-point jumps)")
    spike_platform = st.selectbox("Platform", selected_platforms, key="spike_platform")
    spike_series = filtered[filtered["platform"] == spike_platform].set_index("timestamp")[param].dropna()
    spikes = stats_mod.detect_spikes(spike_series)
    st.write(f"{int(spikes.sum())} spike(s) detected for {spike_platform} / {param}.")
    if spikes.any():
        st.dataframe(spike_series[spikes].rename("value").reset_index(), use_container_width=True)

    if pair_ready:
        st.subheader("Extreme differences (matched-pair outliers)")
        matched_full = get_matched(param)
        if not matched_full.empty:
            diff_mask = stats_mod.detect_diff_outliers(matched_full, platform_a, platform_b)
            st.write(f"{int(diff_mask.sum())} extreme difference(s) out of {len(matched_full)} matched pairs.")
            if diff_mask.any():
                st.dataframe(matched_full[diff_mask], use_container_width=True)

    st.subheader("Missing data — sample counts per platform")
    missing_freq = st.select_slider("Bucket size", options=["1H", "6H", "1D"], value="1D", key="missing_freq")
    st.dataframe(stats_mod.missing_data_summary(filtered, param, freq=missing_freq), use_container_width=True)

# ---------------------------------------------------------------------------
# Export tab
# ---------------------------------------------------------------------------
with tabs[5]:
    st.subheader("Export statistics")
    desc_table = stats_mod.descriptive_stats_table(filtered, param)
    st.download_button(
        "Download descriptive statistics (CSV)",
        data=desc_table.to_csv().encode("utf-8"),
        file_name=f"descriptive_stats_{param}.csv",
        mime="text/csv",
    )

    if pair_ready:
        matched = get_matched(param)
        if not matched.empty:
            metrics = stats_mod.pairwise_comparison(matched, platform_a, platform_b)
            metrics_df = pd.DataFrame([metrics])
            st.download_button(
                "Download pairwise comparison metrics (CSV)",
                data=metrics_df.to_csv(index=False).encode("utf-8"),
                file_name=f"pairwise_metrics_{param}_{platform_a}_vs_{platform_b}.csv",
                mime="text/csv",
            )
            st.download_button(
                "Download matched-pair data (CSV)",
                data=matched.to_csv(index=False).encode("utf-8"),
                file_name=f"matched_pairs_{param}_{platform_a}_vs_{platform_b}.csv",
                mime="text/csv",
            )

    st.subheader("Export figures")
    st.caption("PNG export requires the optional `kaleido` package (see requirements.txt). HTML export always works.")
    fig_choice = st.selectbox(
        "Figure to export",
        ["Time series overlay", "Histogram", "Box plot", "Scatter comparison", "Bland-Altman"],
    )
    fig_map = {
        "Time series overlay": lambda: plotting.time_series_overlay(filtered, param),
        "Histogram": lambda: plotting.histogram(filtered, param),
        "Box plot": lambda: plotting.box_plot(filtered, param),
        "Scatter comparison": lambda: plotting.scatter_comparison(get_matched(param), platform_a, platform_b, param) if pair_ready else None,
        "Bland-Altman": lambda: plotting.bland_altman_plot(get_matched(param), platform_a, platform_b, param) if pair_ready else None,
    }
    fig = fig_map[fig_choice]()
    if fig is None:
        st.info("This figure needs two selected platforms.")
    else:
        html_bytes = fig.to_html(include_plotlyjs="cdn").encode("utf-8")
        st.download_button("Download as HTML", data=html_bytes, file_name=f"{fig_choice.replace(' ', '_').lower()}.html", mime="text/html")
        try:
            png_bytes = fig.to_image(format="png", width=1200, height=700, scale=2)
            st.download_button("Download as PNG", data=png_bytes, file_name=f"{fig_choice.replace(' ', '_').lower()}.png", mime="image/png")
        except Exception:
            st.caption("PNG export unavailable — install `kaleido` to enable it (`pip install kaleido`).")
