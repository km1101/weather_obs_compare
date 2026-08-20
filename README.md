# Weather Platform Comparison

Interactive toolkit for comparing weather-observation platforms against
each other (and, in future, numerical wave-model output). Built around one
question: **do these platforms agree, and if not, how and why?**

## Quick start

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the URL Streamlit prints (usually http://localhost:8501).

Point the "CSV data file" box in the sidebar at any logger export with the
same raw schema (see below) — by default it reads `data/weather_data.csv`.

## Project structure

```
weather_compare/
├── app.py                 # Streamlit dashboard (the only UI-framework-specific file)
├── data/
│   └── weather_data.csv   # your logger export
├── exports/                # suggested output folder for manual exports
├── src/
│   ├── config.py           # platform mapping, scale factors, thresholds — edit this to add platforms/parameters
│   ├── data_loader.py       # raw CSV -> clean typed DataFrame
│   ├── preprocessing.py     # dedup, date filtering, cross-platform timestamp alignment
│   ├── statistics.py        # descriptive stats, pairwise metrics, drift, outliers
│   ├── plotting.py          # every Plotly figure builder
│   └── export.py            # CSV/XLSX/PNG/HTML export helpers
└── requirements.txt
```

Every module is plain pandas/NumPy/SciPy/Plotly and has no Streamlit
import except `app.py` — the analysis code can be reused from a script,
notebook, or a different dashboard framework (Dash/Taipy) without change.

## Data source (sidebar section 1)

Upload a CSV directly in the sidebar, or leave it empty to use the bundled
`data/weather_data.csv`. An "Advanced" expander also lets you point at a
local file path instead (handy when running the dashboard against a large
file you don't want to re-upload each session). Whichever source is
active, the sidebar shows a one-line confirmation of what's currently
loaded.

## Platform mapping (sidebar section 2)

Choose how AWS_ID -> platform-name mapping is resolved:

- **Saved config** — the bundled `PLATFORM_MAPPING` in `src/config.py` (default).
- **Edit manually** — an editable table in the sidebar (add/remove rows freely).
- **Upload mapping file** — a CSV (columns `aws_id`/`id` and `platform_name`/`name`)
  or a JSON file (either `{"62045": "TH2_Sys1", ...}` or
  `[{"aws_id": 62045, "platform_name": "TH2_Sys1"}, ...]`).

A "Current mapping" expander always shows the mapping actually in effect.
Whichever source is chosen, the same dict flows into `data_loader.load_raw_csv`
— nothing else in the pipeline needs to know where it came from.

## Adding a platform permanently

Edit `src/config.py`:

```python
PLATFORM_MAPPING[12345] = "NewSensor_Sys1"
PLATFORM_TYPE[12345] = 3   # optional, informational only
```

No other file needs to change — the loader, dashboard selectors, and
exports all read from this dictionary. For a one-off or per-session
mapping, use the sidebar's "Edit manually" or "Upload mapping file" option
instead — no code edit needed.

## Outlier filtering (sidebar section 3)

A general data-cleaning step, separate from the matched-pair "extreme
differences" control in the Agreement tab. When enabled, it flags outliers
independently per platform and sets them to missing (not drop the whole
row) before every other tab sees the data. Three methods, selectable in
the sidebar:

- **iqr** — Tukey fence (`Q1 - k·IQR`, `Q3 + k·IQR`). No distribution
  assumption; a safe default for skewed quantities like wind speed.
- **zscore** — standard deviations from the mean. Simple, but a few
  extreme values can inflate the mean/SD used to judge them.
- **hampel** — rolling median / median-absolute-deviation. Stays sensitive
  to outliers even under slow drift or trend, unlike the other two which
  use a single global mean/SD or quartiles for the whole selection.

The "Threshold" slider is the fence multiplier (iqr) or the "how many
sigma" cutoff (zscore/hampel) — same slider, method-appropriate meaning.
A summary of how many values were removed per parameter is shown in the
Outliers & Data Quality tab.

## Adding a parameter

If the raw CSV gains a new numeric column, add its scale factor to
`SCALE_FACTORS` in `config.py` (use `1` if it's already in physical
units). It will then automatically appear in the dashboard's parameter
selector and in every statistics/plotting function, which are all
parameter-agnostic.

## Raw data format

The loader expects the schema documented in the project brief:

- `Date` packed as `ddmmyy` (leading zeros dropped, e.g. `80726` = 08/07/26)
- `Time` packed as `hhmm` (e.g. `1000` = 10:00)
- Every measurement column stored as a scaled integer (see
  `SCALE_FACTORS` in `config.py` for the divisor per column)
- `AWS_ID` identifying the platform

Rows with an unrecognised `AWS_ID` (not in `PLATFORM_MAPPING`) or an
unparseable timestamp are dropped during loading and logged — they're
almost always footer/garbage rows the logger appends, not real
observations. **Dates in the dashboard are always displayed as
`dd/mm/yyyy`**, per the spec.

## What each dashboard tab answers

| Tab | Question |
|---|---|
| Summary | What's the typical value, spread, and matched-pair agreement (bias, MAE, RMSE, correlation)? |
| Time Series | Do the platforms track each other over time? Is the gap between them growing (drift)? |
| Distribution | Do the platforms have the same overall spread and shape of readings? |
| Agreement | Scatter vs 1:1 line, Bland–Altman limits of agreement, residuals, cross-parameter correlation |
| Outliers & Data Quality | Physically implausible readings, spikes, extreme matched-pair differences, missing-data gaps |
| Export | Download the current statistics table, matched-pair data, and any figure as CSV/HTML/PNG |

See docstrings in `src/statistics.py` for the reasoning behind each metric
(why Bias vs RMSE vs Scatter Index measure different things, why Bland–Altman
is included alongside Pearson correlation, etc.) — that's the file to read
if you want to understand a specific number, not just look it up.

## Timestamp alignment methods

Two loggers rarely sample on exactly the same clock tick. `src/preprocessing.align_platforms`
offers three strategies, selectable in the sidebar:

- **nearest** (default) — pair each Platform A reading with the closest
  Platform B reading within a configurable tolerance (minutes).
- **exact** — only pair readings that share an identical timestamp. Use
  this if you want timestamp drift itself treated as a data problem
  rather than smoothed over.
- **interpolate** — linearly interpolate Platform B onto Platform A's
  timestamps. Useful when comparing a sparse/irregular source (e.g. a
  future model run) against a densely-sampled platform.

## Optional extras

Two features degrade gracefully if their optional dependency isn't
installed:

- **PNG figure export** needs `kaleido` (`pip install kaleido`), which in
  turn needs a local Chrome/Chromium (`plotly_get_chrome` installs one).
  Without it, HTML export still works from the same button group.
- **Automatic, no-interaction refresh** when the CSV file changes needs
  `streamlit-autorefresh` (`pip install streamlit-autorefresh`). Without
  it, the dashboard still picks up CSV changes on every normal
  interaction (any widget click triggers Streamlit to re-check the file's
  modified time) and via the manual "🔄 Refresh data now" button — true
  hands-off polling just needs the extra package.

## Future expansion: model data

`src/config.py::MODEL_PLATFORM_MAPPING` is a ready-made slot for
numerical wave-model output (WaveWatch III, SWAN, etc.). Model runs
aren't AWS platforms and have no `AWS_ID`, so they're registered with a
negative pseudo-ID — once you add a loader that produces the same clean
schema (`timestamp`, `platform_id`, `platform`, parameter columns) as
`data_loader.load_raw_csv`, every statistics and plotting function in this
project works on it unmodified, enabling Platform-vs-Model and triple
(Platform A / Platform B / Model) comparisons with no changes outside the
new loader.

## Known data-quality notes from the sample file

- `Wave_Height` and `Wave_Period` are present as columns but empty in the
  sample export — the dashboard's parameter selector only lists
  parameters that have at least one non-null value, so they won't appear
  until populated.
- `Visibility` is only populated for the TH2 platforms (Type 3 hardware);
  Mobilis platforms (Type 2) don't carry a visibility sensor. This shows
  up as expected missing data, not a bug.
- A handful of rows have `AWS_ID` values outside the known platform
  mapping (garbage/footer rows) — these are dropped automatically during
  loading, with a count logged.
