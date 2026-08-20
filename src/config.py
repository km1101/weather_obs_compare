"""
Central configuration for the weather-platform comparison toolkit.

Everything that is likely to change as new platforms, sensors, or data
sources are added lives in this one file. Nothing downstream should ever
hard-code an AWS_ID, a scale factor, or a parameter name -- it should all
be looked up from here so that "add a platform" or "add a parameter"
never turns into a multi-file code change.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Platform identity
# ---------------------------------------------------------------------------
# Raw data only ever carries an AWS_ID. Everything user-facing (dashboard
# labels, exported files, plot legends) should go through this mapping
# instead of showing the raw numeric ID. Add new platforms here -- no other
# file needs to change.
PLATFORM_MAPPING: dict[int, str] = {
    62045: "TH2_Sys1",
    62047: "TH2_Sys2",
    36089: "Mobilis_Sys1",
    44912: "Mobilis_Sys2",
}

# Reverse lookup, built automatically.
PLATFORM_NAME_TO_ID: dict[str, int] = {v: k for k, v in PLATFORM_MAPPING.items()}

# Grouping platforms by underlying hardware "Type" column in the raw feed.
# This is informational (e.g. Type 3 units carry a visibility sensor, Type 2
# units don't) and is used to explain missing data, not to filter it.
PLATFORM_TYPE: dict[int, int] = {
    62045: 3,
    62047: 3,
    36089: 2,
    44912: 2,
}

# ---------------------------------------------------------------------------
# Raw-value scale factors
# ---------------------------------------------------------------------------
# The logger stores everything as a scaled integer to avoid floats on the
# wire. Divide the raw column by the factor below to get physical units.
# Source: prompt_weather_compare.docx raw-data table.
SCALE_FACTORS: dict[str, float] = {
    "Min_Since": 1,
    "Latitude": 10000,
    "Longitude": 10000,
    "Temperature": 10,
    "Humidity": 10,
    "Pressure": 10,
    "Sea_Temp": 100,
    "Wind_Direction": 10,
    "Wind_Speed": 100,
    "Gust": 10,
    "COG": 100,
    "SOG": 100,
    "Wave_Height": 10,
    "Wave_Period": 10,
    "Visibility": 1,
}

# ---------------------------------------------------------------------------
# Parameters exposed for comparison in the dashboard
# ---------------------------------------------------------------------------
# The "primary" weather parameters called out in the spec. Any other numeric
# column that appears in the CSV (and isn't an identifier/location column)
# is still analysable -- this list just controls default selector ordering.
PRIMARY_PARAMETERS: list[str] = [
    "Temperature",
    "Humidity",
    "Pressure",
    "Sea_Temp",
    "Wind_Direction",
    "Wind_Speed",
    "Gust",
    "Visibility",
]

# Columns that are identifiers / metadata rather than measurable parameters.
# These are never offered in the parameter selector even though they are
# numeric.
NON_PARAMETER_COLUMNS: set[str] = {
    "Dual",
    "AWS_ID",
    "platform_id",
    "Type",
    "Date",
    "Time",
    "Min_Since",
    "Latitude",
    "Longitude",
    "COG",
    "SOG",
    "timestamp",
    "platform",
}

# Units for display purposes. Unknown parameters just show without a unit.
PARAMETER_UNITS: dict[str, str] = {
    "Temperature": "\u00b0C",
    "Humidity": "%",
    "Pressure": "hPa",
    "Sea_Temp": "\u00b0C",
    "Wind_Direction": "\u00b0",
    "Wind_Speed": "m/s",
    "Gust": "m/s",
    "Wave_Height": "m",
    "Wave_Period": "s",
    "Visibility": "m",
}

# Circular parameters need circular statistics (mean of bearings etc.)
# rather than naive arithmetic mean. Flagged here so the stats module can
# branch on it; full circular-stat support is a documented future
# enhancement (see README).
CIRCULAR_PARAMETERS: set[str] = {"Wind_Direction", "COG"}

# ---------------------------------------------------------------------------
# Data-quality thresholds
# ---------------------------------------------------------------------------
# Sane physical bounds used for the "invalid measurement" outlier check.
# Loose on purpose -- this is a sanity filter, not a climatology model.
VALID_RANGES: dict[str, tuple[float, float]] = {
    "Temperature": (-40, 55),
    "Humidity": (0, 100),
    "Pressure": (850, 1085),
    "Sea_Temp": (-3, 40),
    "Wind_Direction": (0, 360),
    "Wind_Speed": (0, 100),
    "Gust": (0, 120),
    "Visibility": (0, 50000),
}

# Number of standard deviations beyond which a matched-pair difference is
# flagged as an "extreme difference" outlier (used by the outlier module).
DIFF_OUTLIER_SIGMA = 3.0

# General-purpose outlier filtering (sidebar "Outlier filtering" control).
OUTLIER_METHODS: list[str] = ["iqr", "zscore", "hampel"]
DEFAULT_OUTLIER_METHOD = "iqr"
DEFAULT_OUTLIER_THRESHOLD = 3.0

# Default tolerance (minutes) for nearest-neighbour timestamp matching
# between two platforms whose sample clocks don't align exactly.
DEFAULT_MATCH_TOLERANCE_MINUTES = 5

# ---------------------------------------------------------------------------
# Future expansion hook
# ---------------------------------------------------------------------------
# Numerical model output (WaveWatch III, SWAN, ...) is not an AWS platform
# and has no AWS_ID. Model "platforms" are registered here with a negative
# pseudo-ID so they slot into the same PLATFORM_MAPPING-driven code paths
# as real observation platforms without any special-casing downstream.
MODEL_PLATFORM_MAPPING: dict[int, str] = {
    # -1: "WW3_Model",
    # -2: "SWAN_Model",
}
