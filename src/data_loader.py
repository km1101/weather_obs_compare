"""
Data loading and parsing.

Responsible for turning the raw logger CSV -- scaled integers, packed
ddmmyy/hhmm date-time columns, occasional garbage rows -- into a clean,
typed DataFrame with physical units and a proper timestamp column. This is
the only module that should know about the raw file's quirks; everything
downstream works with the clean schema this module produces.

Clean schema produced by ``load_raw_csv``:
    timestamp   : pandas datetime64[ns]
    platform_id : int   (raw AWS_ID)
    platform    : str   (human-readable name via PLATFORM_MAPPING)
    <parameter> : float (physical units, one column per sensor field)
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .config import PLATFORM_MAPPING, SCALE_FACTORS

logger = logging.getLogger(__name__)

# Columns present in the raw feed, in the order documented in the spec.
_RAW_ID_COLUMNS = ["Dual", "AWS_ID", "Type", "Date", "Time"]


def _parse_date(raw: pd.Series) -> pd.Series:
    """Parse the packed ``ddmmyy`` Date column (e.g. 80726 -> 08/07/26).

    The logger drops leading zeros, so 80726 means day=08, month=07, year=26
    -- not 8-digit day/month/year. Zero-pad to 6 digits before splitting.
    """
    as_str = raw.astype("Int64").astype(str).str.zfill(6)
    day = as_str.str.slice(0, 2)
    month = as_str.str.slice(2, 4)
    year = as_str.str.slice(4, 6)
    # 2-digit year: logger data is all 20xx.
    full = "20" + year + "-" + month + "-" + day
    return pd.to_datetime(full, format="%Y-%m-%d", errors="coerce")


def _parse_time(raw: pd.Series) -> pd.Series:
    """Parse the packed ``hhmm`` Time column (e.g. 1000 -> 10:00)."""
    as_str = raw.astype("Int64").astype(str).str.zfill(4)
    hour = as_str.str.slice(0, 2)
    minute = as_str.str.slice(2, 4)
    return pd.to_timedelta(hour + ":" + minute + ":00", errors="coerce")


def load_raw_csv(
    path_or_buffer,
    platform_mapping: dict[int, str] | None = None,
) -> pd.DataFrame:
    """Load the raw logger CSV and return a clean, typed DataFrame.

    ``path_or_buffer`` accepts anything pandas.read_csv accepts: a path
    (str/Path) or a file-like/bytes buffer (e.g. a Streamlit uploaded-file
    object), so the same function serves both "read from disk" and
    "read an in-memory upload" callers without a branch.

    ``platform_mapping`` overrides the bundled ``config.PLATFORM_MAPPING``
    -- pass a dict built from a manually-edited table or an uploaded
    mapping file (see ``mapping.py``) to use a different AWS_ID -> name
    mapping without touching this function.

    Steps applied, in order:
      1. Read the CSV (all columns coerced to numeric where possible).
      2. Drop rows that aren't real observations -- garbage/footer rows
         with no AWS_ID, and the reverse: rows with an AWS_ID that isn't in
         the platform mapping at all (e.g. a stray ``AWS_ID == 1``).
      3. Divide every raw column by its SCALE_FACTORS entry to get
         physical units.
      4. Combine Date + Time into a single timestamp column.
      5. Map AWS_ID -> a human-readable platform name.
      6. Sort chronologically and reset the index.

    Duplicate timestamps and gaps in time are NOT resolved here -- see
    :func:`preprocessing.deduplicate_timestamps`. This function's job is
    purely "raw bytes -> clean typed table", one row per source record.
    """
    platform_mapping = platform_mapping or PLATFORM_MAPPING

    if isinstance(path_or_buffer, (str, Path)):
        df = pd.read_csv(Path(path_or_buffer))
    else:
        df = pd.read_csv(path_or_buffer)

    # Anything not in the platform mapping is either a garbage row (blank
    # footer line the logger appends, AWS_ID missing) or a bogus code we
    # don't recognise (e.g. AWS_ID == 1 seen in sample data). Drop both --
    # they carry no usable measurement.
    known_ids = set(platform_mapping)
    df["AWS_ID"] = pd.to_numeric(df["AWS_ID"], errors="coerce")
    before = len(df)
    df = df[df["AWS_ID"].isin(known_ids)].copy()
    dropped = before - len(df)
    if dropped:
        logger.info("Dropped %d row(s) with unrecognised/missing AWS_ID", dropped)

    df["AWS_ID"] = df["AWS_ID"].astype(int)

    # Physical-unit conversion. Applied to every column we have a factor
    # for; anything else (Dual, Type, and any *new* raw column a user adds
    # later) passes through untouched -- new numeric parameters just need
    # an entry in config.SCALE_FACTORS to get the same treatment, no code
    # change here.
    for col, factor in SCALE_FACTORS.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce") / factor

    # Timestamp assembly.
    date_part = _parse_date(df["Date"])
    time_part = _parse_time(df["Time"])
    df["timestamp"] = date_part + time_part

    # Rows where the timestamp failed to parse are unusable for time-series
    # comparison -- drop them (they overlap heavily with the garbage rows
    # already removed above, but this also catches malformed Date/Time on
    # otherwise-valid rows).
    before = len(df)
    df = df.dropna(subset=["timestamp"])
    dropped = before - len(df)
    if dropped:
        logger.info("Dropped %d row(s) with unparseable timestamp", dropped)

    df["platform"] = df["AWS_ID"].map(platform_mapping)
    df = df.rename(columns={"AWS_ID": "platform_id"})
    df = df.drop(columns=["Date", "Time", "Dual", "Type"], errors="ignore")

    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def numeric_parameter_columns(df: pd.DataFrame) -> list[str]:
    """Return the columns in ``df`` that represent measurable parameters.

    i.e. numeric columns minus identifiers (platform_id) and location
    (Latitude/Longitude), which are metadata, not comparison targets.
    """
    from .config import NON_PARAMETER_COLUMNS

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    return [c for c in numeric_cols if c not in NON_PARAMETER_COLUMNS]
