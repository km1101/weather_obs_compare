"""
pandas frequency-alias compatibility shim.

pandas renamed several resample/rolling offset aliases in 2.2 (H -> h,
M -> ME, T -> min, ...), kept the old spellings working as deprecated
until pandas 3.0, then removed the old spellings entirely in 3.0. There is
no single alias string that works unmodified across "older than 2.2" and
"3.0 or newer" -- so instead of hard-coding a literal anywhere, every
frequency string in this project should be built via :func:`freq` below,
which picks the spelling the installed pandas actually accepts.
"""

from __future__ import annotations

import pandas as pd

_PANDAS_VERSION = tuple(int(p) for p in pd.__version__.split(".")[:2])
_USE_NEW_ALIASES = _PANDAS_VERSION >= (2, 2)

# canonical key -> (old alias, new alias)
_ALIAS_MAP = {
    "H": ("H", "h"),
    "T": ("T", "min"),
    "M": ("M", "ME"),
    "Y": ("Y", "YE"),
}


def freq(spec: str) -> str:
    """Translate a frequency spec written with old-style unit letters
    (e.g. "1H", "6H", "M", "1D") into whatever the installed pandas
    version accepts. Multiples (the leading digits) and units that never
    changed (D, W, S, min already-new-style) pass through unchanged.

    Examples
    --------
    >>> freq("6H")   # pandas < 2.2 -> "6H", pandas >= 2.2 -> "6h"
    >>> freq("M")    # pandas < 2.2 -> "M",  pandas >= 2.2 -> "ME"
    """
    import re

    match = re.fullmatch(r"(\d*)([A-Za-z]+)", spec)
    if not match:
        return spec
    count, unit = match.groups()

    if unit in _ALIAS_MAP:
        old, new = _ALIAS_MAP[unit]
        unit = new if _USE_NEW_ALIASES else old

    return f"{count}{unit}"
