"""Examine a DataFrame and build a per-column profile describing its structure.

The profile stores only *aggregate* information (types, null rate, quantiles,
formats, category frequencies). Real free-text / PII values are never kept.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .semantic import (PUBLIC_EMAIL_DOMAINS, SHAPE_BASED, WEAK,
                       detect_semantic)

QUANTILES = np.linspace(0, 1, 101)

BOOL_PAIRS = [("true", "false"), ("yes", "no"), ("y", "n"), ("t", "f")]

DATE_FORMATS = [
    "%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M",
    "%Y-%m-%dT%H:%M", "%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S.%f",
    "%Y-%m-%dT%H:%M:%SZ", "%Y/%m/%d", "%Y/%m/%d %H:%M:%S",
    "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y", "%m-%d-%Y", "%d.%m.%Y",
    "%d/%m/%Y %H:%M", "%m/%d/%Y %H:%M", "%d/%m/%Y %H:%M:%S", "%m/%d/%Y %H:%M:%S",
    "%d %b %Y", "%d %B %Y", "%d-%b-%Y", "%b %d %Y", "%b %d, %Y", "%B %d, %Y",
]

# Thresholds
MAX_CATEGORIES = 50          # at most this many distinct values ...
MAX_CATEGORY_RATIO = 0.5     # ... and at most this share of rows being distinct
MAX_INT_CATEGORIES = 20
PATTERN_COVERAGE = 0.8
MAX_PATTERN_LEN = 40

_SHAPE_TABLE = str.maketrans(
    {**{c: "A" for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"},
     **{c: "a" for c in "abcdefghijklmnopqrstuvwxyz"},
     **{c: "9" for c in "0123456789"}}
)


@dataclass
class ColumnProfile:
    name: str
    kind: str                 # empty|boolean|integer|float|datetime|categorical|semantic|pattern|text
    null_frac: float
    n_unique: int
    unique: bool
    semantic: str | None = None
    source_dtype: str = "object"
    params: dict[str, Any] = field(default_factory=dict)

    def describe(self) -> str:
        p = self.params
        if self.kind == "datetime":
            return f"format {p['format'] or 'native'}"
        if self.kind in ("integer", "float"):
            q = p["quantiles"]
            return f"range {q[0]:g} .. {q[-1]:g}"
        if self.kind == "categorical":
            vals = ", ".join(str(v) for v in p["values"][:5])
            more = "" if len(p["values"]) <= 5 else f", ... ({len(p['values'])} total)"
            return f"{{{vals}{more}}}"
        if self.kind == "boolean":
            return f"{p['true']}/{p['false']}  p(true)={p['p_true']:.2f}"
        if self.kind == "pattern":
            return f"shape like {p['shapes'][0]!r}"
        if self.kind == "text":
            return f"{p['style']}, ~{int(np.median(p['len_quantiles']))} chars"
        return ""


@dataclass
class Relation:
    """`column` is always >= `base` in the real data (e.g. checkout >= checkin)."""
    column: str
    base: str
    delta_quantiles: list[float]


@dataclass
class TableProfile:
    n_rows: int
    columns: list[ColumnProfile]
    relations: list[Relation]

    def to_dict(self) -> dict:
        return asdict(self)


def _q(arr: np.ndarray) -> list[float]:
    return [float(x) for x in np.quantile(arr.astype(float), QUANTILES)]


def _shape(s: str) -> str:
    return s.translate(_SHAPE_TABLE)


# --------------------------------------------------------------------------- #
# Individual detectors. Each returns (kind, params) or None.
# --------------------------------------------------------------------------- #

def _detect_bool(raw: pd.Series, vals: pd.Series):
    if pd.api.types.is_bool_dtype(raw):
        p = float(raw.dropna().astype(bool).mean())
        return "boolean", {"true": True, "false": False, "p_true": p}
    lower = vals.str.lower()
    uniq = set(lower.unique())
    for t, f in BOOL_PAIRS:
        if uniq <= {t, f} and len(uniq) == 2:
            true_tok = vals[lower == t].mode().iloc[0]
            false_tok = vals[lower == f].mode().iloc[0]
            return "boolean", {"true": true_tok, "false": false_tok,
                               "p_true": float((lower == t).mean())}
    return None


def _parse_dates(raw: pd.Series, vals: pd.Series):
    """Return (parsed datetime Series aligned with vals, format) or None."""
    if pd.api.types.is_datetime64_any_dtype(raw):
        return pd.to_datetime(raw.loc[vals.index]), None
    if pd.to_numeric(vals, errors="coerce").notna().all():
        return None
    sample = vals.sample(min(len(vals), 200), random_state=0)
    for fmt in DATE_FORMATS:
        if pd.to_datetime(sample, format=fmt, errors="coerce").notna().all():
            parsed = pd.to_datetime(vals, format=fmt, errors="coerce")
            if parsed.notna().all():
                return parsed, fmt
    return None


def _detect_numeric(raw: pd.Series, vals: pd.Series):
    num = pd.to_numeric(vals, errors="coerce")
    if not num.notna().all():
        return None
    if vals.str.match(r"^[-+]?0\d").any():          # leading zeros -> it's a code
        return None
    is_int = bool(vals.str.fullmatch(r"[-+]?\d+").all())
    if is_int and vals.str.len().max() > 15:          # long digit strings are IDs
        return None
    arr = num.to_numpy(dtype=float)
    if is_int:
        n_unique = num.nunique()
        if n_unique <= MAX_INT_CATEGORIES and n_unique <= MAX_CATEGORY_RATIO * len(num):
            return None  # let the categorical detector handle it
        return "integer", {"quantiles": _q(arr)}
    decimals = vals.str.extract(r"\.(\d+)$")[0].str.len().fillna(0)
    return "float", {"quantiles": _q(arr), "decimals": int(min(decimals.max(), 15))}


def _detect_categorical(raw: pd.Series, vals: pd.Series):
    n_unique = vals.nunique()
    if n_unique <= MAX_CATEGORIES and n_unique <= MAX_CATEGORY_RATIO * len(vals):
        counts = raw.loc[vals.index].value_counts()
        values = [v.item() if hasattr(v, "item") else v for v in counts.index]
        if isinstance(values[0], str):
            values = [v.strip() for v in values]
        return "categorical", {"values": values,
                               "weights": (counts / counts.sum()).round(6).tolist()}
    return None


def _detect_pattern(vals: pd.Series):
    if vals.str.len().mean() > MAX_PATTERN_LEN:
        return None
    shapes = vals.map(_shape).value_counts()
    top = shapes.head(10)
    if top.sum() / shapes.sum() < PATTERN_COVERAGE:
        return None
    if not any(ch in "Aa9" for ch in top.index[0]):
        return None
    return "pattern", {"shapes": top.index.tolist(),
                       "weights": (top / top.sum()).round(6).tolist()}


def _text_params(vals: pd.Series) -> dict:
    words = vals.str.split().str.len()
    lengths = vals.str.len()
    titled = vals.str.istitle().mean()
    return {
        "style": "sentences" if words.mean() >= 4 else "words",
        "word_quantiles": _q(words.to_numpy()),
        "len_quantiles": _q(lengths.to_numpy()),
        "title_case": bool(titled > 0.5),
        "multiline": bool(vals.str.contains("\n").mean() > 0.5),
    }


def _semantic_params(stype: str, vals: pd.Series) -> dict:
    if stype == "email":
        domains = vals.str.split("@").str[-1].str.lower()
        public = domains[domains.isin(PUBLIC_EMAIL_DOMAINS)]
        counts = public.value_counts()
        return {
            "public_frac": float(len(public) / len(domains)),
            "public_domains": counts.index.tolist(),
            "public_weights": (counts / counts.sum()).round(6).tolist() if len(counts) else [],
        }
    if stype in ("address", "street_address", "credit_card", "phone"):
        return {"multiline": bool(vals.str.contains("\n").mean() > 0.5),
                "digits_only": bool(vals.str.fullmatch(r"\d+").mean() > 0.9)}
    return {}


# --------------------------------------------------------------------------- #

def profile_column(name: str, raw: pd.Series) -> tuple[ColumnProfile, pd.Series | None]:
    """Profile one column. Also returns the parsed datetimes if it is a date column."""
    total = len(raw)
    non_null = raw.dropna()
    vals = non_null.astype(str).str.strip()
    vals = vals[vals != ""]
    null_frac = 1 - len(vals) / total if total else 0.0
    n_unique = int(vals.nunique())
    unique = bool(len(vals) > 1 and n_unique == len(vals))
    base = dict(name=str(name), null_frac=round(null_frac, 6), n_unique=n_unique,
                unique=unique, source_dtype=str(raw.dtype))

    if vals.empty:
        return ColumnProfile(kind="empty", **base), None

    found = _detect_bool(raw, vals)
    if found:
        return ColumnProfile(kind=found[0], params=found[1], **base), None

    stype = detect_semantic(name, vals)
    if stype:
        if stype in WEAK and (cat := _detect_categorical(raw, vals)):
            return ColumnProfile(kind=cat[0], params=cat[1], semantic=stype, **base), None
        if stype in SHAPE_BASED and (pat := _detect_pattern(vals)):
            return ColumnProfile(kind=pat[0], params=pat[1], semantic=stype, **base), None
        return ColumnProfile(kind="semantic", semantic=stype,
                             params=_semantic_params(stype, vals), **base), None

    dates = _parse_dates(raw, vals)
    if dates is not None:
        parsed, fmt = dates
        if parsed.dt.tz is not None:
            parsed = parsed.dt.tz_convert(None)
        secs = (parsed - pd.Timestamp(0)).dt.total_seconds().to_numpy()
        if fmt is None:
            has_time = bool((parsed.dt.normalize() != parsed).any())
        else:
            has_time = any(t in fmt for t in ("%H", "%M", "%S"))
        params = {"format": fmt, "quantiles": _q(secs), "has_time": has_time,
                  "native": fmt is None}
        return ColumnProfile(kind="datetime", params=params, **base), parsed

    for detector in (_detect_numeric, _detect_categorical):
        found = detector(raw, vals)
        if found:
            return ColumnProfile(kind=found[0], params=found[1], **base), None

    found = _detect_pattern(vals)
    if found:
        return ColumnProfile(kind=found[0], params=found[1], **base), None

    return ColumnProfile(kind="text", params=_text_params(vals), **base), None


def _find_relations(parsed: dict[str, pd.Series], total_index: pd.Index) -> list[Relation]:
    """Find date columns that are always on/after an earlier date column."""
    names = list(parsed)
    relations, used = [], set()
    for j, b in enumerate(names):
        for a in names[:j]:
            sa = parsed[a].reindex(total_index)
            sb = parsed[b].reindex(total_index)
            both = sa.notna() & sb.notna()
            if both.sum() < 5 or b in used:
                continue
            delta = (sb[both] - sa[both]).dt.total_seconds()
            if (delta >= 0).all():
                relations.append(Relation(column=b, base=a, delta_quantiles=_q(delta.to_numpy())))
                used.add(b)
    return relations


def profile_table(df: pd.DataFrame) -> TableProfile:
    columns, parsed = [], {}
    for col in df.columns:
        prof, dates = profile_column(col, df[col])
        columns.append(prof)
        if dates is not None:
            parsed[str(col)] = dates
    return TableProfile(n_rows=len(df), columns=columns,
                        relations=_find_relations(parsed, df.index))
