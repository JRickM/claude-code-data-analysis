"""Detect what a column *means* (email, person name, address, ...) from its
name and its values, so we can replace it with a matching Faker provider."""

from __future__ import annotations

import re

import pandas as pd


def normalize(name: object) -> str:
    """'guestEmail' / 'Guest Email' / 'guest-email' -> 'guest_email'."""
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", str(name))
    return re.sub(r"[^0-9a-zA-Z]+", "_", s).strip("_").lower()


def _tok(word: str) -> str:
    """Regex matching `word` as a whole underscore-separated token."""
    return rf"(?:^|_){word}(?:_|$)"


# Ordered: the first matching rule wins (e.g. "email_address" -> email, not address).
NAME_RULES: list[tuple[str, str]] = [
    ("email", r"e_?mail"),
    ("credit_card", r"credit_?card|card_?(?:number|num|no)|" + _tok(r"cc(?:_?(?:number|num|no))?") + "|" + _tok("pan")),
    ("iban", _tok("iban")),
    ("ssn", _tok("ssn") + r"|social_?security"),
    ("phone", r"phone|mobile|" + _tok("tel") + "|" + _tok("cell") + "|" + _tok("fax")),
    ("username", r"user_?name|" + _tok("login") + "|" + _tok("handle")),
    ("first_name", r"first_?name|given_?name|" + _tok("fname") + r"|forename"),
    ("last_name", r"last_?name|surname|family_?name|" + _tok("lname")),
    ("company", r"company|employer|organi[sz]ation|business_?name|" + _tok("firm")),
    ("name", r"^(?:full_?)?name$|(?:customer|guest|client|contact|person|employee|patient|owner|member|student|staff|holder|account)_?name"),
    ("job", _tok("job") + r"|job_?title|occupation|profession"),
    ("postcode", r"zip|post_?code|postal"),
    ("street_address", r"street|address_?line|" + _tok(r"addr(?:ess)?_?1")),
    ("address", r"address|" + _tok("addr")),
    ("city", _tok("city") + "|" + _tok("town")),
    ("state", _tok("state") + r"|province"),
    ("country", r"country"),
    ("url", _tok("url") + r"|website|homepage"),
    ("ipv4", _tok("ip") + r"|ip_?address"),
    ("uuid", _tok("uuid") + "|" + _tok("guid")),
]

VALUE_RES: dict[str, re.Pattern] = {
    "email": re.compile(r"^[^@\s]+@[^@\s]+\.[A-Za-z]{2,}$"),
    "url": re.compile(r"^(?:https?://|www\.)\S+$", re.I),
    "ipv4": re.compile(r"^(?:\d{1,3}\.){3}\d{1,3}$"),
    "uuid": re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I),
    "credit_card": re.compile(r"^\d[\d\s-]{10,22}\d$"),
    "iban": re.compile(r"^[A-Z]{2}\d{2}[A-Z0-9 ]{10,30}$"),
    "ssn": re.compile(r"^\d{3}-?\d{2}-?\d{4}$"),
}

# Types whose values should be words, not numbers (guards against e.g. "job_id").
TEXTUAL = {"first_name", "last_name", "name", "company", "job", "street_address",
           "address", "city", "state", "country", "username"}

# Low-sensitivity types: if they look like a small set of categories, we keep
# the real category distribution instead of inventing random places.
WEAK = {"city", "state", "country"}

# Types better reproduced by copying the *format* (digit/letter shape) of the
# real values with random characters, e.g. "(555) 123-4567" or "SW1A 1AA".
SHAPE_BASED = {"phone", "postcode"}

# Values-only detection (column name gave no hint).
VALUE_ONLY = ("email", "uuid", "url", "ipv4")


def _match_frac(values: pd.Series, regex: re.Pattern) -> float:
    return float(values.str.match(regex).mean()) if len(values) else 0.0


def detect_semantic(name: object, values: pd.Series) -> str | None:
    """Return a semantic type for the column, or None.

    `values` are the non-null values as stripped strings.
    """
    norm = normalize(name)
    numeric_frac = float(pd.to_numeric(values, errors="coerce").notna().mean())
    for stype, pattern in NAME_RULES:
        if not re.search(pattern, norm):
            continue
        validator = VALUE_RES.get(stype)
        if validator is not None:
            if _match_frac(values, validator) >= 0.8:
                return stype
            continue
        if stype in TEXTUAL and numeric_frac > 0.5:
            continue
        return stype
    for stype in VALUE_ONLY:
        if _match_frac(values, VALUE_RES[stype]) >= 0.9:
            return stype
    return None


# Free/public email providers: keeping their distribution is useful for analysis
# (e.g. "most common email domains") and reveals nothing about individuals.
# Any other domain (e.g. a corporate one) is replaced by a fake domain.
PUBLIC_EMAIL_DOMAINS = {
    "gmail.com", "googlemail.com", "yahoo.com", "yahoo.co.uk", "yahoo.fr", "ymail.com",
    "hotmail.com", "hotmail.co.uk", "hotmail.fr", "outlook.com", "live.com", "msn.com",
    "aol.com", "icloud.com", "me.com", "mac.com", "protonmail.com", "proton.me",
    "mail.com", "gmx.com", "gmx.de", "gmx.net", "web.de", "yandex.ru", "yandex.com",
    "zoho.com", "comcast.net", "verizon.net", "att.net", "orange.fr", "free.fr",
    "laposte.net", "libero.it", "qq.com", "163.com", "naver.com",
}
