"""Generate a fake DataFrame from a TableProfile using Faker."""

from __future__ import annotations

import random
from typing import Callable

import numpy as np
import pandas as pd
from faker import Faker
from faker.exceptions import UniquenessException

from .profiler import QUANTILES, ColumnProfile, TableProfile

SEMANTIC_FAKERS: dict[str, Callable[[Faker], object]] = {
    "credit_card": lambda f: f.credit_card_number(),
    "iban": lambda f: f.iban(),
    "ssn": lambda f: f.ssn(),
    "phone": lambda f: f.phone_number(),
    "username": lambda f: f.user_name(),
    "first_name": lambda f: f.first_name(),
    "last_name": lambda f: f.last_name(),
    "name": lambda f: f.name(),
    "company": lambda f: f.company(),
    "job": lambda f: f.job(),
    "postcode": lambda f: f.postcode(),
    "street_address": lambda f: f.street_address(),
    "address": lambda f: f.address(),
    "city": lambda f: f.city(),
    "state": lambda f: f.state() if hasattr(f, "state") else f.city(),
    "country": lambda f: f.country(),
    "url": lambda f: f.url(),
    "ipv4": lambda f: f.ipv4(),
    "uuid": lambda f: f.uuid4(),
}

_UPPER = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_LOWER = "abcdefghijklmnopqrstuvwxyz"
_DIGIT = "0123456789"


class TableFaker:
    def __init__(self, profile: TableProfile, locale: str = "en_US", seed: int | None = None):
        self.profile = profile
        self.fake = Faker(locale)
        if seed is not None:
            self.fake.seed_instance(seed)
        self.rng = np.random.default_rng(seed)
        self.rnd = random.Random(seed)
        self._epochs: dict[str, np.ndarray] = {}   # generated datetimes, for relations

    # ------------------------------------------------------------------ utils
    def _from_quantiles(self, quantiles: list[float], n: int) -> np.ndarray:
        """Inverse-CDF sampling: reproduces the real distribution without copying values."""
        return np.interp(self.rng.random(n), QUANTILES, quantiles)

    def _unique_loop(self, make: Callable[[], object], n: int, unique: bool) -> list:
        if not unique:
            return [make() for _ in range(n)]
        seen, out = set(), []
        for _ in range(n):
            for _attempt in range(100):
                v = make()
                if v not in seen:
                    break
            seen.add(v)
            out.append(v)
        return out

    # ------------------------------------------------------------- generators
    def _gen_boolean(self, c: ColumnProfile, n: int):
        p = c.params
        return np.where(self.rng.random(n) < p["p_true"], p["true"], p["false"]).tolist() \
            if not isinstance(p["true"], bool) else (self.rng.random(n) < p["p_true"]).tolist()

    def _gen_integer(self, c: ColumnProfile, n: int):
        vals = np.rint(self._from_quantiles(c.params["quantiles"], n)).astype("int64")
        if c.unique:  # spread duplicates out while staying near the real range
            vals = pd.Series(vals)
            dup = vals.duplicated()
            nxt = int(vals.max()) + 1
            for i in vals.index[dup]:
                vals[i] = nxt
                nxt += 1
            vals = vals.to_numpy()
        return vals.tolist()

    def _gen_float(self, c: ColumnProfile, n: int):
        return np.round(self._from_quantiles(c.params["quantiles"], n), c.params["decimals"]).tolist()

    def _gen_datetime(self, c: ColumnProfile, n: int):
        secs = self._from_quantiles(c.params["quantiles"], n)
        rel = next((r for r in self.profile.relations if r.column == c.name), None)
        if rel is not None and rel.base in self._epochs:
            base = self._epochs[rel.base]
            ok = ~np.isnan(base)
            secs[ok] = base[ok] + self._from_quantiles(rel.delta_quantiles, int(ok.sum()))
        if not c.params["has_time"]:
            secs = np.floor(secs / 86400) * 86400
        self._epochs[c.name] = secs
        ts = pd.to_datetime(secs, unit="s")
        if c.params["native"]:
            return list(ts)
        return list(ts.strftime(c.params["format"]))

    def _gen_categorical(self, c: ColumnProfile, n: int):
        p = c.params
        idx = self.rng.choice(len(p["values"]), size=n, p=np.array(p["weights"]) / sum(p["weights"]))
        return [p["values"][i] for i in idx]

    def _gen_pattern(self, c: ColumnProfile, n: int):
        p = c.params
        weights = p["weights"]

        def make():
            shape = self.rnd.choices(p["shapes"], weights=weights)[0]
            return "".join(
                self.rnd.choice(_UPPER) if ch == "A" else
                self.rnd.choice(_LOWER) if ch == "a" else
                self.rnd.choice(_DIGIT) if ch == "9" else ch
                for ch in shape)
        return self._unique_loop(make, n, c.unique)

    def _gen_text(self, c: ColumnProfile, n: int):
        p = c.params
        f = self.fake
        if p["style"] == "words":
            counts = np.maximum(1, np.rint(self._from_quantiles(p["word_quantiles"], n))).astype(int)
            it = iter(counts)

            def make():
                s = " ".join(f.words(int(next(it))))
                return s.title() if p["title_case"] else s
            return self._unique_loop(make, n, False)
        lengths = np.maximum(5, np.rint(self._from_quantiles(p["len_quantiles"], n))).astype(int)
        out = []
        for L in lengths:
            s = f.sentence() if L < 20 else f.text(max_nb_chars=int(L))
            if not p["multiline"]:
                s = s.replace("\n", " ")
            out.append(s)
        return out

    def _gen_email(self, c: ColumnProfile, n: int):
        p = c.params
        f = self.fake
        domains, weights = p["public_domains"], p["public_weights"]

        def make():
            if domains and self.rnd.random() < p["public_frac"]:
                domain = self.rnd.choices(domains, weights=weights)[0]
            else:
                domain = f.domain_name()
            user = f.user_name()
            if self.rnd.random() < 0.3:
                user += str(self.rnd.randint(1, 99))
            return f"{user}@{domain}"
        return self._unique_loop(make, n, c.unique)

    def _gen_semantic(self, c: ColumnProfile, n: int):
        if c.semantic == "email":
            return self._gen_email(c, n)
        fn = SEMANTIC_FAKERS[c.semantic]
        p = c.params
        source = self.fake
        if c.unique:
            try:
                out = [fn(self.fake.unique) for _ in range(n)]
                self.fake.unique.clear()
                return self._post_semantic(out, p)
            except (UniquenessException, AttributeError):
                self.fake.unique.clear()
        return self._post_semantic([fn(source) for _ in range(n)], p)

    @staticmethod
    def _post_semantic(values: list, p: dict) -> list:
        if p.get("multiline") is False:
            values = [v.replace("\n", ", ") if isinstance(v, str) else v for v in values]
        if p.get("digits_only"):
            values = ["".join(ch for ch in str(v) if ch.isdigit()) for v in values]
        return values

    # ------------------------------------------------------------------- main
    def generate(self, n_rows: int | None = None) -> pd.DataFrame:
        n = self.profile.n_rows if n_rows is None else n_rows
        data = {}
        for c in self.profile.columns:
            if c.kind == "empty":
                data[c.name] = [None] * n
                continue
            values = getattr(self, f"_gen_{c.kind}")(c, n)
            mask = self.rng.random(n) < c.null_frac
            if mask.any():
                values = [None if m else v for v, m in zip(values, mask)]
                if c.kind == "datetime":
                    self._epochs[c.name][mask] = np.nan
            series = pd.Series(values, dtype=object)
            if c.kind == "integer":
                series = series.astype("Int64")
            elif c.kind == "float":
                series = series.astype("float64")
            data[c.name] = series
        return pd.DataFrame(data)
