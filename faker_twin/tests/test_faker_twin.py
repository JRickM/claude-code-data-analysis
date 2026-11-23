import json

import numpy as np
import pandas as pd

from faker_twin import TableFaker, profile_table
from faker_twin.cli import main


def make_df(n=300, seed=0):
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2023-01-01") + pd.to_timedelta(rng.integers(0, 365, n), unit="D")
    end = start + pd.to_timedelta(rng.integers(1, 30, n), unit="D")
    return pd.DataFrame({
        "customer_id": [f"CUS-{i:05d}" for i in range(n)],
        "firstName": rng.choice(["Ann", "Bob", "Cara", "Dan", "Eve", "Finn", "Gus"] * 50, n),
        "email": [f"user{i}@{d}" for i, d in enumerate(rng.choice(["gmail.com", "acme-corp.com"], n))],
        "phone": [f"({rng.integers(200, 999)}) {rng.integers(100, 999)}-{rng.integers(1000, 9999)}" for _ in range(n)],
        "zip": [f"{z:05d}" for z in rng.integers(0, 99999, n)],
        "age": rng.integers(18, 90, n).astype(str),
        "score": rng.normal(50, 10, n).round(2).astype(str),
        "rating": rng.choice(["1", "2", "3", "4", "5"], n),
        "active": rng.choice(["yes", "no"], n),
        "start": start.strftime("%d/%m/%Y"),
        "end": end.strftime("%d/%m/%Y"),
        "notes": [" ".join(rng.choice(["good", "late", "paid", "called", "again", "soon"], 6)) for _ in range(n)],
    }).astype(str)


def kinds(df):
    return {c.name: (c.kind, c.semantic) for c in profile_table(df).columns}


def test_detection():
    k = kinds(make_df())
    assert k["customer_id"] == ("pattern", None)
    assert k["firstName"][0] == "categorical" or k["firstName"][1] == "first_name"
    assert k["email"] == ("semantic", "email")
    assert k["phone"] == ("pattern", "phone")
    assert k["zip"] == ("pattern", "postcode")
    assert k["age"] == ("integer", None)
    assert k["score"] == ("float", None)
    assert k["rating"] == ("categorical", None)
    assert k["active"] == ("boolean", None)
    assert k["start"] == ("datetime", None)
    assert k["notes"] == ("text", None)


def test_generation_keeps_structure():
    real = make_df()
    prof = profile_table(real)
    fake = TableFaker(prof, seed=1).generate(500)
    assert list(fake.columns) == list(real.columns)
    assert len(fake) == 500
    assert fake["customer_id"].str.fullmatch(r"[A-Z]{3}-\d{5}").all()
    assert fake["customer_id"].is_unique
    assert fake["zip"].str.fullmatch(r"\d{5}").all()
    assert set(fake["active"]) <= {"yes", "no"}
    assert set(fake["rating"]) <= set(real["rating"])
    s = pd.to_datetime(fake["start"], format="%d/%m/%Y")
    e = pd.to_datetime(fake["end"], format="%d/%m/%Y")
    assert (e >= s).all()                                   # relation preserved
    assert not set(fake["email"]) & set(real["email"])      # no real values leaked
    assert fake["email"].str.endswith("@acme-corp.com").sum() == 0   # private domain replaced
    assert 18 <= fake["age"].min() and fake["age"].max() <= 89


def test_nulls_preserved():
    real = make_df(1000)
    real.loc[real.sample(frac=0.2, random_state=0).index, "score"] = None
    fake = TableFaker(profile_table(real), seed=2).generate()
    assert 0.15 < fake["score"].isna().mean() < 0.25


def test_seed_is_reproducible():
    prof = profile_table(make_df())
    a = TableFaker(prof, seed=7).generate()
    b = TableFaker(prof, seed=7).generate()
    pd.testing.assert_frame_equal(a, b)


def test_cli_end_to_end(tmp_path):
    src = tmp_path / "data"
    (src / "sub").mkdir(parents=True)
    make_df(50).to_csv(src / "customers.csv", index=False)
    make_df(20).to_csv(src / "sub" / "more.csv", sep=";", index=False)
    make_df(10).to_json(src / "records.json", orient="records")
    out = tmp_path / "out"
    assert main([str(src), "-o", str(out), "-r", "-s", "3", "--report"]) == 0
    fake = pd.read_csv(out / "customers.csv", dtype=str)
    assert list(fake.columns) == list(make_df(1).columns) and len(fake) == 50
    assert len(pd.read_csv(out / "sub" / "more.csv", sep=";")) == 20
    assert len(json.loads((out / "records.json").read_text())) == 10
    assert (out / "_profiles" / "customers.csv.json").exists()
