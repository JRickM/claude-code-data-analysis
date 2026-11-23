# faker-twin

Point it at a folder of data files. It examines each file and writes a **fake twin** with the same
columns, types, formats and statistical shape. Every value is newly generated with
[Faker](https://faker.readthedocs.io/).

The output is safe to share with an AI assistant or a colleague when you need code written against
your data without exposing the real records.

## Install

```bash
cd faker_twin
pip install -e .                 # core (CSV / TSV / JSON / JSONL)
pip install -e ".[excel,parquet]"  # + .xlsx and .parquet support
```

## Usage

```bash
faker-twin                         # opens a folder picker
faker-twin ../data                 # writes ../data_fake/<same file names>
faker-twin ../data -o ./fake -n 100 -s 42     # 100 rows per file, reproducible
faker-twin ../data --profile-only  # just show what was detected, write nothing
faker-twin ../data -r -l fr_FR     # include sub-folders, French names/addresses
```

`python -m faker_twin ...` works the same way.

| Option | Meaning |
|---|---|
| `input` | Data folder. If you leave it out, a dialog lets you pick one. |
| `-o, --output` | Output folder. Default: `<input>_fake`, next to the input folder. |
| `-n, --rows` | Rows per file. Default: the same number as the source. |
| `-l, --locale` | Faker locale (`en_US`, `fr_FR`, `de_DE`, ...). |
| `-s, --seed` | Seed for reproducible output. |
| `-r, --recursive` | Also process sub-folders, keeping the folder layout. |
| `--profile-only` | Print the detected structure only. |
| `--report` | Also write `_profiles/*.json` with the detected schema. |

Supported files: `.csv` (delimiter auto-detected), `.tsv`, delimited `.txt`, `.json` (records),
`.jsonl`, `.xlsx` (all sheets), and `.parquet`.

## What gets detected and how it is faked

| Detected | Example | Fake values |
|---|---|---|
| Person and contact PII, found by column name and value checks | `guest_email`, `first_name`, `billing_address`, `credit_card_number`, `iban`, `ssn` | Matching Faker provider |
| Email | `x@gmail.com` | Faker usernames. Public providers (gmail, yahoo...) keep their real share; other domains are replaced with fake ones |
| Phone, post code | `(555) 123-4567`, `02134` | Same character shape, random digits |
| Codes and IDs | `CUS-00042` | Same shape (`AAA-99999`), random characters, unique if the original was |
| Boolean | `True/False`, `yes/no`, `Y/N` | Same tokens and the same true rate |
| Integer and float | `146.83` | Sampled from the real distribution (101 quantiles), same decimal places |
| Date and datetime | `04 Apr 2020`, `2023-01-31 14:05:00` | Same format and distribution |
| Categories | `BASIC / DELUXE / SUITE` | Same values and frequencies |
| Free text | comments, notes | Faker words or sentences of similar length |

The tool also keeps:
- the share of missing values in every column;
- which columns are unique;
- date pairs where one is always on or after the other (e.g. `end >= start`), along with the
  real gap distribution.

## Privacy notes

- The fake files contain **no real values**, except category labels (such as `BASIC`), boolean
  tokens and public email domains. Before sharing, check the `--profile-only` output for
  categorical columns whose labels might be sensitive.
- The `--report` JSON files contain statistics of the **real** data (quantiles, category values).
  Keep them local.
- Numbers and dates follow the real distribution of each column on its own. Correlations between
  columns are not reproduced, apart from the date-pair rule above. If your analysis depends on
  those correlations, use a model-based tool such as SDV.

## Tests

```bash
pip install -e ".[dev]"
pytest
```
