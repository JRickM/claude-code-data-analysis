"""Read and write the supported tabular file formats."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

SUPPORTED = {".csv", ".tsv", ".txt", ".xlsx", ".xlsm", ".xls", ".json", ".jsonl", ".parquet"}


@dataclass
class LoadedFile:
    path: Path
    tables: dict[str, pd.DataFrame]          # sheet name -> data (one entry for non-Excel)
    options: dict = field(default_factory=dict)


def find_files(folder: Path, recursive: bool, exclude: Path | None = None) -> list[Path]:
    pattern = "**/*" if recursive else "*"
    files = []
    for p in sorted(folder.glob(pattern)):
        if not p.is_file() or p.suffix.lower() not in SUPPORTED or p.name.startswith(("~$", ".")):
            continue
        if exclude is not None and exclude in p.parents:
            continue
        files.append(p)
    return files


def _read_text(path: Path) -> tuple[str, str]:
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return path.read_text(encoding=enc)[:65536], enc
        except UnicodeDecodeError:
            continue
    raise ValueError("could not decode file")


def read_file(path: Path) -> LoadedFile:
    ext = path.suffix.lower()
    if ext in (".csv", ".tsv", ".txt"):
        sample, enc = _read_text(path)
        if ext == ".tsv":
            sep = "\t"
        else:
            try:
                sep = csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
            except csv.Error:
                if ext == ".txt":
                    raise ValueError("not a delimited text file")
                sep = ","
        # Read everything as text so we keep leading zeros, exact formats, etc.
        df = pd.read_csv(path, sep=sep, dtype=str, encoding=enc)
        return LoadedFile(path, {"data": df}, {"sep": sep})
    if ext in (".xlsx", ".xlsm", ".xls"):
        sheets = pd.read_excel(path, sheet_name=None)
        return LoadedFile(path, sheets)
    if ext == ".jsonl":
        return LoadedFile(path, {"data": pd.read_json(path, lines=True)}, {"lines": True})
    if ext == ".json":
        return LoadedFile(path, {"data": pd.read_json(path)})
    if ext == ".parquet":
        return LoadedFile(path, {"data": pd.read_parquet(path)})
    raise ValueError(f"unsupported file type: {ext}")


def write_file(loaded: LoadedFile, tables: dict[str, pd.DataFrame], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ext = out_path.suffix.lower()
    if ext in (".csv", ".tsv", ".txt"):
        tables["data"].to_csv(out_path, sep=loaded.options.get("sep", ","), index=False,
                              encoding="utf-8")
    elif ext in (".xlsx", ".xlsm", ".xls"):
        if ext == ".xls":   # pandas can't write legacy .xls
            out_path = out_path.with_suffix(".xlsx")
        with pd.ExcelWriter(out_path) as writer:
            for sheet, df in tables.items():
                df.to_excel(writer, sheet_name=sheet, index=False)
    elif ext == ".jsonl":
        tables["data"].to_json(out_path, orient="records", lines=True, date_format="iso",
                               force_ascii=False)
    elif ext == ".json":
        tables["data"].to_json(out_path, orient="records", indent=2, date_format="iso",
                               force_ascii=False)
    elif ext == ".parquet":
        tables["data"].to_parquet(out_path, index=False)
