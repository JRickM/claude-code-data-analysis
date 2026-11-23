"""Command-line entry point: examine a folder and write fake twins of every file."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .fileio import find_files, read_file, write_file
from .generator import TableFaker
from .profiler import TableProfile, profile_table


def choose_folder() -> Path | None:
    """Open a folder picker; fall back to typing the path."""
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        chosen = filedialog.askdirectory(title="Choose the folder with your data")
        root.destroy()
        return Path(chosen) if chosen else None
    except Exception:
        typed = input("Path to the data folder: ").strip().strip('"')
        return Path(typed) if typed else None


def print_profile(name: str, prof: TableProfile) -> None:
    print(f"\n  [{name}]  {prof.n_rows} rows x {len(prof.columns)} columns")
    w = max([len(c.name) for c in prof.columns] + [6])
    for c in prof.columns:
        kind = c.kind if not c.semantic else f"{c.kind}:{c.semantic}"
        flags = " unique" if c.unique else ""
        print(f"    {c.name:<{w}}  {kind:<22} nulls {c.null_frac:>6.1%}{flags:<7}  {c.describe()}")
    for r in prof.relations:
        print(f"    relation: {r.column} >= {r.base} (gap preserved)")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="faker-twin",
        description="Examine every data file in a folder and create a fake twin "
                    "with the same structure using Faker.")
    p.add_argument("input", nargs="?", type=Path,
                   help="folder with your data (omit to pick one in a dialog)")
    p.add_argument("-o", "--output", type=Path,
                   help="output folder (default: <input>_fake next to the input folder)")
    p.add_argument("-n", "--rows", type=int,
                   help="rows to generate per file (default: same as the source file)")
    p.add_argument("-l", "--locale", default="en_US", help="Faker locale, e.g. fr_FR, de_DE")
    p.add_argument("-s", "--seed", type=int, help="random seed for reproducible output")
    p.add_argument("-r", "--recursive", action="store_true", help="include sub-folders")
    p.add_argument("--profile-only", action="store_true",
                   help="only examine and print the detected structure, write nothing")
    p.add_argument("--report", action="store_true",
                   help="also write <output>/_profiles/*.json (contains statistics of the "
                        "REAL data: don't share it)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    src = args.input or choose_folder()
    if src is None:
        print("No folder chosen.")
        return 1
    src = src.expanduser().resolve()
    if not src.is_dir():
        print(f"Not a folder: {src}")
        return 1
    out = (args.output or src.with_name(src.name + "_fake")).expanduser().resolve()
    if out == src:
        print("Output folder must differ from the input folder.")
        return 1

    files = find_files(src, args.recursive, exclude=out)
    if not files:
        print(f"No supported data files found in {src}")
        return 1

    print(f"Examining {len(files)} file(s) in {src}")
    ok = 0
    for path in files:
        rel = path.relative_to(src)
        print(f"\n{rel}")
        try:
            loaded = read_file(path)
            fake_tables, profiles = {}, {}
            for name, df in loaded.tables.items():
                prof = profile_table(df)
                profiles[name] = prof
                print_profile(name, prof)
                if not args.profile_only:
                    faker = TableFaker(prof, locale=args.locale, seed=args.seed)
                    fake_tables[name] = faker.generate(args.rows)
            if not args.profile_only:
                target = out / rel
                write_file(loaded, fake_tables, target)
                print(f"  -> wrote {target}")
                if args.report:
                    rpt = out / "_profiles" / rel.with_suffix(rel.suffix + ".json")
                    rpt.parent.mkdir(parents=True, exist_ok=True)
                    rpt.write_text(json.dumps({k: v.to_dict() for k, v in profiles.items()},
                                              indent=2, default=str), encoding="utf-8")
            ok += 1
        except Exception as exc:  # keep going with the other files
            print(f"  !! skipped: {type(exc).__name__}: {exc}")

    print(f"\nDone: {ok}/{len(files)} file(s) processed."
          + ("" if args.profile_only else f" Fake data in {out}"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
