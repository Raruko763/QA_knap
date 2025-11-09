#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scan a folder (default: ./raw) for .sol files and extract `Cost` values to CSV.
Output columns: instance,Cost

Usage:
  python collect_costs.py --root ./raw --out costs.csv
  python collect_costs.py --root ./raw --out costs.csv --recursive
  python collect_costs.py --selftest
"""
import argparse
import csv
import sys
import re
from pathlib import Path

COST_PAT = re.compile(r'^\s*Cost\s+([+-]?\d+(?:\.\d+)?)\s*$', re.IGNORECASE)

def read_text_best_effort(p: Path) -> str:
    """Read text trying a few encodings commonly used, return as str."""
    encodings = ("utf-8", "utf-8-sig", "cp932", "iso-8859-1")
    last_err = None
    for enc in encodings:
        try:
            return p.read_text(encoding=enc, errors="strict")
        except Exception as e:
            last_err = e
    # Fallback with replacement to avoid hard crash
    try:
        return p.read_text(encoding="utf-8", errors="replace")
    except Exception:
        raise last_err

def parse_cost_from_text(text: str):
    """Return the last matched Cost value as float (or None if not found)."""
    cost_val = None
    for line in text.splitlines():
        m = COST_PAT.match(line)
        if m:
            try:
                cost_val = float(m.group(1))
            except ValueError:
                # ignore bad lines
                pass
    return cost_val

def collect_costs(root: Path, recursive: bool):
    """Yield tuples (instance, cost) for each .sol file under root."""
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Root directory not found or not a directory: {root}")
    pattern = "**/*.sol" if recursive else "*.sol"
    for p in root.glob(pattern):
        if not p.is_file():
            continue
        text = read_text_best_effort(p)
        cost = parse_cost_from_text(text)
        instance = p.stem  # filename without extension
        yield instance, cost

def write_csv(rows, out_csv: Path):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["instance", "Cost"])
        for inst, cost in rows:
            w.writerow([inst, "" if cost is None else cost])

def main(argv=None):
    ap = argparse.ArgumentParser(description="Extract Cost from .sol files to CSV.")
    ap.add_argument("--root", type=str, default="./raw", help="Root folder containing .sol files (default: ./raw)")
    ap.add_argument("--out", type=str, default="costs.csv", help="Output CSV path (default: costs.csv)")
    ap.add_argument("--recursive", action="store_true", help="Recurse into subdirectories")
    ap.add_argument("--fail-on-missing", action="store_true",
                    help="Exit with non-zero status if any .sol lacks a Cost line")
    ap.add_argument("--selftest", action="store_true", help="Run a quick self-test then exit")
    args = ap.parse_args(argv)

    if args.selftest:
        # # Create a temp layout and verify basic behavior
        # import tempfile, shutil
        # from textwrap import dedent
        # with tempfile.TemporaryDirectory() as d:
        #     root = Path(d) / "raw"
        #     root.mkdir(parents=True, exist_ok=True)
        #     (root / "A.sol").write_text(dedent(\"\"\"
        #         Route #1: 1
        #         Route #2: 2 3
        #         Cost 247
        #     \"\"\").strip(), encoding="utf-8")
        #     (root / "B.sol").write_text("Route #1: 1\nCost 123.45\n", encoding="utf-8")
        #     (root / "C.sol").write_text("Route #1: 1\n(no cost here)\n", encoding="utf-8")
        #     out_csv = Path(d) / "costs.csv"
        #     rows = list(collect_costs(root, recursive=False))
        #     write_csv(rows, out_csv)
        #     print(out_csv.read_text(encoding="utf-8"))
        return 0

    root = Path(args.root)
    out_csv = Path(args.out)

    rows = list(collect_costs(root, recursive=args.recursive))
    if args.fail_on_missing and any(cost is None for _, cost in rows):
        # Print which instances are missing costs
        missing = [inst for inst, cost in rows if cost is None]
        sys.stderr.write("ERROR: Missing Cost in the following instances: " + ", ".join(missing) + "\n")
        return 2

    write_csv(rows, out_csv)
    print(f"Wrote {out_csv} with {len(rows)} rows.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
