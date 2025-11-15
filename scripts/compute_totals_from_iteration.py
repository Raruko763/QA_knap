#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Summarize per-iteration total distances for all runs under a BASE directory.
- Detects any "<timestamp>/*_before_data" directories that contain iteration_*.json.
- For each run dir, computes totals using ONLY iteration files via the delta algorithm:
    * iteration_1.json: must contain ALL clusters (cluster_id, total_distance)
    * iteration_k.json (k>=2): contains touched clusters only
- Writes per-run CSV: "<run>/total_distance_by_iteration.csv"
- Appends to master CSV at BASE: "iteration_total_route_summary.csv"
Usage:
    python summarize_totals_for_base.py /path/to/out/ortools_test
"""

import sys
import json
import re
from pathlib import Path

ITER_FILE_RE = re.compile(r"^iteration_(\d+)\.json$")

def find_run_dirs(base: Path):
    """Yield directories that directly contain iteration_*.json (i.e., '*_before_data')."""
    if not base.is_dir():
        return
    for ts_dir in base.iterdir():
        if not ts_dir.is_dir():
            continue
        for sub in ts_dir.iterdir():
            if not sub.is_dir():
                continue
            if sub.name.endswith("_before_data"):
                if any(ITER_FILE_RE.match(p.name) for p in sub.glob("iteration_*.json")):
                    yield sub

def load_iteration(path: Path):
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    items = []
    for obj in data:
        try:
            cid = int(obj["cluster_id"])
            dist = float(obj.get("total_distance", 0.0))
            items.append((cid, dist))
        except Exception:
            continue
    return items

def compute_totals_for_run(run_dir: Path):
    """Return list of (iteration, total_distance). Also writes per-run CSV."""
    iter_files = []
    for p in run_dir.glob("iteration_*.json"):
        m = ITER_FILE_RE.match(p.name)
        if not m:
            continue
        k = int(m.group(1))
        if k >= 1:
            iter_files.append((k, p))
    iter_files.sort(key=lambda t: t[0])
    if not iter_files:
        return []

    if iter_files[0][0] != 1:
        raise RuntimeError(f"First iteration must be iteration_1.json with all clusters. Found iteration_{iter_files[0][0]}.json in {run_dir}")

    # Baseline
    k1, p1 = iter_files[0]
    items = load_iteration(p1)
    if not items:
        raise RuntimeError(f"iteration_1.json empty or malformed in {run_dir}")
    prev_map = {cid: dist for cid, dist in items}
    total_prev = sum(prev_map.values())

    out_csv = run_dir / "total_distance_by_iteration.csv"
    with out_csv.open("w", encoding="utf-8") as f:
        f.write("iteration,total_distance\n")
        f.write(f"{k1},{total_prev}\n")

    results = [(k1, total_prev)]

    # Subsequent iterations
    for k, p in iter_files[1:]:
        touched = load_iteration(p)
        sum_old = 0.0
        sum_new = 0.0
        for cid, newd in touched:
            oldd = float(prev_map.get(cid, 0.0))
            sum_old += oldd
            sum_new += newd
            prev_map[cid] = newd
        total_k = total_prev - sum_old + sum_new
        with out_csv.open("a", encoding="utf-8") as f:
            f.write(f"{k},{total_k}\n")
        results.append((k, total_k))
        total_prev = total_k

    return results

def main():
    if len(sys.argv) < 2:
        print("Usage: python summarize_totals_for_base.py /path/to/out/ortools_test")
        sys.exit(1)
    base = Path(sys.argv[1]).resolve()
    if not base.is_dir():
        print("Not a directory:", base)
        sys.exit(1)

    master_path = base / "iteration_total_route_summary.csv"
    with master_path.open("w", encoding="utf-8") as mf:
        mf.write("timestamp_dir,instance_dir,iteration,total_distance\n")

    found_any = False
    for run_dir in find_run_dirs(base):
        found_any = True
        ts_dir = run_dir.parent.name
        instance_dir = run_dir.name
        try:
            totals = compute_totals_for_run(run_dir)
            with master_path.open("a", encoding="utf-8") as mf:
                for it, tot in totals:
                    mf.write(f"{ts_dir},{instance_dir},{it},{tot}\n")
        except Exception as e:
            # Record a simple error row (leave total blank)
            with master_path.open("a", encoding="utf-8") as mf:
                mf.write(f"{ts_dir},{instance_dir},ERROR,\n")

    if not found_any:
        print("No runs found under:", base)
    else:
        print("Master summary path:", master_path)

if __name__ == "__main__":
    main()
