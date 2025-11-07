
import os
import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import csv
import re

def extract_total_distance(json_file):
    """Return sum of 'total_distance(s)' from iteration_X.json (list of per-cluster dicts)."""
    try:
        with open(json_file, "r") as f:
            data = json.load(f)
        if isinstance(data, list):
            # accept either "total_distance" or "total_distances"
            total = 0.0
            for d in data:
                if not isinstance(d, dict):
                    continue
                if "total_distance" in d and isinstance(d["total_distance"], (int, float)):
                    total += float(d["total_distance"])
                elif "total_distances" in d and isinstance(d["total_distances"], (int, float)):
                    total += float(d["total_distances"])
            return total
        return 0.0
    except Exception as e:
        print(f"⚠️ 読み込み失敗: {json_file} ({e})")
        return 0.0

def plot_curve(xs, ys, title, out_png):
    plt.figure(figsize=(8, 5))
    plt.plot(xs, ys, marker='o', linewidth=2, label="Total Distance")
    plt.title(title)
    plt.xlabel("Iteration")
    plt.ylabel("Total Distance")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

def safe_name(s: str) -> str:
    """ファイル名に安全な形へ（英数・._-以外は_へ）"""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)

def scan_and_plot(base_dir, output_index=True):
    base = Path(base_dir)
    if not base.exists():
        print(f"❌ 指定ディレクトリが存在しません: {base}")
        return

    run_roots = []
    for p in base.iterdir():
        if p.is_dir():
            try:
                if any((p / child).is_dir() and child.endswith("_before_data") for child in os.listdir(p)):
                    run_roots.append(p)
            except PermissionError:
                continue
    try:
        if any(child.endswith("_before_data") for child in os.listdir(base)):
            run_roots.append(base)
    except PermissionError:
        pass

    run_roots = sorted(set(run_roots))
    if not run_roots:
        print("⚠️ 対象となる run ディレクトリが見つかりませんでした。")
        return

    summary_rows = []
    for run_root in run_roots:
        for inst_dir in sorted(run_root.iterdir()):
            if not inst_dir.is_dir() or not inst_dir.name.endswith("_before_data"):
                continue

            ts = run_root.name
            instance = inst_dir.name.replace("_before_data", "")
            # ここでファイル名用の安全なベース名を作る
            fname_base = f"{safe_name(instance)}__{safe_name(ts)}"

            itr_files = sorted(
                [f for f in inst_dir.iterdir() if f.is_file() 
                 and f.name.startswith("iteration_") and f.suffix == ".json" 
                 and "timings" not in f.name
                 ],
                key=lambda p: int(p.stem.split("_")[1])
            )
            if not itr_files:
                continue

            xs, ys = [], []
            for f in itr_files:
                it = int(f.stem.split("_")[1])
                dist = extract_total_distance(f)
                xs.append(it)
                ys.append(dist)
                summary_rows.append({
                    "timestamp": ts,
                    "instance": instance,
                    "iteration": it,
                    "total_distance": dist,
                    "json_path": str(f)
                })

            # ▼ 出力ファイル名にインスタンス名＆タイムスタンプを含める
            png = inst_dir / f"improvement_curve__{fname_base}.png"
            title = f"Improvement Curve — {instance} ({ts})"
            plot_curve(xs, ys, title, png)
            print(f"📈 Saved: {png}")

            csv_path = inst_dir / f"improvement_curve__{fname_base}.csv"
            with open(csv_path, "w", newline="") as cf:
                writer = csv.writer(cf)
                writer.writerow(["iteration", "total_distance"])
                writer.writerows(zip(xs, ys))
            print(f"🧾 Saved: {csv_path}")

    if summary_rows:
        out_csv = base / "all_runs_summary.csv"  # ここは全体まとめなので固定名のままでもOK
        with open(out_csv, "w", newline="") as cf:
            fieldnames = ["timestamp", "instance", "iteration", "total_distance", "json_path"]
            writer = csv.DictWriter(cf, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"🧮 全体サマリを書き出しました: {out_csv}")
        
def main():
    ap = argparse.ArgumentParser(description="Plot improvement curves for all instances under an ex1-style folder")
    ap.add_argument("-b", "--base", required=True, help="Base folder that contains timestamp folders and *_before_data dirs (e.g., ./ex1)")
    args = ap.parse_args()
    scan_and_plot(args.base)

if __name__ == "__main__":
    main()
