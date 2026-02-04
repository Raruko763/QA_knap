import os
import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import csv
import re


def extract_total_distance(json_file: Path) -> float:
    """iteration_X.json (list of per-cluster dicts) から total_distance(s) を合計して返す。"""
    try:
        with json_file.open("r") as f:
            data = json.load(f)

        if isinstance(data, list):
            total = 0.0
            for d in data:
                if not isinstance(d, dict):
                    continue
                if isinstance(d.get("total_distance"), (int, float)):
                    total += float(d["total_distance"])
                elif isinstance(d.get("total_distances"), (int, float)):
                    total += float(d["total_distances"])
            return total

        return 0.0

    except Exception as e:
        print(f"⚠️ 読み込み失敗: {json_file} ({e})")
        return 0.0


def plot_curve(xs, ys, title, out_png: Path):
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(xs, ys, marker="o", linewidth=2, label="Total Distance")
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


def parse_lam_alpha_from_path(p: Path):
    """
    lam_*/alpha_* を含むパスなら拾う（無ければ None）
    """
    lam = None
    alpha = None
    for part in p.parts:
        if part.startswith("lam_"):
            lam = part.replace("lam_", "").replace("_", ".")
        if part.startswith("alpha_"):
            alpha = part.replace("alpha_", "").replace("_", ".")
    try:
        lam_f = float(lam) if lam is not None else None
    except Exception:
        lam_f = None
    try:
        alpha_f = float(alpha) if alpha is not None else None
    except Exception:
        alpha_f = None
    return lam_f, alpha_f


def find_timestamp_for_run_dir(run_dir: Path) -> str:
    """
    run_dir = .../<timestamp>/<instance>_before_data  or  .../<timestamp>/<instance>_sweep_qubo
    timestamp を 1 つ上のディレクトリ名として取得する。
    """
    return run_dir.parent.name if run_dir.parent is not None else ""


def is_iteration_json(p: Path) -> bool:
    """
    iteration_<num>.json だけを True にする。
    除外:
      - iteration_<num>_swap.json
      - iteration_<num>_meta.json
      - iteration_<num>_swap_timings.json 等
    """
    if not (p.is_file() and p.suffix == ".json"):
        return False
    m = re.fullmatch(r"iteration_(\d+)\.json", p.name)
    return m is not None


def iteration_index(p: Path) -> int:
    m = re.fullmatch(r"iteration_(\d+)\.json", p.name)
    return int(m.group(1)) if m else -1


def scan_and_plot(base_dir: str, output_index=True):
    base = Path(base_dir).resolve()
    if not base.exists():
        print(f"❌ 指定ディレクトリが存在しません: {base}")
        return

    # ✅ *_before_data と *_sweep_qubo を両方拾う
    run_dirs = []
    run_dirs += [p for p in base.rglob("*_before_data") if p.is_dir()]
    run_dirs += [p for p in base.rglob("*_sweep_qubo") if p.is_dir()]
    run_dirs = sorted(set(run_dirs))

    if not run_dirs:
        print("⚠️ *_before_data / *_sweep_qubo が見つかりませんでした。base を確認してください。")
        print(f"   base={base}")
        return

    summary_rows = []

    for run_dir in run_dirs:
        # instance name
        if run_dir.name.endswith("_before_data"):
            instance = run_dir.name.replace("_before_data", "")
        elif run_dir.name.endswith("_sweep_qubo"):
            instance = run_dir.name.replace("_sweep_qubo", "")
        else:
            instance = run_dir.name

        ts = find_timestamp_for_run_dir(run_dir)
        lam, alpha = parse_lam_alpha_from_path(run_dir)

        # iteration_<num>.json だけ拾う
        itr_files = sorted(
            [f for f in run_dir.iterdir() if is_iteration_json(f)],
            key=iteration_index
        )

        if not itr_files:
            continue

        xs, ys = [], []
        for f in itr_files:
            it = iteration_index(f)
            dist = extract_total_distance(f)
            xs.append(it)
            ys.append(dist)

            summary_rows.append({
                "lam": lam,
                "alpha": alpha,
                "timestamp": ts,
                "instance": instance,
                "iteration": it,
                "total_distance": dist,
                "json_path": str(f),
                "run_dir": str(run_dir),
            })

        lam_s = "lamNA" if lam is None else f"lam{safe_name(str(lam))}"
        alpha_s = "alphaNA" if alpha is None else f"alpha{safe_name(str(alpha))}"
        fname_base = f"{safe_name(instance)}__{safe_name(ts)}__{lam_s}__{alpha_s}"

        png = run_dir / f"improvement_curve__{fname_base}.png"
        title = f"Improvement — {instance} | ts={ts} | lam={lam} | alpha={alpha}"
        plot_curve(xs, ys, title, png)
        print(f"📈 Saved: {png}")

        csv_path = run_dir / f"improvement_curve__{fname_base}.csv"
        with csv_path.open("w", newline="") as cf:
            writer = csv.writer(cf)
            writer.writerow(["iteration", "total_distance"])
            writer.writerows(zip(xs, ys))
        print(f"🧾 Saved: {csv_path}")

    # 全体サマリ
    if summary_rows and output_index:
        out_csv = base / "all_runs_summary.csv"
        with out_csv.open("w", newline="") as cf:
            fieldnames = ["lam", "alpha", "timestamp", "instance", "iteration", "total_distance", "json_path", "run_dir"]
            writer = csv.DictWriter(cf, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"🧮 全体サマリを書き出しました: {out_csv}")


def main():
    ap = argparse.ArgumentParser(
        description="Plot improvement curves under folders containing *_before_data or *_sweep_qubo"
    )
    ap.add_argument(
        "-b", "--base", required=True,
        help="Base folder that contains timestamp/*_before_data or timestamp/*_sweep_qubo"
    )
    args = ap.parse_args()
    scan_and_plot(args.base)


if __name__ == "__main__":
    main()
