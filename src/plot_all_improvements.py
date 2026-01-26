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
    ex19_sweep_lam_alpha のパス例:
      .../ex19_sweep_lam_alpha/lam_0_5/alpha_10_0/2026.../Leuven2_before_data/...
    ここから lam/alpha を float として抽出する。
    見つからなければ None。
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


def find_timestamp_for_before_data(before_dir: Path):
    """
    before_dir = .../<timestamp>/<instance>_before_data
    timestamp を 1 つ上のディレクトリ名として取得する。
    """
    if before_dir.parent is not None:
        return before_dir.parent.name
    return ""


def scan_and_plot(base_dir: str, output_index=True):
    base = Path(base_dir).resolve()
    if not base.exists():
        print(f"❌ 指定ディレクトリが存在しません: {base}")
        return

    # ✅ ここが肝：再帰的に *_before_data を全部拾う（ex19構造に対応）
    before_dirs = sorted([p for p in base.rglob("*_before_data") if p.is_dir()])

    if not before_dirs:
        print("⚠️ *_before_data が見つかりませんでした。base を確認してください。")
        print(f"   base={base}")
        return

    summary_rows = []

    for inst_dir in before_dirs:
        instance = inst_dir.name.replace("_before_data", "")
        ts = find_timestamp_for_before_data(inst_dir)  # 例: 20260121_155519
        lam, alpha = parse_lam_alpha_from_path(inst_dir)

        # iteration_X.json を拾う（timings除外）
        itr_files = sorted(
            [
                f for f in inst_dir.iterdir()
                if f.is_file()
                and f.name.startswith("iteration_")
                and f.suffix == ".json"
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
                "lam": lam,
                "alpha": alpha,
                "timestamp": ts,
                "instance": instance,
                "iteration": it,
                "total_distance": dist,
                "json_path": str(f),
            })

        # 出力ファイル名（lam/alphaも入れて衝突しにくく）
        lam_s = "lamNA" if lam is None else f"lam{safe_name(str(lam))}"
        alpha_s = "alphaNA" if alpha is None else f"alpha{safe_name(str(alpha))}"
        fname_base = f"{safe_name(instance)}__{safe_name(ts)}__{lam_s}__{alpha_s}"

        png = inst_dir / f"improvement_curve__{fname_base}.png"
        title = f"Improvement — {instance} | ts={ts} | lam={lam} | alpha={alpha}"
        plot_curve(xs, ys, title, png)
        print(f"📈 Saved: {png}")

        csv_path = inst_dir / f"improvement_curve__{fname_base}.csv"
        with csv_path.open("w", newline="") as cf:
            writer = csv.writer(cf)
            writer.writerow(["iteration", "total_distance"])
            writer.writerows(zip(xs, ys))
        print(f"🧾 Saved: {csv_path}")

    # 全体サマリ
    if summary_rows and output_index:
        out_csv = base / "all_runs_summary.csv"
        with out_csv.open("w", newline="") as cf:
            fieldnames = ["lam", "alpha", "timestamp", "instance", "iteration", "total_distance", "json_path"]
            writer = csv.DictWriter(cf, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"🧮 全体サマリを書き出しました: {out_csv}")


def main():
    ap = argparse.ArgumentParser(
        description="Plot improvement curves under ex19 (lam/alpha/timestamp/*_before_data) folders"
    )
    ap.add_argument(
        "-b", "--base", required=True,
        help="Base folder that contains lam_*/alpha_*/timestamp/*_before_data (e.g., ./out/ex19_sweep_lam_alpha)"
    )
    args = ap.parse_args()
    scan_and_plot(args.base)


if __name__ == "__main__":
    main()
