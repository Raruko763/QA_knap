import os
import json
import matplotlib.pyplot as plt
import argparse


def extract_total_distance(json_file):
    """iteration_X.json から total_distances を合計して返す"""
    try:
        with open(json_file, "r") as f:
            data = json.load(f)
        if isinstance(data, list):
            return sum(d.get("total_distances", 0) for d in data if isinstance(d, dict))
        return 0
    except Exception as e:
        print(f"⚠️ 読み込み失敗: {json_file} ({e})")
        return 0


def main():
    parser = argparse.ArgumentParser(description="Plot improvement curve for iterative QA optimization")
    parser.add_argument("-p", "--path", help="Target directory (e.g. ./out/20251027_2330/E-n51-k5)", required=True)
    parser.add_argument("-o", "--output", help="Output image filename", default="improvement_curve.png")
    args = parser.parse_args()

    base_dir = args.path
    if not os.path.exists(base_dir):
        print(f"❌ 指定パスが存在しません: {base_dir}")
        return

    files = sorted(
        [f for f in os.listdir(base_dir) if f.startswith("iteration_") and f.endswith(".json")],
        key=lambda x: int(x.split("_")[1].split(".")[0])
    )
    if not files:
        print(f"❌ iteration_X.json が見つかりません: {base_dir}")
        return

    iteration_nums = []
    total_distances = []
    for f in files:
        path = os.path.join(base_dir, f)
        total = extract_total_distance(path)
        iteration = int(f.split("_")[1].split(".")[0])
        iteration_nums.append(iteration)
        total_distances.append(total)
        print(f"Iteration {iteration}: Total distance = {total:.3f}")

    plt.figure(figsize=(8, 5))
    plt.plot(iteration_nums, total_distances, marker='o', color='tab:blue', linewidth=2, label="Total Distance")
    plt.title("Improvement Curve (Total Distance vs Iteration)")
    plt.xlabel("Iteration")
    plt.ylabel("Total Distance")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    out_path = os.path.join(base_dir, args.output)
    plt.savefig(out_path)
    plt.close()
    print(f"\n📈 改善曲線を保存しました: {out_path}")


if __name__ == "__main__":
    main()
