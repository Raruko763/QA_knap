import pandas as pd
import matplotlib.pyplot as plt

# === ファイル読込 ===
all_runs = pd.read_csv("all_runs_summary.csv")

# 最適解一覧
best_known = {
    "Leuven2": 111395,
    "X-n856-k95": 89965,
    "E-n101-k14": 1067,
    "E-n101-k8": 815,
    "E-n76-k14": 1021,
    "E-n76-k10": 830,
    "E-n76-k8": 735,
    "E-n76-k7": 682,
    "E-n51-k5": 521
}

# === 各インスタンスの最短距離と最終距離 ===
agg = (
    all_runs.sort_values(["instance", "iteration"])
    .groupby("instance")
    .agg(
        best_iteration_distance=("total_distance", "min"),
        last_iteration_distance=("total_distance", "last")
    )
    .reset_index()
)

# 最適解を追加
agg["best_known"] = agg["instance"].map(best_known)

# === グラフ①：最終距離 vs 最適解 ===
plt.figure(figsize=(9,6))
x = range(len(agg))
plt.bar(x, agg["last_iteration_distance"], label="Final iteration", alpha=0.7)
plt.bar(x, agg["best_known"], label="Best known", alpha=0.5)
plt.xticks(x, agg["instance"], rotation=45, ha="right")
plt.ylabel("Total Distance")
plt.title("Final Distance vs Best Known")
plt.legend()
plt.tight_layout()
plt.savefig("compare_final_vs_best.png", dpi=200)
plt.close()

# === グラフ②：最短距離 vs 最終距離 ===
plt.figure(figsize=(9,6))
plt.bar(x, agg["best_iteration_distance"], label="Shortest (min iteration)", alpha=0.7)
plt.bar(x, agg["last_iteration_distance"], label="Final iteration", alpha=0.5)
plt.xticks(x, agg["instance"], rotation=45, ha="right")
plt.ylabel("Total Distance")
plt.title("Shortest vs Final (per instance)")
plt.legend()
plt.tight_layout()
plt.savefig("compare_shortest_vs_final.png", dpi=200)
plt.close()

print("✅ 出力:")
print("compare_final_vs_best.png")
print("compare_shortest_vs_final.png")
