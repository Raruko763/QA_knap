import os
import json
import csv

# === 設定 ===
BASE_DIR = "./ex2"  # ex2フォルダを探索
OUTPUT_CSV = "./swap_timings_summary.csv"

records = []

for root, dirs, files in os.walk(BASE_DIR):
    for file in files:
        if file.endswith("_swap_timings.json"):
            json_path = os.path.join(root, file)
            timestamp = os.path.basename(os.path.dirname(os.path.dirname(json_path)))
            instance_name = os.path.basename(os.path.dirname(json_path))

            try:
                with open(json_path, "r") as f:
                    data = json.load(f)

                    # データが1件 or 複数件でも対応
                    if isinstance(data, dict):
                        data = [data]

                    for entry in data:
                        records.append({
                            "instance_name": instance_name,
                            "timestamp": timestamp,
                            "iteration": entry.get("iteration"),
                            "swap_index": entry.get("swap_index"),
                            "from_cluster": entry.get("from_cluster"),
                            "to_cluster": entry.get("to_cluster"),
                            "qa_ms": entry.get("qa_ms"),
                            "move_ms": entry.get("move_ms"),
                            "block_ms": entry.get("block_ms"),
                        })
            except Exception as e:
                print(f"⚠️ 読み込み失敗: {json_path} ({e})")

# === CSV出力 ===
if records:
    with open(OUTPUT_CSV, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=[
            "instance_name", "timestamp", "iteration",
            "swap_index", "from_cluster", "to_cluster",
            "qa_ms", "move_ms", "block_ms"
        ])
        writer.writeheader()
        writer.writerows(records)

    print(f"✅ {len(records)} 件のデータを {OUTPUT_CSV} に保存しました。")
else:
    print("⚠️ _swap_timings.json が見つかりませんでした。")
