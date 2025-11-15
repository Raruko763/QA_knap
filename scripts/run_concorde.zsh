#!/bin/zsh
# =============================================
# TSPLIB + Concorde ベンチマーク一括実行スクリプト
# 対象: 複数の .tsp を tsplib_concorde_test.py 経由で解く
# 出力:
#   - out_concorde/<インスタンス名>_concorde.csv
#   - out_concorde/all_concorde_merged.csv（結合版）
# =============================================

set -euo pipefail

# ---------- 設定 ----------
# CVRP のインスタンス名をそのまま使う想定
INSTANCES=(
  "E-n101-k14"
  "E-n101-k8"
  "E-n76-k14"
  "E-n76-k10"
  "E-n76-k8"
  "E-n76-k7"
  "E-n51-k5"
  "X-n856-k95"
  "Leuven2"
  "X-n1001-k43"
)

# .tsp ファイルが置いてあるディレクトリ
# 例: ./instance/tsp/E-n101-k14.tsp
TSP_DIR="./instance/tsp"

# 出力先ディレクトリ
OUT_DIR="./out_concorde"
mkdir -p "${OUT_DIR}"

# 1ファイルごとに CSV を作るか？
MAKE_PER_INSTANCE_CSV=true

# ---------- 実行 ----------
CSV_LIST=()

for inst in "${INSTANCES[@]}"; do
  echo "====================================="
  echo "🎯 インスタンス: ${inst}"
  echo "====================================="

  TSP_PATH="${TSP_DIR}/${inst}.tsp"

  if [[ ! -f "${TSP_PATH}" ]]; then
    echo "⚠️ .tsp ファイルが見つかりません: ${TSP_PATH}"
    continue
  fi

  # 個別 CSV のファイル名
  OUT_CSV="${OUT_DIR}/${inst}_concorde.csv"

  echo "🚀 Concorde 実行 (tsplib_concorde_test.py 経由)..."
  python3 tsplib_concorde_test.py \
    "${TSP_PATH}" \
    --max_dim 100000 \
    --output "${OUT_CSV}"

  if [[ $? -eq 0 ]]; then
    echo "✅ 完了: ${inst} → ${OUT_CSV}"
    CSV_LIST+=("${OUT_CSV}")
  else
    echo "❌ エラー発生: ${inst}"
  fi
done

# ---------- CSV 結合（オプション） ----------
if [[ "${#CSV_LIST[@]}" -gt 0 ]]; then
  MERGED_CSV="${OUT_DIR}/all_concorde_merged.csv"
  echo "🧩 CSV を結合します → ${MERGED_CSV}"

  # 1つ目のヘッダ + 以降はデータ行だけ結合
  head -n 1 "${CSV_LIST[1]}" > "${MERGED_CSV}"
  for csv in "${CSV_LIST[@]}"; do
    tail -n +2 "${csv}" >> "${MERGED_CSV}"
  done

  echo "🎉 結合 CSV 完成: ${MERGED_CSV}"
else
  echo "⚠️ 有効な CSV が 1 件も生成されませんでした。"
fi

echo "🎉 すべてのインスタンスで Concorde 実行が完了しました。"
