#!/bin/zsh
# =====================================================
# Qknapcore.py を複数インスタンスまとめて実行
# 出力先: out/ex15/
# =====================================================

set -u

# ---------- 設定 ----------
INSTANCES=(
  "E-n101-k14"
  "X-n856-k95"
  "Leuven2"
  "X-n1001-k43"
  "E-n101-k8"
  "E-n76-k14"
  "E-n76-k10"
  "E-n76-k8"
  "E-n76-k7"
  "E-n51-k5"
)

# インスタンスディレクトリ群の root
ROOT_DIR="/home/toshiya1048/experiments/ex0/knapsack"

# 出力先
OUT_DIR="./out/ex15"

# Qknapcore パラメータ
ANNEAL_MS=10000
NT=1
MAX_ITER=100
TSP_SOLVER="concorde"

mkdir -p "$OUT_DIR"

# ---------- 実行 ----------
for inst in "${INSTANCES[@]}"; do
  echo ""
  echo "====================================="
  echo "🎯 インスタンス: ${inst}"
  echo "====================================="

  INST_DIR="${ROOT_DIR}/${inst}vrp"
  JSON_PATH="${INST_DIR}/${inst}_before_data.json"

  if [[ ! -f "$JSON_PATH" ]]; then
    echo "⚠️ before_data.json が見つかりません: $JSON_PATH"
    continue
  fi

  echo "📄 JSON: $JSON_PATH"
  echo "📂 出力先: $OUT_DIR"
  echo "🚀 実行開始..."

  python3 src/Qknapcore.py \
    -j "$JSON_PATH" \
    -sp "$OUT_DIR" \
    --t "$ANNEAL_MS" \
    -nt "$NT" \
    --max_iter "$MAX_ITER" \
    --tsp_solver "$TSP_SOLVER"

  RET=$?
  if [[ $RET -eq 0 ]]; then
    echo "✅ 完了: ${inst}"
  else
    echo "❌ エラー (${RET}): ${inst}"
  fi
done

echo ""
echo "🎉 すべてのインスタンスの実行が終了しました"
