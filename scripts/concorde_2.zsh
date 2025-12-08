#!/bin/zsh
# =============================================
# qaxapcore をまとめて実行するスクリプト
# - before_data.json を順番に読んで Qknapcore.py を実行
# - Concorde でクラスタ内 TSP を解く設定
# =============================================

set -euo pipefail

# ---------- 設定 ----------
# 実験したいインスタンス名
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

# before_data.json が置いてあるディレクトリ
OUT_DIR="./out"

# Qknapcore のパラメータ
ANNEAL_MS=10000      # --t （アニーリング時間 ms）
NT=1                 # -nt （QA の繰り返し回数）
MAX_ITER=100         # --max_iter
TSP_SOLVER="concorde"  # "ortools" / "concorde" / "amplify" から選択

# ---------- 実行 ----------
for inst in "${INSTANCES[@]}"; do
  echo "====================================="
  echo "🎯 インスタンス: ${inst}"
  echo "====================================="

  JSON_PATH="${OUT_DIR}/${inst}_before_data.json"

  if [[ ! -f "$JSON_PATH" ]]; then
    echo "⚠️ before_data.json が見つかりません: $JSON_PATH"
    continue
  fi

  echo "🚀 Qknapcore.py 実行開始..."
  python3 src/Qknapcore.py \
    -j "$JSON_PATH" \
    -sp "$OUT_DIR" \
    --t "$ANNEAL_MS" \
    -nt "$NT" \
    --max_iter "$MAX_ITER" \
    --tsp_solver "$TSP_SOLVER"

  if [[ $? -eq 0 ]]; then
    echo "✅ 完了: ${inst}"
  else
    echo "❌ エラー発生: ${inst}"
  fi
done

echo "🎉 すべてのインスタンスで Qknapcore 実行が完了しました。"
