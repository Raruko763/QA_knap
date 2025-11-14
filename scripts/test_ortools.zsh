#!/bin/zsh
# =============================================
# OR-Tools ベース CVRP 実験スクリプト
# 対象: 複数の before_data.json を連続実行
# 出力: out/<日付時刻>/<インスタンス名>_before_data/iteration_X.json
# 実行本体: src/Qknapcore.py（tsp_solver=ortools）
# =============================================

set -euo pipefail

# ---------- 設定 ----------
# 実験対象インスタンス
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

OUT_DIR="./out"

# Qknapcore のパラメータ
QA_SOLVES=1              # -nt（1スワップあたりの QA 実行回数）
QA_TIME_MS=10000         # --t（QA のアニーリング時間 ms）
MAX_ITER=100             # --max_iter（反復上限）

TSP_SOLVER="ortools"     # --tsp_solver
TSP_TIME_LIMIT_MS=10000   # --tsp_time_limit_ms（クラスタ内 TSP の OR-Tools 制限時間）

# ---------- 実行 ----------
for inst in "${INSTANCES[@]}"; do
  echo "====================================="
  echo "🎯 インスタンス: ${inst}"
  echo "====================================="

  JSON_PATH="${OUT_DIR}/${inst}_before_data.json"

  if [[ ! -f "$JSON_PATH" ]]; then
    echo "⚠️ ファイルが見つかりません: $JSON_PATH"
    continue
  fi

  echo "🚀 実行開始..."
  python3 src/Qknapcore.py \
    -j "$JSON_PATH" \
    -sp "$OUT_DIR" \
    -nt "$QA_SOLVES" \
    --t "$QA_TIME_MS" \
    --max_iter "$MAX_ITER" \
    --tsp_solver "$TSP_SOLVER" \
    --tsp_time_limit_ms "$TSP_TIME_LIMIT_MS"

  if [[ $? -eq 0 ]]; then
    echo "✅ 完了: ${inst}"
  else
    echo "❌ エラー発生: ${inst}"
  fi
done

echo "🎉 すべてのインスタンスで OR-Tools 実験が完了しました。"
