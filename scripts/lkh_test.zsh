#!/bin/zsh
# =============================================
# QA_KNAP 量子アニーリング後処理 自動実験スクリプト
# 対象: 複数の before_data.json を連続実行
# 出力: out/<日付時刻>/<インスタンス名>/iteration_X.json
# TSPソルバ: OR-Tools / LKH / Amplify (今回は LKH を使用)
# =============================================

set -euo pipefail

# ---------- パス設定 ----------
# このスクリプトの位置からリポジトリルートを解決
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

OUT_DIR="${REPO_ROOT}/out"

# LKH 実行ファイルのパスを環境変数に設定
export LKH_BIN="${REPO_ROOT}/LKH3-3.0.6/LKH"

echo "📂 REPO_ROOT = ${REPO_ROOT}"
echo "📂 OUT_DIR   = ${OUT_DIR}"
echo "🔑 LKH_BIN   = ${LKH_BIN}"

# ---------- インスタンス一覧 ----------
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
  python3 "${REPO_ROOT}/src/Qknapcore.py" \
    -j "$JSON_PATH" \
    -sp "$OUT_DIR" \
    -nt 1 \
    --t 10000 \
    --tsp_solver lkh \
    --tsp_time_limit_ms 50000

  echo "✅ 完了: ${inst}"
done

echo "🎉 すべてのインスタンスで QA + LKH-TSP 実行が完了しました。"
