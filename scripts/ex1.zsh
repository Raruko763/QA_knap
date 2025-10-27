#!/bin/zsh
# =============================================
# QA_KNAP 量子アニーリング後処理 自動実験スクリプト
# 対象: 複数の before_data.json を連続実行
# 出力: out/<日付時刻>/<インスタンス名>/iteration_X.json
# =============================================

set -euo pipefail

# ---------- 設定 ----------
# INSTANCES=(

#   "E-n51-k5"

# )
INSTANCES=(
  "X-n856-k95"
  "Leuven2"
  "E-n51-k5"
  "E-n101-k14"
  "X-n1001-k43"
)
OUT_DIR="./out"

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
    -nt 3 \
    --t 10000

  if [[ $? -eq 0 ]]; then
    echo "✅ 完了: ${inst}"
  else
    echo "❌ エラー発生: ${inst}"
  fi
done

echo "🎉 すべてのインスタンスで QA 実行が完了しました。"
