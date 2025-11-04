#!/bin/zsh
# =============================================
# QA_KNAP 量子アニーリング後処理 自動実験スクリプト
# 対象: 複数の before_data.json を連続実行
# 出力: out/<日付時刻>/<インスタンス名>/iteration_X.json
# =============================================

# ex3と同じ条件で大規模インスタンスを解く

set -euo pipefail

# ---------- 設定 ----------
# INSTANCES=(

#   "E-n51-k5"

# )
# INSTANCES=(
#   "X-n856-k95"
#   "Leuven2"
#   "X-n1001-k43"
# )
INSTANCES=(
  "X-n101-k25"
  "X-n153-k22"
  "X-n308-k13"
  "X-n367-k17"
  "X-n459-k26"
  "X-n561-k42"
  "X-n685-k75"
  "X-n733-k159"
  "X-n801-k40"
  "X-n936-k151"
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
    --t 10000 \
    --max_iter 100
 
  if [[ $? -eq 0 ]]; then
    echo "✅ 完了: ${inst}"
  else
    echo "❌ エラー発生: ${inst}"
  fi
done

echo "🎉 すべてのインスタンスで QA 実行が完了しました。"
