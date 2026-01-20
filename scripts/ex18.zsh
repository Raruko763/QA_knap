#!/bin/zsh
# =====================================================
# Qknapcore.py を複数インスタンス × lam(角度重み) でまとめて実行
# 出力先: OUT_DIR/lam_*/<instance>...
# =====================================================

set -u

# ---------- 設定 ----------
INSTANCES=(
  "Leuven2"
  "E-n101-k8"
  "E-n51-k5"
  "X-n1001-k43"
)

# ★ 角度重み lam の5候補
LAMS=(
5.0
7.5
10.0
100
1000
)

# インスタンスディレクトリ群の root
ROOT_DIR="/home/toshiya1048/experiments/ex0/knapsack"

# 出力先（実験名）
OUT_DIR="./out/ex18_fixdist_sweep_lam"

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

  for lam in "${LAMS[@]}"; do
    # lam ごとに出力先を分ける（. を _ にして見やすく）
    lam_tag="${lam//./_}"
    OUT_LAM_DIR="${OUT_DIR}/lam_${lam_tag}"
    mkdir -p "$OUT_LAM_DIR"

    echo ""
    echo "-------------------------------------"
    echo "🧪 lam=${lam}  → 出力先: ${OUT_LAM_DIR}"
    echo "-------------------------------------"
    echo "🚀 実行開始..."

    python3 src/Qknapcore.py \
      -j "$JSON_PATH" \
      -sp "$OUT_LAM_DIR" \
      --t "$ANNEAL_MS" \
      -nt "$NT" \
      --max_iter "$MAX_ITER" \
      --tsp_solver "$TSP_SOLVER" \
      --lam "$lam"

    RET=$?
    if [[ $RET -eq 0 ]]; then
      echo "✅ 完了: ${inst} (lam=${lam})"
    else
      echo "❌ エラー (${RET}): ${inst} (lam=${lam})"
    fi
  done
done

echo ""
echo "🎉 すべてのインスタンス × lam の実行が終了しました"
