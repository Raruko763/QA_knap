#!/bin/zsh
# =====================================================
# ex19: Qknapcore.py を
#   lam（目的関数の角度重み）× alpha（制約ペナルティ）
#   の 2 重ループで実行
#
# 出力構造:
# out/ex19_sweep_lam_alpha/
#   lam_<lam>/alpha_<alpha>/<timestamp>/<instance>_before_data/...
# =====================================================

set -u

# ---------- 設定 ----------
INSTANCES=(
  "Leuven2"
  "E-n101-k8"
  "E-n51-k5"
  "X-n1001-k43"
)

# ★ 角度重み lam（目的関数）
LAMS=(
  0.5
  1.0
  2.0
  5.0
  7.5
  10.0
)

# ★ 制約ペナルティ alpha
ALPHAS=(
  0.5
  1.0
  2.0
  5.0
  10.0
)

ROOT_DIR="/home/toshiya1048/experiments/ex0/knapsack"
OUT_DIR="./out/ex19_sweep_lam_alpha"

ANNEAL_MS=10000
NT=1
MAX_ITER=100
TSP_SOLVER="concorde"

mkdir -p "$OUT_DIR"

echo "====================================="
echo "🧪 ex19: sweep lam × alpha"
echo "📂 OUT_DIR: ${OUT_DIR}"
echo "====================================="

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
    lam_tag="${lam//./_}"
    OUT_LAM_DIR="${OUT_DIR}/lam_${lam_tag}"
    mkdir -p "$OUT_LAM_DIR"

    for alpha in "${ALPHAS[@]}"; do
      alpha_tag="${alpha//./_}"
      OUT_ALPHA_DIR="${OUT_LAM_DIR}/alpha_${alpha_tag}"
      mkdir -p "$OUT_ALPHA_DIR"

      echo ""
      echo "-------------------------------------"
      echo "🧪 lam=${lam}, alpha=${alpha}"
      echo "📂 出力先: ${OUT_ALPHA_DIR}"
      echo "-------------------------------------"
      echo "🚀 実行開始..."

      python3 src/Qknapcore.py \
        -j "$JSON_PATH" \
        -sp "$OUT_ALPHA_DIR" \
        --t "$ANNEAL_MS" \
        -nt "$NT" \
        --max_iter "$MAX_ITER" \
        --tsp_solver "$TSP_SOLVER" \
        --lam "$lam" \
        --alpha "$alpha"

      RET=$?
      if [[ $RET -eq 0 ]]; then
        echo "✅ 完了: ${inst} (lam=${lam}, alpha=${alpha})"
      else
        echo "❌ エラー (${RET}): ${inst} (lam=${lam}, alpha=${alpha})"
      fi
    done
  done
done

echo ""
echo "🎉 ex19: lam × alpha の全実験が終了しました"
