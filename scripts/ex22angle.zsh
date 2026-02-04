#!/bin/zsh
# =====================================================
# ex21: core_sweep_qubo を
#   lam（角度重み）× alpha（制約ペナルティ）
#   の 2 重ループで実行
#
# ex19 と lam / alpha を完全一致させる
#
# 出力構造:
# out/ex21_sweep_lam_alpha/
#   lam_<lam>/alpha_<alpha>/<instance>/...
# =====================================================

set -u
set -o pipefail

# ---------- 設定 ----------
INSTANCES=(
  "Leuven2"
  "E-n101-k8"
  "E-n51-k5"
  "X-n1001-k43"
)

# ★ ex19 と同じ lam
LAMS=(
  0.5
  1.0
  2.0
  5.0
  7.5
  10.0
)

# ★ ex19 と同じ alpha
ALPHAS=(
  0.5
  1.0
  2.0
  5.0
  10.0
)

OUT_DIR="./out/ex22_sweep_lam_alpha_angle"

ANNEAL_MS=3000
NT=1
MAX_ITER=10

STAGE2_MODE="ang"     # dist / dist+ang に変えたければここ

mkdir -p "$OUT_DIR"
[[ -f "src/__init__.py" ]] || touch src/__init__.py

echo "====================================="
echo "🧪 ex21: sweep + QUBO (lam × alpha)"
echo "📂 OUT_DIR: ${OUT_DIR}"
echo "⚙ stage2_mode: ${STAGE2_MODE}"
echo "====================================="

# ---------- 実行 ----------
for inst in "${INSTANCES[@]}"; do
  echo ""
  echo "====================================="
  echo "🎯 インスタンス: ${inst}"
  echo "====================================="

  VRP_PATH="data/raw/${inst}.vrp"
  if [[ ! -f "$VRP_PATH" ]]; then
    echo "⚠️ vrp が見つかりません: $VRP_PATH"
    continue
  fi

  for lam in "${LAMS[@]}"; do
    lam_tag="${lam//./_}"
    OUT_LAM_DIR="${OUT_DIR}/lam_${lam_tag}"
    mkdir -p "$OUT_LAM_DIR"

    for alpha in "${ALPHAS[@]}"; do
      alpha_tag="${alpha//./_}"
      OUT_ALPHA_DIR="${OUT_LAM_DIR}/alpha_${alpha_tag}/${inst}"
      mkdir -p "$OUT_ALPHA_DIR"

      echo ""
      echo "-------------------------------------"
      echo "🧪 lam=${lam}, alpha=${alpha}"
      echo "📂 出力先: ${OUT_ALPHA_DIR}"
      echo "-------------------------------------"
      echo "🚀 実行開始..."

      python3 -m src.core_sweep_qubo \
        -i "$VRP_PATH" \
        -sp "$OUT_ALPHA_DIR" \
        --start_angle 0.0 \
        --anneal_ms "$ANNEAL_MS" \
        --nt "$NT" \
        --p 1.0 \
        --q 1.0 \
        --lam "$lam" \
        --alpha "$alpha" \
        --stage2_mode "$STAGE2_MODE" \
        --max_iter "$MAX_ITER"

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
echo "🎉 ex21: sweep + QUBO lam × alpha angle の全実験が終了しました"
