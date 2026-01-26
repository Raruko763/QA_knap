#!/bin/zsh
# =====================================================
# test_sweep.zsh
# - data/raw/*.vrp を Sweep-only + Concorde で一括テスト
# - src.test_sweep を module 実行する（import安定）
# =====================================================

set -euo pipefail

RAW_DIR="./data/raw"
OUT_BASE="./out/test_sweep"
START_ANGLE=0.0

# Concorde 設定
# CONCORDE_SEED=1
export CONCORDE_BIN="/home/toshiya1048/tools/concorde/TSP/concorde"


# ---------- 事前チェック ----------
if [[ ! -d "$RAW_DIR" ]]; then
  echo "❌ RAW_DIR not found: $RAW_DIR" >&2
  exit 1
fi

mkdir -p "$OUT_BASE"

# src をパッケージとして扱えるように（無ければ作る）
if [[ ! -f "src/__init__.py" ]]; then
  touch src/__init__.py
fi

echo "============================================="
echo "Sweep + Concorde VRP batch test"
echo "RAW_DIR      : $RAW_DIR"
echo "OUT_BASE     : $OUT_BASE"
echo "START_ANGLE : $START_ANGLE (rad)"
# echo "CONCORDE_SEED: $CONCORDE_SEED"
echo "============================================="

# ---------- 実行 ----------
for vrp in "$RAW_DIR"/*.vrp; do
  name=$(basename "$vrp")
  echo ""
  echo "---------------------------------------------"
  echo "🚚 Instance: $name"
  echo "---------------------------------------------"

  python3 -m src.test_sweep \
    -i "$vrp" \
    -sp "$OUT_BASE" \
    --start_angle "$START_ANGLE" \
    --solve_tsp \
    # --seed "$CONCORDE_SEED" || echo "⚠️ skipped: $name"
done

echo ""
echo "✅ All Sweep + Concorde VRP tests finished."
echo "📂 Results saved under: $OUT_BASE"
