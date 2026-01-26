#!/bin/zsh
# =====================================================
# test_sweep_vrp.zsh
# - data/raw/*.vrp を Sweep-only + Concorde で一括テスト
# - test_sweep.py を呼び出す
# =====================================================

set -euo pipefail

# ---------- 設定 ----------
# .vrp が置いてあるディレクトリ
RAW_DIR="./data/raw"

# 出力先（timestamp/instance_sweep_only が作られる）
OUT_BASE="./out/test_sweep"

# Sweep + Concorde core
CORE_SWEEP_VRP="./src/test_sweep.py"

# Sweep 開始角（rad）
START_ANGLE=0.0

# Concorde 設定（必要なら調整）
# export CONCORDE_BIN="/path/to/concorde"
CONCORDE_SEED=1

# ---------- 事前チェック ----------
if [[ ! -d "$RAW_DIR" ]]; then
  echo "❌ RAW_DIR not found: $RAW_DIR" >&2
  exit 1
fi

if [[ ! -f "$CORE_SWEEP_VRP" ]]; then
  echo "❌ test_sweep.py not found: $CORE_SWEEP_VRP" >&2
  exit 1
fi

mkdir -p "$OUT_BASE"

echo "============================================="
echo "Sweep + Concorde VRP batch test"
echo "RAW_DIR      : $RAW_DIR"
echo "OUT_BASE     : $OUT_BASE"
echo "CORE_SWEEP   : $CORE_SWEEP_VRP"
echo "START_ANGLE : $START_ANGLE (rad)"
echo "CONCORDE_SEED: $CONCORDE_SEED"
echo "============================================="

# ---------- 実行 ----------
for vrp in "$RAW_DIR"/*.vrp; do
  name=$(basename "$vrp")
  echo ""
  echo "---------------------------------------------"
  echo "🚚 Instance: $name"
  echo "---------------------------------------------"

  # ❗ Python 側で
  #   - node_coord 無し
  #   - MemoryError
  #   - Concorde 失敗
  # を全部 catch して skip するので zsh は止まらない
  python3 "$CORE_SWEEP_VRP" \
    -i "$vrp" \
    -sp "$OUT_BASE" \
    --start_angle "$START_ANGLE" \
    --solve_tsp \
    --seed "$CONCORDE_SEED" || echo "⚠️ skipped: $name"

done

echo ""
echo "✅ All Sweep + Concorde VRP tests finished."
echo "📂 Results saved under: $OUT_BASE"
