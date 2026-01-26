#!/bin/zsh
# =====================================================
# test_sweep_vrp.zsh
# - data/raw/*.vrp を Sweep-only で一括テスト
# - Pythonが落ちても止めずに次へ（skip）
# =====================================================

set -u  # -e は使わない（落ちたら止まるので）

RAW_DIR="./data/raw"
OUT_BASE="./out/test_sweep"
CORE_SWEEP_VRP="./src/test_sweep.py"
START_ANGLE=0.0

mkdir -p "$OUT_BASE"
SKIP_LOG="$OUT_BASE/skipped.log"
ERR_LOG="$OUT_BASE/errors.log"

echo "Sweep-only batch start: $(date)" | tee -a "$ERR_LOG"

for vrp in "$RAW_DIR"/*.vrp; do
  name=$(basename "$vrp")
  echo ""
  echo "---------------------------------------------"
  echo "🚚 Instance: $name"
  echo "---------------------------------------------"

  # Python実行（stdout/stderrは両方ログにも残す）
  python3 "$CORE_SWEEP_VRP" \
    -i "$vrp" \
    -sp "$OUT_BASE" \
    --start_angle "$START_ANGLE" \
    >> "$ERR_LOG" 2>&1

  rc=$?
  if [[ $rc -ne 0 ]]; then
    echo "SKIP $name (exit=$rc)" | tee -a "$SKIP_LOG"
    echo "---- error tail ($name) ----" >> "$SKIP_LOG"
    tail -n 5 "$ERR_LOG" >> "$SKIP_LOG"
    echo "----------------------------" >> "$SKIP_LOG"
    continue
  fi
done

echo ""
echo "✅ Finished. Results: $OUT_BASE"
echo "📝 Skip log: $SKIP_LOG"
echo "📝 Error log: $ERR_LOG"
