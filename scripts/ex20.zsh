#!/bin/zsh
set -euo pipefail

RAW_DIR="./data/raw"
OUT_BASE="./out/ex20_前処理sweep法のみ"
START_ANGLE=0.0

mkdir -p "$OUT_BASE"
[[ -f "src/__init__.py" ]] || touch src/__init__.py

INSTANCES=(
  "Leuven2"
  "E-n101-k8"
  "E-n51-k5"
  "X-n1001-k43"
)

echo "============================================="
echo "Sweep + Concorde (selected instances)"
echo "OUT_BASE: $OUT_BASE"
echo "============================================="

for inst in "${INSTANCES[@]}"; do
  vrp="$RAW_DIR/${inst}.vrp"
  if [[ ! -f "$vrp" ]]; then
    echo "⚠️ missing: $vrp (skip)"
    continue
  fi

  echo ""
  echo "---------------------------------------------"
  echo "🚚 Instance: ${inst}.vrp"
  echo "---------------------------------------------"

  python3 -m src.test_sweep \
    -i "$vrp" \
    -sp "$OUT_BASE" \
    --start_angle "$START_ANGLE" \
    --solve_tsp || echo "⚠️ skipped: $inst"
done

echo ""
echo "✅ Finished. Results: $OUT_BASE"
