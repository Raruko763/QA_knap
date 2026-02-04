#!/bin/zsh
set -euo pipefail

OUT_BASE="./out/ex21_前処理sweep後処理自分_angle"
mkdir -p "$OUT_BASE"
[[ -f "src/__init__.py" ]] || touch src/__init__.py

INSTANCES=(
  "Leuven2"
  "E-n101-k8"
  "E-n51-k5"
  "X-n1001-k43"
)

for inst in "${INSTANCES[@]}"; do
  vrp="data/raw/${inst}.vrp"
  echo "🚚 $inst"

  python3 -m src.core_sweep_qubo \
    -i "$vrp" \
    -sp "$OUT_BASE" \
    --start_angle 0.0 \
    --anneal_ms 3000 \
    --nt 1 \
    --p 1.0 \
    --q 1.0 \
    --lam 1.0 \
    --alpha 1.0 \
    --stage2_mode ang \
    --max_iter 50 \
    || echo "⚠️ skipped: $inst"
done
