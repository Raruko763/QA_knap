#!/bin/zsh
# =============================================
# qaxapcore をまとめて実行するスクリプト
# - before_data.json を順番に読んで Qknapcore.py を実行
# - Concorde でクラスタ内 TSP を解く設定
# =============================================
# p の範囲を変える場合は P_EXPONENTS（0.9^n の n 群）を編集してください。

set -uo pipefail

# ---------- 設定 ----------
# 実験したいインスタンス名
INSTANCES=(
  "E-n101-k14"
  "X-n856-k95"
  "Leuven2"
  "X-n1001-k43"
  "E-n101-k8"
  "E-n76-k14"
  "E-n76-k10"
  "E-n76-k8"
  "E-n76-k7"
  "E-n51-k5"
)

# before_data.json が置いてあるディレクトリ
OUT_DIR="./out"

# Qknapcore のパラメータ
ANNEAL_MS=10000      # --t （アニーリング時間 ms）
NT=1                 # -nt （QA の繰り返し回数）
MAX_ITER=100         # --max_iter
TSP_SOLVER="concorde"  # "ortools" / "concorde" / "amplify" から選択

# p = 0.9^n の n を指定（ここを編集すると p の範囲を変更できる）
P_EXPONENTS=(0 1 2 3 4 5)

# p 値を計算する（安定のため Python で指数計算し、末尾の 0 を削除）
calc_p() {
  python3 - "$1" <<'PY'
import sys
n = int(sys.argv[1])
val = 0.9 ** n
s = f"{val:.10f}".rstrip("0").rstrip(".")
print(s if s else "0")
PY
}

# ---------- 実行 ----------
for n in "${P_EXPONENTS[@]}"; do
  p_value="$(calc_p "$n")"
  p_dir="${OUT_DIR}/p_${p_value}"
  log_file="${p_dir}/run.log"
  mkdir -p "$p_dir"

  echo "#####################################"
  echo "🎯 p = 0.9^${n} => ${p_value}"
  echo "📂 出力先: ${p_dir}"
  echo "#####################################"

  {
    echo "========== $(date '+%Y-%m-%d %H:%M:%S') =========="
    echo "p = 0.9^${n} => ${p_value}"
  } >>"$log_file"

  for inst in "${INSTANCES[@]}"; do
    echo "-------------------------------------"
    echo "🎯 インスタンス: ${inst} (p=${p_value})"

    JSON_PATH="${OUT_DIR}/${inst}_before_data.json"

    if [[ ! -f "$JSON_PATH" ]]; then
      echo "⚠️ before_data.json が見つかりません: $JSON_PATH"
      echo "⚠️ SKIP ${inst} (p=${p_value})" | tee -a "$log_file"
      continue
    fi

    cmd=(
      python3 src/Qknapcore.py
      -j "$JSON_PATH"
      -sp "$p_dir"
      --p "$p_value"
      --t "$ANNEAL_MS"
      -nt "$NT"
      --max_iter "$MAX_ITER"
      --tsp_solver "$TSP_SOLVER"
    )
    cmd_str=$(printf "%q " "${cmd[@]}")

    echo "🚀 実行開始: ${cmd_str}"
    {
      echo ""
      echo "----- $(date '+%Y-%m-%d %H:%M:%S') inst=${inst} p=${p_value} -----"
      echo "CMD: ${cmd_str}"
    } >>"$log_file"

    if "${cmd[@]}" >>"$log_file" 2>&1; then
      echo "✅ 完了: ${inst} (p=${p_value})" | tee -a "$log_file"
    else
      echo "❌ エラー: ${inst} (p=${p_value})" | tee -a "$log_file"
    fi
  done
done

echo "🎉 すべてのインスタンスで Qknapcore 実行が完了しました。"
