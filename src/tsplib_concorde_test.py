#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TSPLIB の .tsp をまとめて Concorde（tsp_concorde.solve_tsp_concorde）で解いて
BEST_KNOWN と比較し、CSV にまとめるベンチマークスクリプト。

前提:
- 同じディレクトリに tsp_concorde.py があり、以下の関数が定義されていること:
    solve_tsp_concorde(dist_matrix, work_dir, seed=None, concorde_bin=None)
- solve_tsp_concorde は:
    - ルート補完を一切行わない
    - solver_status != "SUCCESS" の場合 route は None になる
"""

import sys
import math
import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

from tsp_concorde import solve_tsp_concorde

# --- 既知の最適解 (BEST_KNOWN) ---
BEST_KNOWN: Dict[str, Optional[int]] = {
    "att48": 10628,
    "berlin52": 7542,
    "bier127": 118282,
    "brazil58": 25395,
    "brg180": 1950,
    "ch130": 6110,
    "ch150": 6528,
    "dantzig42": 699,
    "eil51": 426,
    "eil76": 538,
    "eil101": 629,
    "fri26": 937,
    "gr17": 2085,
    "gr21": 2707,
    "gr24": 1272,
    "gr48": 5046,
    "gr96": 55209,
    "gr120": 6942,
    "gr137": 69853,
    "gr202": 40160,    # 202 → 200制限ギリ外なので外してもOK
    "hk48": 11461,
    "kroA100": 21282,
    "kroB100": 22141,
    "kroC100": 20749,
    "kroD100": 21294,
    "kroE100": 22068,
    "kroA150": 26524,
    "kroB150": 26130,
    "kroA200": 29368,
    "lin105": 14379,
    "lin318": None,    # 318 → 除外推奨
    "pa561": None,     # 561 → 除外
    "pr76": 108159,
    "pr107": 44303,
    "pr124": 59030,
    "pr136": 96772,
    "pr144": 58537,
    "pr152": 73682,
    "rat99": 1211,
    "rat195": 2323,
    "rat783": None,    # 除外
    "rd100": 7910,
    "rd400": None,     # 除外
    "st70": 675,
    "ts225": None,     # 225 → 200制限外
    "tsp225": None,    # 同上
    "ulysses16": 6859,
    "ulysses22": 7013,
    "a280": 2579,
    "ali535": None,    # 535 → 除外
}

def save_results_to_csv(csv_path: str | Path, rows: List[Dict[str, Any]]):
    """
    solve_tsp_concorde → tsplib_concorde_test 側で集めた結果 rows を CSV に保存する
    """
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "Instance",
        "DIMENSION",
        "Best_Known",
        "Optimal_STDOUT",
        "Calculated_Cost",
        "Best_Known_Diff",
        "Stdout_Diff",
        "GAP_Pct",
        "Status",
        "Time_sec",
        "Message",
    ]


    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"📄 CSV 保存完了: {csv_path}")


# --- TSPLIB 読み取り（座標 or FULL_MATRIX） ---
def read_tsplib(tsp_path: Path) -> Dict[str, Any]:
    """
    TSPLIB TSP ファイルの必要な情報だけ読み取る。

    戻り値:
        {
            "name": str | None,
            "dim": int,
            "edge_weight_type": str,      # 例: "EUC_2D", "EXPLICIT"
            "edge_weight_format": str | None,
            "coords": List[(x, y)] | None,
            "dist_matrix": List[List[int]] | None,
        }
    """
    name: Optional[str] = None
    dim: Optional[int] = None
    edge_weight_type: Optional[str] = None
    edge_weight_format: Optional[str] = None
    coords: List[Tuple[float, float]] = []
    dist_matrix: Optional[List[List[int]]] = None

    if not tsp_path.is_file():
        raise FileNotFoundError(f"ファイルが見つからないか、ディレクトリです: {tsp_path}")

    lines = tsp_path.read_text().splitlines()

    # ヘッダ部
    for line in lines:
        s = line.strip()
        if not s or s.upper().startswith("COMMENT"):
            continue
        if ":" in s:
            k, v = [x.strip() for x in s.split(":", 1)]
            ku = k.upper()
            if ku == "NAME":
                name = v
            elif ku == "DIMENSION":
                dim = int(v)
            elif ku == "EDGE_WEIGHT_TYPE":
                edge_weight_type = v.upper()
            elif ku == "EDGE_WEIGHT_FORMAT":
                edge_weight_format = v.upper()

    if dim is None:
        raise ValueError("DIMENSION が見つかりません")
    if edge_weight_type is None:
        raise ValueError("EDGE_WEIGHT_TYPE が見つかりません")

    # 座標読み (NODE_COORD_SECTION)
    if any(line.upper().strip() == "NODE_COORD_SECTION" for line in lines):
        reading = False
        for line in lines:
            s = line.strip()
            u = s.upper()
            if u == "NODE_COORD_SECTION":
                reading = True
                continue
            if u in ("EOF", "DEMAND_SECTION", "CAPACITY"):
                break
            if ":" in s and not reading:
                continue
            if reading:
                parts = s.split()
                if len(parts) >= 3 and parts[0].isdigit():
                    x = float(parts[1])
                    y = float(parts[2])
                    coords.append((x, y))
        coords = coords[:dim]

    # 距離行列読み (EXPLICIT/FULL_MATRIX)
    if edge_weight_type == "EXPLICIT" and edge_weight_format == "FULL_MATRIX":
        dist_matrix = [[0] * dim for _ in range(dim)]
        reading = False
        row = col = 0
        for line in lines:
            s = line.strip()
            u = s.upper()
            if u == "EDGE_WEIGHT_SECTION":
                reading = True
                continue
            if u == "EOF":
                break
            if not reading:
                continue
            for p in s.split():
                if row >= dim:
                    break
                dist_matrix[row][col] = int(float(p))
                col += 1
                if col >= dim:
                    col = 0
                    row += 1
            if row >= dim:
                break

    return {
        "name": name,
        "dim": dim,
        "edge_weight_type": edge_weight_type,
        "edge_weight_format": edge_weight_format,
        "coords": coords if coords else None,
        "dist_matrix": dist_matrix,
    }


# --- EUC_2D 用の距離行列生成（TSPLIB の丸め仕様に合わせる） ---
def build_euc2d_dist_matrix(coords: List[Tuple[float, float]]) -> List[List[int]]:
    """
    TSPLIB の EUC_2D 距離:
        d(i,j) = int( sqrt( (xi-xj)^2 + (yi-yj)^2 ) + 0.5 )
    """
    n = len(coords)
    dmat = [[0] * n for _ in range(n)]
    for i in range(n):
        x1, y1 = coords[i]
        for j in range(n):
            if i == j:
                continue
            x2, y2 = coords[j]
            dij = math.hypot(x1 - x2, y1 - y2)
            dmat[i][j] = int(dij + 0.5)
    return dmat


# --- メイン: TSPLIB → Concorde → CSV ---
def main() -> None:
    ap = argparse.ArgumentParser(
        description="TSPLIB .tsp を tsp_concorde.solve_tsp_concorde で解いて CSV にまとめる"
    )
    ap.add_argument("path", help="処理対象のファイル、またはディレクトリパス", type=str)
    ap.add_argument(
        "--max_dim",
        help="処理する都市数(DIMENSION)の最大上限 (この値以下の問題のみ処理)",
        type=int,
        default=sys.maxsize,
    )
    ap.add_argument(
        "--output",
        help="結果を保存するCSVファイル名",
        type=str,
        default="concorde_results.csv",
    )
    ap.add_argument(
        "--workdir",
        help="Concorde 実行用の一時ディレクトリ",
        type=str,
        default="_concorde_work_tsplib",
    )
    args = ap.parse_args()

    target_path = Path(args.path).resolve()
    max_dim = args.max_dim
    workdir = args.workdir

    # 処理対象 .tsp リスト
    tsp_files: List[Path] = []
    if target_path.is_dir():
        print(f"📂 ディレクトリ '{target_path}' 内の .tsp ファイルを再帰的に検索中...")
        tsp_files.extend(sorted(target_path.rglob("*.tsp")))
    elif target_path.is_file():
        tsp_files.append(target_path)
    else:
        print(f"❌ パスが見つからないか、無効です: {args.path}")
        return

    results_list: List[Dict[str, Any]] = []

    print(f"🔍 処理対象のファイル数: {len(tsp_files)}")
    if max_dim != sys.maxsize:
        print(f"📏 都市数上限: {max_dim} を超えるファイルはスキップします。")

    for tsp_file_path in tsp_files:
        instance_stem = tsp_file_path.stem
        print(f"\n--- 処理中: {tsp_file_path.name} ---")

        try:
            info = read_tsplib(tsp_file_path)
            dim = int(info["dim"])
            etype = info["edge_weight_type"]
            eformat = info["edge_weight_format"]
            coords = info["coords"]
            explicit_matrix = info["dist_matrix"]

            if dim > max_dim:
                print(f"⏭️ スキップ: DIMENSION={dim} > max_dim={max_dim}")
                results_list.append({
                    "Instance": instance_stem,
                    "DIMENSION": dim,
                    "Best_Known": BEST_KNOWN.get(instance_stem, "N/A"),
                    "Calculated_Cost": "N/A",
                    "GAP_Pct": "N/A",
                    "Status": "SKIP_DIM",
                    "Time_sec": "N/A",
                    "Message": f"DIMENSION {dim} > max_dim {max_dim}",
                })
                continue

            best_known_cost = BEST_KNOWN.get(instance_stem)
            print(f"✅ DIMENSION={dim}, EDGE_WEIGHT_TYPE={etype}, BEST_KNOWN={best_known_cost}")

            # --- 距離行列の構築 ---
            if etype == "EUC_2D":
                if coords is None:
                    raise ValueError("coords が必要です (EUC_2D)")
                dist_matrix = build_euc2d_dist_matrix(coords)
            elif etype == "EXPLICIT" and eformat == "FULL_MATRIX":
                if explicit_matrix is None:
                    raise ValueError("dist_matrix が必要です (EXPLICIT/FULL_MATRIX)")
                dist_matrix = explicit_matrix
            else:
                msg = f"EDGE_WEIGHT_TYPE={etype}, FORMAT={eformat} は未対応のためスキップ"
                print(f"⏭️ {msg}")
                results_list.append({
                    "Instance": instance_stem,
                    "DIMENSION": dim,
                    "Best_Known": best_known_cost if best_known_cost is not None else "N/A",
                    "Calculated_Cost": "N/A",
                    "GAP_Pct": "N/A",
                    "Status": "SKIP_UNSUPPORTED_EDGE_WEIGHT",
                    "Time_sec": "N/A",
                    "Message": msg,
                })
                continue

            # --- Concorde (tsp_concorde) で解く ---
            res = solve_tsp_concorde(dist_matrix, work_dir=workdir)

            solver_status = res.get("solver_status", "UNKNOWN")
            route = res.get("route")
            total_distance = res.get("total_distance")
            elapsed_sec = res.get("solve_time_ms", 0) / 1000.0

            if solver_status != "SUCCESS" or route is None or total_distance is None:
                msg = f"solver_status={solver_status}"
                print(f"❌ FAIL: {msg}")
                results_list.append({
                    "Instance": instance_stem,
                    "DIMENSION": dim,
                    "Best_Known": best_known_cost if best_known_cost is not None else "N/A",
                    "Calculated_Cost": "N/A",
                    "GAP_Pct": "N/A",
                    "Status": "FAILED",
                    "Time_sec": f"{elapsed_sec:.3f}",
                    "Message": msg,
                })
                continue

            calculated_cost = float(total_distance)
            gap_pct_str = "N/A"
            if best_known_cost is not None and best_known_cost > 0:
                gap = (calculated_cost - best_known_cost) * 100.0 / best_known_cost
                gap_pct_str = f"{gap:.4f}"
            best_known_diff = (
                calculated_cost - best_known_cost
                if best_known_cost is not None else "N/A"
            )

            stdout_optimal = res.get("optimal_value_stdout")
            stdout_diff = (
                calculated_cost - stdout_optimal
                if stdout_optimal is not None else "N/A"
            )
     

            result_row = {
                "Instance": instance_stem,
                "DIMENSION": dim,
                "Best_Known": best_known_cost if best_known_cost is not None else "N/A",
                "Optimal_STDOUT": stdout_optimal if stdout_optimal is not None else "N/A",
                "Calculated_Cost": f"{calculated_cost:.4f}",
                "Best_Known_Diff": best_known_diff,
                "Stdout_Diff": stdout_diff,
                "GAP_Pct": gap_pct_str,
                "Status": "SUCCESS",
                "Time_sec": f"{elapsed_sec:.3f}",
                "Message": "",
            }

            results_list.append(result_row)

            print(
                f"    ✅ Status=SUCCESS, Cost={result_row['Calculated_Cost']}, "
                f"GAP={gap_pct_str} %, Time={result_row['Time_sec']} sec"
            )

        except Exception as e:
            print(f"❌ 処理中にエラー: {type(e).__name__}: {e}")
            results_list.append({
                "Instance": instance_stem,
                "DIMENSION": "N/A",
                "Best_Known": BEST_KNOWN.get(instance_stem) if BEST_KNOWN.get(instance_stem) is not None else "N/A",
                "Calculated_Cost": "N/A",
                "GAP_Pct": "N/A",
                "Status": "ERROR",
                "Time_sec": "N/A",
                "Message": str(e),
            })
            continue

    # --- CSV 書き出し ---
    if results_list:
        save_results_to_csv(args.output, results_list)
    else:
        print("\n⚠️ 処理されたファイルがありませんでした。")


if __name__ == "__main__":
    main()
