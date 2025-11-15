import sys
import math
import time
import subprocess
import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

# --- 既知の最適解 (BEST_KNOWN) ---
BEST_KNOWN = {
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


# --- (1) 既存の read_tsplib 関数 ---
def read_tsplib(tsp_path: Path) -> Dict:
    name = None
    dim = None
    edge_weight_type = None
    edge_weight_format = None
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

    # 座標読み (EUC_2D)
    if any(line.upper().strip() == "NODE_COORD_SECTION" for line in lines):
        reading = False
        for line in lines:
            s = line.strip()
            u = s.upper()
            if u == "NODE_COORD_SECTION":
                reading = True
                continue
            if u == "EOF" or u == "DEMAND_SECTION" or u == "CAPACITY":
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

# --- (2) 既存の calc_tour_cost 関数 ---
def calc_tour_cost(info: Dict, tour_zero_based: List[int]) -> float:
    dim = info["dim"]
    if len(tour_zero_based) == dim + 1 and tour_zero_based[0] == tour_zero_based[-1]:
        tour_zero_based = tour_zero_based[:-1]

    if len(tour_zero_based) != dim:
        seen = set(tour_zero_based)
        missing = [i for i in range(dim) if i not in seen]
        tour_zero_based = tour_zero_based + missing

    etype = info["edge_weight_type"]
    coords = info["coords"]
    dist_matrix = info["dist_matrix"]

    total = 0.0
    if etype == "EUC_2D":
        if coords is None:
            raise ValueError("coords が必要です (EUC_2D)")
        for i in range(dim):
            a = tour_zero_based[i]
            b = tour_zero_based[(i + 1) % dim]
            x1, y1 = coords[a]
            x2, y2 = coords[b]
            total += math.hypot(x1 - x2, y1 - y2) 
    elif etype == "EXPLICIT":
        if dist_matrix is None:
            raise ValueError("dist_matrix が必要です (EXPLICIT)")
        for i in range(dim):
            a = tour_zero_based[i]
            b = tour_zero_based[(i + 1) % dim]
            total += dist_matrix[a][b]
    else:
        raise NotImplementedError(f"EDGE_WEIGHT_TYPE={etype} は未対応")

    return total

# --- (3) 既存の solve_tsplib_with_concorde 関数 ---
def solve_tsplib_with_concorde(tsp_file: str, workdir: str = "concorde_tsplib_test") -> Dict[str, Any]:
    tsp_path = Path(tsp_file).resolve()
    workdir_path = Path(workdir).resolve()
    workdir_path.mkdir(parents=True, exist_ok=True)

    local_tsp = workdir_path / tsp_path.name
    
    if local_tsp != tsp_path:
        local_tsp.write_bytes(tsp_path.read_bytes())

    start = time.perf_counter()
    proc = subprocess.run(
        ["concorde", local_tsp.name],
        cwd=str(workdir_path),
        capture_output=True,
        text=True,
    )
    elapsed = time.perf_counter() - start

    if proc.returncode != 0:
        return {
            "ok": False,
            "elapsed": elapsed,
            "tour_file": None,
            "cost": None,
            "msg": f"Concorde failed: {proc.stderr}",
        }

    # tour ファイル(.sol or .tour) 探す
    stem = local_tsp.stem
    tour_path = None
    for ext in (".sol", ".tour"):
        cand = workdir_path / f"{stem}{ext}"
        if cand.exists():
            tour_path = cand
            break

    if tour_path is None:
        return {
            "ok": False,
            "elapsed": elapsed,
            "tour_file": None,
            "cost": None,
            "msg": "tour file not found",
        }

    # TOUR_SECTION 読み取り
    tour_idx = []
    reading = False
    for line in tour_path.read_text().splitlines():
        s = line.strip()
        if s == "TOUR_SECTION":
            reading = True
            continue
        if not reading:
            continue
        if s in ("-1", "EOF"):
            break
        try:
            tour_idx.append(int(s) - 1)
        except ValueError:
            pass

    info = read_tsplib(local_tsp) 
    cost = calc_tour_cost(info, tour_idx)

    return {
        "ok": True,
        "elapsed": elapsed,
        "tour_file": str(tour_path),
        "cost": cost,
        "msg": "",
    }


# --- (4) 拡張された main 関数 ---
def main():
    ap = argparse.ArgumentParser(
        description="指定されたディレクトリ内のTSPLIBファイルをConcordeで解き、結果をCSVに出力します。"
    )
    ap.add_argument("path", help="処理対象のファイル、またはディレクトリパス", type=str)
    ap.add_argument(
        "--max_dim", 
        help="処理する都市数(DIMENSION)の最大上限 (この値以下の問題のみ処理)", 
        type=int, 
        default=sys.maxsize
    )
    ap.add_argument(
        "--output",
        help="結果を保存するCSVファイル名",
        type=str,
        default="concorde_results.csv",
    )
    args = ap.parse_args()
    
    target_path = Path(args.path).resolve()
    max_dim = args.max_dim
    
    # 処理対象のファイルリストを決定
    tsp_files = []
    if target_path.is_dir():
        print(f"📂 ディレクトリ '{target_path.name}' 内の .tsp ファイルを検索中...")
        tsp_files.extend(target_path.rglob("*.tsp"))
    elif target_path.is_file():
        tsp_files.append(target_path)
    else:
        print(f"❌ パスが見つからないか、無効です: {args.path}")
        return

    results_list: List[Dict[str, Any]] = []
    
    print(f"🔍 処理対象のファイル数: {len(tsp_files)}")
    if max_dim != sys.maxsize:
        print(f"📏 都市数上限: {max_dim} を超えるファイルはスキップされます。")

    # ファイルを一つずつ処理
    for tsp_file_path in tsp_files:
        print(f"\n--- 処理中: {tsp_file_path.name} ---")
        instance_stem = tsp_file_path.stem
        
        try:
            # 都市数(DIMENSION)を読み込み、フィルタリング
            info = read_tsplib(tsp_file_path)
            dim = info['dim']
            
            if dim > max_dim:
                print(f"⏭️ スキップ (都市数 {dim} > 上限 {max_dim})")
                continue
                
            # BEST_KNOWNの取得
            best_known_cost = BEST_KNOWN.get(instance_stem)
            
            print(f"✅ 都市数: {dim} / BEST_KNOWN: {best_known_cost}")

            # Concordeで解く
            res = solve_tsplib_with_concorde(str(tsp_file_path))
            
            calculated_cost = res['cost']
            gap_pct = "N/A"
            
            # GAPの計算 (BEST_KNOWN が None ではなく、0 よりも大きい場合)
            if res['ok'] and calculated_cost is not None and best_known_cost is not None and best_known_cost > 0:
                # Concordeは正確な解を出すため、通常はコスト >= BEST_KNOWN となる
                gap = (calculated_cost - best_known_cost) / best_known_cost * 100.0
                gap_pct = f"{gap:.4f}"
            
            result = {
                "Instance": instance_stem,
                "DIMENSION": dim,
                "Best_Known": best_known_cost if best_known_cost is not None else "N/A",
                "Calculated_Cost": f"{calculated_cost:.4f}" if calculated_cost is not None else "N/A",
                "GAP_Pct": gap_pct,
                "Status": "SUCCESS" if res['ok'] else "FAILED",
                "Time_sec": f"{res['elapsed']:.3f}",
                "Message": res['msg'].strip() if res['msg'] else "",
            }
            results_list.append(result)
            
            print(f"    結果: {result['Status']}, コスト: {result['Calculated_Cost']}, GAP: {result['GAP_Pct']} %, 時間: {result['Time_sec']} sec")

        except Exception as e:
            # 処理失敗時のログ
            print(f"❌ 処理中にエラーが発生しました: {type(e).__name__}: {e}")
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

    # --- CSVへの書き出し ---
    if results_list:
        csv_path = Path(args.output).resolve()
        fieldnames = ["Instance", "DIMENSION", "Best_Known", "Calculated_Cost", "GAP_Pct", "Status", "Time_sec", "Message"]
        
        try:
            with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(results_list)
            
            print(f"\n🎉 処理完了。結果は '{csv_path.name}' に保存されました。")
        except Exception as e:
             print(f"\n❌ CSV書き出し中にエラーが発生しました: {e}")
    else:
        print("\n⚠️ 処理されたファイルはありませんでした。")

if __name__ == "__main__":
    main()