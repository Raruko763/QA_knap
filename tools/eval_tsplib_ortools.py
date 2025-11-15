# tools/eval_tsplib_ortools.py
import time, os, sys, csv, re
from pathlib import Path
from typing import List, Optional, Tuple
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.tsplib_parser import parse_tsplib
from src.tsp_ortools_solver import solve_tsp_ortools
# ... 既存 import の下に追記
from pathlib import Path

def read_bks_table(csv_path: Path) -> dict[str, int]:
    """tools/tsplib_bks.csv を {インスタンス名: 最適値} の dict に読む"""
    if not csv_path.exists():
        return {}
    m = {}
    for line in csv_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = line.strip()
        if not s or s.startswith("#"): 
            continue
        parts = [p.strip() for p in s.split(",")]
        if len(parts) >= 2 and parts[1].isdigit():
            m[parts[0]] = int(parts[1])
    return m
def read_bks_table(csv_path: Path) -> dict[str, int]:
    """tsplib_bks.csv を読み込んで {ファイル名: 最適値} の辞書を返す"""
    if not csv_path.exists():
        return {}
    m = {}
    for line in csv_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line or line.startswith("#") or "," not in line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2 and parts[1].isdigit():
            m[parts[0]] = int(parts[1])
    return m
def parse_tsplib_tour(tour_path: Path) -> list[int] | None:
    """
    .opt.tour / .tour を読んで TOUR_SECTION 最初のツアーを返す（1-based → 0-based）
    - 形式例:
        TYPE : TOUR
        DIMENSION : 52
        TOUR_SECTION
        1 34 23 ... 1
        -1
        EOF
    """
    lines = tour_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    tour = []
    in_section = False
    for line in lines:
        s = line.strip()
        if not s:
            continue
        u = s.upper()
        if u.startswith("TOUR_SECTION"):
            in_section = True
            continue
        if not in_section:
            continue
        if s == "-1" or u.startswith("EOF"):
            break
        for tok in s.split():
            v = int(tok)
            # TSPLIBは都市番号1始まり
            if v == -1:
                in_section = False
                break
            tour.append(v - 1)
    return tour if tour else None

def find_opt_tour_file(tsp_path: Path) -> Path | None:
    base = tsp_path.with_suffix("")  # remove .tsp
    for cand in [base.with_suffix(".opt.tour"), base.with_suffix(".tour")]:
        if cand.exists():
            return cand
    return None

def tour_length_from_tourfile(tsp_path: Path, dist_matrix) -> int | None:
    cand = find_opt_tour_file(tsp_path)
    if not cand:
        return None
    tour = parse_tsplib_tour(cand)
    if not tour:
        return None
    # 閉路にする（末尾→先頭）
    total = 0
    for i in range(len(tour) - 1):
        total += int(dist_matrix[tour[i]][tour[i+1]])
    total += int(dist_matrix[tour[-1]][tour[0]])
    return total

def read_bks_table(csv_path: Path) -> dict[str, int]:
    if not csv_path.exists():
        return {}
    m = {}
    for row in csv_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        row = row.strip()
        if not row or row.startswith("#"):
            continue
        parts = [p.strip() for p in row.split(",")]
        if len(parts) >= 2 and parts[1].isdigit():
            m[parts[0]] = int(parts[1])
    return m


def tour_length(route: List[int], dist):
    if not route: return None
    s = 0
    for i in range(len(route)-1):
        s += int(dist[route[i]][route[i+1]])
    return s

def read_tsplib_header(path: Path) -> Tuple[Optional[int], Optional[str], Optional[str]]:
    """距離行列を作らずに DIMENSION / EDGE_WEIGHT_TYPE / FORMAT だけ読む（軽量）"""
    n = None; ew_type = None; ew_fmt = None
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            up = line.upper()
            if up.startswith("NODE_COORD_SECTION") or up.startswith("EDGE_WEIGHT_SECTION") or up.startswith("EOF"):
                break
            if up.startswith("DIMENSION"):
                n = int(line.split(":",1)[1].strip())
            elif up.startswith("EDGE_WEIGHT_TYPE"):
                ew_type = line.split(":",1)[1].strip().upper()
            elif up.startswith("EDGE_WEIGHT_FORMAT"):
                ew_fmt = line.split(":",1)[1].strip().upper()
    return n, ew_type, ew_fmt

def extract_best_known(path: Path) -> Optional[int]:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
        m = re.search(r"(BEST_KNOWN|OPTIMAL_VALUE|OPTIMUM)\s*[:=]\s*(\d+)", text, re.I)
        return int(m.group(2)) if m else None
    except Exception:
        return None

def main(
    root="/home/toshiya1048/dev/amplify_var_limit_check/instance/tsp",
    time_limit_ms=3000,
    seed=42,
    out_csv="tsplib_ortools_results.csv",
    max_n=100,  # ← ここで閾値を調整
):
    root = Path(root)
    tsp_files = sorted(p for p in root.rglob("*.tsp"))
    bks_map = read_bks_table(Path("tools/tsplib_bks.csv"))
    rows = []
    for tsp in tsp_files:
        # 1) まず軽量にヘッダだけ読む（ここでスキップ判定）
        n, ew_type, ew_fmt = read_tsplib_header(tsp)
        if n is None:
            print(f"[PARSE-ERR] {tsp.name}: DIMENSION not found in header")
            rows.append([tsp.name, None, None, None, None, None, None, "PARSE_ERR: no DIMENSION"])
            continue

        if n > max_n:
            print(f"[SKIP] {tsp.name} (too large: n={n})")
            rows.append([tsp.name, n, ew_type, ew_fmt, None, None, extract_best_known(tsp), "SKIP: too large"])
            continue

        # 2) ここで初めて重い parse を呼ぶ（n が閾値以下）
        try:
            inst = parse_tsplib(str(tsp))
        except Exception as e:
            print(f"[PARSE-ERR] {tsp.name}: {e}")
            rows.append([tsp.name, n, ew_type, ew_fmt, None, None, None, f"PARSE_ERR: {e}"])
            continue

        print(f"[RUN] {tsp.name}  n={inst.n}  type={inst.edge_weight_type}  fmt={inst.edge_weight_format}")
        t0 = time.perf_counter()
        route = solve_tsp_ortools(inst.distance_matrix, time_limit_ms=time_limit_ms, seed=seed)
        t1 = time.perf_counter()

        length = tour_length(route, inst.distance_matrix) if route else None
        elapsed_ms = int((t1 - t0) * 1000)
        best_known = tour_length_from_tourfile(tsp, inst.distance_matrix)
        if best_known is None:
            best_known = extract_best_known(tsp)
        if best_known is None:
            best_known = bks_map.get(tsp.name)
        gap = (length - best_known) / best_known * 100.0 if (best_known is not None and length is not None) else None

        rows.append([tsp.name, inst.n, inst.edge_weight_type, inst.edge_weight_format,
                     elapsed_ms, length, best_known, gap])

    # 3) 最後に一括保存
    out = Path(out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["instance","n","edge_weight_type","edge_weight_format",
                    "solve_time_ms","route_length","best_known","gap_percent"])
        w.writerows(rows)

    print(f"[DONE] wrote {out.resolve()}  ({len(rows)} rows)")

if __name__ == "__main__":
    main()
