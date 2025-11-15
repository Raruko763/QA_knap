# tsplib_concorde_test.py
import sys
import math
import time
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional


def read_tsplib(tsp_path: Path) -> Dict:
    name = None
    dim = None
    edge_weight_type = None
    edge_weight_format = None
    coords: List[Tuple[float, float]] = []
    dist_matrix: Optional[List[List[int]]] = None

    lines = tsp_path.read_text().splitlines()

    # ------- ヘッダ部 -------
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

    # ------- 座標読み (EUC_2D) -------
    if any(line.upper().strip() == "NODE_COORD_SECTION" for line in lines):
        reading = False
        for line in lines:
            s = line.strip()
            u = s.upper()
            if u == "NODE_COORD_SECTION":
                reading = True
                continue
            if u == "EOF":
                break
            if ":" in s and not reading:
                continue
            if reading:
                parts = s.split()
                if len(parts) >= 3:
                    x = float(parts[1])
                    y = float(parts[2])
                    coords.append((x, y))
        coords = coords[:dim]

    # ------- 距離行列読み (EXPLICIT/FULL_MATRIX) -------
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


def solve_tsplib_with_concorde(tsp_file: str, workdir: str = "."):
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


def main():
    if len(sys.argv) < 2:
        print("Usage: python tsplib_concorde_test.py path/to/instance.tsp [...]")
        sys.exit(0)

    for arg in sys.argv[1:]:
        res = solve_tsplib_with_concorde(arg, workdir="concorde_tsplib_test")
        print(f"Instance: {Path(arg).stem}")
        print(f"  ok      : {res['ok']}")
        print(f"  elapsed : {res['elapsed']:.3f} sec")
        print(f"  tour    : {res['tour_file']}")
        print(f"  cost    : {res['cost']}")
        print(f"  msg     : {res['msg']}")
        print()


if __name__ == "__main__":
    main()
