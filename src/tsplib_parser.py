# src/tsplib_parser.py
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import math

class TSPLIBInstance:
    def __init__(self, name: str, n: int, ew_type: str, ew_format: Optional[str], dist: List[List[int]]):
        self.name = name
        self.n = n
        self.edge_weight_type = (ew_type or "").upper()
        self.edge_weight_format = (ew_format or "").upper() if ew_format else None
        self.distance_matrix = dist  # int の正方行列 [n][n]

def _read_k_numbers(lines_iter, k: int) -> List[int]:
    """EDGE_WEIGHT_SECTION 等から合計k個の整数を順に読む"""
    vals: List[int] = []
    while len(vals) < k:
        line = next(lines_iter).strip()
        if not line or line.upper().startswith(("EOF", "DISPLAY_DATA_SECTION", "TOUR_SECTION")):
            break
        vals.extend(int(x) for x in line.split())
    return vals

def _distance_att(x1, y1, x2, y2) -> int:
    # TSPLIB pseudo-Euclidean（ATT）
    rij = math.sqrt(((x1 - x2)**2 + (y1 - y2)**2) / 10.0)
    tij = int(rij + 0.5)
    return tij if tij >= rij else tij + 1

def _build_matrix_from_coords(coords: List[Tuple[float, float]], ew_type: str) -> List[List[int]]:
    n = len(coords)
    M = [[0]*n for _ in range(n)]
    for i in range(n):
        x1, y1 = coords[i]
        for j in range(i+1, n):
            x2, y2 = coords[j]
            if ew_type == "EUC_2D":
                dij = int(round(math.hypot(x1 - x2, y1 - y2)))
            elif ew_type == "CEIL_2D":
                dij = int(math.ceil(math.hypot(x1 - x2, y1 - y2)))
            elif ew_type == "ATT":
                dij = _distance_att(x1, y1, x2, y2)
            else:
                raise NotImplementedError(f"EDGE_WEIGHT_TYPE {ew_type} is not supported in this parser")
            M[i][j] = M[j][i] = dij
    return M

def parse_tsplib(path: str) -> TSPLIBInstance:
    """
    TSPLIB .tsp を読み、整数距離の正方行列を返す。
    対応:
      - NODE_COORD_SECTION + (EUC_2D / CEIL_2D / ATT)
      - EDGE_WEIGHT_TYPE: EXPLICIT + (FULL_MATRIX / LOWER_DIAG_ROW / UPPER_ROW / LOWER_DIAG_COL / UPPER_COL)
    """
    p = Path(path)
    text = p.read_text(encoding="utf-8", errors="ignore").splitlines()
    it = iter(text)

    name = p.stem
    n = None
    ew_type = None
    ew_format = None

    coords: List[Tuple[float, float]] = []
    matrix: Optional[List[List[int]]] = None

    # 1) ヘッダ読取
    for line in it:
        s = line.strip()
        if not s:
            continue
        up = s.upper()
        if up.startswith("NAME"):
            name = s.split(":", 1)[1].strip() if ":" in s else name
        elif up.startswith("TYPE"):
            pass
        elif up.startswith("DIMENSION"):
            n = int(s.split(":", 1)[1].strip())
        elif up.startswith("EDGE_WEIGHT_TYPE"):
            ew_type = s.split(":", 1)[1].strip().upper()
        elif up.startswith("EDGE_WEIGHT_FORMAT"):
            ew_format = s.split(":", 1)[1].strip().upper()
        elif up.startswith("NODE_COORD_SECTION"):
            if n is None:
                # 次の行で数えてもよいが素直に必須とする
                raise ValueError("DIMENSION not found before NODE_COORD_SECTION")
            # 2) 座標を読む
            for _ in range(n):
                parts = next(it).strip().split()
                # 1-based index, x, y
                if len(parts) < 3:
                    raise ValueError("Invalid NODE_COORD_SECTION line")
                x = float(parts[-2]); y = float(parts[-1])
                coords.append((x, y))
            # 3) 距離行列生成
            if ew_type not in ("EUC_2D", "CEIL_2D", "ATT"):
                raise NotImplementedError(f"NODE_COORD_SECTION with {ew_type} is not supported by this parser")
            matrix = _build_matrix_from_coords(coords, ew_type)
        elif up.startswith("EDGE_WEIGHT_SECTION"):
            if n is None:
                raise ValueError("DIMENSION not found before EDGE_WEIGHT_SECTION")
            # EXPLICIT 前提
            fmt = (ew_format or "").upper()
            vals = []
            # 読み方分岐
            if fmt in ("FULL_MATRIX", ""):
                vals = _read_k_numbers(it, n*n)
                if len(vals) < n*n:
                    raise ValueError("Not enough numbers in FULL_MATRIX")
                matrix = [vals[i*n:(i+1)*n] for i in range(n)]
            elif fmt in ("LOWER_DIAG_ROW", "LOWER_ROW"):
                # 下三角(対角含む)をrow-wiseで与える
                need = n*(n+1)//2 if "DIAG" in fmt else n*(n-1)//2
                vals = _read_k_numbers(it, need)
                if len(vals) < need:
                    raise ValueError("Not enough numbers in LOWER_*")
                matrix = [[0]*n for _ in range(n)]
                idx = 0
                for i in range(n):
                    jmax = i if "DIAG" in fmt else i-1
                    for j in range(jmax+1):
                        vij = vals[idx]; idx += 1
                        if i == j:
                            matrix[i][j] = 0 if "DIAG" in fmt else vij
                        else:
                            matrix[i][j] = matrix[j][i] = vij
            elif fmt in ("UPPER_DIAG_ROW", "UPPER_ROW"):
                need = n*(n+1)//2 if "DIAG" in fmt else n*(n-1)//2
                vals = _read_k_numbers(it, need)
                if len(vals) < need:
                    raise ValueError("Not enough numbers in UPPER_*")
                matrix = [[0]*n for _ in range(n)]
                idx = 0
                for i in range(n):
                    jmin = i if "DIAG" in fmt else i+1
                    for j in range(jmin, n):
                        vij = vals[idx]; idx += 1
                        if i == j:
                            matrix[i][j] = 0 if "DIAG" in fmt else vij
                        else:
                            matrix[i][j] = matrix[j][i] = vij
            else:
                raise NotImplementedError(f"EDGE_WEIGHT_FORMAT {fmt} not supported in this parser")
        elif up.startswith("EOF"):
            break

    if n is None or matrix is None:
        raise ValueError("Failed to parse TSPLIB file (DIMENSION/Matrix not found).")

    return TSPLIBInstance(name=name, n=n, ew_type=ew_type or "EXPLICIT", ew_format=ew_format, dist=matrix)
