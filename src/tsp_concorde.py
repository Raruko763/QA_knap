# src/tsp_concorde.py
import os
import time
import subprocess
from pathlib import Path
from shutil import which

import numpy as np


def _resolve_concorde_bin(concorde_bin: str | None = None) -> str:
    """
    Concorde 実行ファイルのパスを解決するヘルパ。

    優先順位:
      1. 引数 concorde_bin
      2. 環境変数 CONCORDE_BIN
      3. PATH 上の "concorde"
    """
    if concorde_bin:
        p = Path(concorde_bin)
        if not p.is_file():
            raise FileNotFoundError(f"指定された concorde_bin が存在しません: {concorde_bin}")
        return str(p)

    env_bin = os.environ.get("CONCORDE_BIN")
    if env_bin:
        p = Path(env_bin)
        if not p.is_file():
            raise FileNotFoundError(f"環境変数 CONCORDE_BIN のパスが存在しません: {env_bin}")
        return str(p)

    found = which("concorde")
    if not found:
        raise RuntimeError(
            "Concorde 実行ファイルが見つかりません。\n"
            "- PATH に 'concorde' を通すか\n"
            "- 環境変数 CONCORDE_BIN を設定するか\n"
            "- solve_tsp_concorde(concorde_bin=...) で明示してください。"
        )
    return found


def solve_tsp_concorde(
    dist_matrix,
    work_dir,
    seed: int | None = None,
    concorde_bin: str | None = None,
):
    """
    距離行列を TSPLIB (FULL_MATRIX) に変換して Concorde を 1 回実行し、
    経路と総距離などを返す。

    Parameters
    ----------
    dist_matrix : array-like (N, N)
        対称 or 非対称でもよいが、ここでは行列の値をそのまま距離として使う。
    work_dir : str | Path
        一時ファイル(.tsp, .sol/.tour)を置く作業ディレクトリ。
    seed : int | None
        Concorde の -s オプションに渡す乱数シード。None の場合は指定しない。
    concorde_bin : str | None
        Concorde 実行ファイルのパス。None の場合は PATH / CONCORDE_BIN から解決。

    Returns
    -------
    result : dict
        {
            "route": List[int],        # 0-based 巡回順
            "total_distance": int,     # D_int に基づく総距離
            "solver": "concorde",
            "solver_status": "SUCCESS" or "FAIL",
            "solve_time_ms": int,      # 実測実行時間(ms)
            "raw_stdout": str,
            "raw_stderr": str,
            "tsp_file": str,
            "tour_file": str | None,
        }
    """
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    # ---- 距離行列を整数化 ----
    D_int = np.array(dist_matrix, dtype=int)
    if D_int.ndim != 2 or D_int.shape[0] != D_int.shape[1]:
        raise ValueError(
            f"dist_matrix は NxN の正方行列である必要がありますが、shape={D_int.shape} でした。"
        )
    n = int(D_int.shape[0])
    np.fill_diagonal(D_int, 0)

    # ---- 一意なファイル名 ----
    uniq = f"{int(time.time() * 1000)}_{os.getpid()}"
    tsp_path = work_dir / f"cluster_{uniq}.tsp"

    # ---- TSPLIB FULL_MATRIX 問題ファイル作成 ----
    with tsp_path.open("w") as f:
        f.write(f"NAME: cluster_{uniq}\n")
        f.write("TYPE: TSP\n")
        f.write(f"DIMENSION: {n}\n")
        f.write("EDGE_WEIGHT_TYPE: EXPLICIT\n")
        f.write("EDGE_WEIGHT_FORMAT: FULL_MATRIX\n")
        f.write("EDGE_WEIGHT_SECTION\n")
        for i in range(n):
            row = " ".join(str(int(v)) for v in D_int[i])
            f.write(row + "\n")
        f.write("EOF\n")

    concorde_exec = _resolve_concorde_bin(concorde_bin)

    cmd = [concorde_exec, tsp_path.name]
    if seed is not None:
        cmd += ["-s", str(int(seed))]

    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=str(work_dir),
        capture_output=True,
        text=True,
        check=False,
    )
    t1 = time.perf_counter()
    solve_ms = int((t1 - t0) * 1000)

    # ---- tour ファイル探索 (.sol / .tour のどちらか) ----
    stem = tsp_path.stem
    tour_path = None
    for ext in (".sol", ".tour"):
        cand = work_dir / f"{stem}{ext}"
        if cand.exists():
            tour_path = cand
            break

    if proc.returncode != 0 or tour_path is None:
        # 失敗時も情報は返す
        return {
            "route": None,
            "total_distance": None,
            "solver": "concorde",
            "solver_status": "FAIL",
            "solve_time_ms": solve_ms,
            "raw_stdout": proc.stdout,
            "raw_stderr": proc.stderr,
            "tsp_file": str(tsp_path),
            "tour_file": str(tour_path) if tour_path else None,
        }

    # ---- TOUR_SECTION 読み取り ----
    route_idx: list[int] = []
    with tour_path.open("r") as f:
        reading = False
        for line in f:
            s = line.strip()
            if s == "TOUR_SECTION":
                reading = True
                continue
            if not reading:
                continue
            if s in ("-1", "EOF"):
                break
            try:
                node = int(s)
            except ValueError:
                continue
            # TSPLIB は 1-based なので 0-based に変換
            route_idx.append(node - 1)

    # ---- 長さ補正（LKH ラッパと同じノリで安全側に補完）----
    if len(route_idx) > n:
        route_idx = route_idx[:n]

    if len(route_idx) < n:
        seen = set(route_idx)
        missing = [i for i in range(n) if i not in seen]
        route_idx.extend(missing)

    # 念のため 0..n-1 の置換にしておく
    if len(route_idx) != n:
        raise RuntimeError(
            f"Concorde の tour 長がおかしいです: len(route_idx)={len(route_idx)}, n={n}"
        )

    # ---- 総距離計算 ----
    total = 0
    for i in range(n):
        a = route_idx[i]
        b = route_idx[(i + 1) % n]
        total += int(D_int[a, b])

    return {
        "route": route_idx,
        "total_distance": int(total),
        "solver": "concorde",
        "solver_status": "SUCCESS",
        "solve_time_ms": solve_ms,
        "raw_stdout": proc.stdout,
        "raw_stderr": proc.stderr,
        "tsp_file": str(tsp_path),
        "tour_file": str(tour_path),
    }
