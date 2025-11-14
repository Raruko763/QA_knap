import os
import time
import subprocess
from pathlib import Path
from shutil import which

import numpy as np


def solve_tsp_lkh(dist_matrix, work_dir, time_limit_ms=2000,
                  runs=1, seed=12345, lkh_bin=None):
    """
    LKH を使って TSP を解く（距離行列 → EXPLICIT TSPLIB → LKH実行）

    dist_matrix: NxN の距離行列（depot含む）
    work_dir   : 出力フォルダ（Path or str）
    """
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    # ---- LKH の実行ファイルを探す ----
    lkh_exec = lkh_bin or os.environ.get("LKH_BIN") or "LKH"
    if which(lkh_exec) is None:
        raise FileNotFoundError(
            f"LKH executable '{lkh_exec}' が見つかりません。\n"
            "PATH を通すか、環境変数 LKH_BIN を設定してください。"
        )

    # ---- 距離行列を整数化（LKHは整数前提） ----
    D = np.asarray(dist_matrix, dtype=float)
    n = int(D.shape[0])
    # ---- ノード数チェックの追加 ----
    # MAX_NODES = 200
    # if n > MAX_NODES:
    #     raise ValueError(
    #         f"ノード数 ({n}) が許容される最大値 ({MAX_NODES}) を超えています。\n"
    #         "200都市を超えるTSP問題の解決は許可されていません。"
    #     )
    # --------------------------------
    D_int = np.rint(D).astype(int)
    np.fill_diagonal(D_int, 0)

    # ---- ファイル名が被らないように一意化 ----
    uniq = f"{int(time.time() * 1000)}_{os.getpid()}"
    tsp_path = work_dir / f"cluster_{uniq}.tsp"
    par_path = work_dir / f"cluster_{uniq}.par"
    tour_path = work_dir / f"cluster_{uniq}.tour"

    # ---- TSPLIB 問題ファイル ----
    with open(tsp_path, "w") as f:
        f.write(f"NAME: cluster_{uniq}\n")
        f.write("TYPE: TSP\n")
        f.write(f"DIMENSION: {n}\n")
        f.write("EDGE_WEIGHT_TYPE: EXPLICIT\n")
        f.write("EDGE_WEIGHT_FORMAT: FULL_MATRIX\n")
        f.write("EDGE_WEIGHT_SECTION\n")
        for i in range(n):
            f.write(" ".join(str(int(v)) for v in D_int[i]) + "\n")
        f.write("EOF\n")

    # ---- パラメータファイル ----
    time_limit_sec = max(1, int(round(time_limit_ms / 1000.0)))
    with open(par_path, "w") as f:
        f.write(f"PROBLEM_FILE = {tsp_path.name}\n")
        f.write(f"OUTPUT_TOUR_FILE = {tour_path.name}\n")
        f.write(f"RUNS = {int(runs)}\n")
        f.write(f"TIME_LIMIT = {time_limit_sec}\n")    # 秒
        f.write(f"SEED = {int(seed)}\n")
        f.write("TRACE_LEVEL = 0\n")

    # ---- LKH 実行 ----
    t0 = time.perf_counter()
    proc = subprocess.run(
        [lkh_exec, str(par_path)],
        cwd=str(work_dir),
        capture_output=True,
        text=True,
        check=False,
    )
    t1 = time.perf_counter()
    solve_ms = int((t1 - t0) * 1000)

    if proc.returncode != 0 or not tour_path.exists():
        raise RuntimeError(
            "LKH execution failed.\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )

    # ---- TOUR_SECTION 読み取り ----
    route_idx = []
    with open(tour_path, "r") as f:
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
                route_idx.append(int(s) - 1)  # 1-index → 0-index
            except:
                pass

    # ---- 閉路・欠損ノード補正 ----
    if len(route_idx) == n + 1 and route_idx[0] == route_idx[-1]:
        route_idx = route_idx[:-1]
    if len(route_idx) != n:
        route_idx = list(dict.fromkeys(route_idx))
        missing = [i for i in range(n) if i not in route_idx]
        route_idx += missing

    # ---- 合計距離計算（渡した距離行列で） ----
    total = 0
    for i in range(n):
        a = route_idx[i]
        b = route_idx[(i + 1) % n]
        total += int(D_int[a][b])

    return {
        "route": route_idx,
        "total_distance": int(total),
        "solver": "lkh",
        "solver_status": "SUCCESS",
        "solve_time_ms": solve_ms,
        "raw_stdout": proc.stdout,
    }
