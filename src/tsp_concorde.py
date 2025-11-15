# src/tsp_concorde.py
import os
import re
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

    ※ ルート補完は一切しない。
      - route 長さ != n
      - 0..n-1 の置換になっていない
      いずれかなら FAIL 扱いにして route=None を返す。
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
    stem = tsp_path.stem  # ★ ここで stem を定義するのがポイント

    # ---- TSPLIB FULL_MATRIX 問題ファイル作成 ----
    with tsp_path.open("w") as f:
        f.write(f"NAME: {stem}\n")
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

    # ★ TOUR_SECTION 形式の .tour を強制生成 (-x で .mas/.sav も消す)
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

    # ---- stdout から最適値を拾う（距離チェック用）----
    optimal_value_stdout = None
    m = re.search(r"Optimal Solution:\s*([0-9.+\-Ee]+)", proc.stdout)
    if m:
        try:
            optimal_value_stdout = float(m.group(1))
        except ValueError:
            optimal_value_stdout = None

    # ---- tour ファイル探索 (.tour を優先) ----
    tour_path = None
    for ext in (".tour", ".sol"):
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
            "solver_status": "FAIL_RUN_OR_NO_TOUR",
            "solve_time_ms": solve_ms,
            "raw_stdout": proc.stdout,
            "raw_stderr": proc.stderr,
            "tsp_file": str(tsp_path),
            "tour_file": str(tour_path) if tour_path else None,
            "optimal_value_stdout": optimal_value_stdout,
            "cost": None,
        }

    # ==============================
    #   TOUR の読み取り（補完なし）
    # ==============================
    route_idx: list[int] = []

    # ---- ① TSPLIB TOUR_SECTION 形式を優先 ----
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
            # 一行に複数数字がある場合にも対応
            for tok in s.split():
                try:
                    node = int(tok)
                except ValueError:
                    continue
                # TSPLIB は 1-based
                route_idx.append(node - 1)

    # ---- ② フォールバック: TOUR_SECTION が無い場合、正の整数全部を読む ----
    if not route_idx:
        tokens: list[int] = []
        with tour_path.open("r") as f:
            for line in f:
                for tok in line.strip().split():
                    try:
                        node = int(tok)
                    except ValueError:
                        continue
                    tokens.append(node)

        if not tokens:
            return {
                "route": None,
                "total_distance": None,
                "solver": "concorde",
                "solver_status": "FAIL_PARSE_TOUR_EMPTY",
                "solve_time_ms": solve_ms,
                "raw_stdout": proc.stdout,
                "raw_stderr": proc.stderr,
                "tsp_file": str(tsp_path),
                "tour_file": str(tour_path),
                "optimal_value_stdout": optimal_value_stdout,
                "cost": None,
            }

        # パターンB想定:
        #   [n, v0, v1, ..., v{n-1}]  (0-based)
        #   もしくは [v0, ..., v{n-1}] (0-based / 1-based)
        if tokens[0] == n and len(tokens) == n + 1:
            cand = tokens[1:]
        elif len(tokens) == n:
            cand = tokens[:]
        else:
            return {
                "route": None,
                "total_distance": None,
                "solver": "concorde",
                "solver_status": "FAIL_PARSE_TOUR_LEN_MISMATCH",
                "solve_time_ms": solve_ms,
                "raw_stdout": proc.stdout,
                "raw_stderr": proc.stderr,
                "tsp_file": str(tsp_path),
                "tour_file": str(tour_path),
                "optimal_value_stdout": optimal_value_stdout,
                "cost": None,
            }

        # 0-based / 1-based 判定
        mn, mx = min(cand), max(cand)
        if mn == 0 and mx == n - 1:
            route_idx = cand
        elif mn == 1 and mx == n:
            route_idx = [x - 1 for x in cand]
        else:
            return {
                "route": None,
                "total_distance": None,
                "solver": "concorde",
                "solver_status": "FAIL_PARSE_TOUR_RANGE",
                "solve_time_ms": solve_ms,
                "raw_stdout": proc.stdout,
                "raw_stderr": proc.stderr,
                "tsp_file": str(tsp_path),
                "tour_file": str(tour_path),
                "optimal_value_stdout": optimal_value_stdout,
                "cost": None,
            }

    # ---- ③ 厳密チェック（補完なし）----
    if len(route_idx) != n:
        return {
            "route": None,
            "total_distance": None,
            "solver": "concorde",
            "solver_status": "FAIL_TOUR_LEN",
            "solve_time_ms": solve_ms,
            "raw_stdout": proc.stdout,
            "raw_stderr": proc.stderr,
            "tsp_file": str(tsp_path),
            "tour_file": str(tour_path),
            "optimal_value_stdout": optimal_value_stdout,
            "cost": None,
        }

    if set(route_idx) != set(range(n)):
        return {
            "route": None,
            "total_distance": None,
            "solver": "concorde",
            "solver_status": "FAIL_TOUR_PERM",
            "solve_time_ms": solve_ms,
            "raw_stdout": proc.stdout,
            "raw_stderr": proc.stderr,
            "tsp_file": str(tsp_path),
            "tour_file": str(tour_path),
            "optimal_value_stdout": optimal_value_stdout,
            "cost": None,
        }

    # ---- 総距離計算 ----
        # ---- 総距離計算（ルートからのコスト）----
    total = 0
    for i in range(n):
        a = route_idx[i]
        b = route_idx[(i + 1) % n]
        total += int(D_int[a, b])

    cost_from_route = int(total)

    # ---- Concorde 表示値との比較用 ----
    cost_from_stdout = None
    if optimal_value_stdout is not None:
        # TSPLIB 用の整数距離になっているはずなので丸めておく
        cost_from_stdout = int(round(optimal_value_stdout))

    cost_diff = None
    if cost_from_stdout is not None:
        cost_diff = cost_from_route - cost_from_stdout

    # メインで返す cost は「自前の距離行列に対するコスト」でそろえる
    # （クラスタ内距離など、こっち側の定義が真実だから）
    return {
        "route": route_idx,
        "total_distance": cost_from_route,          # = route から計算した値
        "solver": "concorde",
        "solver_status": "SUCCESS",
        "solve_time_ms": solve_ms,
        "raw_stdout": proc.stdout,
        "raw_stderr": proc.stderr,
        "tsp_file": str(tsp_path),
        "tour_file": str(tour_path),

        # 両方残しておく
        "optimal_value_stdout": optimal_value_stdout,  # 生の float
        "cost_from_stdout": cost_from_stdout,          # int に丸めたもの
        "cost_from_route": cost_from_route,
        "cost_diff": cost_diff,                        # route - stdout

        # 互換用エイリアス（今まで通り cost を見るコード向け）
        "cost": cost_from_route,
    }
