import os
import time
import subprocess
from pathlib import Path
from shutil import which

import numpy as np


def solve_tsp_lkh(
    dist_matrix,
    work_dir,
    time_limit_ms=3000,
    runs=20,
    seed=0,  # デフォルト値を0に変更（多様な探索のため）
    lkh_bin=None,
    max_trials=None,
    max_candidates=None,
):
    """
    LKH を使って TSP を解く（距離行列 → EXPLICIT TSPLIB → LKH 実行）

    dist_matrix   : NxN の距離行列（depot 含む）
    work_dir    : 出力フォルダ（Path or str）
    time_limit_ms: LKHに与える実行時間制限（ミリ秒）
    runs        : 実行回数
    seed        : 乱数シード (0でランダム)
    max_trials  : 解の改善が見られないときの最大試行回数 (LKH設定)
    max_candidates: 探索の候補数 (LKH設定)

    ※ 距離の整数化は Amplify 側と揃えて np.array(dist_matrix, dtype=int) で行う。
    """
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    # ---- LKH の実行ファイルを探す ----
    lkh_exec = lkh_bin or os.environ.get("LKH_BIN") or "LKH"
    if which(lkh_exec) is None and not Path(lkh_exec).exists():
        raise FileNotFoundError(
            f"LKH executable '{lkh_exec}' が見つかりません。\n"
            "PATH を通すか、環境変数 LKH_BIN を設定してください。"
        )

    # ---- 距離行列を整数化（Amplify と同じルール）----
    D_int = np.array(dist_matrix, dtype=int)
    if D_int.ndim != 2 or D_int.shape[0] != D_int.shape[1]:
        raise ValueError(
            f"dist_matrix は NxN の正方行列である必要がありますが、shape={D_int.shape} でした。"
        )
    n = int(D_int.shape[0])
    np.fill_diagonal(D_int, 0)

    # ---- ファイル名が被らないように一意化 ----
    uniq = f"{int(time.time() * 1000)}_{os.getpid()}"
    tsp_path = work_dir / f"cluster_{uniq}.tsp"
    par_path = work_dir / f"cluster_{uniq}.par"
    tour_path = work_dir / f"cluster_{uniq}.tour"

    # ---- TSPLIB 問題ファイル (.tsp) ----
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

    time_limit_sec = max(1, int(round(time_limit_ms / 1000.0)))
    
    # ---- パラメータファイル (.par) ----
    with open(par_path, "w") as f:
        f.write(f"PROBLEM_FILE = {tsp_path.name}\n")
        f.write(f"OUTPUT_TOUR_FILE = {tour_path.name}\n")
        # RUNS は引数から取得し、多様な探索を可能にする
        f.write(f"RUNS = {int(runs)}\n")
        
        # TIME_LIMIT を有効化し、時間制約を設ける
        f.write(f"TIME_LIMIT = {time_limit_sec}\n")
        
        # SEED を有効化（デフォルト 0: 実行ごとに異なるシード）
        f.write(f"SEED = {int(seed)}\n")
        
        # LKHの探索パラメータを追加
        if max_trials is not None:
            # 解の改善が見られない場合の最大試行回数を設定 (局所最適解からの脱出を試みる)
            f.write(f"MAX_TRIALS = {int(max_trials)}\n")
            
        if max_candidates is not None:
            # 探索の候補数を設定
            f.write(f"MAX_CANDIDATES = {int(max_candidates)}\n")
            # 候補リストの作成方法を距離だけでなく角度も考慮する方式に変更してみる
            f.write("CANDIDATE_SET_TYPE = ALPHA\n")
            
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
        # LKHのエラー内容を詳細に表示
        error_message = (
            "LKH execution failed.\n"
            f"Command: {lkh_exec} {par_path}\n"
            f"Working directory: {work_dir}\n"
            f"Return code: {proc.returncode}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}\n"
            f"Tour file exists: {tour_path.exists()}"
        )
        raise RuntimeError(error_message)

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
                # TSPLIB は 1 始まり → Python は 0 始まり
                route_idx.append(int(s) - 1)
            except Exception:
                pass

    # ---- 閉路・欠損ノード補正 ----
    n_found = len(route_idx)
    # 1) 先頭 = 末尾 の閉路形式なら末尾を落とす
    if n_found == n + 1 and route_idx[0] == route_idx[-1]:
        route_idx = route_idx[:-1]
        n_found -= 1

    # 2) ノード数がおかしい場合は、重複削除＋欠損ノード追加
    if n_found != n:
        route_idx = list(dict.fromkeys(route_idx))  # 順序保持で重複削除
        missing = [i for i in range(n) if i not in route_idx]
        route_idx += missing

    # ---- 合計距離計算（渡した距離行列で）----
    total = 0
    # ルートのノード数がnであることを保証（上記補正処理により）
    for i in range(n):
        a = route_idx[i]
        b = route_idx[(i + 1) % n]
        total += int(D_int[a, b])

    return {
        "route": route_idx,
        "total_distance": int(total),
        "solver": "lkh",
        "solver_status": "SUCCESS",
        "solve_time_ms": solve_ms,
        "raw_stdout": proc.stdout,
    }