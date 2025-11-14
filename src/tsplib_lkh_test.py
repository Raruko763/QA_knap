import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Optional, Dict, Any


def parse_best_cost_from_log(log_path: str) -> Optional[int]:
    """
    LKH のログファイルから「最良コスト」をパースするざっくり版。
    - Cost.min = 7542
    - Best = 7542
    - Tour cost = 7542
    みたいな行を拾う。
    """
    log = Path(log_path)
    if not log.exists():
        return None

    best: Optional[int] = None
    pat = re.compile(r"\b(Cost\.min|Best|Tour\s+cost)\s*=\s*([-+]?\d+)")

    for line in log.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = pat.search(line)
        if m:
            cost = int(m.group(2))
            best = cost  # 最後に出てきた値を採用

    return best


def get_tsp_dimension(tsp_path: Path) -> Optional[int]:
    """
    TSPLIBファイルから DIMENSION (ノード数) をパースして取得する。
    """
    if not tsp_path.exists():
        return None
    
    # DIMENSION : XXXX のパターンを検索
    pat = re.compile(r"DIMENSION\s*:\s*(\d+)", re.IGNORECASE)
    
    try:
        # ファイル全体を読み込み
        content = tsp_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        # 読み込みエラーが発生した場合はNoneを返す
        return None

    for line in content.splitlines():
        m = pat.search(line)
        if m:
            return int(m.group(1))
            
    return None


def solve_tsplib_with_lkh(
    tsp_path: str,
    *,
    lkh_bin: Optional[str] = None,
    workdir: str = "lkh_tsplib_test",
    runs: int = 20,
    seed: Optional[int] = None,
    extra_par: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    TSPLIB の .tsp をそのまま LKH に渡して解くテスト用ユーティリティ。

    - tsp_path : TSPLIB の .tsp ファイルへのパス
    - lkh_bin  : LKH 実行ファイルのパス。
                  None の場合は環境変数 LKH_BIN → "LKH" の順で探す。
    - workdir  : .par / .tour / .log を置く作業ディレクトリ
    - runs     : LKH の RUNS パラメータ
    - seed     : LKH の SEED パラメータ（None なら指定しない）
    - extra_par: 追加で書きたい LKH パラメータの dict
                  例: {"MAX_TRIALS": 10000, "MOVE_TYPE": 5}

    戻り値:
        {
            "ok": bool,
            "tour_file": str or None,
            "log_file": str or None,
            "msg": str,
            "elapsed": float or None,   # 実行時間 [秒]
            "cost": int or None,        # ログから読んだベストコスト
        }
    """
    tsp = Path(tsp_path).resolve()
    if not tsp.exists():
        return {
            "ok": False,
            "tour_file": None,
            "log_file": None,
            "msg": f"not found: {tsp_path}",
            "elapsed": None,
            "cost": None,
        }

    # --- ノード数チェックの追加 ---
    dimension = get_tsp_dimension(tsp)
    MAX_NODES =200 
    
    if dimension is None:
        # DIMENSIONが読めない場合はエラーとはせず、警告メッセージを出す
        print(f"[WARN] DIMENSION not found in {tsp_path}. Skipping size check.")
    elif dimension > MAX_NODES:
        return {
            "ok": False,
            "tour_file": None,
            "log_file": None,
            "msg": f"ノード数 ({dimension}) が最大許容値 ({MAX_NODES}) を超えています。実行をスキップしました。",
            "elapsed": None,
            "cost": None,
        }
    # ----------------------------

    # LKH 実行ファイルの解決
    real_lkh = lkh_bin or os.environ.get("LKH_BIN") or "LKH"
    if shutil.which(real_lkh) is None and not Path(real_lkh).exists():
        return {
            "ok": False,
            "tour_file": None,
            "log_file": None,
            "msg": f"LKH not found: {real_lkh}",
            "elapsed": None,
            "cost": None,
        }

    wdir = Path(workdir).resolve()
    wdir.mkdir(parents=True, exist_ok=True)

    name = tsp.stem
    par_path  = wdir / f"{name}.par"
    tour_path = wdir / f"{name}.tour"
    log_path  = wdir / f"{name}.log"

    # --- LKH パラメータファイル生成 ---
    lines = [
        f"PROBLEM_FILE = {tsp}",
        f"OUTPUT_TOUR_FILE = {tour_path}",
        f"RUNS = {int(runs)}",
        "TRACE_LEVEL = 1",
    ]
    if seed is not None:
        lines.append(f"SEED = {int(seed)}")
    if extra_par:
        for k, v in extra_par.items():
            if v is not None:
                lines.append(f"{k} = {v}")
    par_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # --- LKH 実行 ---
    t0 = time.perf_counter()
    with open(log_path, "w", encoding="utf-8") as lf:
        proc = subprocess.run(
            [real_lkh, str(par_path)],
            stdout=lf,
            stderr=subprocess.STDOUT,
            cwd=str(wdir),
        )
    t1 = time.perf_counter()
    elapsed = t1 - t0

    if proc.returncode != 0 or not tour_path.exists():
        return {
            "ok": False,
            "tour_file": None if not tour_path.exists() else str(tour_path),
            "log_file": str(log_path),
            "msg": f"LKH failed (code={proc.returncode}). See {log_path}",
            "elapsed": elapsed,
            "cost": None,
        }

    # ログからベストコストを取得
    cost = parse_best_cost_from_log(str(log_path))

    return {
        "ok": True,
        "tour_file": str(tour_path),
        "log_file": str(log_path),
        "msg": "OK",
        "elapsed": elapsed,
        "cost": cost,
    }


if __name__ == "__main__":
    # 実行環境にLKHがないため、常に失敗する（またはLKHが見つからない）ことを想定したデモコードです。
    print("--- LKH TSPLIB テストユーティリティのデモンストレーション ---")
    
    # 1. 存在しないファイルのテスト
    res_not_found = solve_tsplib_with_lkh("non_existent_file.tsp")
    print("\n[テスト1] ファイル非存在:")
    print(res_not_found)

    # 2. ダミーのTSPLIBファイルを作成して DIMENSION チェックをテスト
    dummy_tsp_path = Path("lkh_tsplib_test") / "dummy_test_problem.tsp"
    dummy_tsp_path.parent.mkdir(exist_ok=True)

    # 201ノード（制限オーバー）のダミーファイルを作成
    content_too_large = """
NAME : TOO_LARGE
TYPE : TSP
COMMENT : Problem with 201 cities
DIMENSION : 201
EDGE_WEIGHT_TYPE : EUC_2D
NODE_COORD_SECTION
1 1 1
...
EOF
"""
    dummy_tsp_path.write_text(content_too_large.strip())

    print("\n[テスト2] ノード数オーバーのチェック (DIMENSION: 201):")
    res_large = solve_tsplib_with_lkh(str(dummy_tsp_path))
    print(res_large)
    
    # 3. 正常なファイルパス（ただしLKHは実行できない）のテスト
    # 正常なノード数（例: 50）のダミーファイルを作成
    content_ok = content_too_large.replace("DIMENSION : 201", "DIMENSION : 50")
    dummy_tsp_path.write_text(content_ok.strip())
    
    print("\n[テスト3] 正常ノード数での実行 (DIMENSION: 50):")
    res_ok = solve_tsplib_with_lkh(str(dummy_tsp_path))
    print(res_ok) # LKHが見つからない旨のメッセージが出力されるはず

    # クリーンアップ
    shutil.rmtree(dummy_tsp_path.parent, ignore_errors=True)
    
    print("\nデモンストレーション終了。")