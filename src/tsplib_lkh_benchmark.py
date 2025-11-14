# tsplib_lkh_benchmark.py
import csv
from pathlib import Path

from tsplib_lkh_test import solve_tsplib_with_lkh
from parse_log import parse_best_cost_from_log

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
    "gr202": 40160,   # 202 → 200制限ギリ外なので外してもOK
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
    "lin318": None,   # 318 → 除外推奨
    "pa561": None,    # 561 → 除外
    "pr76": 108159,
    "pr107": 44303,
    "pr124": 59030,
    "pr136": 96772,
    "pr144": 58537,
    "pr152": 73682,
    "rat99": 1211,
    "rat195": 2323,
    "rat783": None,   # 除外
    "rd100": 7910,
    "rd400": None,    # 除外
    "st70": 675,
    "ts225": None,    # 225 → 200制限外
    "tsp225": None,   # 同上
    "ulysses16": 6859,
    "ulysses22": 7013,

    # あなたが使ってるであろう TSPLIB 基本セットから
    "a280": 2579,
    "ali535": None,  # 535 → 除外
}


def main():
    tsplib_dir = Path("/home/toshiya1048/dev/amplify_var_limit_check/instance/tsp")
    out_csv = Path("tsplib_lkh_results.csv")

    tsp_files = sorted(tsplib_dir.glob("*.tsp"))
    if not tsp_files:
        print("No .tsp files found in", tsplib_dir)
        return

    # ▼ ① 最初にヘッダだけ書く（上書き）
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "instance", "runs", "elapsed_sec", "cost",
            "best_known", "gap_percent", "log_file", "tour_file", "ok", "msg"
        ])

    # ▼ ② 各インスタンスごとに append 書き込み
    for tsp in tsp_files:
        name = tsp.stem
        print(f"\n=== Instance: {name} ===")

        try:
            res = solve_tsplib_with_lkh(
                str(tsp),
                lkh_bin=None,
                workdir="lkh_tsplib_test",
                runs=20,
                seed=1234,
            )

            cost = parse_best_cost_from_log(res["log_file"]) if res["ok"] else None
            best = BEST_KNOWN.get(name)
            gap = None
            if cost is not None and best is not None and best > 0:
                gap = (cost - best) * 100.0 / best

            # append で追記
            with out_csv.open("a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    name,
                    20,
                    res.get("elapsed", None),
                    cost,
                    best,
                    gap,
                    res.get("log_file"),
                    res.get("tour_file"),
                    res["ok"],
                    res.get("msg", ""),
                ])

            print(f"→ saved result for {name}")

        except Exception as e:
            print(f"❌ Error in {name}: {e}")
            # error も CSV に残す
            with out_csv.open("a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    name, 20, None, None, BEST_KNOWN.get(name),
                    None, None, None, False, f"error: {e}"
                ])


    print("\n🎉 すべての実行が終了しました")
    print("📄 結果: ", out_csv)


if __name__ == "__main__":
    main()
