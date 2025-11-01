import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from amplify import FixstarsClient
from src.vrpfactory import vrpfactory
from src.knap_divpro import knap_dippro
from TSP import TSP

import time
import json
from datetime import timedelta, datetime
import numpy as np


class Core:
    def __init__(self):
        """Fixstars Amplifyクライアント設定"""
        self.client = FixstarsClient()
       
        self.client.token = "AE/Y0TY3dM834BNw0YGdHlkIg8oLsCvAsXB"
        print("🔑 FixstarsClient initialized.")

    def to_native(o):
        """NumPy系をPythonのプリミティブ型に正規化"""
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        return o
    
    
    def main(self):
        import argparse
        parser = argparse.ArgumentParser(
            description="Iterative QA-based CVRP optimizer with detailed timing logs"
        )
        parser.add_argument("-j",   help="Path to before_data.json",             type=str, required=True)
        parser.add_argument("-sp",  help="Base output directory (e.g. ./out)",   type=str, required=True)
        parser.add_argument("--t",  help="Annealing time (ms)",                  type=int, default=3000)
        parser.add_argument("-nt",  help="QA solves per swap (num_solve)",       type=int, default=3)
        parser.add_argument("--p",  help="QA parameter p",                       type=float, default=1.0)
        parser.add_argument("--q",  help="QA parameter q",                       type=float, default=1.0)
        parser.add_argument("--max_iter", help="Max iterations (safety cap)",    type=int, default=50)
        parser.add_argument("--eps", help="Stop if improvement < eps",           type=float, default=1e-3)
        args = parser.parse_args()

        # === 出力ルート ===
        instance_name = os.path.splitext(os.path.basename(args.j))[0]
        timestamp     = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir      = os.path.join(args.sp, timestamp, instance_name)
        os.makedirs(save_dir, exist_ok=True)

        print(f"\n🚀 実験開始: {instance_name}")
        print(f"📂 出力先: {save_dir}")

        # === before_data.json 読み込み ===
        VRPfactory = vrpfactory()
        (
            cluster_nums, grax, gray, gra_distances,
            x, y, distances, demands, capacity,
            clusters, clusters_coordx, clusters_coordy, cluster_demands,
            gra_clusters_coordx, gra_clusters_coordy, depo_x, depo_y
        ) = VRPfactory.get_gluster_gravity_info(args.j)

        # 固定設定
        nvehicle = 1
        depo_x, depo_y = depo_x[0], depo_y[0]
        self.client.parameters.timeout = timedelta(milliseconds=args.t)

        # === 初期クラスタ順序（重心TSPで決定） ===
        tsp_over_clusters = TSP(
            self.client, gra_distances, demands, capacity,
            nvehicle, args.nt, cluster_nums, save_dir, grax, gray, args.j
        )
        gra_result = tsp_over_clusters.des_TSP(args.p, args.q)
        perms = gra_result["route"][1:]  # depot(0)を除く
        print(f"🧭 Initial cluster order: {perms}")

        prev_total_distance = None
        iteration = 0

        while True:
            iteration += 1
            print(f"\n===== Iteration {iteration} =====")

            # 1) クラスタ間再配置（QA）— 詳細計測ログ
            swap_time_log = []

            for idx, current_cluster_index in enumerate(perms):
                next_cluster_index = perms[(idx + 1) % len(perms)]

                # 計測開始：クラスタペア確定直後
                t_block_start = time.perf_counter()

                # 現在と次のクラスタ情報を取り出し
                current_x, current_y = x[current_cluster_index], y[current_cluster_index]
                current_demands      = demands[current_cluster_index]
                current_grax, current_gray = grax[current_cluster_index], gray[current_cluster_index]

                next_x, next_y       = x[next_cluster_index], y[next_cluster_index]
                next_demands         = demands[next_cluster_index]
                next_grax, next_gray = grax[next_cluster_index], gray[next_cluster_index]

                # 次クラスタの残積載量
                restcapacity = float(capacity - sum(next_demands))

                # 2つのQA入力距離行列
                distances_from_mycluster  = vrpfactory.make_distances(current_x, current_y, current_grax, current_gray)
                distances_from_nextcluster = vrpfactory.make_distances(current_x, current_y, next_grax, next_gray)

                # QA 実行（内部で total_time / execution_time / response_time / route / total_distances を返す想定）
                proccesor = knap_dippro(
                    self.client,
                    distances_from_mycluster,
                    distances_from_nextcluster,
                    current_demands,
                    restcapacity,
                    args.nt,
                    current_x,   # 都市IDリスト相当（あなたの実装に合わせて）
                    args.j
                )
                # ...（ループ内：record作成の直前あたりを修正）
            # QA 実行
            pro_result = proccesor.QA_processors()

            # moved_indices を取り出して Python の list[int] に正規化
            moved_raw = pro_result.get("route", [])
            if isinstance(moved_raw, np.ndarray):
                moved_indices = moved_raw.tolist()
            else:
                moved_indices = list(moved_raw) if not isinstance(moved_raw, list) else moved_raw

            # 要素をできるだけ int 化（失敗したらそのまま）
            try:
                moved_indices = [int(x) for x in moved_indices]
            except Exception:
                # 例えば [0. 0. 0.] のような float なら int に落ちるはずだが、
                # 何か混在していたら to_native で最低限のシリアライズは保証
                moved_indices = [to_native(x) for x in moved_indices]

            # 計測
            t_block_end = time.perf_counter()
            block_ms = float((t_block_end - t_block_start) * 1000.0)

            qa_total_time = pro_result.get("total_time", 0.0)
            qa_ms = float(qa_total_time) * 1000.0  # total_time が秒想定。ミリ秒ならここはそのまま float(qa_total_time)

            move_ms = max(block_ms - qa_ms, 0.0)

            record = {
                "iteration":     int(iteration),
                "swap_index":    int(idx),
                "from_cluster":  int(current_cluster_index),
                "to_cluster":    int(next_cluster_index),
                "qa_ms":         float(qa_ms),
                "move_ms":       float(move_ms),
                "block_ms":      float(block_ms),
                "moved_indices": moved_indices
            }
            swap_time_log.append(record)

            print(f"[swap {idx}] QA={qa_ms:.2f}ms | move={move_ms:.2f}ms | total={block_ms:.2f}ms | moved={moved_indices}")


            # スワップ時間ログJSON保存（集計なし・生配列）
            swap_log_path = os.path.join(save_dir, f"iteration_{iteration}_swap_timings.json")
            with open(swap_log_path, "w") as f:
                json.dump(swap_time_log, f, indent=2)
            print(f"🕒 スワップ詳細を保存: {swap_log_path}")

            # 2) 各クラスタ内のTSPを解き直す
            total_distance = 0.0
            tsp_routes = []
            for cluster_id in range(len(clusters)):
                # 各クラスタの座標／需要をデポ込みで整形
                coordx = [depo_x] + clusters_coordx[cluster_id]
                coordy = [depo_y] + clusters_coordy[cluster_id]
                cluster_demand = [0] + cluster_demands[cluster_id]
                city_list      = [0] + clusters[cluster_id]

                cluster_distance = vrpfactory.make_cluster_distance_matrix(coordx, coordy)

                tsp_solver = TSP(
                    self.client,
                    cluster_distance,
                    cluster_demand,
                    capacity,
                    1,               # 1車両（クラス内TSP）
                    args.nt,
                    city_list,
                    save_dir,
                    coordx,
                    coordy,
                    args.j
                )
                result = tsp_solver.solve_TSP(args.p, args.q)
                tsp_routes.append({
                    "cluster_id":      cluster_id,
                    "route":           result["route"],
                    "total_time":      result.get("total_time", None),
                    "execution_time":  result.get("execution_time", None),
                    "response_time":   result.get("response_time", None),
                    "total_distance":  result.get("total_distances", None)
                })
                total_distance += float(result.get("total_distances", 0.0))

            print(f"📏 Total distance after iteration {iteration}: {total_distance:.6f}")

            # 3) 改善判定（保存は毎回行う）
            # TSP結果を保存
            iteration_path = os.path.join(save_dir, f"iteration_{iteration}.json")
            with open(iteration_path, "w") as f:
                json.dump(tsp_routes, f, indent=2)
            print(f"💾 保存: {iteration_path}")

            # 収束判定
            if prev_total_distance is not None:
                improvement = prev_total_distance - total_distance
                print(f"🟢 Improvement: {improvement:.6f}")
                if abs(improvement) < args.eps:
                    print("⚠️ 改善が停止したため終了。")
                    break

            prev_total_distance = total_distance

            if iteration >= args.max_iter:
                print("⚠️ 最大イテレーション数に達したため停止。")
                break

        print("\n✅ Optimization completed.")
        print(f"📂 Results saved in: {save_dir}")


if __name__ == "__main__":
    core = Core()
    core.main()
