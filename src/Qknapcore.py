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
import argparse
import numpy as np


def to_native(o):
    """NumPy なども含め、JSON 直列化しやすい素の型へ変換"""
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    return o


class Core:
    def __init__(self):
        """Fixstars Amplify クライアント設定"""
        self.client = FixstarsClient()
        # ★必要に応じて差し替え
        self.client.token = "AE/Y0TY3dM834BNw0YGdHlkIg8oLsCvAsXB"
        print("🔑 FixstarsClient initialized.")

    def main(self):
        ap = argparse.ArgumentParser(
            description="Iterative QA-based CVRP optimizer with skip-by-capacity and stop-when-no-move"
        )
        ap.add_argument("-j",   help="Path to before_data.json",             type=str, required=True)
        ap.add_argument("-sp",  help="Base output directory (e.g. ./out)",   type=str, required=True)
        ap.add_argument("--t",  help="Annealing time (ms)",                  type=int, default=3000)
        ap.add_argument("-nt",  help="QA solves per swap (num_solve)",       type=int, default=3)
        ap.add_argument("--p",  help="QA parameter p",                       type=float, default=1.0)
        ap.add_argument("--q",  help="QA parameter q",                       type=float, default=1.0)
        ap.add_argument("--max_iter", help="Max iterations (safety cap)",    type=int, default=50)
        ap.add_argument("--eps", help="(unused now) keep for compat",        type=float, default=1e-3)
        args = ap.parse_args()

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

        # === 重心 TSP（クラスタ順序の初期化 & 計測保存） ===
        tsp_over_clusters = TSP(
            self.client, gra_distances, demands, capacity,
            nvehicle, args.nt, cluster_nums, save_dir, grax, gray, args.j
        )
        gra_result = tsp_over_clusters.des_TSP(args.p, args.q)
        perms = gra_result["route"][1:]  # depot(0)を除く
        perms_native = [int(v) for v in np.array(perms).tolist()]
        print(f"🧭 Initial cluster order: {perms}")

        # 初期重心情報を保存（centroid_init.json）
        centroid_payload = {
            "instance": instance_name,
            "params": {"p": args.p, "q": args.q, "nt": args.nt, "anneal_ms": args.t},
            "clusters": to_native(np.array(cluster_nums)),
            "centroids": {
                "x": to_native(np.array(grax)),
                "y": to_native(np.array(gray)),
            },
            "route_over_centroids": {
                "with_depot": to_native(np.array(gra_result["route"])),
                "without_depot": perms_native,  # ← ここを置き換え
            },
            "metrics": {
                "total_time": gra_result.get("total_time", None),
                "execution_time": gra_result.get("execution_time", None),
                "response_time": gra_result.get("response_time", None),
                "total_distances": gra_result.get("total_distances", None),
            },
            "centroid_distance_shape": list(np.array(gra_distances).shape)
        }

        with open(os.path.join(save_dir, "centroid_init.json"), "w") as f:
            json.dump(centroid_payload, f, indent=2)
        print(f"💾 保存: {os.path.join(save_dir, 'centroid_init.json')}")

        # === 反復最適化 ===
        iteration = 0
        while True:
            iteration += 1
            print(f"\n===== Iteration {iteration} =====")
            swap_time_log = []
            moved_total = 0  # ← このイテレーション内で動いた都市が一度でもあれば >0

            # --- クラスタ間再配置（QA） ---
            for idx, current_cluster_index in enumerate(perms):
                next_cluster_index = perms[(idx + 1) % len(perms)]

                # 次クラスタの残積載量
                restcapacity = float(capacity - sum(demands[next_cluster_index]))

                # スキップ（容量なし）
                if restcapacity <= 0:
                    record = {
                        "iteration":     int(iteration),
                        "swap_index":    int(idx),
                        "from_cluster":  int(current_cluster_index),
                        "to_cluster":    int(next_cluster_index),
                        "qa_ms":         0.0,
                        "move_ms":       0.0,
                        "block_ms":      0.0,
                        "moved_indices": [],
                        "n_city":        int(len(x[current_cluster_index])),
                        "skipped":       True,
                        "skip_reason":   "no_remaining_capacity_in_next_cluster"
                    }
                    swap_time_log.append(record)
                    print(f"[swap {idx}] ⏭️ skip (next cluster {next_cluster_index} has no remaining capacity)")
                    continue

                # 計測開始（クラスタペア確定直後）
                t_block_start = time.perf_counter()

                # 現在と次のクラスタ情報
                current_x, current_y = x[current_cluster_index], y[current_cluster_index]
                current_demands      = demands[current_cluster_index]
                current_grax, current_gray = grax[current_cluster_index], gray[current_cluster_index]

                next_x, next_y       = x[next_cluster_index], y[next_cluster_index]
                next_demands         = demands[next_cluster_index]
                next_grax, next_gray = grax[next_cluster_index], gray[next_cluster_index]

                # 2つのQA入力距離行列
                distances_from_mycluster   = vrpfactory.make_distances(current_x, current_y, current_grax, current_gray)
                distances_from_nextcluster = vrpfactory.make_distances(current_x, current_y, next_grax, next_gray)

                # QA 実行
                proccesor = knap_dippro(
                    self.client,
                    distances_from_mycluster,
                    distances_from_nextcluster,
                    current_demands,
                    restcapacity,
                    capacity,
                    args.nt,
                    current_x,   # あなたの実装に合わせた「都市ID配列」相当
                    args.j
                )
                pro_result = proccesor.QA_processors()

                # moved_indices（0/1配列 or インデックス配列）を正規化
                moved_raw = pro_result.get("route", [])
                moved_arr = np.array(moved_raw)
                # 0/1ベクトルとみなし、合計>0 で「何か動いた」と定義（要件に合わせて調整可）
                moved_count = int(np.sum(moved_arr))
                did_move = moved_count > 0
                if did_move:
                    moved_total += 1

                # 計測
                t_block_end = time.perf_counter()
                block_ms = float((t_block_end - t_block_start) * 1000.0)

                qa_total_time = pro_result.get("total_time", 0.0)
                # 注意：total_time を「秒」で返しているなら *1000、ミリ秒ならそのまま
                # ここでは「秒」想定のため *1000
                qa_ms = float(qa_total_time) * 1000.0

                move_ms = max(block_ms - qa_ms, 0.0)
                n_city = pro_result.get("n_city", len(current_x))

                record = {
                    "iteration":     int(iteration),
                    "swap_index":    int(idx),
                    "from_cluster":  int(current_cluster_index),
                    "to_cluster":    int(next_cluster_index),
                    "qa_ms":         float(qa_ms),
                    "move_ms":       float(move_ms),
                    "block_ms":      float(block_ms),
                    "moved_indices": to_native(moved_arr),
                    "n_city":        int(n_city),
                    "skipped":       False
                }
                swap_time_log.append(record)

                print(f"[swap {idx}] QA={qa_ms:.2f}ms | move={move_ms:.2f}ms | total={block_ms:.2f}ms | moved_count={moved_count}")

                # 実際のスワップ反映（移動がある場合のみ）
                if did_move:
                    (
                        clusters, clusters_coordx, clusters_coordy, cluster_demands,
                        gra_clusters_coordx, gra_clusters_coordy, distances
                    ) = VRPfactory.process_swap(
                        pro_result["route"],
                        clusters, clusters_coordx, clusters_coordy, cluster_demands,
                        gra_clusters_coordx, gra_clusters_coordy,
                        current_cluster_index, next_cluster_index, distances
                    )

            # スワップ時間ログ（生配列）保存
            swap_log_path = os.path.join(save_dir, f"iteration_{iteration}_swap_timings.json")
            with open(swap_log_path, "w") as f:
                json.dump(swap_time_log, f, indent=2)
            print(f"🕒 スワップ詳細を保存: {swap_log_path}")

            # 「このイテレーションで都市が1件も動かなかったら」終了
            if moved_total == 0:
                print("🟡 No city moved in this iteration → stop optimization.")
                break

            # --- 各クラスタ内 TSP を解き直す（参考：距離の推移を残したい場合） ---
            total_distance = 0.0
            tsp_routes = []
            for cluster_id in range(len(clusters)):
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
                    "total_distance":  result.get("total_distances", None),
                    "overall": result.get("overall", {}),
                    "runs": result.get("runs", []),

                })
                total_distance += float(result.get("total_distances", 0.0))

            print(f"📏 Total distance after iteration {iteration}: {total_distance:.6f}")

            # 保存（毎回）
            iteration_path = os.path.join(save_dir, f"iteration_{iteration}.json")
            with open(iteration_path, "w") as f:
                json.dump(tsp_routes, f, indent=2)
            print(f"💾 保存: {iteration_path}")

            if iteration >= args.max_iter:
                print("⚠️ 最大イテレーション数に達したため停止。")
                break

        print("\n✅ Optimization completed.")
        print(f"📂 Results saved in: {save_dir}")


if __name__ == "__main__":
    core = Core()
    core.main()
