import os
import sys
import time
import json
import argparse
import numpy as np
from datetime import timedelta, datetime

# プロジェクト直下の src を import パスに追加
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from amplify import FixstarsClient
from src.vrpfactory import vrpfactory
from src.knap_divpro import knap_dippro
from TSP import TSP


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
        # ★推奨：環境変数 AMPLIFY_TOKEN から読む。無ければフォールバック
        self.client.token = os.getenv("AMPLIFY_TOKEN", "AE/Y0TY3dM834BNw0YGdHlkIg8oLsCvAsXB")
        print("🔑 FixstarsClient initialized.")

    def main(self):
        ap = argparse.ArgumentParser(
            description="Iterative QA-based CVRP optimizer with per-swap logging (before/after centroid-distance sums)"
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
                "without_depot": perms_native,
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
            moved_total = 0  # このイテレーションで1回でも移動があれば >0

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
                        "skip_reason":   "no_remaining_capacity_in_next_cluster",

                        # 求める指標はスキップ時は None
                        "sum_dist_current_before": None,
                        "sum_dist_current_after":  None
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

                # 2つのQA入力距離行列（都市→重心 距離ベクトル想定）
                distances_from_mycluster   = vrpfactory.make_distances(current_x, current_y, current_grax, current_gray)
                distances_from_nextcluster = vrpfactory.make_distances(current_x, current_y, next_grax, next_gray)

                # === QA 実行 ===
                proccesor = knap_dippro(
                    self.client,
                    distances_from_mycluster,
                    distances_from_nextcluster,
                    current_demands,
                    restcapacity,
                    capacity,
                    args.nt,
                    current_x,   # “都市ID配列相当”
                    args.j
                )
                pro_result = proccesor.QA_processors()

                # --- スワップ前：今のクラスタ（from_cluster）の重心距離合計 ---
                sum_dist_current_before = float(
                    np.sum(np.asarray(distances_from_mycluster, dtype=float).flatten())
                )

                # moved 正規化（0/1ベクトル or インデックス配列）
                def _normalize_moved(raw, length):
                    arr = np.array(raw, dtype=float)
                    if arr.ndim == 1 and arr.size == length and np.isin(arr, [0, 1]).all():
                        return arr
                    mask = np.zeros(length, dtype=float)
                    try:
                        idx_ = np.array(raw, dtype=int)
                        idx_ = idx_[(idx_ >= 0) & (idx_ < length)]
                        mask[idx_] = 1.0
                    except Exception:
                        pass
                    return mask

                moved_raw = pro_result.get("route", [])
                moved_arr = _normalize_moved(moved_raw, len(current_x))
                moved_idx = np.where(moved_arr > 0.5)[0]
                moved_count = int(moved_idx.size)
                did_move = moved_count > 0
                if did_move:
                    moved_total += 1

                # 都市ごとの距離差（旧→新）
                city_deltas = []
                if moved_count > 0:
                    prev_vals = np.asarray(distances_from_mycluster, dtype=float)[moved_idx]
                    new_vals  = np.asarray(distances_from_nextcluster, dtype=float)[moved_idx]
                    deltas    = new_vals - prev_vals
                    avg_delta = float(np.mean(deltas))
                    improved_ratio = float(np.mean(deltas < 0.0))
                    for i_local, prev_d, new_d, dlt in zip(moved_idx, prev_vals, new_vals, deltas):
                        try:
                            city_global = int(current_x[int(i_local)])
                        except Exception:
                            city_global = int(i_local)
                        city_deltas.append({
                            "city_local_index": int(i_local),
                            "city_global_id":   city_global,
                            "prev_dist":        float(prev_d),
                            "new_dist":         float(new_d),
                            "delta":            float(dlt)
                        })
                else:
                    avg_delta = None
                    improved_ratio = None

                # 計測（ブロック時間）
                t_block_end = time.perf_counter()
                block_ms = float((t_block_end - t_block_start) * 1000.0)

                qa_total_time = pro_result.get("total_time", 0.0)
                qa_ms = float(qa_total_time) * 1000.0  # total_time は秒想定
                move_ms = max(block_ms - qa_ms, 0.0)
                n_city = pro_result.get("n_city", len(current_x))

                # ★スワップ反映（動いたときのみ）→ クラスタと重心が更新
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

                # --- スワップ後：今のクラスタ（from_cluster）の重心距離合計（重心更新後を反映） ---
                cur_after_xs = clusters_coordx[current_cluster_index]
                cur_after_ys = clusters_coordy[current_cluster_index]
                cur_after_cx = gra_clusters_coordx[current_cluster_index]
                cur_after_cy = gra_clusters_coordy[current_cluster_index]
                dist_vec_after = vrpfactory.make_distances(cur_after_xs, cur_after_ys, cur_after_cx, cur_after_cy)
                sum_dist_current_after = float(np.sum(np.asarray(dist_vec_after, dtype=float).flatten()))

                # 記録
                record = {
                    "iteration":     int(iteration),
                    "swap_index":    int(idx),
                    "from_cluster":  int(current_cluster_index),
                    "to_cluster":    int(next_cluster_index),
                    "qa_ms":         float(qa_ms),
                    "move_ms":       float(move_ms),
                    "block_ms":      float(block_ms),
                    "moved_indices": to_native(moved_arr),  # 0/1 ベクトルで保存
                    "n_city":        int(n_city),
                    "skipped":       False,

                    # 都市ごとの改善
                    "city_deltas":    city_deltas,
                    "avg_delta":      avg_delta,
                    "improved_ratio": improved_ratio,

                    # ★欲しい指標（今のクラスタの重心距離合計：前/後）
                    "sum_dist_current_before": sum_dist_current_before,
                    "sum_dist_current_after":  sum_dist_current_after
                }
                swap_time_log.append(record)

                print(f"[swap {idx}] QA={qa_ms:.2f}ms | move={move_ms:.2f}ms | total={block_ms:.2f}ms | "
                      f"moved_count={moved_count} | avgΔ={avg_delta if avg_delta is not None else 'NA'} | "
                      f"sumBefore={sum_dist_current_before:.3f} | sumAfter={sum_dist_current_after:.3f}")

            # スワップ時間ログ（生配列）保存
            swap_log_path = os.path.join(save_dir, f"iteration_{iteration}_swap_timings.json")
            with open(swap_log_path, "w") as f:
                json.dump(swap_time_log, f, indent=2)
            print(f"🕒 スワップ詳細を保存: {swap_log_path}")

            # このイテレーションで都市が1件も動かなかったら終了
            if moved_total == 0:
                print("🟡 No city moved in this iteration → stop optimization.")
                break

            # --- 各クラスタ内 TSP を解き直す（距離の推移を残す） ---
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
                    "overall":         result.get("overall", {}),
                    "runs":            result.get("runs", []),
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
