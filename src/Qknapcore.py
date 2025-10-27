import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from amplify import FixstarsClient
from src.vrpfactory import vrpfactory
from src.knap_divpro import knap_dippro
from TSP import TSP

import matplotlib.pyplot as plt
import json
from datetime import timedelta, datetime


class Core:
    def __init__(self):
        """Fixstars Amplifyクライアント設定"""
        self.client = FixstarsClient()
        self.client.token = "AE/Y0TY3dM834BNw0YGdHlkIg8oLsCvAsXB"
        print("🔑 FixstarsClient initialized.")

    def main(self):
        import argparse
        parser = argparse.ArgumentParser(description="Iterative QA-based CVRP optimizer (until convergence)")
        parser.add_argument("-j", help="Path to before_data.json", type=str, required=True)
        parser.add_argument("-sp", help="Base output directory (e.g. ./out)", type=str, required=True)
        parser.add_argument("--t", help="Annealing time (ms)", default=3000, type=int)
        parser.add_argument("-nt", help="Number of QA solves per iteration", default=3, type=int)
        parser.add_argument("--p", help="QA parameter p", default=1.0, type=float)
        parser.add_argument("--q", help="QA parameter q", default=1.0, type=float)
        args = parser.parse_args()

        # === 基本情報 ===
        instance_name = os.path.splitext(os.path.basename(args.j))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.join(args.sp, timestamp, instance_name)
        os.makedirs(save_dir, exist_ok=True)

        print(f"\n🚀 実験開始: {instance_name}")
        print(f"📂 出力先: {save_dir}")

        # === before_data.json から情報をロード ===
        VRPfactory = vrpfactory()
        (
            cluster_nums, grax, gray, gra_distances,
            x, y, distances, demands, capacity,
            clusters, clusters_coordx, clusters_coordy, cluster_demands,
            gra_clusters_coordx, gra_clusters_coordy, depo_x, depo_y
        ) = VRPfactory.get_gluster_gravity_info(args.j)

        nvehicle = 1
        depo_x, depo_y = depo_x[0], depo_y[0]
        self.client.parameters.timeout = timedelta(milliseconds=args.t)

        # === 初期クラスタ順序 ===
        tsp = TSP(self.client, gra_distances, demands, capacity, nvehicle, args.nt, cluster_nums, save_dir, grax, gray, args.j)
        gra_result = tsp.des_TSP(args.p, args.q)
        perms = gra_result["route"][1:]  # depot除外
        print(f"🧭 Initial cluster order: {perms}")

        prev_total_distance = None
        iteration = 0
        improvement_threshold = 1e-3
        max_iterations = 50  # 念のため上限

        while True:
            iteration += 1
            print(f"\n===== Iteration {iteration} =====")

            # --- クラスタ間再配置 (QA) ---
            for idx, current_cluster_index in enumerate(perms):
                next_cluster_index = perms[(idx + 1) % len(perms)]
                current_x, current_y = x[current_cluster_index], y[current_cluster_index]
                current_demands = demands[current_cluster_index]
                current_grax, current_gray = grax[current_cluster_index], gray[current_cluster_index]
                next_x, next_y = x[next_cluster_index], y[next_cluster_index]
                next_demands = demands[next_cluster_index]
                next_grax, next_gray = grax[next_cluster_index], gray[next_cluster_index]

                restcapacity = float(capacity - sum(next_demands))
                distances_from_mycluster = vrpfactory.make_distances(current_x, current_y, current_grax, current_gray)
                distances_from_nextcluster = vrpfactory.make_distances(current_x, current_y, next_grax, next_gray)

                proccesor = knap_dippro(
                    self.client, distances_from_mycluster, distances_from_nextcluster,
                    current_demands, restcapacity, args.nt, current_x, args.j
                )
                pro_result = proccesor.QA_processors()

                clusters, clusters_coordx, clusters_coordy, cluster_demands, gra_clusters_coordx, gra_clusters_coordy, distances = \
                    VRPfactory.process_swap(
                        pro_result["route"],
                        clusters, clusters_coordx, clusters_coordy, cluster_demands,
                        gra_clusters_coordx, gra_clusters_coordy,
                        current_cluster_index, next_cluster_index, distances
                    )

            # --- 各クラスタ内TSP再計算 ---
            total_distance = 0
            tsp_routes = []
            for cluster_id in range(len(clusters)):
                coordx = [depo_x] + clusters_coordx[cluster_id]
                coordy = [depo_y] + clusters_coordy[cluster_id]
                cluster_demand = [0] + cluster_demands[cluster_id]
                city_list = [0] + clusters[cluster_id]
                cluster_distance = vrpfactory.make_cluster_distance_matrix(coordx, coordy)

                tsp_solver = TSP(
                    self.client, cluster_distance, cluster_demand, capacity,
                    1, args.nt, city_list, save_dir, coordx, coordy, args.j
                )
                result = tsp_solver.solve_TSP(args.p, args.q)
                tsp_routes.append(result)
                total_distance += result["total_distances"]

            print(f"📏 Total distance after iteration {iteration}: {total_distance:.3f}")

            # --- 改善チェック ---
            if prev_total_distance is not None:
                improvement = prev_total_distance - total_distance
                print(f"🟢 Improvement: {improvement:.6f}")

                # まず結果を保存してから改善判定
                iteration_path = os.path.join(save_dir, f"iteration_{iteration}.json")
                with open(iteration_path, "w") as f:
                    json.dump(tsp_routes, f, indent=4)
                print(f"💾 Saved: {iteration_path}")

                if abs(improvement) < improvement_threshold:
                    print("⚠️ 改善が停止したため終了。")
                    break
            else:
                # 1回目（prev_total_distanceがまだないとき）
                iteration_path = os.path.join(save_dir, f"iteration_{iteration}.json")
                with open(iteration_path, "w") as f:
                    json.dump(tsp_routes, f, indent=4)
                print(f"💾 Saved: {iteration_path}")

            prev_total_distance = total_distance

            if iteration >= max_iterations:
                print("⚠️ 最大イテレーション数に達したため停止。")
                break

        print("\n✅ Optimization completed.")
        print(f"📂 Results saved in: {save_dir}")


if __name__ == "__main__":
    core = Core()
    core.main()
