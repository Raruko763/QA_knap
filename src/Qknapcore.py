import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from amplify import FixstarsClient
from src.vrpfactory import vrpfactory
from src.knap_divpro import knap_dippro
from TSP import TSP

import matplotlib.pyplot as plt
import numpy as np
import argparse
import json
from datetime import timedelta, datetime

class Core:
    def __init__(self):
        '''
        コンストラクタ
        '''
        
        # ソルバーの設定
        self.client = FixstarsClient()

        self.client.token = "AE/Y0TY3dM834BNw0YGdHlkIg8oLsCvAsXB"

        
    def main(self):
        parser = argparse.ArgumentParser(description='CVRP')
        parser.add_argument('-i', help='Path to instance file', type=str)  # インスタンスファイルのパス
        parser.add_argument('--p', help='Parameter: p', default=1.0, type=float)
        parser.add_argument('--q', help='Parameter: q', default=1.0, type=float)
        parser.add_argument('--r', help='Parameter: r', default=1.0, type=float)
        parser.add_argument('-j', help='Path to JSON file', type=str, required=False)  # JSONファイルのパス
        parser.add_argument('-sp',help='Path to JSON file', type=str, required=False)
        parser.add_argument('--t',help='anniiring_time',default=1000,type=int)
        parser.add_argument('-nt',help='num_solve',default=1,type=int)

        args = parser.parse_args()

        # データの取得方法を選択
        if args.i:  # インスタンスファイルを使用
            distances, demand, capacity, nvehicle, x, y = vrpfactory.makedata(args.i)
            ncity = len(distances)
            city = [i for i in range(ncity)]
        elif args.j:  # JSONファイルを使用
            VRPfactory = vrpfactory()
            (cluster_nums, grax, gray, gra_distances,
                x, y, distances, demands, capacity,
                clusters, clusters_coordx, clusters_coordy, cluster_demands,
                gra_clusters_coordx, gra_clusters_coordy,depo_x,depo_y) = VRPfactory.get_gluster_gravity_info(args.j)
            # print(distances)
            
            # capacity = 0
            nvehicle = 1
            depo_x = depo_x[0]
            depo_y = depo_y[0]
            print("cluster",cluster_nums)
            print("x",depo_x)
            print("y",depo_y)
            # ncity = len(city)
            # ncluster = len(cluster_nums)
        else:
            raise ValueError("Either -i (instance file) or -j (JSON file) must be provided.")

        # 問題のパラメータ設定

        p = args.p
        q = args.q
        r = args.r
        p = 1
        q=1
        save_dir = args.sp
        num_solve = args.nt
        anniring_time = args.t
        self.client.parameters.timeout = timedelta(milliseconds=anniring_time)  # タイムアウト2秒
        results = []
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = os.path.join(save_dir, f"TSP_cluster_results_{now}.json")
        # 最初に重心でTSPしてTSPの重心を決めるのインスタンスを作成
        tsp = TSP(self.client, gra_distances, demands, capacity,nvehicle,num_solve , cluster_nums,save_dir,grax,gray,args.j)
        # 順番を求める
        gra_result = tsp.des_TSP(p,q)
        perms = gra_result["route"]
        gra_result_row = {
            "gra_total_time": gra_result["total_time"],
            "gra_execution_time": gra_result["execution_time"],
            "gra_response_time": gra_result["response_time"]
        }
        results.append(gra_result_row)
        perms = perms[1:]
        print("prems",perms)
        # self.plot_clusters_only(clusters_coordx, clusters_coordy)
        #後処理をする関数
        process_num = 2
        for i in range(process_num):
            for idx, current_cluster_index in enumerate(perms):
                next_cluster_index = perms[(idx + 1)%len(perms)]

                print("current_cluster_index",current_cluster_index)
                print("x",x)
                print("xcurent",x[current_cluster_index])
                current_x = x[current_cluster_index]
                current_y = y[current_cluster_index]
                current_demands = demands[current_cluster_index]
                current_distance = distances[current_cluster_index] 
                current_grax = grax[current_cluster_index]
                current_gray = gray[current_cluster_index]
                current_capacity = capacity
                city = [x for i in range(len(x))]

                next_x = x[next_cluster_index]
                next_y = y[next_cluster_index]
                next_demands = demands[next_cluster_index]
                next_distance = distances[next_cluster_index]
                next_grax = grax[next_cluster_index]
                next_gray = gray[next_cluster_index]
                next_capacity = capacity

                distances_from_mycluster = vrpfactory.make_distances(current_x,current_y,current_grax,current_gray)
                distances_from_nextcluster = vrpfactory.make_distances(current_x,current_y,next_grax,next_gray)
                restcapacity_of_nextcluster = next_capacity - sum(next_demands)
                
                print("next_capacity",next_capacity)
                print("demands",next_demands)
                print("restcapacity_of_nextcluster",restcapacity_of_nextcluster)
                print("clussters",clusters)
                restcapacity_of_nextcluster = float(restcapacity_of_nextcluster)
                proccesor = knap_dippro(self.client, distances_from_mycluster,distances_from_nextcluster,current_demands,restcapacity_of_nextcluster,num_solve,city,args.j)
                pro_result = proccesor.QA_processors()
                swap_perms = pro_result["route"]
                pro_result_row = {
                    "pro_total_time": pro_result["total_time"],
                    "pro_execution_time": pro_result["execution_time"],
                    "pro_response_time": pro_result["response_time"]
                }
                results.append(pro_result_row)
                clusters, clusters_coordx, clusters_coordy, cluster_demands, gra_clusters_coordx, gra_clusters_coordy, distances = vrpfactory.process_swap(
                    swap_perms,
                    clusters,
                    clusters_coordx,
                    clusters_coordy,
                    cluster_demands,
                    gra_clusters_coordx,
                    gra_clusters_coordy,
                    current_cluster_index,
                    next_cluster_index,
                    distances 
                )
            self.plot_clusters_only(clusters_coordx, clusters_coordy)
        for cluster_id in range(len(clusters)):
            city_list = clusters[cluster_id]
            print("city_list",city_list)
            city_list = [0] + clusters[cluster_id]
            print("city_list",city_list)
            cluster_distance = distances[cluster_id]
            cluster_demand = [0] + cluster_demands[cluster_id] 
            coordx = [depo_x] + clusters_coordx[cluster_id]
            coordy = [depo_y] + clusters_coordy[cluster_id]
            print(coordy)
            cluster_distance = vrpfactory.make_cluster_distance_matrix(coordx,coordy)
            tsp_solver = TSP(
                self.client,
                cluster_distance,
                cluster_demand,
                capacity,
                1,
                num_solve,
                city_list,
                save_dir,
                coordx,
                coordy,
                args.j
            )

            result = tsp_solver.solve_TSP(p, q)

            # 結果出力
            print(f"Cluster {cluster_id}")
            print("  📍 Route:", result["route"])
            print("  🕒 Total Time:", result["total_time"])
            print("  ⚙️ Execution Time:", result["execution_time"])
            print("  🔁 Response Time:", result["response_time"])
            print("  📏 Total Distance:", result["total_distances"] )
            # 1行分のデータを辞書形式で格納
            result_row = {
                "cluster_id": cluster_id,
                "route": result["route"],
                "total_time": result["total_time"],
                "execution_time": result["execution_time"],
                "response_time": result["response_time"],
                "total_distance": result["total_distances"]
            }
            results.append(result_row)
            # results.append(pro_result_row)
        # 保存ファイル名
        # csv_file_path = os.path.join(save_dir, "TSP_cluster_results.csv")

        with open(json_path, "w") as f:
            json.dump(results, f, indent=4)
        print(f"TSP + QA の結果を {json_path} に保存しました。")
        # print(f"クラスタ別のTSP結果を {csv_file_path} に保存しました。")

    def plot_clusters_only(self,clusters_coordx, clusters_coordy):
        plt.figure(figsize=(8, 6))
        colors = plt.cm.tab20.colors  # カラーパレット（最大20クラスタ）

        for i, (xs, ys) in enumerate(zip(clusters_coordx, clusters_coordy)):
            plt.scatter(xs, ys, color=colors[i % len(colors)], label=f"Cluster {i+1}", s=50)

        plt.title("City Coordinates by Cluster")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend()
        plt.axis("equal")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("process2.pdf")

        
        

if __name__ == "__main__":
    core = Core()  # 引数を渡して Core インスタンスを作成
    core.main()  # メイン処理を呼び出す