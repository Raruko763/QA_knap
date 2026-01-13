import numpy as np

from typing import List, Tuple, Dict
from amplify import VariableGenerator, one_hot, einsum, less_equal, ConstraintList, Poly, Model,solve,clamp,equal_to
from amplify.client import FixstarsClient
from datetime import timedelta

from datetime import datetime 
import matplotlib.pyplot as plt
import pandas as pd
import src.vrpfactory as vrpfactory
import os
import json

import re

class knap_dippro:

    def __init__(
        self,
        client,
        distances_from_mycluster,
        distances_from_nextcluster,
        demands,
        restcapacity_of_nextcluster,
        max_capacity,
        num_solve,
        city,
        file_path,
        depot_xy=None,
        cur_xs=None,
        cur_ys=None,
        next_xs=None,
        next_ys=None,
    ):
        self.client = client
        self.distances_from_mycluster = np.array(distances_from_mycluster,dtype=int)
        self.distances_from_nextcluster = np.array(distances_from_nextcluster,dtype=int)
        self.demands = demands
        self.num_solve = num_solve
        self.restcapacity_of_nextcluster =restcapacity_of_nextcluster
        self.maxcapacity = max_capacity
        # self.x = x
        # self.y = y
        self.city = city
        self.depot_xy = depot_xy
        self.cur_xs = np.array(cur_xs) if cur_xs is not None else None
        self.cur_ys = np.array(cur_ys) if cur_ys is not None else None
        self.next_xs = np.array(next_xs) if next_xs is not None else None
        self.next_ys = np.array(next_ys) if next_ys is not None else None

        # self.clu_path = clu_path

        self._colors = [
                        "green",
                        "orange",
                        "blue",
                        "red",
                        "purple",
                        "pink",
                        "darkblue",
                        "cadetblue",
                        "darkred",
                        "lightred",
                        "darkgreen",
                        "lightgreen",
                        "lightblue",
                        "darkpurple",
                    ]
        self.file_path = file_path
    
    def calculate_total_distance(self,route: np.ndarray, distance_matrix: np.ndarray) -> float:
        total_distance = 0
        for i in range(len(route) - 1):
            total_distance += distance_matrix[route[i]%len(distance_matrix), route[i + 1]%len(distance_matrix)]
        return total_distance
    
    # def distance(sequence: Dict[int, List[int]]):
    #     for vehicle, route in best_tour.items():
    #         total_distance = calculate_total_distance(route, distance_matrix)
    #         print(f"Total distance for vehicle {vehicle}: {total_distance}")

    def _unit_vectors(self, xs, ys, origin):
        eps = 1e-12
        dx = xs - origin[0]
        dy = ys - origin[1]
        norms = np.sqrt(dx * dx + dy * dy) + eps
        return np.stack([dx / norms, dy / norms], axis=1)

    def _mean_direction(self, us):
        vec = np.sum(us, axis=0)
        norm = np.linalg.norm(vec)
        if norm < 1e-12:
            return np.array([0.0, 0.0])
        return vec / norm

    def solve_stage2_reassignment(
        self,
        lam: float,
        alpha: float = 1.0,
        p: float | None = None,
        mode: str = "dist+ang",
    ) -> dict:
        n_mycluster = len(self.demands)
        gen = VariableGenerator()
        x = gen.array("Binary", shape=(n_mycluster))

        if mode not in {"dist", "dist+ang", "ang"}:
            raise ValueError(f"Invalid mode: {mode}")

        dist_norm = None
        if mode in {"dist", "dist+ang"}:
            dist_diff = self.distances_from_nextcluster.astype(float) - self.distances_from_mycluster.astype(float)
            dist_scale = float(np.max(np.abs(dist_diff))) + 1e-12
            dist_norm = dist_diff / dist_scale

        ang_norm = None
        if mode in {"ang", "dist+ang"}:
            if self.depot_xy is None or self.cur_xs is None or self.cur_ys is None:
                raise ValueError("depot_xy/cur_xs/cur_ys are required for stage2 angle objective.")
            if self.next_xs is None or self.next_ys is None:
                raise ValueError("next_xs/next_ys are required for stage2 angle objective.")

            cur_us = self._unit_vectors(self.cur_xs, self.cur_ys, self.depot_xy)
            next_us = self._unit_vectors(self.next_xs, self.next_ys, self.depot_xy)
            v_a = self._mean_direction(cur_us)
            v_b = self._mean_direction(next_us)

            ang_diff = np.dot(cur_us, v_a) - np.dot(cur_us, v_b)
            ang_norm = ang_diff / 2.0

        if mode == "dist":
            coef = dist_norm
        elif mode == "ang":
            coef = ang_norm
        else:
            coef = dist_norm + lam * ang_norm
        objective = einsum("i,i->", coef, x)

        demands = np.array(self.demands)
        weight_sums = einsum("i,i->", demands, x)
        capacity_constraints: ConstraintList = less_equal(
            weight_sums,
            self.restcapacity_of_nextcluster,
            penalty_formulation="Relaxation",
            label='weight_sum'
        )
        cap_penalty = alpha * max(float(np.sum(np.abs(coef))), 1e-9)
        if p is not None:
            cap_penalty *= p
        capacity_constraints *= cap_penalty
        model = Model(objective, capacity_constraints)

        result = solve(model, self.client)
        x_values = result.best.values

        swap_perms = x.evaluate(x_values)
        total_time = result.total_time.total_seconds()
        execution_time = result.execution_time.total_seconds()
        response_time = result.response_time.total_seconds()
        total_objective = result.best.objective
        return {
            "route": swap_perms,
            "total_time": total_time,
            "execution_time": execution_time,
            "response_time": response_time,
            "total_objective": total_objective,
            "n_city": n_mycluster,
            "mode": mode,
            "lam": lam,
            "alpha": alpha,
            "p": p,
        }

    def QA_processors(self, p: float = None):
        """Deprecated: use solve_stage2_reassignment instead."""
        return self.solve_stage2_reassignment(lam=0.3, alpha=1.0, p=p)
     

    def des_TSP(self, p, q):
        """
        TSP (1クラスタ内) を QUBO で解く — 直列実行(num_solves回)。
        返り値: dict
        - best: 最良解（全試行の中で objective が最小）
        - runs: 各試行の最良解一覧（分布解析用）
        - overall: solve呼び出し全体の時間など
        """
        print("Solution is TSP.",self.num_solve)
        NUM_CITIES = len(self.city)

        # 変数定義
        gen = VariableGenerator()
        x = gen.array("Binary", shape=(NUM_CITIES + 1, NUM_CITIES))
        x[NUM_CITIES, :] = x[0, :]  # 巡回の最後→最初を同一行に縛る

        # 目的関数：連続都市間の距離総和
        objective: Poly = einsum("ij,ni,nj->", self.distances, x[:-1], x[1:])  # type: ignore

        # 制約：各時刻に1都市だけ訪問（行制約）＆各都市は1度だけ訪問（列制約）
        row_constraints = one_hot(x[:-1], axis=1, label="one_trip_constraint")
        col_constraints = one_hot(x[:-1], axis=0, label="one_visit_constraint")

        # 制約強度（ペナルティ）を距離スケールに合わせて調整
        constraints = p * row_constraints + q * col_constraints
        constraints *= float(np.amax(self.distances))

        # QUBOモデル
        model = objective + constraints

        # 直列実行（num_solve回）
        result = solve(model, self.client, num_solves=self.num_solve)

        if len(result) == 0:
            raise RuntimeError("At least one of the constraints is not satisfied.")

        # ---- 最良解（全試行の中で最も良い） ----
        best_sol = result.best
        best_q = x.evaluate(best_sol.values)
        best_route_idx = list(np.where(np.array(best_q) == 1)[1])  # 各時刻に選ばれた都市の列インデックス
        # 必要なら巡回開始点を整形（任意）
        if hasattr(self, "repair_circle_shift"):
            best_route_idx = self.repair_circle_shift(best_route_idx)
        best_route = [self.city[r] for r in best_route_idx]

        # どの試行が最良だったか（インデックス）
        best_run_index = min(
            range(len(result.split)),
            key=lambda i: result.split[i].best.objective
        )
        print("solvetime", len(result.split))
        # ---- 全試行の結果（分布解析用） ----
        runs = []
        for i, r_i in enumerate(result.split):
            best_i = r_i.best
            q_i = x.evaluate(best_i.values)
            route_i_idx = list(np.where(np.array(q_i) == 1)[1])
            if hasattr(self, "repair_circle_shift"):
                route_i_idx = self.repair_circle_shift(route_i_idx)
            route_i = [self.city[r] for r in route_i_idx]

            runs.append({
                "run": i,
                "total_distances": float(best_i.objective),
                "total_time": r_i.total_time.total_seconds(),
                "execution_time": r_i.execution_time.total_seconds(),
                "response_time": r_i.response_time.total_seconds(),
                "route_idx": route_i_idx,
                "route": route_i,
            })

        # ---- 直列全体の時間情報 ----
        overall = {
            "total_time": result.total_time.total_seconds(),         # solve全体（壁時計）
            "execution_time": result.execution_time.total_seconds(), # 直列合計
            "response_time": result.response_time.total_seconds(),   # 直列合計
            "num_solves": result.num_solves
        }

        # デバッグ出力（任意）
        print(f"[BEST] run={best_run_index}, objective={float(best_sol.objective)}")
        # print("best route idx:", best_route_idx)

        return {
            "route": best_route,
            "total_time": result.total_time.total_seconds(),
            "execution_time": result.execution_time.total_seconds(),
            "response_time": result.response_time.total_seconds(),
            "total_distances": float(best_sol.objective),
            "runs": runs,          # 各試行の最良解
            "overall": overall     # 直列全体の時間情報
        }


    # def plot(self, route, total_distance,savedir,p,q,i):
    #     # seaborn のスタイル設定
    #     sns.set_theme(style="whitegrid")
    
    # # グラフをプロット
    #     plt.figure(figsize=(10, 6))
        
    #     # 経路を描画
    #     x = self.x
    #     y = self.y
    #     x_coords = [self.x[r] for r in route]  
    #     y_coords = [self.y[r] for r in route]  

        
    #     plt.plot(x_coords + [x_coords[0]], y_coords + [y_coords[0]], 'b-', marker='o')
        
    #     # 都市を描画
    #     plt.scatter(x_coords, y_coords, c='r', zorder=5)
        
    #     # 都市番号をプロット
    #     for j in range(len(route[:-1])):
    #         plt.text(x_coords[j] + 0.1, y_coords[j] + 0.1, str(self.city[route[j]]), fontsize=16, ha='center', va='center')
        
    #     # タイトルにtotal_distanceを追加
    #     plt.title(f'TSP Solution - Total Distance: {total_distance:.2f}', fontsize=18)
    
    #     # 軸ラベルの設定
    #     plt.xlabel('X')
    #     plt.ylabel('Y')
    #     # グラフの表示

    #     # savedir の親ディレクトリを取得
    #     parent_dir = os.path.dirname(savedir)

    #     match = re.search(r"cluster_\d+", self.clu_path)
    #     if match:
    #         # 最終的な保存先のパスを作成
    #         save_path = os.path.join(parent_dir, f"route_{match.group()}_{i}_{p}_{q}.pdf")

    #     # 親ディレクトリが存在しない場合は作成
    #     # os.makedirs(os.path.dirname(save_path), exist_ok=True)
    #     plt.savefig("E-n51-k5_gravity_plot.pdf")



    
    def save_to_json(self,p,q,i,data,file_path):

        results = self.load_or_create_json(file_path)

        results[f"p={p}_q={q}_experiment={i}"] = {
            "data":data
        }
        try:
            with open(file_path, 'w') as json_file:
                json.dump(results, json_file, indent=4, ensure_ascii=False)
            # print(f"JSONファイルとして保存されました: {file_path}")
        except Exception as e:
            print(f"エラーが発生しました: {e}")


    def append_fault_result_to_json(self,p,q,r,i,fault_constraints,save_path):
        # print(f"save_path,{save_path}/QCVRPS")
        
        file_path = f"{save_path}"
        # ファイルを読み込むか新しいデータを作成
        results = self.load_or_create_json(file_path)
        # p, q, r のキーに対応するデータを追加
        results[f"p={p}_q={q}_r={r}_experiment={i}"] = fault_constraints

        # JSONファイルに保存
        with open(file_path, "w") as json_file:
            json.dump(results, json_file, indent=4, ensure_ascii=False)

        # print(f"Results appended to {file_path}")

    def load_or_create_json(self,file_path):
        self.ensure_directory_exists(file_path)
        if os.path.exists(file_path):
            with open(file_path, "r") as json_file:
                return json.load(json_file)
        else:
                    empty_data = {}  # 空の辞書を定義
        with open(file_path, "w") as json_file:  # ファイルを作成して書き込む
            json.dump(empty_data, json_file, indent=4)  # 空のデータを書き込む
        return empty_data  # 空の辞書を返す
    
    def append_results_to_json(self, p, q, r,i, vehicle_data, other_datas, save_path):
    # ファイルを読み込むか新しいデータを作成

        file_path = f"{save_path}"
        results = self.load_or_create_json(file_path)

        # 新しい結果を追加
        results[f"p={p}_q={q}_r={r}_experiment={i}"] = {
            "vehicles": {
                f"vehicle{data['vehicle']}": {
                    "route": str(data["route"]),  # route はリスト形式になっている
                    "distance": data["distance"],
                }
                for data in vehicle_data
            },
            "total_distances": int(other_datas["total_distances"]),  # シンプルにアクセス
            "solve_time":{"total_time": other_datas["total_time"],
                          "execution_time": other_datas["execution_time"],
                          "response_time": other_datas["response_time"]
                          }
        }

        # ファイルに保存
        with open(file_path, "w") as json_file:
            json.dump(results, json_file, indent=4, ensure_ascii=False)
        
        # print(f"Results appended to {file_path}")

    def ensure_directory_exists(self,file_path):
        directory = os.path.dirname(file_path)  # ファイルパスからディレクトリ部分を取得
        if not os.path.exists(directory):  # ディレクトリが存在しない場合
            os.makedirs(directory)  # ディレクトリを作成
            
    def make_QA_df(self, vehicledata,odata,p,q,r): 
    # 空のデータフレームを初期化
        QA_distance_df = pd.DataFrame()
        if vehicledata is None:
            print("No feasible solution found. Skipping this iteration.")
        else:
            for data in vehicledata:
                # vehicle_dataを使用した処理
                data_df = pd.DataFrame(data)
                odata_df = pd.DataFrame(odata)
            # QA_distance_dfにdata_dfを追加
            QA_distance_df = pd.concat([QA_distance_df, data_df], axis=0, ignore_index=True)
            QA_distance_df = pd.concat([QA_distance_df, odata_df], axis=0, ignore_index=True)
        # 現在の日付と時間を取得
        now = datetime.now().strftime("%Y%m%d%H")
        results_dir = f"/home/toshiya1048/QA_CVRP/result/{now}/solve"
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        # パスを指定してCSVファイルとして保存

        path = f"{results_dir}/{p,q,r}_QA_distance_df.csv"
        QA_distance_df.to_csv(path, index=False)
    
   
