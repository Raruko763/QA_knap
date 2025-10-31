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

    def __init__(self,client,distances_from_mycluster,distances_from_nextcluster,demands,restcapacity_of_nextcluster,num_solve,city,file_path):
        self.client = client
        self.distances_from_mycluster = np.array(distances_from_mycluster,dtype=int)
        self.distances_from_nextcluster = np.array(distances_from_nextcluster,dtype=int)
        self.demands = demands
        self.num_solve = num_solve
        self.restcapacity_of_nextcluster =restcapacity_of_nextcluster
        # self.x = x
        # self.y = y
        self.city = city

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

    def QA_processors(self):
        n_mycluster = len(self.demands)
        gen = VariableGenerator()
        x = gen.array("Binary", shape=(n_mycluster))
        # print("x",x)
        # print("demand",self.demands)
        #都市追加に関する目的関数
        objective = einsum("i,i->", self.distances_from_nextcluster, x) + einsum("i,i->", self.distances_from_mycluster, (1 - x))
        demands = np.array(self.demands)
        weight_sums = einsum("i,i->", demands, x)
        capacity_constraints: ConstraintList = less_equal(weight_sums, self.restcapacity_of_nextcluster, penalty_formulation="Relaxation",label='weight_sum')
        maxdit = max(np.amax(self.distances_from_mycluster),np.amax(self.distances_from_nextcluster))
        capacity_constraints *= maxdit
        model= Model(objective,capacity_constraints)

        result = solve(model,self.client)
        x_values = result.best.values
        # result.filter_solution = False
        # result.best.feasible
        # #解の取得と処理
        # falcons = list(c for c in model.constraints if not c.is_satisfied(result.best.values))
        # print(falcons)
        # 結果の表示
        swap_perms = x.evaluate(x_values)
        total_time= result.total_time.total_seconds()
        execution_time = result.execution_time.total_seconds()
        response_time = result.response_time.total_seconds()
        total_distances =result.best.objective
        return  {
                    "route": swap_perms,
                    "total_time": total_time,
                    "execution_time": execution_time,
                    "response_time": response_time,
                    "total_distances": total_distances,
                    "n_city":n_mycluster
                }
     

    def des_TSP(self,p,q):
        print("Solution is TSP.")
        NUM_CITIES = len(self.city)
        # print("NUM_CITY",NUM_CITIES)
        gen = VariableGenerator()
        x = gen.array("Binary", shape=(NUM_CITIES + 1, NUM_CITIES))
        x[NUM_CITIES, :] = x[0, :]
        # print("QUBO",x)
        objective: Poly = einsum("ij,ni,nj->", self.distances, x[:-1], x[1:])  # type: ignore
        row_constraints = one_hot(x[:-1], axis=1,label='one_trip_constraint')
    # 最後の行を除いた q の各列のうち一つのみが 1 である制約
        col_constraints = one_hot(x[:-1], axis=0,label='one_visit_constraint')
            
        constraints = p*row_constraints + q*col_constraints

        # print("objective",objective)
        # print("rowconstraint",row_constraints)
        # print("colconstraint",col_constraints)
        constraints *= np.amax(self.distances)  # 制約条件の強さを設定
        model = objective + constraints
        for i in range(self.num_solve):

            try:
                result = solve(model, self.client)
                if len(result) == 0:
                    raise RuntimeError("At least one of the constraints is not satisfied.")

                # result.best.objective
                q_values = x.evaluate(result.best.values)
                route = np.where(np.array(q_values) == 1)[1]
                route = list(route)
                print(route)
                self.plot(route,result.best.objective,self.file_path,p,q,i)

            except RuntimeError:
                # print("No feasible solution found.")
                result.filter_solution = False
                result.best.feasible
                #解の取得と処理
                falcons = list(c for c in model.constraints if not c.is_satisfied(result.best.values))
                
                constraints_data = []

                for constraint in falcons:
                    constraints_data.append({
                        "label": constraint.label,
                        "conditional": str(constraint.conditional),  # 必要に応じて文字列化
                        "weight": constraint.weight
                    })
                    self.ensure_directory_exists(self.file_path)
                    self.save_to_json(p,q,i,constraints_data,self.file_path)
                    
        return route

    def plot(self, route, total_distance,savedir,p,q,i):
        # seaborn のスタイル設定
        sns.set_theme(style="whitegrid")
    
    # グラフをプロット
        plt.figure(figsize=(10, 6))
        
        # 経路を描画
        x = self.x
        y = self.y
        x_coords = [self.x[r] for r in route]  
        y_coords = [self.y[r] for r in route]  

        
        plt.plot(x_coords + [x_coords[0]], y_coords + [y_coords[0]], 'b-', marker='o')
        
        # 都市を描画
        plt.scatter(x_coords, y_coords, c='r', zorder=5)
        
        # 都市番号をプロット
        for j in range(len(route[:-1])):
            plt.text(x_coords[j] + 0.1, y_coords[j] + 0.1, str(self.city[route[j]]), fontsize=16, ha='center', va='center')
        
        # タイトルにtotal_distanceを追加
        plt.title(f'TSP Solution - Total Distance: {total_distance:.2f}', fontsize=18)
    
        # 軸ラベルの設定
        plt.xlabel('X')
        plt.ylabel('Y')
        # グラフの表示

        # savedir の親ディレクトリを取得
        parent_dir = os.path.dirname(savedir)

        match = re.search(r"cluster_\d+", self.clu_path)
        if match:
            # 最終的な保存先のパスを作成
            save_path = os.path.join(parent_dir, f"route_{match.group()}_{i}_{p}_{q}.pdf")

        # 親ディレクトリが存在しない場合は作成
        # os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig("E-n51-k5_gravity_plot.pdf")



    
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
    
   
