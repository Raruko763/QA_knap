import numpy as np

import math
# from geopy.distance import geodesic
import matplotlib.pyplot as plt
import json

import vrplib


class vrpfactory:
    
    @staticmethod
    def make_gravity( coordx, coordy):
        if len(coordx) == 0 or len(coordy) == 0:
            return 0.0, 0.0
        gravity_x = sum(coordx) / len(coordx)
        gravity_y = sum(coordy) / len(coordy)
        return gravity_x, gravity_y
    
    @staticmethod
    def sort_cluster(cluster, coordx, coordy, demand):
        if len(cluster) == 0:
            return
        sorted_data = sorted(zip(cluster, coordx, coordy, demand), key=lambda x: x[0])
        cluster[:], coordx[:], coordy[:], demand[:] = zip(*sorted_data)

    @staticmethod
    def process_swap(
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
    ):
        # current cluster の情報
        current_cluster = clusters[current_cluster_index]
        current_coordx = clusters_coordx[current_cluster_index]
        current_coordy = clusters_coordy[current_cluster_index]
        current_demands = cluster_demands[current_cluster_index]

        # next cluster の情報
        next_cluster = clusters[next_cluster_index]
        next_coordx = clusters_coordx[next_cluster_index]
        next_coordy = clusters_coordy[next_cluster_index]
        next_demands = cluster_demands[next_cluster_index]

        # 交換する都市のインデックス（逆順にソートして pop 対応）
        move_indices = [idx for idx, flag in enumerate(swap_perms) if flag == 1]
        move_indices.sort(reverse=True)

        for idx in move_indices:
            city = current_cluster[idx]
            city_x = current_coordx[idx]
            city_y = current_coordy[idx]
            city_demand = current_demands[idx]

            # current から削除
            current_cluster.pop(idx)
            current_coordx.pop(idx)
            current_coordy.pop(idx)
            current_demands.pop(idx)

            # next に追加
            next_cluster.append(city)
            next_coordx.append(city_x)
            next_coordy.append(city_y)
            next_demands.append(city_demand)

        # 🔽 ここでソートを実行
        vrpfactory.sort_cluster(current_cluster, current_coordx, current_coordy, current_demands)
        vrpfactory.sort_cluster(next_cluster, next_coordx, next_coordy, next_demands)

        # 重心再計算
        gra_clusters_coordx[current_cluster_index], gra_clusters_coordy[current_cluster_index] = vrpfactory.make_gravity(current_coordx, current_coordy)
        gra_clusters_coordx[next_cluster_index], gra_clusters_coordy[next_cluster_index] = vrpfactory.make_gravity(next_coordx, next_coordy)

        # 距離行列も更新（必要ならここで）
        distances[current_cluster_index] = vrpfactory.make_cluster_distance_matrix(current_coordx, current_coordy)
        distances[next_cluster_index] = vrpfactory.make_cluster_distance_matrix(next_coordx, next_coordy)

        print(f"Processed swap from cluster {current_cluster_index} to {next_cluster_index}.")
        print(f"Moved cities indices: {move_indices}")

        return clusters, clusters_coordx, clusters_coordy, cluster_demands, gra_clusters_coordx, gra_clusters_coordy, distances


    # @staticmethod
    # def process_swap(
    #     swap_perms,
    #     clusters,
    #     clusters_coordx,
    #     clusters_coordy,
    #     cluster_demands,
    #     gra_clusters_coordx,
    #     gra_clusters_coordy,
    #     current_cluster_index,
    #     next_cluster_index,
    #     distances  # ← 距離行列リストを受け取る
    # ):
    #     # current cluster の情報
    #     current_cluster = clusters[current_cluster_index]
    #     current_coordx = clusters_coordx[current_cluster_index]
    #     current_coordy = clusters_coordy[current_cluster_index]
    #     current_demands = cluster_demands[current_cluster_index]

    #     # next cluster の情報
    #     next_cluster = clusters[next_cluster_index]
    #     next_coordx = clusters_coordx[next_cluster_index]
    #     next_coordy = clusters_coordy[next_cluster_index]
    #     next_demands = cluster_demands[next_cluster_index]

    #     # 交換する都市のインデックス（逆順にソートして pop 対応）
    #     move_indices = [idx for idx, flag in enumerate(swap_perms) if flag == 1]
    #     move_indices.sort(reverse=True)

    #     for idx in move_indices:
    #         # 都市データ
    #         city = current_cluster[idx]
    #         city_x = current_coordx[idx]
    #         city_y = current_coordy[idx]
    #         city_demand = current_demands[idx]

    #         # current クラスタから削除
    #         current_cluster.pop(idx)
    #         current_coordx.pop(idx)
    #         current_coordy.pop(idx)
    #         current_demands.pop(idx)

    #         # next クラスタに追加
    #         next_cluster.append(city)
    #         next_coordx.append(city_x)
    #         next_coordy.append(city_y)
    #         next_demands.append(city_demand)

    #     # 重心を再計算
    #     gra_clusters_coordx[current_cluster_index], gra_clusters_coordy[current_cluster_index] = vrpfactory.make_gravity(current_coordx, current_coordy)
    #     gra_clusters_coordx[next_cluster_index], gra_clusters_coordy[next_cluster_index] = vrpfactory.make_gravity(next_coordx, next_coordy)

    #     # 距離行列を再計算して上書き
    #     distances[current_cluster_index] = vrpfactory.make_cluster_distance_matrix(current_coordx, current_coordy)
    #     distances[next_cluster_index] = vrpfactory.make_cluster_distance_matrix(next_coordx, next_coordy)

    #     print(f"Processed swap from cluster {current_cluster_index} to {next_cluster_index}.")
    #     print(f"Moved cities indices: {move_indices}")

    #     return clusters, clusters_coordx, clusters_coordy, cluster_demands, gra_clusters_coordx, gra_clusters_coordy, distances

    def make_distances(x,y,gra_x,gra_y):
        
        distances = []

        for i in range(len(x)):
            dist = np.sqrt((x[i]-gra_x)**2 + (y[i]-gra_y)**2)
            distances.append(dist)
        return distances
    
    @staticmethod

    def make_cluster_distance_matrix(x_list, y_list):
        """
        クラスタ内の距離行列を作成する関数
        :param x_list: 都市の x 座標のリスト
        :param y_list: 都市の y 座標のリスト
        :return: 距離行列（numpy配列）
        """
        n = len(x_list)
        dist_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                dist_matrix[i][j] = vrpfactory.distance(x_list[i], y_list[i], x_list[j], y_list[j])

        return dist_matrix
    def get_gluster_gravity_info(self, file_name):
        with open(file_name, "r") as f:
            data = json.load(f)

        cluster_nums = []
        graxs = []
        grays = []
        all_x_coords = []
        all_y_coords = []
        demands = []
        capacities = []
        cluster_distances_list = []

        clusters = []
        cluster_demands = []
        clusters_coordx = []
        clusters_coordy = []
        gra_clusters_coordx = []
        gra_clusters_coordy = []
        depo_x = 0
        depo_y = 0

        for key, value in data.items():
            if key.startswith("cluster_"):
                cluster_num = int(key.split("_")[1])

                gravity = value.get("gravity", {})
                grax = gravity.get("x")
                gray = gravity.get("y")

                cluster_nums.append(cluster_num)
                graxs.append(grax)
                grays.append(gray)

                coordinates = value.get("coordinates", [])[1:]  # ← デポ以外
                depo_coordinate = value.get("coordinates", [])[:1]  # ← デポ以外
                depo_x = [coordinate.get("x") for coordinate in depo_coordinate]
                depo_y = [coordinate.get("y") for coordinate in depo_coordinate]
                
                x_list = [coord.get("x") for coord in coordinates]
                y_list = [coord.get("y") for coord in coordinates]
                all_x_coords.append(x_list)
                all_y_coords.append(y_list)

                demand = value.get("demand", [])[1:]  # ← デポ以外
                demands.append(demand)

                capacity = value.get("capacity")
                capacities.append(capacity)

                cluster_distance = value.get("cluster_distance", [])[1:]
                cluster_distance = [row[1:] for row in cluster_distance]  # ← デポ以外
                cluster_distances_list.append(cluster_distance)

                cities = value.get("cities", [])[1:]  # ← デポ以外
                clusters.append(cities)
                cluster_demands.append(demand)
                clusters_coordx.append(x_list)
                clusters_coordy.append(y_list)
                gra_clusters_coordx.append(grax)
                gra_clusters_coordy.append(gray)

        # 重心間距離行列作成
        n = len(cluster_nums)
        gra_distances = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(n):
                if i != j:
                    distance_value = self.distance(graxs[i], grays[i], graxs[j], grays[j])
                    gra_distances[i][j] = distance_value

        return (
            cluster_nums, graxs, grays, gra_distances,
            all_x_coords, all_y_coords, cluster_distances_list,
            demands, capacity,
            clusters, clusters_coordx, clusters_coordy, cluster_demands,
            gra_clusters_coordx, gra_clusters_coordy,
            depo_x,depo_y
        )



    def cluster_data(file_name):

        #jsonファイルからデータを読み込む
        f = open(file_name,"r")
        data = json.load(f)
        
        cities = data["cities"]
        distance_matrix = data["cluster_distance"]
        demand = data["demand"][1:]
        demand = np.array(demand)
        # print(type(demand))
        capacity = data["capacity"]

        coordinates = data["coordinates"]
        x_coord = [coord["x"] for coord in coordinates]  # x座標のリスト
        y_coord = [coord["y"] for coord in coordinates] 
        nvheicle = data["required_trucks"]

        return cities,distance_matrix,demand,capacity,nvheicle,x_coord,y_coord


    def makedata(file_name):
        nodes = vrplib.read_instance(file_name)

        distance_matrix = nodes['edge_weight']
        demand = nodes['demand'][1:]          # depot除外
        capacity = nodes['capacity']

        coord = nodes.get('node_coord')
        if coord is None:
            raise ValueError("This .vrp has no node_coord. Sweep needs coordinates.")

        # coord は depot を含む (0..n) が普通
        x_coord = [float(coord[i][0]) for i in range(len(coord))]
        y_coord = [float(coord[i][1]) for i in range(len(coord))]

        # nvehicle は .vrp の情報から取れないことが多いのでダミー or None にする
        nvehicle = None

        return distance_matrix, np.array(demand, dtype=float), float(capacity), nvehicle, x_coord, y_coord

    
    @staticmethod
    def distance(x1, y1, x2, y2):
        """distance: euclidean distance between (x1,y1) and (x2,y2)"""
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def makecvrp(instance_name):

        args = instance_name
        
        # file_name = "E-n22-k4.vrp"
        file_name = f"{args}"
        # file_name = "E-n51-k5.vrp"
        f = open(file_name)
        data = f.readlines()
        f.close()
        n = int(data[3].split()[-1])
        Q = int(data[5].split()[-1])
        #print("n=", n, "Q=", Q)
        x, y, q =[],[],[]
        pos = {}
        for i, row in enumerate(data[7 : 7 + n]):
            #print(row.split()[2])
            x.append(int(row.split()[1])) 
            y.append(int(row.split()[2]))
            #depotのdemandは省いている
        for i, row in enumerate(data[9 + n : 8 + 2 * n]):
            q.append(int(row.split()[1]))
        m = 3
        # 距離行列の作成 (2次元リストまたはNumPy配列で初期化)
        c = np.zeros((n, n), dtype=float)  # n×nの距離行列をゼロで初期化
        # n個の地点間のユークリッド距離を計算して行列に格納
        for i in range(n):
            for j in range(n):
                if i != j:
                    distance_value = vrpfactory.make_cluster_distance_matrix(x,y)  # ユークリッド距離を計算
                    

        # 結果を確認
        #print(c)# 距離を整数に変換して行列に格納
        distances = np.array(c,dtype=float)
        demand = np.array(q)
        capacity = Q
        nvehicle = m
        # plt.scatter(x,y)
        # plt.savefig("E-n30-k3.pdf")
        
        # plt.clf()
        return distances,demand,capacity,nvehicle,x,y
    def makeGraph(distances,x,y):

        edges = []
        weights = []

        for i in range(len(x)):
            for j in range(len(y)):
                if(i !=j):
                    edges.append((i,j))
                    weights.append(distances[i,j])
        # print("edges", len(edges))
        # print("weights",len(weights))
        return edges, weights
        
    def makerandom(ncity, nvehicle):
        avg_cities_per_vehicle = ncity // nvehicle
        # 乱数シードの固定
        seed = 12345
        rng = np.random.default_rng(seed)
        # 各都市における配送需要（重量）をランダムに決定
        demand = rng.integers(1, 100, size=ncity)
        demand_max = np.max(demand)
        demand_mean = demand.mean()
        # 全体的な需要に合わせ、車両の積載可能量 Q を設定する。
        capacity = int(demand_max) + int(demand_mean) * avg_cities_per_vehicle
        # 座標の取り得る範囲を設定
        lat_range = [35.7014, 35.968]
        lon_range = [139.34, 140.04]
        
    # デポと各都市の座標をランダムに決定
        ind2coord = [
            (
                rng.uniform(lon_range[0], lon_range[1]),
                rng.uniform(lat_range[0], lat_range[1]),
            )
            for i in range(ncity + 1)
        ]
            
        # 2都市間の座標距離行列 D
        distances = np.array(
            [
                [geodesic(coord_i[::-1], coord_j[::-1]).m for coord_j in ind2coord]
                for coord_i in ind2coord
            ]
        )
        return distances,demand,capacity,nvehicle

    def readmaxcutBenchimark():
        with open("G1.txt") as f:

            lines = f.read().splitlines()
            #n,mの設定
            n, m =  lines[0].split()
            #print(n,m)
                
            edges = []
            weights = []
            location = np.zeros((int(m),int(m)))
            for row in lines[1:]:
                u,v = list(row.split()[:2])
                edges.append([int(u),int(v)])
                w = list(row.split()[2:]) 
                weights.append(w)
            #print(weights)
            
            for i,j in edges:
                u,v = int(i),int(j)
                location[u][v] = 1
                location[v][u] = 1
                #print(location[u][v])
            edges = np.array(edges)
            weights = np.array(weights)
            m = int(m)
            print("location",location.shape)
            print("edges",edges.shape)
            print("weights",weights.shape)
            if(sum(sum(location)) == 2*m):
                print("true")
            else:
                print("false",m,sum(sum(location)))
        return weights,edges,location

    def makedistances():
        '''
        QAPの距離行列を作成    
        '''
        matrix = []
        #テストデータrc208Ano.txt 
        with open("rc208Ano.txt") as f:
            lines = f.read().splitlines()
            lines_rstrip = [line.rstrip("'") for line in lines]
            targetlines = lines_rstrip[9:]
            # num_vehicle_line = lines_rstrip[5]
            # num_vehicle, cap_vehicle =  num_vehicle_line.split()[1:]
            
            matrix =  np.array([
                [int(x) for x in row.split()]  # Convert each string element to integer
                for row in targetlines
                ])
            matrix = matrix.T

            id_num = matrix[0]
            x = matrix[1]
            y = matrix[2]
            demand = matrix[3]
            readytime = matrix[4]
            duedate = matrix[5]
            servicetime = matrix[6]
            weight = []
            edges= []
            distances = []
            location = []
            for i in range(len(id_num)):
                for j in range(len(id_num)):
                    
                    edges.append([i,j])
                    dist = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                    distances.append(dist)
                    
                location.append([x[i],y[i]])
                
            
            distances=np.array(distances)
            # for i in range(len(x)):
            #     for j in range(len(x)):
            #         if i!=j:
            #             distances[i,j] = distances[i,j] + servicetime[i]
            time_window = []
            for ready,due in zip(readytime,duedate):
                time_window.append((ready,due))
            #print(edges)
            return distances,demand,demand,servicetime,time_window,location,edges

    def testvrp():
        s=0
        w = [1,1,1,1,1,1,1,1,1,1]
        edge = [
            (s,1),
            (s,2),
            (s,3),
            (s,4),
            (s,5),
            (1,2),
            (3,4),
            (6,7),
            (0,6),
            (0,7)
        ]

        return w,edge
    def testrpb():
        edge = []
        weight = []
        w = [[0,40,60,75,90,200,100,160,80],
            [40,0,65,40,100,50,75,110,100],
            [60,65,0,75,100,100,75,75,75],
            [75,40,75,0,100,50,90,90,150],
            [90,100,100,100,0,100,75,75,100],
            [200,50,100,50,100,0,70,90,75],
            [100,75,75,90,75,70,0,70,100],
            [160,110,75,90,75,90,70,0,100],
            [80,100,75,150,100,75,100,100,0]
            ]


        for i in range(9):
            for j in range(i,9):
                if (i!=j):
                    weight.append(w[i][j])
                    edge.append((i,j))
            
        demand = [0,2,1.5,4.5,3,1.5,4,2.5,3]
        pickup = [0,3,1,2,2,3,4,1.5,3]
        servise_time = [0,1,0.5,1,1,1,1.5,1,0.8]
        time_window = [[0,0],
                    [6,7],
                    [5,7],
                    [1,3],
                    [4,7],
                    [3,5],
                    [2,5],
                    [4,6],
                    [1.5,4]
                    ]

        return edge,weight,demand,pickup,servise_time,time_window,w

    def Testrpb():
        edge = []
        weight = []
        # w = [[0,40,60,75,90,200,100,160,80],
        #     [40,0,65,40,100,50,75,110,100],
        #     [60,65,0,75,100,100,75,75,75],
        #     [75,40,75,0,100,50,90,90,150],
        #     [90,100,100,100,0,100,75,75,100],
        #     [200,50,100,50,100,0,70,90,75],
        #     [100,75,75,90,75,70,0,70,100],
        #     [160,110,75,90,75,90,70,0,100],
        #     [80,100,75,150,100,75,100,100,0]
        #     ]
        w = [[0,1,0,1],
            [1,0,1,0],
            [0,1,0,1],
            [1,0,1,0]
        ]

        for i in range(4):
            for j in range(i,4):
                if (i!=j):
                    weight.append(w[i][j])
                    edge.append((i,j))
        
        demand = [0,2,1.5,4.5,3,1.5,4,2.5,3]
        pickup = [0,3,1,2,2,3,4,1.5,3]
        servise_time = [0,1,0.5,1,1,1,1.5,1,0.8]
        time_window = [[0,0],
                    [6,7],
                    [5,7],
                    [1,3],
                    [4,7],
                    [3,5],
                    [2,5],
                    [4,6],
                    [1.5,4]
                    ]
        edge = [[0,1],
                [0,3],
                [1,2],
                [2,3],
                
                ]
        return edge,weight,demand,pickup,servise_time,time_window,w


