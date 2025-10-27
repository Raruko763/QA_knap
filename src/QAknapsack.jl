# using JuMP
# using HiGHS
using PyCall
using Statistics
# using Distances
using JSON  
using DotEnv
using Plots
using Dates


function calc_0_to_distnce(x::Vector{Int},y::Vector{Int})
    depo_x = x[1]
    depo_y = y[1]
    # println("depo_x is $depo_x")
    distances = []
    for i in eachindex(x)
        dis = sqrt((depo_x - x[i])^2 + (depo_y - y[i])^2)
        push!(distances,dis)
    end
        return

end
##  配列の要素をインデックスとする  
function read_instance(instance_dir::String, instance_name::String)
    
    #read cvrp instance
    vrplib = pyimport("vrplib")
    instance_path = joinpath(instance_dir,instance_name)
    
    problem = vrplib.read_instance(instance_path)
    
end

function read_knapsack_instance(instance_dir::String, instance_name::String)

    #file_path
    filepath = joinpath(instance_dir,instance_name)
    
    f = open(filepath,"r") do f
        readlines(f)
    end
    # println(f)
end

function select_farest_customer(problem::Dict{Any, Any},delcities ::Vector{Int})
    
    #条件を設定
    distances = problem["edge_weight"]
    distances = distances[1,:]
    # println(distances)
    # println(length(distances))
    From_0_to_distances = []
    
    for i in delcities
        append!(From_0_to_distances,distances[i])
    end

    # println(length(From_0_to_distances))
    #デポから1番遠い都市を選ぶ

    # println("こここ,$From_0_to_distances")
    # println(From_0_to_distances)
    tmp = 0
    far_city = 0
    
    # for i in delcities
    #     append!(from_0_to_distances,From_0_to_distances[i])
    # end

    for i in eachindex(delcities)
        # println("i,$i")
        if From_0_to_distances[i] > tmp
            
            tmp = From_0_to_distances[i]
            far_city = delcities[i]
        end
    end
    # println("これこれお$(far_city)")

    return far_city
end

function select_most_valuable_customer(problem::Dict{Any, Any},delcities::Vector{Int})
    
    #条件を設定
    depot = problem["depot"]
    demand = problem["demand"]

    #価値が高い都市を選ぶ
    valuable_demand = 0
    valuable_customer = 0

    for i in delcities
        if demand[i] > valuable_demand
            valuable_demand = demand[i]
            valuable_customer = i
        end
    end

    return valuable_customer
end



function solve_knapsack(problem::Dict{Any, Any})
    depot = problem["depot"]
    distances = problem["edge_weight"]
    demand = problem["demand"]
    capacity = problem["capacity"]
    L = length(problem["node_coord"][:,1])
    _x = problem["node_coord"][:,1]
    _y = problem["node_coord"][:,2]
    x = problem["node_coord"][2:L,1]
    y = problem["node_coord"][2:L,2]
    depo = 1
    depox = problem["node_coord"][1,1]
    depoy = problem["node_coord"][1,2]
    delcities = [i for i in 2:length(x) + 1]

    clusters = []
    gra_clusters_coordx = []
    gra_clusters_coordy = []
    clusters_coordx = []
    clusters_coordy = []
    cluster_demands = []

    while !isempty(delcities)
        citie_coordx = [depox]
        citie_coordy = [depoy]
        cluster = [depo]
        cluster_demand = [demand[1]]
        #unique_city = select_most_valuable_customer(problem, delcities)
        unique_city = select_farest_customer(problem, delcities)

        push!(cluster, unique_city)
        push!(citie_coordx, x[unique_city-1])
        push!(citie_coordy, y[unique_city-1])
        push!(cluster_demand, demand[unique_city])
        filter!(del_city -> del_city != unique_city, delcities)
        total_demand = demand[unique_city]

        while total_demand < capacity
            if isempty(delcities)
                break
            end

            next_city, next_citie_coordx, next_citie_coordy = find_nearest_city(delcities, x, y, citie_coordx, citie_coordy)

            if total_demand + demand[next_city] > capacity
                break
            end

            push!(cluster, next_city)
            push!(citie_coordx, next_citie_coordx)
            push!(citie_coordy, next_citie_coordy)
            push!(cluster_demand, demand[next_city])
            filter!(del_city -> del_city != next_city, delcities)
            total_demand += demand[next_city]

            if total_demand > capacity
                println("⚠️ クラスタ内で容量オーバー: $total_demand > $capacity")
            end
        end

        actual_total = sum(cluster_demand)
        if actual_total > capacity
            println("⚠️ クラスタ完成後に容量オーバー確認: $actual_total > $capacity")
        end

        gra_x, gra_y = make_gravity(citie_coordx, citie_coordy)
        append!(clusters, [cluster])
        append!(clusters_coordx, [citie_coordx])
        append!(clusters_coordy, [citie_coordy])
        append!(gra_clusters_coordx, [gra_x])
        append!(gra_clusters_coordy, [gra_y])
        append!(cluster_demands, [cluster_demand])
    end

    return clusters, clusters_coordx, clusters_coordy, gra_clusters_coordx, gra_clusters_coordy, cluster_demands
end

function save_clusters_to_json(
    clusters, clusters_coordx, clusters_coordy,
    cluster_demands,
    gra_clusters_coordx, gra_clusters_coordy,
    time_clustering,
    output_file, instance_name, problem
)


    cluster_data = Dict()
    capacity = problem["capacity"]
    distances = problem["edge_weight"]

    # 時間を保存
    cluster_data["time_clustering"] = time_clustering
    
    
    cluster_data["instance"] = instance_name

    for i in eachindex(clusters)
        cluster = clusters[i]
        cluster_coords_x = clusters_coordx[i]
        cluster_coords_y = clusters_coordy[i]
        cluster_demand = cluster_demands[i]

        # ソート（都市番号順）
        perms = sortperm(cluster)
        cluster = [cluster[idx] for idx in perms]
        cluster_coords_x = [cluster_coords_x[idx] for idx in perms]
        cluster_coords_y = [cluster_coords_y[idx] for idx in perms]
        cluster_demand = [cluster_demand[idx] for idx in perms]

        gravity_x = gra_clusters_coordx[i]
        gravity_y = gra_clusters_coordy[i]
        cluster_distance = create_cluster_distance_matrix(distances, cluster)
        total_demand = sum(cluster_demand)
        is_carriable = total_demand <= capacity

        cluster_info = Dict(
            "capacity" => capacity,
            "cities" => cluster,
            "demand" => cluster_demand,
            "total_demand" => total_demand,
            "is_carriable" => is_carriable,
            "gravity" => Dict("x" => gravity_x, "y" => gravity_y),
            "cluster_distance" => cluster_distance,
            "coordinates" => [Dict("x" => cluster_coords_x[j], "y" => cluster_coords_y[j]) for j in eachindex(cluster)],
            "required_trucks" => 1
        )

        cluster_data["cluster_$i"] = cluster_info
    end

    open(output_file, "w") do file
        JSON.print(file, cluster_data, 4)
    end
end


function create_cluster_distance_matrix(distance_matrix::Matrix{Float64}, cluster_indices::Vector{Int})
    # クラスタに含まれる都市の数
    cluster_size = length(cluster_indices)
    
    # クラスタの距離行列を初期化
    cluster_matrix = zeros(cluster_size, cluster_size)
    
    # クラスタ内の各都市間の距離をコピー
    for i in 1:cluster_size
        for j in 1:cluster_size
            cluster_matrix[i, j] = distance_matrix[cluster_indices[i], cluster_indices[j]]

        end
    end
    
    return cluster_matrix
end




# 距離が近い都市を見つける関数
function find_nearest_city(delcities::Vector{Int},x::Vector{Int},y::Vector{Int},citie_coordx,citie_coordy)
    
    # 現在のクラスタの中間地点を求める
    center_coordx = mean(citie_coordx)  # 各列（xとy座標）の平均
    center_coordy = mean(citie_coordy)  # 各列（xとy座標）の平均
    nearest_city = 0

    # L =length(x)
    # println("length:x is,$L")
    # L = length(delcities)
    # println("length:delcities is,$L")
    min_dis = Inf
    for i in delcities 
        # println("i.$i")

        dis = sqrt((center_coordx - x[i-1])^2 + (center_coordy - y[i-1])^2)

        if(dis < min_dis)
            nearest_city = i
            min_dis = dis
        end
       
    end
    nearest_city = nearest_city
    nearest_coordx = x[nearest_city-1]
    nearest_coordy = y[nearest_city-1]
    
    return nearest_city, nearest_coordx, nearest_coordy
end

# # クラスタ内の都市を最適化する関数（入れ替え改善）
# function optimize_cluster!(problem::Dict{Any,Any},clusters::Vector{Int},clusters_coordx::Vector{Int},clusters_coordy::Vector{Int},gra_clusters_coordx::Vector{Int},gra_clusters_coordx::Vector{Int})
#     # クラスタ内での都市入れ替えを行い、総需用量が容量を超えないように
#     for cluster in clusters

#         for city in cluster
#             Euclidean
#         end

#     end
# end

# クラスタの重心を返す関数
function make_gravity(cluster_x::Vector{Int},cluster_y::Vector{Int})
    
    gra_x = 0
    gra_y = 0
    
    gra_x = mean(cluster_x)
    gra_y = mean(cluster_y)

    return gra_x,gra_y
end

function plot_clusters(clusters, clusters_coordx, clusters_coordy, gra_clusters_coordx, gra_clusters_coordy, folder, filename)
    num_clusters = length(clusters)
    all_colors = distinguishable_colors(num_clusters + 10)
    colors = filter(c -> c != RGB(0,0,0) && c != RGB(0.2,0.2,0.2), all_colors)[1:num_clusters] # 黒と暗い色を除外

    plt = plot(legend = false)  # 凡例を非表示

    for i in 1:num_clusters
        scatter!(plt, clusters_coordx[i], clusters_coordy[i], color=colors[i], markersize=8)
        scatter!(plt, [gra_clusters_coordx[i]], [gra_clusters_coordy[i]], color=colors[i], marker=:diamond, markersize=12)

        for (idx, (x, y)) in enumerate(zip(clusters_coordx[i], clusters_coordy[i]))
            annotate!(x, y, text("$(clusters[i][idx])", 10, :black))
        end
    end

    mkpath(folder)  # フォルダを作成（存在しない場合）
    savefig(plt, joinpath(folder, filename))
end




# 特定のクラスタ内で最も近い都市を探す
function find_nearest_city_in_cluster(x, y, cluster, coordx, coordy, demands)
    min_dist = Inf
    nearest_city_idx = -1
    nearest_city_x = -1
    nearest_city_y = -1
    nearest_city_demand = -1
    for i in 2:length(cluster) # デポを除く
        dist = sqrt((coordx[i] - x)^2 + (coordy[i] - y)^2)
        if dist < min_dist
            min_dist = dist
            nearest_city_idx = i
            nearest_city_x = coordx[i]
            nearest_city_y = coordy[i]
            nearest_city_demand = demands[i]
        end
    end
    return nearest_city_idx, nearest_city_x, nearest_city_y, nearest_city_demand
end

# 交換後の需要が積載容量を超えないか確認
function cluster_demand_with_swap(cluster_demands, swap_idx, new_demand, capacity)
    total_demand = sum(cluster_demands) - cluster_demands[swap_idx] + new_demand
    return total_demand <= capacity
end

# 都市の交換を実行
# function perform_swap!(
#     clusters, clusters_coordx, clusters_coordy, cluster_demands,
#     i, idx1, j, idx2
# )
#     # クラスタiの都市を保存
#     city_i = clusters[i][idx1]
#     city_x_i = clusters_coordx[i][idx1]
#     city_y_i = clusters_coordy[i][idx1]
#     city_demand_i = cluster_demands[i][idx1]

#     # クラスタjの都市を保存
#     city_j = clusters[j][idx2]
#     city_x_j = clusters_coordx[j][idx2]
#     city_y_j = clusters_coordy[j][idx2]
#     city_demand_j = cluster_demands[j][idx2]

#     # 都市の交換
#     clusters[i][idx1] = city_j
#     clusters_coordx[i][idx1] = city_x_j
#     clusters_coordy[i][idx1] = city_y_j
#     cluster_demands[i][idx1] = city_demand_j

#     clusters[j][idx2] = city_i
#     clusters_coordx[j][idx2] = city_x_i
#     clusters_coordy[j][idx2] = city_y_i
#     cluster_demands[j][idx2] = city_demand_i
# end

function perform_swap!(
    clusters, clusters_coordx, clusters_coordy, cluster_demands,
    i, idx1, j, idx2
)
    # クラスタ i の都市を保存
    city_i = clusters[i][idx1]
    city_x_i = clusters_coordx[i][idx1]
    city_y_i = clusters_coordy[i][idx1]
    city_demand_i = cluster_demands[i][idx1]

    # クラスタ j の都市を保存
    city_j = clusters[j][idx2]
    city_x_j = clusters_coordx[j][idx2]
    city_y_j = clusters_coordy[j][idx2]
    city_demand_j = cluster_demands[j][idx2]

    # println("Before swap: Cluster $i: ", clusters[i])
    # println("Before swap: Cluster $j: ", clusters[j])
    # println("Swapping city $(clusters[i][idx1]) in cluster $i with city $(clusters[j][idx2]) in cluster $j")

    # クラスタから都市を削除
    deleteat!(clusters[i], idx1)
    deleteat!(clusters_coordx[i], idx1)
    deleteat!(clusters_coordy[i], idx1)
    deleteat!(cluster_demands[i], idx1)

    deleteat!(clusters[j], idx2)
    deleteat!(clusters_coordx[j], idx2)
    deleteat!(clusters_coordy[j], idx2)
    deleteat!(cluster_demands[j], idx2)

    # クラスタに都市を追加
    push!(clusters[i], city_j)
    push!(clusters_coordx[i], city_x_j)
    push!(clusters_coordy[i], city_y_j)
    push!(cluster_demands[i], city_demand_j)

    push!(clusters[j], city_i)
    push!(clusters_coordx[j], city_x_i)
    push!(clusters_coordy[j], city_y_i)
    push!(cluster_demands[j], city_demand_i)

    # println("After swap: Cluster $i: ", clusters[i])
    # println("After swap: Cluster $j: ", clusters[j])
end



# ファイルパスの設定
DotEnv.config()
instance_dir = "/home/toshiya1048/dev/QA_knap/data/raw/"
save_dir = ARGS[1]
instance_name = ARGS[2]

println("instance_name: $instance_name")

before_data = joinpath(save_dir, string(splitext(instance_name)[1], "_before_data.json"))

# インスタンス読み込み
problem = read_instance(instance_dir, instance_name)
capacity = problem["capacity"]

# ------------------------
# 🕒 クラスタリング時間計測
# ------------------------
println("🚀 クラスタ分割を開始します...")
elapsed_sec = @elapsed begin
    clusters, clusters_coordx, clusters_coordy,
    gra_clusters_coordx, gra_clusters_coordy, clusters_demands =
        solve_knapsack(problem)
end
println("✅ 分割完了: 所要時間 = $(round(elapsed_sec, digits=3)) 秒")


# JSON保存
save_clusters_to_json(
    clusters, clusters_coordx, clusters_coordy,
    clusters_demands,
    gra_clusters_coordx, gra_clusters_coordy,
    elapsed_sec,                 # ← 経過秒数
    before_data, instance_name, problem
)

println("💾 before_data.json 保存完了: $before_data")
