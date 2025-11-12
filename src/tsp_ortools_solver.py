# src/tsp_ortools_solver.py
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
import numpy as np

# src/tsp_ortools_solver.py
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

def solve_tsp_ortools(distance_matrix, time_limit_ms=10, seed=42):
    n = len(distance_matrix)
    manager = pywrapcp.RoutingIndexManager(n, 1, 0)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(i, j):
        return int(distance_matrix[manager.IndexToNode(i)][manager.IndexToNode(j)] * 1000)
    transit_idx = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_idx)

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.time_limit.FromMilliseconds(time_limit_ms)
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    params.log_search = False

    # --- 互換化: OR-Tools の版によって random_seed が無い ---
    try:
        params.random_seed = seed                # ある版
    except AttributeError:
        # 代替：存在する場合のみ設定（例: use_random_number_generator 等）
        if hasattr(params, "use_random_number_generator"):
            setattr(params, "use_random_number_generator", True)
        # 乱数種は未設定のままでOK（解は普通に出ます）

    solution = routing.SolveWithParameters(params)

    route = []
    if solution:
        idx = routing.Start(0)
        while not routing.IsEnd(idx):
            route.append(manager.IndexToNode(idx))
            idx = solution.Value(routing.NextVar(idx))
        route.append(0)  # 閉路
    return route
