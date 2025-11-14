from ortools.constraint_solver import pywrapcp, routing_enums_pb2
import time

def solve_tsp_ortools(distance_matrix, time_limit_ms=10000):
    """Solve TSP with OR-Tools and return route + distance + timing."""
    n = len(distance_matrix)
    if n == 0:
        return {"route": [], "total_distance": 0.0, "solver_status": "EMPTY"}

    manager = pywrapcp.RoutingIndexManager(n, 1, 0)
    routing = pywrapcp.RoutingModel(manager)

    def dist_callback(from_index, to_index):
        f, t = manager.IndexToNode(from_index), manager.IndexToNode(to_index)
        return int(distance_matrix[f][t])

    transit_callback_index = routing.RegisterTransitCallback(dist_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    search_params = pywrapcp.DefaultRoutingSearchParameters()

    # 初期解を強く（ユークリッド/対称距離なら有効）
    search_params.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.CHRISTOFIDES
    )

    # 近傍探索を強化（デフォルトだと弱い）
    search_params.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )

    # 主要オペレータを明示的にON（Lin-Kernighan + 2-opt/Or-opt）
    search_params.local_search_operators.use_lin_kernighan = pywrapcp.BOOL_TRUE
    search_params.local_search_operators.use_two_opt       = pywrapcp.BOOL_TRUE
    search_params.local_search_operators.use_or_opt  
    search_params.time_limit.seconds = time_limit_ms // 1000
    search_params.time_limit.seconds = int(time_limit_ms / 1000)

    start_time = time.perf_counter()
    solution = routing.SolveWithParameters(search_params)
    end_time = time.perf_counter()

    if not solution:
        return {
            "route": [],
            "total_distance": None,
            "solver_status": "NO_SOLUTION",
            "solve_time_ms": (end_time - start_time) * 1000.0
        }

    route, total_distance = [], 0
    index = routing.Start(0)
    while not routing.IsEnd(index):
        route.append(manager.IndexToNode(index))
        next_index = solution.Value(routing.NextVar(index))
        total_distance += routing.GetArcCostForVehicle(index, next_index, 0)
        index = next_index
    route.append(manager.IndexToNode(index))

    return {
        "route": route,
        "total_distance": float(total_distance),
        "solver_status": "SUCCESS",
        "solve_time_ms": (end_time - start_time) * 1000.0
    }
