#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Core experiment driver (hardened serialization + correct logging):
- Always convert NumPy types to native Python for JSON.
- Use latest cluster state (clusters / cluster_demands / centroids) consistently.
- "after" is recomputed from state; if no move occurred, it naturally equals "before" without explicit assignment.
- Skip writing iteration files only if *really* desired; here we DO write when there were moves.
"""

import os
import sys
import json
import time
import argparse
from datetime import timedelta, datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

# Make project root importable if running from scripts/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from amplify import FixstarsClient
except Exception as e:
    raise RuntimeError("Failed to import 'amplify'. Install Fixstars Amplify SDK.") from e

from src.vrpfactory import vrpfactory
from src.knap_divpro import knap_dippro
# from TSP import TSP  # ← QUBO版TSPは使わない
from src.tsp_ortools import solve_tsp_ortools


# ---------- utils ----------
def to_native(o: Any):
    """Convert NumPy scalars/arrays to plain Python for JSON serialization."""
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    return o


def normalize_moved(raw, length: int) -> np.ndarray:
    """
    Normalize 'moved' to a 0/1 mask of length 'length'.
    Accepts either a 0/1 vector or an index list.
    """
    arr = np.array(raw)
    # Case 1: already 0/1 vector of correct length
    if arr.ndim == 1 and arr.size == length and np.isin(arr, [0, 1, 0.0, 1.0]).all():
        return arr.astype(float)
    # Case 2: treat as indices
    mask = np.zeros(length, dtype=float)
    try:
        idx_ = arr.astype(int)
        idx_ = idx_[(idx_ >= 0) & (idx_ < length)]
        mask[idx_] = 1.0
    except Exception:
        pass
    return mask


def compute_route_distance(route: List[int], distance_matrix) -> float:
    """Return total distance for a route (auto-closes if needed)."""
    if not route:
        return 0.0
    total = 0.0
    for i in range(len(route) - 1):
        total += float(distance_matrix[route[i]][route[i + 1]])
    if route[0] != route[-1]:
        total += float(distance_matrix[route[-1]][route[0]])
    return total


# ---------- core ----------
class Core:
    def __init__(self):
        """Fixstars Amplify client config (env var preferred)."""
        self.client = FixstarsClient()
        token = os.environ.get("AMPLIFY_TOKEN")
        if token:
            self.client.token = token
            print("🔑 FixstarsClient token loaded from AMPLIFY_TOKEN.")
        else:
            print("⚠️ AMPLIFY_TOKEN not set. (Using default client config)")

    def main(self):
        ap = argparse.ArgumentParser(
            description="Iterative QA-based CVRP optimizer (robust JSON + corrected before/after logging)"
        )
        ap.add_argument("-j",   help="Path to before_data.json",             type=str, required=True)
        ap.add_argument("-sp",  help="Base output directory (e.g. ./out)",   type=str, required=True)
        ap.add_argument("--t",  help="Annealing time (ms)",                  type=int, default=3000)
        ap.add_argument("-nt",  help="QA solves per swap (num_solve)",       type=int, default=3)
        ap.add_argument("--p",  help="QA parameter p",                       type=float, default=1.0)
        ap.add_argument("--q",  help="QA parameter q",                       type=float, default=1.0)
        ap.add_argument("--max_iter", help="Max iterations",                 type=int, default=50)
        ap.add_argument("--tsp_solver", choices=["ortools", "amplify"], default="ortools",
                        help="TSP solver: use 'ortools' to avoid QUBO")
        ap.add_argument("--tsp_time_limit_ms", type=int, default=2000,
                        help="OR-Tools time limit per cluster (ms)")
        ap.add_argument("--eps", help="(unused, compat)",                    type=float, default=1e-3)
        args = ap.parse_args()

        # === Output dir ===
        before_path = Path(args.j).resolve()
        parent_name = before_path.parent.name
        instance_name = before_path.stem.replace("_before_data", "")
        # インスタンス名が空になるのを防ぐため、念のためチェック
        if not instance_name:
             instance_name = before_path.stem
        timestamp     = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir      = Path(args.sp) / timestamp / f"{instance_name}_before_data"
        save_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n🚀 Start experiment: {instance_name}")
        print(f"📂 Output: {save_dir}")

        # === Load via vrpfactory ===
        VRPfactory = vrpfactory()
        (
            cluster_nums, grax, gray, gra_distances,
            x, y, distances, demands, capacity,
            clusters, clusters_coordx, clusters_coordy, cluster_demands,
            gra_clusters_coordx, gra_clusters_coordy, depo_x, depo_y
        ) = VRPfactory.get_gluster_gravity_info(str(before_path))

        # Fixed settings
        nvehicle = 1
        depo_x, depo_y = depo_x[0], depo_y[0]
        self.client.parameters.timeout = timedelta(milliseconds=args.t)

        # === Initial centroid-level TSP (cluster order) ===
        # 既存の「重心TSP」部分は変更せずAmplify版を利用
        from TSP import TSP
        tsp_over_clusters = TSP(
            self.client, gra_distances, demands, capacity,
            nvehicle, args.nt, cluster_nums, str(save_dir), grax, gray, str(before_path)
        )
        gra_result = tsp_over_clusters.des_TSP(args.p, args.q)

        # perms: Python int list (no NumPy scalars)
        route_with_depot = [int(v) for v in np.array(gra_result["route"]).tolist()]
        perms_list = route_with_depot[1:]  # exclude depot(0)
        perms = [int(v) for v in perms_list]
        print(f"🧭 Initial cluster order: {perms}")

        # Save centroid init (robust types)
        centroid_payload = {
            "instance": instance_name,
            "params": {"p": args.p, "q": args.q, "nt": args.nt, "anneal_ms": args.t},
            "clusters": to_native(np.array(cluster_nums)),
            "centroids": {
                "x": to_native(np.array(grax)),
                "y": to_native(np.array(gray)),
            },
            "route_over_centroids": {
                "with_depot": route_with_depot,
                "without_depot": perms,
            },
            "metrics": {
                "total_time": to_native(gra_result.get("total_time")),
                "execution_time": to_native(gra_result.get("execution_time")),
                "response_time": to_native(gra_result.get("response_time")),
                "total_distances": to_native(gra_result.get("total_distances")),
            },
            "centroid_distance_shape": list(np.array(gra_distances).shape)
        }
        with open(save_dir / "centroid_init.json", "w") as f:
            json.dump(centroid_payload, f, indent=2, default=to_native)
        print(f"💾 Saved: {save_dir / 'centroid_init.json'}")

        # === Iterations ===
        iteration = 0
        while True:
            iteration += 1
            print(f"\n===== Iteration {iteration} =====")
            swap_time_log: List[Dict[str, Any]] = []
            moved_total = 0
            touched_clusters = set()  # この反復で内容が変わった(=TSP解く対象)

            # Adjacency along perms
            for idx, current_cluster_index in enumerate(perms):
                next_cluster_index = perms[(idx + 1) % len(perms)]

                # Remaining capacity of the next cluster (use LATEST demand)
                restcapacity = float(capacity - sum(cluster_demands[next_cluster_index]))

                # Skip if no capacity in next cluster
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
                        "n_city":        int(len(clusters[current_cluster_index])),
                        "skipped":       True,
                        "skip_reason":   "no_remaining_capacity_in_next_cluster",
                        "sum_dist_current_before": None,
                        "sum_dist_current_after":  None,
                    }
                    swap_time_log.append(record)
                    print(f"[swap {idx}] ⏭️ skip (next={next_cluster_index})")
                    continue

                # --- Start timing ---
                t_block_start = time.perf_counter()

                # Latest state snapshot for current cluster
                cur_ids = clusters[current_cluster_index]               # city ids
                cur_xs  = clusters_coordx[current_cluster_index]
                cur_ys  = clusters_coordy[current_cluster_index]
                cur_cx  = gra_clusters_coordx[current_cluster_index]
                cur_cy  = gra_clusters_coordy[current_cluster_index]

                # Distance vectors to "current" centroid (before) and "next" centroid
                dist_vec_before  = vrpfactory.make_distances(cur_xs, cur_ys, cur_cx, cur_cy)
                sum_before = float(np.sum(np.asarray(dist_vec_before, dtype=float).flatten()))

                next_cx = gra_clusters_coordx[next_cluster_index]
                next_cy = gra_clusters_coordy[next_cluster_index]
                dist_vec_to_next = vrpfactory.make_distances(cur_xs, cur_ys, next_cx, next_cy)

                # Latest demands to pass into knap
                demand_current = cluster_demands[current_cluster_index]

                # === QA execution ===
                proccesor = knap_dippro(
                    self.client,
                    dist_vec_before,        # current centroid distances (latest)
                    dist_vec_to_next,       # next centroid distances (latest)
                    demand_current,         # latest demand of current cluster
                    restcapacity,
                    capacity,
                    args.nt,
                    cur_ids,                # latest city id list
                    str(before_path)
                )
                pro_result = proccesor.QA_processors()

                # Normalize moved -> 0/1 mask
                moved_arr = normalize_moved(pro_result.get("route", []), len(cur_ids))
                did_move = bool(moved_arr.sum() > 0.5)
                if did_move:
                    moved_total += 1
                    # from側とto側の2クラスタは内容が変わるので記録
                    touched_clusters.add(int(current_cluster_index))
                    touched_clusters.add(int(next_cluster_index))

                # Timing
                t_block_end = time.perf_counter()
                block_ms = float((t_block_end - t_block_start) * 1000.0)
                qa_total_time = float(pro_result.get("total_time", 0.0))  # seconds expected
                qa_ms = max(0.0, qa_total_time * 1000.0)
                move_ms = max(0.0, block_ms - qa_ms)
                n_city = int(pro_result.get("n_city", len(cur_ids)))

                # Apply swap only if something moved
                if did_move:
                    (
                        clusters, clusters_coordx, clusters_coordy, cluster_demands,
                        gra_clusters_coordx, gra_clusters_coordy, distances
                    ) = VRPfactory.process_swap(
                        moved_arr,
                        clusters, clusters_coordx, clusters_coordy, cluster_demands,
                        gra_clusters_coordx, gra_clusters_coordy,
                        current_cluster_index, next_cluster_index, distances
                    )

                # Recompute "after" from the (possibly) updated latest state.
                # If did_move==False, latest==before-state -> natural equality (no explicit assignment needed).
                cur_after_xs = clusters_coordx[current_cluster_index]
                cur_after_ys = clusters_coordy[current_cluster_index]
                cur_after_cx = gra_clusters_coordx[current_cluster_index]
                cur_after_cy = gra_clusters_coordy[current_cluster_index]
                dist_vec_after = vrpfactory.make_distances(cur_after_xs, cur_after_ys, cur_after_cx, cur_after_cy)
                sum_after = float(np.sum(np.asarray(dist_vec_after, dtype=float).flatten()))

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
                    "skipped":       False,
                    "sum_dist_current_before": float(sum_before),
                    "sum_dist_current_after":  float(sum_after),
                }
                swap_time_log.append(record)

                print(f"[swap {idx}] move={did_move} | qa={qa_ms:.1f}ms | total={block_ms:.1f}ms | "
                      f"before={sum_before:.3f} | after={sum_after:.3f}")

            # ---- ここから挙動変更部分 ----
            # まず swap タイミングのログを保存（改善の有無に関わらず）
            swap_log_path = save_dir / f"iteration_{iteration}_swap_timings.json"
            with open(swap_log_path, "w") as f:
                json.dump(swap_time_log, f, indent=2, default=to_native)
            print(f"🕒 Saved swap details: {swap_log_path}")

            # 都市交換が発生しなかった場合でも、このイテレーションで
            # 一度 TSP を解いてから終了するようにする
            if moved_total == 0:
                print("🟡 No city moved in this iteration → solve final TSP over all clusters and stop.")
                # 全クラスタを TSP の対象にする
                touched_clusters = set(range(len(clusters)))
            # ---- 変更ここまで ----

            # 交換があったクラスタ（または収束時は全クラスタ）で TSP を解く
            total_distance = 0.0
            tsp_routes: List[Dict[str, Any]] = []
            if args.tsp_solver == "ortools":
                for cluster_id in sorted(touched_clusters):
                    coordx = [depo_x] + clusters_coordx[cluster_id]
                    coordy = [depo_y] + clusters_coordy[cluster_id]
                    cluster_distance = vrpfactory.make_cluster_distance_matrix(coordx, coordy)

                    ort = solve_tsp_ortools(cluster_distance, time_limit_ms=args.tsp_time_limit_ms)
                    solver_status = ""
                    if isinstance(ort, dict):
                        route = ort.get("route", [])
                        tot = ort.get("total_distance")
                        solver_status = ort.get("solver_status", "")
                        solve_time_ms = ort.get("solve_time_ms", None)
                    else:
                        route = ort
                        tot = None
                        solver_status = ""
                        solve_time_ms = None

                    tsp_routes.append({
                        "cluster_id":     int(cluster_id),
                        "route":          route,
                        "total_distance": tot,
                        "solver":         "ortools",
                        "solver_status":  solver_status,
                        "solve_time_ms":  solve_time_ms
                    })
                    if tot is not None:
                        total_distance += tot
            else:
                # 互換のために amplify(TSP) を選べるよう残す（必要なら）
                from TSP import TSP
                for cluster_id in sorted(touched_clusters):
                    coordx = [depo_x] + clusters_coordx[cluster_id]
                    coordy = [depo_y] + clusters_coordy[cluster_id]
                    cluster_demand = [0] + cluster_demands[cluster_id]
                    city_list = [0] + clusters[cluster_id]
                    cluster_distance = vrpfactory.make_cluster_distance_matrix(coordx, coordy)
                    tsp_solver = TSP(self.client, cluster_distance, cluster_demand, capacity,
                                     1, args.nt, city_list, str(save_dir), coordx, coordy, str(before_path))
                    result = tsp_solver.solve_TSP(args.p, args.q)
                    dist_val = float(result.get("total_distances", 0.0))
                    tsp_routes.append({
                        "cluster_id":     int(cluster_id),
                        "route":          to_native(result.get("route")),
                        "total_distance": dist_val,
                        "solver":         "amplify",
                    })
                    total_distance += dist_val

            print(f"📏 Total distance (touched clusters only) after iteration {iteration}: {total_distance:.6f}")
            iteration_path = save_dir / f"iteration_{iteration}.json"
            with open(iteration_path, "w") as f:
                json.dump(tsp_routes, f, indent=2, default=to_native)
            print(f"💾 Saved: {iteration_path}")

            # 収束 or max_iter で終了
            if moved_total == 0:
                print("✅ No further moves → optimization finished.")
                break

            if iteration >= args.max_iter:
                print("⚠️ Reached max iterations. Stop.")
                break

        print("\n✅ Optimization completed.")
        print(f"📂 Results saved in: {save_dir}")


if __name__ == "__main__":
    Core().main()
