#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Core experiment driver (fixed):
- Uses *latest* cluster centroids for BOTH "before" and "after" measurements.
- When no move occurs in a swap, sum_dist_current_after == sum_dist_current_before (ratio 1.0).
- Skips writing the last iteration files if *no* swap moved any city in that iteration.
- QA inputs compare current cluster vs next cluster centroids (latest state).
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
    raise RuntimeError("Failed to import 'amplify'. Make sure Fixstars Amplify SDK is installed.") from e

from src.vrpfactory import vrpfactory
from src.knap_divpro import knap_dippro
from TSP import TSP


def to_native(o: Any):
    """Convert to plain Python types for JSON serialization."""
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    return o


def normalize_moved(raw, length: int) -> np.ndarray:
    """
    Normalize moved indices to a 0/1 vector of length 'length'.
    Accepts either a 0/1 vector or a list of indices.
    """
    arr = np.array(raw, dtype=float)
    if arr.ndim == 1 and arr.size == length and np.isin(arr, [0.0, 1.0]).all():
        return arr
    mask = np.zeros(length, dtype=float)
    try:
        idx_ = np.array(raw, dtype=int)
        idx_ = idx_[(idx_ >= 0) & (idx_ < length)]
        mask[idx_] = 1.0
    except Exception:
        pass
    return mask


class Core:
    def __init__(self):
        """Fixstars Amplify client config using environment variable if present."""
        self.client = FixstarsClient()
        token = os.environ.get("AMPLIFY_TOKEN")
        if token:
            self.client.token = token
            print("🔑 FixstarsClient token loaded from AMPLIFY_TOKEN.")
        else:
            print("⚠️ AMPLIFY_TOKEN not set. Set it via environment for authenticated runs.")
        # (Other client parameters can be configured here if needed.)

    def main(self):
        ap = argparse.ArgumentParser(
            description="Iterative QA-based CVRP optimizer (fixed logging of before/after distances)"
        )
        ap.add_argument("-j",   help="Path to before_data.json",             type=str, required=True)
        ap.add_argument("-sp",  help="Base output directory (e.g. ./out)",   type=str, required=True)
        ap.add_argument("--t",  help="Annealing time (ms)",                  type=int, default=3000)
        ap.add_argument("-nt",  help="QA solves per swap (num_solve)",       type=int, default=3)
        ap.add_argument("--p",  help="QA parameter p",                       type=float, default=1.0)
        ap.add_argument("--q",  help="QA parameter q",                       type=float, default=1.0)
        ap.add_argument("--max_iter", help="Max iterations (safety cap)",    type=int, default=50)
        ap.add_argument("--eps", help="(unused, kept for compat)",           type=float, default=1e-3)
        args = ap.parse_args()

        # === Output directory structure ===
        before_path = Path(args.j).resolve()
        # Instance name from parent folder (e.g., E-n101-k14_before_data -> E-n101-k14)
        parent_name = before_path.parent.name
        instance_name = parent_name.replace("_before_data", "") or before_path.stem
        timestamp     = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir      = Path(args.sp) / timestamp / f"{instance_name}_before_data"
        save_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n🚀 Start experiment: {instance_name}")
        print(f"📂 Output: {save_dir}")

        # === Load before_data.json via vrpfactory ===
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

        # === Initial centroid-level TSP (for cluster order) ===
        tsp_over_clusters = TSP(
            self.client, gra_distances, demands, capacity,
            nvehicle, args.nt, cluster_nums, str(save_dir), grax, gray, str(before_path)
        )
        gra_result = tsp_over_clusters.des_TSP(args.p, args.q)
        perms_list = gra_result["route"][1:]  # exclude depot(0)
        perms = [int(v) for v in np.array(perms_list).tolist()]
        print(f"🧭 Initial cluster order: {perms}")

        # Save centroid init
        centroid_payload = {
            "instance": instance_name,
            "params": {"p": args.p, "q": args.q, "nt": args.nt, "anneal_ms": args.t},
            "clusters": to_native(np.array(cluster_nums)),
            "centroids": {"x": to_native(np.array(grax)), "y": to_native(np.array(gray))},
            "route_over_centroids": {
                "with_depot": to_native(np.array(gra_result["route"])),
                "without_depot": perms,
            },
            "metrics": {
                "total_time": gra_result.get("total_time"),
                "execution_time": gra_result.get("execution_time"),
                "response_time": gra_result.get("response_time"),
                "total_distances": gra_result.get("total_distances"),
            },
            "centroid_distance_shape": list(np.array(gra_distances).shape)
        }
        with open(save_dir / "centroid_init.json", "w") as f:
            json.dump(centroid_payload, f, indent=2)
        print(f"💾 Saved: {save_dir / 'centroid_init.json'}")

        # === Iterative optimization ===
        iteration = 0
        while True:
            iteration += 1
            print(f"\n===== Iteration {iteration} =====")
            swap_time_log = []
            moved_total = 0  # if any swap moves, this becomes > 0

            # --- Swap across adjacent clusters in perms ---
            for idx, current_cluster_index in enumerate(perms):
                next_cluster_index = perms[(idx + 1) % len(perms)]

                # Remaining capacity of the next cluster
                restcapacity = float(capacity - sum(demands[next_cluster_index]))

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
                        "n_city":        int(len(x[current_cluster_index])),
                        "skipped":       True,
                        "skip_reason":   "no_remaining_capacity_in_next_cluster",
                        "sum_dist_current_before": None,
                        "sum_dist_current_after":  None,
                    }
                    swap_time_log.append(record)
                    print(f"[swap {idx}] ⏭️ skip (next cluster {next_cluster_index} has no remaining capacity)")
                    continue

                # --- Start timing for this swap ---
                t_block_start = time.perf_counter()

                # Snapshot current cluster (latest state)
                cur_before_xs = clusters_coordx[current_cluster_index]
                cur_before_ys = clusters_coordy[current_cluster_index]
                cur_before_cx = gra_clusters_coordx[current_cluster_index]
                cur_before_cy = gra_clusters_coordy[current_cluster_index]

                # Before: sum of distances to *latest* centroid of current cluster
                dist_vec_before = vrpfactory.make_distances(cur_before_xs, cur_before_ys, cur_before_cx, cur_before_cy)
                sum_dist_current_before = float(np.sum(np.asarray(dist_vec_before, dtype=float).flatten()))

                # To-next vector uses *latest* centroid of next cluster
                next_cx = gra_clusters_coordx[next_cluster_index]
                next_cy = gra_clusters_coordy[next_cluster_index]
                dist_vec_to_next = vrpfactory.make_distances(cur_before_xs, cur_before_ys, next_cx, next_cy)

                # === QA execution ===
                proccesor = knap_dippro(
                    self.client,
                    dist_vec_before,        # from (latest)
                    dist_vec_to_next,       # to   (latest)
                    demands[current_cluster_index],
                    restcapacity,
                    capacity,
                    args.nt,
                    cur_before_xs,          # your implementation’s city-id list
                    str(before_path)
                )
                pro_result = proccesor.QA_processors()

                # Normalize moved to 0/1 vector
                moved_arr = normalize_moved(pro_result.get("route", []), len(cur_before_xs))
                did_move = bool(moved_arr.sum() > 0.5)
                if did_move:
                    moved_total += 1

                # Timing
                t_block_end = time.perf_counter()
                block_ms = float((t_block_end - t_block_start) * 1000.0)
                qa_total_time = float(pro_result.get("total_time", 0.0))  # seconds expected
                qa_ms = max(0.0, qa_total_time * 1000.0)
                move_ms = max(0.0, block_ms - qa_ms)
                n_city = int(pro_result.get("n_city", len(cur_before_xs)))

                # Apply swap to state only if something moved
                if did_move:
                    (
                        clusters, clusters_coordx, clusters_coordy, cluster_demands,
                        gra_clusters_coordx, gra_clusters_coordy, distances
                    ) = VRPfactory.process_swap(
                        moved_arr,  # 0/1 vector acceptable if your process_swap supports it; otherwise adapt.
                        clusters, clusters_coordx, clusters_coordy, cluster_demands,
                        gra_clusters_coordx, gra_clusters_coordy,
                        current_cluster_index, next_cluster_index, distances
                    )

                # After: recompute using *latest* state; if no move, equals before
                if did_move:
                    cur_after_xs = clusters_coordx[current_cluster_index]
                    cur_after_ys = clusters_coordy[current_cluster_index]
                    cur_after_cx = gra_clusters_coordx[current_cluster_index]
                    cur_after_cy = gra_clusters_coordy[current_cluster_index]
                    dist_vec_after = vrpfactory.make_distances(cur_after_xs, cur_after_ys, cur_after_cx, cur_after_cy)
                    sum_dist_current_after = float(np.sum(np.asarray(dist_vec_after, dtype=float).flatten()))
                else:
                    sum_dist_current_after = sum_dist_current_before

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
                    "sum_dist_current_before": float(sum_dist_current_before),
                    "sum_dist_current_after":  float(sum_dist_current_after),
                }
                swap_time_log.append(record)

                print(f"[swap {idx}] move={did_move} | qa={qa_ms:.1f}ms | total={block_ms:.1f}ms | "
                      f"before={sum_dist_current_before:.3f} | after={sum_dist_current_after:.3f}")

            # If no swap moved anything, stop without writing iteration files
            if moved_total == 0:
                print("🟡 No city moved in this iteration → stop optimization (no files written for this iter).")
                break

            # Save swap timings for this iteration
            swap_log_path = save_dir / f"iteration_{iteration}_swap_timings.json"
            with open(swap_log_path, "w") as f:
                json.dump(swap_time_log, f, indent=2)
            print(f"🕒 Saved swap details: {swap_log_path}")

            # Re-solve TSP per cluster and save per-iteration summary
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
                    1,               # single vehicle within cluster
                    args.nt,
                    city_list,
                    str(save_dir),
                    coordx,
                    coordy,
                    str(before_path)
                )
                result = tsp_solver.solve_TSP(args.p, args.q)
                tsp_routes.append({
                    "cluster_id":      int(cluster_id),
                    "route":           result["route"],
                    "total_time":      result.get("total_time"),
                    "execution_time":  result.get("execution_time"),
                    "response_time":   result.get("response_time"),
                    "total_distance":  result.get("total_distances"),
                    "overall":         result.get("overall", {}),
                    "runs":            result.get("runs", []),
                })
                try:
                    total_distance += float(result.get("total_distances", 0.0))
                except Exception:
                    pass

            print(f"📏 Total distance after iteration {iteration}: {total_distance:.6f}")
            iteration_path = save_dir / f"iteration_{iteration}.json"
            with open(iteration_path, "w") as f:
                json.dump(tsp_routes, f, indent=2)
            print(f"💾 Saved: {iteration_path}")

            if iteration >= args.max_iter:
                print("⚠️ Reached max iterations. Stop.")
                break

        print("\n✅ Optimization completed.")
        print(f"📂 Results saved in: {save_dir}")


if __name__ == "__main__":
    Core().main()
