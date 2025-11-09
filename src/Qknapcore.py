#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Core experiment driver (修正版):
- else節を削除（did_move=False時もbefore==afterが自然に成り立つ前提）
- 最新の cluster_demands / clusters を使用して restcapacity と都市IDを算出
"""
import os
import sys
import json
import time
import argparse
from datetime import timedelta, datetime
from pathlib import Path
from typing import Any
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from amplify import FixstarsClient
from src.vrpfactory import vrpfactory
from src.knap_divpro import knap_dippro
from TSP import TSP


def to_native(o: Any):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    return o


def normalize_moved(raw, length: int) -> np.ndarray:
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
        self.client = FixstarsClient()
        token = os.environ.get("AMPLIFY_TOKEN")
        if token:
            self.client.token = token
            print("🔑 FixstarsClient token loaded from AMPLIFY_TOKEN.")
        else:
            print("⚠️ AMPLIFY_TOKEN not set.")

    def main(self):
        ap = argparse.ArgumentParser(description="Iterative QA-based CVRP optimizer (fixed distance logging)")
        ap.add_argument("-j", help="Path to before_data.json", type=str, required=True)
        ap.add_argument("-sp", help="Base output directory (e.g. ./out)", type=str, required=True)
        ap.add_argument("--t", help="Annealing time (ms)", type=int, default=3000)
        ap.add_argument("-nt", help="QA solves per swap", type=int, default=3)
        ap.add_argument("--p", help="QA parameter p", type=float, default=1.0)
        ap.add_argument("--q", help="QA parameter q", type=float, default=1.0)
        ap.add_argument("--max_iter", help="Max iterations", type=int, default=50)
        args = ap.parse_args()

        before_path = Path(args.j).resolve()
        parent_name = before_path.parent.name
        instance_name = parent_name.replace("_before_data", "") or before_path.stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = Path(args.sp) / timestamp / f"{instance_name}_before_data"
        save_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n🚀 Start experiment: {instance_name}")
        print(f"📂 Output: {save_dir}")

        VRPfactory = vrpfactory()
        (
            cluster_nums, grax, gray, gra_distances,
            x, y, distances, demands, capacity,
            clusters, clusters_coordx, clusters_coordy, cluster_demands,
            gra_clusters_coordx, gra_clusters_coordy, depo_x, depo_y
        ) = VRPfactory.get_gluster_gravity_info(str(before_path))

        depo_x, depo_y = depo_x[0], depo_y[0]
        self.client.parameters.timeout = timedelta(milliseconds=args.t)

        tsp_over_clusters = TSP(
            self.client, gra_distances, demands, capacity,
            1, args.nt, cluster_nums, str(save_dir), grax, gray, str(before_path)
        )
        gra_result = tsp_over_clusters.des_TSP(args.p, args.q)
        perms_list = gra_result["route"][1:]
        perms = [int(v) for v in np.array(perms_list).tolist()]

        print(f"🧭 Initial cluster order: {perms}")

        centroid_payload = {
            "instance": instance_name,
            "params": {"p": args.p, "q": args.q, "nt": args.nt, "anneal_ms": args.t},
            "centroids": {"x": to_native(np.array(grax)), "y": to_native(np.array(gray))},
            "route_over_centroids": {"with_depot": gra_result["route"], "without_depot": perms},
        }
        with open(save_dir / "centroid_init.json", "w") as f:
            json.dump(centroid_payload, f, indent=2)

        iteration = 0
        while True:
            iteration += 1
            print(f"\n===== Iteration {iteration} =====")
            swap_time_log = []
            moved_total = 0

            for idx, current_cluster_index in enumerate(perms):
                next_cluster_index = perms[(idx + 1) % len(perms)]

                # 残容量（最新の需要で）
                restcapacity = float(capacity - sum(cluster_demands[next_cluster_index]))

                # スキップ（容量なし）
                if restcapacity <= 0:
                    record = {
                        "iteration": int(iteration),
                        "swap_index": int(idx),
                        "from_cluster": int(current_cluster_index),
                        "to_cluster": int(next_cluster_index),
                        "skipped": True,
                        "sum_dist_current_before": None,
                        "sum_dist_current_after": None
                    }
                    swap_time_log.append(record)
                    continue

                t_block_start = time.perf_counter()

                cur_before_xs = clusters_coordx[current_cluster_index]
                cur_before_ys = clusters_coordy[current_cluster_index]
                cur_before_cx = gra_clusters_coordx[current_cluster_index]
                cur_before_cy = gra_clusters_coordy[current_cluster_index]

                dist_vec_before = vrpfactory.make_distances(cur_before_xs, cur_before_ys, cur_before_cx, cur_before_cy)
                sum_dist_current_before = float(np.sum(np.asarray(dist_vec_before).flatten()))

                next_cx = gra_clusters_coordx[next_cluster_index]
                next_cy = gra_clusters_coordy[next_cluster_index]
                dist_vec_to_next = vrpfactory.make_distances(cur_before_xs, cur_before_ys, next_cx, next_cy)

                # knap_dippro 呼び出し直前で都市IDと需要を取得
                city_ids_current = clusters[current_cluster_index]        # 都市ID
                demand_current   = cluster_demands[current_cluster_index] # 最新の需要

                proccesor = knap_dippro(
                    self.client,
                    dist_vec_before,
                    dist_vec_to_next,
                    demand_current,
                    restcapacity,
                    capacity,
                    args.nt,
                    city_ids_current,
                    str(before_path)
                )
                pro_result = proccesor.QA_processors()

                moved_arr = normalize_moved(pro_result.get("route", []), len(cur_before_xs))
                did_move = bool(moved_arr.sum() > 0.5)
                if did_move:
                    moved_total += 1
                    (
                        clusters, clusters_coordx, clusters_coordy, cluster_demands,
                        gra_clusters_coordx, gra_clusters_coordy, distances
                    ) = VRPfactory.process_swap(
                        moved_arr,
                        clusters, clusters_coordx, clusters_coordy, cluster_demands,
                        gra_clusters_coordx, gra_clusters_coordy,
                        current_cluster_index, next_cluster_index, distances
                    )

                t_block_end = time.perf_counter()
                qa_ms = pro_result.get("total_time", 0.0) * 1000.0
                block_ms = (t_block_end - t_block_start) * 1000.0
                move_ms = max(block_ms - qa_ms, 0.0)

                cur_after_xs = clusters_coordx[current_cluster_index]
                cur_after_ys = clusters_coordy[current_cluster_index]
                cur_after_cx = gra_clusters_coordx[current_cluster_index]
                cur_after_cy = gra_clusters_coordy[current_cluster_index]
                dist_vec_after = vrpfactory.make_distances(cur_after_xs, cur_after_ys, cur_after_cx, cur_after_cy)
                sum_dist_current_after = float(np.sum(np.asarray(dist_vec_after).flatten()))

                record = {
                    "iteration": int(iteration),
                    "swap_index": int(idx),
                    "from_cluster": int(current_cluster_index),
                    "to_cluster": int(next_cluster_index),
                    "qa_ms": float(qa_ms),
                    "move_ms": float(move_ms),
                    "block_ms": float(block_ms),
                    "moved_indices": to_native(moved_arr),
                    "skipped": False,
                    "sum_dist_current_before": sum_dist_current_before,
                    "sum_dist_current_after": sum_dist_current_after
                }
                swap_time_log.append(record)

            if moved_total == 0:
                print("🟡 No city moved → stop optimization.")
                break

            with open(save_dir / f"iteration_{iteration}_swap_timings.json", "w") as f:
                json.dump(swap_time_log, f, indent=2)

        print("\n✅ Optimization completed.")
        print(f"📂 Results saved in: {save_dir}")


if __name__ == "__main__":
    Core().main()
