#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
driver_sweep_qubo.py
Flow:
  1) load .vrp (vrplib)
  2) Sweep initial clustering (capacity-feasible) + time_sweep
  3) Build centroid graph + perms (Concorde TSP on centroids)
  4) Run existing QUBO iteration (knap_dippro + process_swap)
  5) Solve final per-cluster TSP (Concorde) for total route length + time_routing

Notes:
- No routing is computed right after Sweep (only clustering + centroid perms).
- All timings are stored as float milliseconds (not int).
"""

import os
import re
import json
import math
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import vrplib

from amplify import FixstarsClient

from src.vrpfactory import vrpfactory
from src.knap_divpro import knap_dippro
from src.tsp_concorde import solve_tsp_concorde


# ---------- utils ----------
def to_native(o: Any):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    return o


def ms(t0: float, t1: float) -> float:
    return (t1 - t0) * 1000.0


def centroid(xs: List[float], ys: List[float]) -> Tuple[float, float]:
    if not xs:
        return 0.0, 0.0
    return float(np.mean(xs)), float(np.mean(ys))


def make_dist_matrix_from_points(xs: List[float], ys: List[float]) -> np.ndarray:
    n = len(xs)
    D = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            D[i, j] = math.sqrt(dx * dx + dy * dy)
    return D


def load_vrp_instance(vrp_path: str) -> Dict[str, Any]:
    """
    Read .vrp via vrplib.
    Assumption: node 0 is depot, customers are 1..n.
    """
    inst = vrplib.read_instance(vrp_path)

    coord = inst.get("node_coord")
    if coord is None:
        raise ValueError("This .vrp has no node_coord. Sweep needs coordinates.")

    demand_all = inst.get("demand")
    if demand_all is None:
        raise ValueError("This .vrp has no demand vector.")
    if len(demand_all) != len(coord):
        raise ValueError("Mismatch: len(demand) != len(node_coord)")

    capacity = inst.get("capacity")
    if capacity is None:
        raise ValueError("This .vrp has no capacity field.")

    depot_x, depot_y = float(coord[0][0]), float(coord[0][1])

    n_customers = len(coord) - 1
    city_ids = list(range(1, n_customers + 1))
    xs = [float(coord[i][0]) for i in range(1, n_customers + 1)]
    ys = [float(coord[i][1]) for i in range(1, n_customers + 1)]
    ds = [float(demand_all[i]) for i in range(1, n_customers + 1)]

    return {
        "depot": (depot_x, depot_y),
        "city_ids": city_ids,
        "xs": xs,
        "ys": ys,
        "demands": ds,
        "capacity": float(capacity),
    }


def sweep_capacity_split(
    city_ids: List[int],
    xs: List[float],
    ys: List[float],
    demands: List[float],
    capacity: float,
    depot_x: float,
    depot_y: float,
    start_angle: float = 0.0,
) -> Dict[str, Any]:
    """
    Classic sweep clustering:
      - sort customers by angle around depot
      - pack sequentially under capacity
    Returns clusters + coords + demands + centroids + centroid distance matrix
    """
    items = []
    for cid, x, y, d in zip(city_ids, xs, ys, demands):
        if d > capacity:
            raise ValueError(f"demand of city {cid} ({d}) exceeds capacity ({capacity})")
        ang = math.atan2(y - depot_y, x - depot_x)
        ang = (ang - start_angle) % (2 * math.pi)
        items.append((int(cid), float(x), float(y), float(d), ang))
    items.sort(key=lambda t: t[4])

    clusters: List[List[int]] = []
    clusters_coordx: List[List[float]] = []
    clusters_coordy: List[List[float]] = []
    cluster_demands: List[List[float]] = []

    cur_ids: List[int] = []
    cur_xs: List[float] = []
    cur_ys: List[float] = []
    cur_ds: List[float] = []
    load = 0.0

    for cid, x, y, d, _ in items:
        if load + d <= capacity + 1e-12:
            cur_ids.append(cid); cur_xs.append(x); cur_ys.append(y); cur_ds.append(d)
            load += d
        else:
            clusters.append(cur_ids)
            clusters_coordx.append(cur_xs)
            clusters_coordy.append(cur_ys)
            cluster_demands.append(cur_ds)

            cur_ids, cur_xs, cur_ys, cur_ds = [cid], [x], [y], [d]
            load = d

    if cur_ids:
        clusters.append(cur_ids)
        clusters_coordx.append(cur_xs)
        clusters_coordy.append(cur_ys)
        cluster_demands.append(cur_ds)

    # centroids
    grax, gray = [], []
    for xs_i, ys_i in zip(clusters_coordx, clusters_coordy):
        cx, cy = centroid(xs_i, ys_i)
        grax.append(cx); gray.append(cy)

    gra_distances = make_dist_matrix_from_points(grax, gray)

    return {
        "cluster_nums": list(range(len(clusters))),
        "clusters": clusters,
        "clusters_coordx": clusters_coordx,
        "clusters_coordy": clusters_coordy,
        "cluster_demands": cluster_demands,
        "gra_clusters_coordx": grax,
        "gra_clusters_coordy": gray,
        "gra_distances": gra_distances,
    }


def build_cluster_distance_matrices(
    depot_x: float, depot_y: float,
    clusters_coordx: List[List[float]],
    clusters_coordy: List[List[float]],
) -> List[np.ndarray]:
    mats = []
    for xs_i, ys_i in zip(clusters_coordx, clusters_coordy):
        coordx = [depot_x] + xs_i
        coordy = [depot_y] + ys_i
        mats.append(vrpfactory.make_cluster_distance_matrix(coordx, coordy))
    return mats


def main():
    ap = argparse.ArgumentParser(description="Sweep init + QUBO iteration + routing driver")
    ap.add_argument("-i", "--input_vrp", required=True, type=str, help="Path to .vrp")
    ap.add_argument("-sp", required=True, type=str, help="Base output directory")
    ap.add_argument("--start_angle", type=float, default=0.0, help="Sweep start angle (rad)")

    # QUBO iteration params (match your core)
    ap.add_argument("--anneal_ms", type=int, default=3000)
    ap.add_argument("--nt", type=int, default=3)
    ap.add_argument("--p", type=float, default=1.0)
    ap.add_argument("--q", type=float, default=1.0)
    ap.add_argument("--lam", type=float, default=0.3)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--stage2_mode", choices=["dist", "dist+ang", "ang"], default="dist+ang")
    ap.add_argument("--max_iter", type=int, default=50)

    # routing (final evaluation only)
    ap.add_argument("--routing", action="store_true", help="Solve final per-cluster TSP (Concorde) to get total distance")
    ap.add_argument("--concorde_bin", type=str, default="", help="Path to concorde binary (or set CONCORDE_BIN)")

    args = ap.parse_args()

    vrp_path = Path(args.input_vrp).resolve()
    instance_name = vrp_path.stem

    base_out = Path(args.sp)
    base_out.mkdir(parents=True, exist_ok=True)
    skipped_log = base_out / "skipped.log"

    t_all0 = time.perf_counter()

    try:
        # ---- output dir ----
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = base_out / timestamp / f"{instance_name}_sweep_qubo"
        save_dir.mkdir(parents=True, exist_ok=True)

        # ---- Amplify client ----
        client = FixstarsClient()
        token = os.environ.get("AMPLIFY_TOKEN")
        if token:
            client.token = token
        client.parameters.timeout = args.anneal_ms  # ms (Amplify expects ms timedelta in other code; keep consistent if needed)

        # ---- load vrp ----
        info = load_vrp_instance(str(vrp_path))
        depot_x, depot_y = info["depot"]
        capacity = float(info["capacity"])

        # ---- (1) sweep init + time ----
        t0 = time.perf_counter()
        sweep_out = sweep_capacity_split(
            city_ids=info["city_ids"],
            xs=info["xs"],
            ys=info["ys"],
            demands=info["demands"],
            capacity=capacity,
            depot_x=depot_x,
            depot_y=depot_y,
            start_angle=float(args.start_angle),
        )
        t1 = time.perf_counter()
        time_sweep_ms = ms(t0, t1)

        cluster_nums = sweep_out["cluster_nums"]
        clusters = sweep_out["clusters"]
        clusters_coordx = sweep_out["clusters_coordx"]
        clusters_coordy = sweep_out["clusters_coordy"]
        cluster_demands = sweep_out["cluster_demands"]
        gra_clusters_coordx = sweep_out["gra_clusters_coordx"]
        gra_clusters_coordy = sweep_out["gra_clusters_coordy"]
        gra_distances = np.array(sweep_out["gra_distances"], dtype=float)

        # initial cluster distance matrices (for later updates)
        distances = build_cluster_distance_matrices(depot_x, depot_y, clusters_coordx, clusters_coordy)

        # save sweep init (NO routing here)
        sweep_init_path = save_dir / "sweep_init.json"
        with sweep_init_path.open("w") as f:
            json.dump({
                "instance": instance_name,
                "input_vrp": str(vrp_path),
                "depot": {"x": depot_x, "y": depot_y},
                "capacity": capacity,
                "start_angle_rad": float(args.start_angle),
                "K": len(clusters),
                "cluster_sizes": [len(c) for c in clusters],
                "cluster_demands_sum": [float(sum(ds)) for ds in cluster_demands],
                "time_sweep_ms": float(time_sweep_ms),
            }, f, indent=2, default=to_native)

        # ---- (2) perms by centroid TSP (Concorde) ----
        # NOTE: this is NOT VRP route evaluation; it's just adjacency order for iteration.
        t2 = time.perf_counter()
        work_dir = save_dir / "centroid_tsp_work"
        cres = solve_tsp_concorde(
            dist_matrix=gra_distances,
            work_dir=str(work_dir),
            concorde_bin=(args.concorde_bin.strip() or None),
        )
        t3 = time.perf_counter()
        time_perms_ms = ms(t2, t3)

        if cres.get("route") is None:
            raise RuntimeError(f"Centroid TSP failed: {cres.get('solver_status')}")

        # route is a permutation of 0..K-1. Use it directly as perms.
        perms = [int(v) for v in cres["route"]]
        centroid_perms_path = save_dir / "centroid_perms.json"
        with centroid_perms_path.open("w") as f:
            json.dump({
                "instance": instance_name,
                "K": len(clusters),
                "perms": perms,
                "time_perms_ms": float(time_perms_ms),
                "centroid_tsp_status": cres.get("solver_status"),
                "centroid_tsp_cost": cres.get("total_distance"),
            }, f, indent=2, default=to_native)

        # ---- (3) iteration (reuse your core logic style) ----
        t_iter0 = time.perf_counter()

        iteration = 0
        while True:
            iteration += 1
            moved_total = 0
            swap_time_log: List[Dict[str, Any]] = []

            for idx, current_cluster_index in enumerate(perms):
                next_cluster_index = perms[(idx + 1) % len(perms)]

                restcapacity = float(capacity - sum(cluster_demands[next_cluster_index]))
                if restcapacity <= 0:
                    swap_time_log.append({
                        "iteration": iteration,
                        "swap_index": idx,
                        "from_cluster": int(current_cluster_index),
                        "to_cluster": int(next_cluster_index),
                        "skipped": True,
                        "skip_reason": "no_remaining_capacity_in_next_cluster",
                    })
                    continue

                t_block0 = time.perf_counter()

                cur_ids = clusters[current_cluster_index]
                cur_xs = clusters_coordx[current_cluster_index]
                cur_ys = clusters_coordy[current_cluster_index]
                cur_cx = gra_clusters_coordx[current_cluster_index]
                cur_cy = gra_clusters_coordy[current_cluster_index]

                # distance vectors to current centroid and next centroid
                dist_vec_before = vrpfactory.make_distances(cur_xs, cur_ys, cur_cx, cur_cy)
                next_cx = gra_clusters_coordx[next_cluster_index]
                next_cy = gra_clusters_coordy[next_cluster_index]
                dist_vec_to_next = vrpfactory.make_distances(cur_xs, cur_ys, next_cx, next_cy)

                demand_current = cluster_demands[current_cluster_index]

                # QUBO solve (stage2 reassignment)
                proc = knap_dippro(
                    client,
                    dist_vec_before,
                    dist_vec_to_next,
                    demand_current,
                    restcapacity,
                    capacity,
                    args.nt,
                    cur_ids,
                    str(vrp_path),
                    depot_xy=(depot_x, depot_y),
                    cur_xs=cur_xs,
                    cur_ys=cur_ys,
                    next_xs=clusters_coordx[next_cluster_index],
                    next_ys=clusters_coordy[next_cluster_index],
                )
                pro_result = proc.solve_stage2_reassignment(
                    lam=args.lam,
                    alpha=args.alpha,
                    p=args.p,
                    mode=args.stage2_mode,
                )

                moved_raw = pro_result.get("route", [])
                moved_arr = np.array(moved_raw, dtype=float)

                # normalize moved_arr: accept either 0/1 vector or indices
                if not (moved_arr.ndim == 1 and moved_arr.size == len(cur_ids) and np.isin(moved_arr, [0, 1, 0.0, 1.0]).all()):
                    mask = np.zeros(len(cur_ids), dtype=float)
                    try:
                        idxs = np.array(moved_raw, dtype=int)
                        idxs = idxs[(idxs >= 0) & (idxs < len(cur_ids))]
                        mask[idxs] = 1.0
                    except Exception:
                        pass
                    moved_arr = mask

                did_move = bool(moved_arr.sum() > 0.5)
                if did_move:
                    moved_total += 1
                    (
                        clusters, clusters_coordx, clusters_coordy, cluster_demands,
                        gra_clusters_coordx, gra_clusters_coordy, distances
                    ) = vrpfactory.process_swap(
                        moved_arr,
                        clusters, clusters_coordx, clusters_coordy, cluster_demands,
                        gra_clusters_coordx, gra_clusters_coordy,
                        current_cluster_index, next_cluster_index, distances
                    )

                t_block1 = time.perf_counter()

                swap_time_log.append({
                    "iteration": iteration,
                    "swap_index": idx,
                    "from_cluster": int(current_cluster_index),
                    "to_cluster": int(next_cluster_index),
                    "did_move": did_move,
                    "block_ms": float(ms(t_block0, t_block1)),
                    "moved_count": int(moved_arr.sum()),
                })

            # save per-iteration swap log
            with (save_dir / f"iteration_{iteration}_swap.json").open("w") as f:
                json.dump(swap_time_log, f, indent=2, default=to_native)

            if moved_total == 0:
                break
            if iteration >= args.max_iter:
                break

        t_iter1 = time.perf_counter()
        time_iteration_ms = ms(t_iter0, t_iter1)

        # ---- (4) routing (final evaluation only) ----
        routing_payload = None
        time_routing_ms = None
        total_distance = None

        if args.routing:
            t_r0 = time.perf_counter()
            work_dir = save_dir / "concorde_work_final"
            routes = []
            total = 0
            for cluster_id, (xs_i, ys_i, cities_i) in enumerate(zip(clusters_coordx, clusters_coordy, clusters)):
                coordx = [depot_x] + xs_i
                coordy = [depot_y] + ys_i
                D = vrpfactory.make_cluster_distance_matrix(coordx, coordy)
                res = solve_tsp_concorde(D, work_dir=str(work_dir), concorde_bin=(args.concorde_bin.strip() or None))
                if res.get("route") is None:
                    routes.append({
                        "cluster_id": cluster_id,
                        "solver_status": res.get("solver_status"),
                        "total_distance": None,
                    })
                    continue
                route_local = res["route"]
                route_global = [0 if v == 0 else int(cities_i[v - 1]) for v in route_local]
                td = int(res.get("total_distance"))
                total += td
                routes.append({
                    "cluster_id": cluster_id,
                    "solver_status": res.get("solver_status"),
                    "total_distance": td,
                    "route_local": route_local,
                    "route_global": route_global,
                })
            t_r1 = time.perf_counter()
            time_routing_ms = ms(t_r0, t_r1)
            total_distance = int(total)
            routing_payload = routes

            with (save_dir / "final_routes.json").open("w") as f:
                json.dump(routing_payload, f, indent=2, default=to_native)

        # ---- summary ----
        t_all1 = time.perf_counter()
        summary = {
            "instance": instance_name,
            "input_vrp": str(vrp_path),
            "K_init": len(cluster_nums),
            "K_final": len(clusters),
            "time_sweep_ms": float(time_sweep_ms),
            "time_perms_ms": float(time_perms_ms),
            "time_iteration_ms": float(time_iteration_ms),
            "time_routing_ms": float(time_routing_ms) if time_routing_ms is not None else None,
            "time_total_ms": float(ms(t_all0, t_all1)),
            "params": {
                "anneal_ms": args.anneal_ms,
                "nt": args.nt,
                "p": args.p,
                "q": args.q,
                "lam": args.lam,
                "alpha": args.alpha,
                "stage2_mode": args.stage2_mode,
                "max_iter": args.max_iter,
            },
            "final_total_distance": total_distance,
        }
        with (save_dir / "final_summary.json").open("w") as f:
            json.dump(summary, f, indent=2, default=to_native)

        print(f"\n✅ Done: {instance_name}")
        print(f"📂 Output: {save_dir}")
        print(f"   sweep_ms={time_sweep_ms:.3f}  perms_ms={time_perms_ms:.3f}  iter_ms={time_iteration_ms:.3f}")
        if args.routing:
            print(f"   routing_ms={time_routing_ms:.3f}  total_dist={total_distance}")

    except (ValueError, MemoryError) as e:
        msg = f"SKIP {instance_name}: {type(e).__name__}: {e}"
        print(f"⚠️ {msg}")
        with skipped_log.open("a") as f:
            f.write(msg + "\n")
        return
    except Exception as e:
        msg = f"SKIP {instance_name}: {type(e).__name__}: {e}"
        print(f"⚠️ {msg}")
        with skipped_log.open("a") as f:
            f.write(msg + "\n")
        return


if __name__ == "__main__":
    main()
