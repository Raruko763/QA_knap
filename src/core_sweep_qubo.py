#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
driver_sweep_qubo.py

Flow:
  1) load .vrp (vrplib)
  2) Sweep initial clustering (capacity-feasible) + time_sweep_ms
  3) Build centroid graph + perms (Concorde TSP on centroids) + time_perms_ms
  4) Evaluate (Concorde) on clusters -> iteration_0.json  (your expected format)
  5) Run QUBO iteration (knap_dippro + process_swap)
     After each iteration, evaluate (Concorde) -> iteration_k.json
  6) Save final_summary.json

Outputs under:
  out/<timestamp>/<instance>_sweep_qubo/
    sweep_init.json
    centroid_perms.json
    iteration_0.json
    iteration_0_meta.json
    iteration_1_swap.json
    iteration_1.json
    iteration_1_meta.json
    ...
    final_summary.json
"""

import os
import sys
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

try:
    from amplify import FixstarsClient
except Exception as e:
    raise RuntimeError("Failed to import 'amplify'. Install Fixstars Amplify SDK.") from e

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


def rotate_route_to_start(route: List[int], start_node: int = 0) -> List[int]:
    """Rotate a Hamiltonian cycle representation so that it starts with start_node."""
    if not route:
        return route
    try:
        k = route.index(start_node)
    except ValueError:
        return route
    return route[k:] + route[:k]


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


def normalize_moved(raw, length: int) -> np.ndarray:
    """
    Normalize 'moved' to a 0/1 mask of length 'length'.
    Accepts either a 0/1 vector or an index list.
    """
    arr = np.array(raw)
    # already 0/1 vector
    if arr.ndim == 1 and arr.size == length and np.isin(arr, [0, 1, 0.0, 1.0]).all():
        return arr.astype(float)
    # treat as indices
    mask = np.zeros(length, dtype=float)
    try:
        idx_ = arr.astype(int)
        idx_ = idx_[(idx_ >= 0) & (idx_ < length)]
        mask[idx_] = 1.0
    except Exception:
        pass
    return mask


# ---------- VRP loader ----------
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


# ---------- Sweep clustering ----------
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


# ---------- Evaluation: Concorde per-cluster TSP ----------
def eval_clusters_concorde(
    depot_x: float,
    depot_y: float,
    clusters: List[List[int]],
    clusters_coordx: List[List[float]],
    clusters_coordy: List[List[float]],
    work_dir: Path,
    concorde_bin: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], int, float]:
    """
    Returns:
      routes_list: list of dicts (your expected iteration_k.json format)
      total_sum: sum of total_distance over SUCCESS clusters
      eval_time_ms: wall time for evaluating all clusters
    """
    work_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    routes: List[Dict[str, Any]] = []
    total_sum = 0

    for cluster_id, (cities_i, xs_i, ys_i) in enumerate(zip(clusters, clusters_coordx, clusters_coordy)):
        coordx = [depot_x] + xs_i
        coordy = [depot_y] + ys_i
        D = vrpfactory.make_cluster_distance_matrix(coordx, coordy)

        cres = solve_tsp_concorde(
            dist_matrix=D,
            work_dir=str(work_dir),
            concorde_bin=concorde_bin,
        )

        status = cres.get("solver_status")
        solve_time_ms = cres.get("solve_time_ms")

        route_local = cres.get("route")
        if route_local is not None:
            route_local = rotate_route_to_start([int(v) for v in route_local], start_node=0)

        td = cres.get("total_distance")
        if isinstance(td, (int, float)) and status == "SUCCESS":
            td_int = int(td)
            total_sum += td_int
        else:
            td_int = None

        route_global = None
        if route_local is not None:
            route_global = []
            for node in route_local:
                if node == 0:
                    route_global.append(0)
                else:
                    # node is local index in [1..len(cities_i)]
                    route_global.append(int(cities_i[node - 1]))

        routes.append({
            "cluster_id": int(cluster_id),
            "route_local": route_local,
            "route_global": route_global,
            "total_distance": td_int,
            "solver": "concorde",
            "solver_status": status,
            "solve_time_ms": solve_time_ms,
            # keep debug/consistency fields if solver provides them
            "optimal_value_stdout": cres.get("optimal_value_stdout"),
            "cost_from_stdout": cres.get("cost_from_stdout"),
            "cost_from_route": cres.get("cost_from_route"),
            "cost_diff": cres.get("cost_diff"),
        })

    t1 = time.perf_counter()
    return routes, int(total_sum), ms(t0, t1)


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="Sweep init + perms + QUBO iteration + Concorde evaluation each iteration")

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

    # Concorde
    ap.add_argument("--concorde_bin", type=str, default="", help="Path to concorde binary (or set CONCORDE_BIN env)")

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

        # IMPORTANT: your other code uses timedelta(milliseconds=...)
        # Here we follow that style:
        from datetime import timedelta
        client.parameters.timeout = timedelta(milliseconds=int(args.anneal_ms))

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

        # save sweep init (NO routing cost here)
        sweep_init_path = save_dir / "sweep_init.json"
        with sweep_init_path.open("w") as f:
            json.dump({
                "instance": instance_name,
                "input_vrp": str(vrp_path),
                "depot": {"x": depot_x, "y": depot_y},
                "capacity": capacity,
                "start_angle_rad": float(args.start_angle),
                "K": int(len(clusters)),
                "cluster_sizes": [int(len(c)) for c in clusters],
                "cluster_demands_sum": [float(sum(ds)) for ds in cluster_demands],
                "time_sweep_ms": float(time_sweep_ms),
            }, f, indent=2, default=to_native)
        print("前処理終了")
        # ---- (2) perms by centroid TSP (Concorde) ----
        # this is for adjacency order, not VRP evaluation
        t2 = time.perf_counter()
        centroid_work_dir = save_dir / "centroid_tsp_work"
        cres = solve_tsp_concorde(
            dist_matrix=gra_distances,
            work_dir=str(centroid_work_dir),
            concorde_bin=(args.concorde_bin.strip() or None),
        )
        t3 = time.perf_counter()
        time_perms_ms = ms(t2, t3)

        if cres.get("route") is None:
            raise RuntimeError(f"Centroid TSP failed: {cres.get('solver_status')}")

        perms = [int(v) for v in cres["route"]]
        perms = rotate_route_to_start(perms, start_node=0)

        centroid_perms_path = save_dir / "centroid_perms.json"
        with centroid_perms_path.open("w") as f:
            json.dump({
                "instance": instance_name,
                "K": int(len(clusters)),
                "perms": perms,
                "time_perms_ms": float(time_perms_ms),
                "centroid_tsp_status": cres.get("solver_status"),
                "centroid_tsp_cost": cres.get("total_distance"),
            }, f, indent=2, default=to_native)

        # ---- (2.5) evaluation right after sweep => iteration_0.json ----
        it0_work = save_dir / "concorde_work_iter_0"
        routes0, total0, eval0_ms = eval_clusters_concorde(
            depot_x=depot_x,
            depot_y=depot_y,
            clusters=clusters,
            clusters_coordx=clusters_coordx,
            clusters_coordy=clusters_coordy,
            work_dir=it0_work,
            concorde_bin=(args.concorde_bin.strip() or None),
        )
        with (save_dir / "iteration_0.json").open("w") as f:
            json.dump(routes0, f, indent=2, default=to_native)
        with (save_dir / "iteration_0_meta.json").open("w") as f:
            json.dump({
                "iteration": 0,
                "total_distance": int(total0),
                "eval_time_ms": float(eval0_ms),
                "K": int(len(clusters)),
            }, f, indent=2, default=to_native)

        # ---- (3) iteration ----
        t_iter0 = time.perf_counter()

        iteration = 0
        while True:
            iteration += 1
            moved_total = 0
            swap_time_log: List[Dict[str, Any]] = []

            for idx, current_cluster_index in enumerate(perms):
                next_cluster_index = perms[(idx + 1) % len(perms)]

                # remaining capacity in next cluster
                restcapacity = float(capacity - sum(cluster_demands[next_cluster_index]))
                if restcapacity <= 0:
                    swap_time_log.append({
                        "iteration": iteration,
                        "swap_index": idx,
                        "from_cluster": int(current_cluster_index),
                        "to_cluster": int(next_cluster_index),
                        "restcapacity": restcapacity,
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

                # distance vectors
                dist_vec_before = vrpfactory.make_distances(cur_xs, cur_ys, cur_cx, cur_cy)
                next_cx = gra_clusters_coordx[next_cluster_index]
                next_cy = gra_clusters_coordy[next_cluster_index]
                dist_vec_to_next = vrpfactory.make_distances(cur_xs, cur_ys, next_cx, next_cy)

                demand_current = cluster_demands[current_cluster_index]

                # QUBO solve
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

                moved_arr = normalize_moved(pro_result.get("route", []), len(cur_ids))
                did_move = bool(moved_arr.sum() > 0.5)
                if did_move:
                    moved_total += 1

                    # apply swap
                    (
                        clusters, clusters_coordx, clusters_coordy, cluster_demands,
                        gra_clusters_coordx, gra_clusters_coordy, _distances_dummy
                    ) = vrpfactory.process_swap(
                        moved_arr,
                        clusters, clusters_coordx, clusters_coordy, cluster_demands,
                        gra_clusters_coordx, gra_clusters_coordy,
                        current_cluster_index, next_cluster_index, distances=[None]*len(clusters)  # distances unused here
                    )

                t_block1 = time.perf_counter()

                swap_time_log.append({
                    "iteration": iteration,
                    "swap_index": idx,
                    "from_cluster": int(current_cluster_index),
                    "to_cluster": int(next_cluster_index),
                    "restcapacity": float(restcapacity),
                    "did_move": did_move,
                    "moved_count": int(moved_arr.sum()),
                    "block_ms": float(ms(t_block0, t_block1)),
                })

            # save swap log
            with (save_dir / f"iteration_{iteration}_swap.json").open("w") as f:
                json.dump(swap_time_log, f, indent=2, default=to_native)

            # ---- evaluate after this iteration => iteration_k.json ----
            it_work = save_dir / f"concorde_work_iter_{iteration}"
            routes_k, total_k, eval_k_ms = eval_clusters_concorde(
                depot_x=depot_x,
                depot_y=depot_y,
                clusters=clusters,
                clusters_coordx=clusters_coordx,
                clusters_coordy=clusters_coordy,
                work_dir=it_work,
                concorde_bin=(args.concorde_bin.strip() or None),
            )
            with (save_dir / f"iteration_{iteration}.json").open("w") as f:
                json.dump(routes_k, f, indent=2, default=to_native)
            with (save_dir / f"iteration_{iteration}_meta.json").open("w") as f:
                json.dump({
                    "iteration": int(iteration),
                    "total_distance": int(total_k),
                    "eval_time_ms": float(eval_k_ms),
                    "K": int(len(clusters)),
                    "moved_total": int(moved_total),
                }, f, indent=2, default=to_native)

            # stop
            if moved_total == 0:
                break
            if iteration >= args.max_iter:
                break

        t_iter1 = time.perf_counter()
        time_iteration_ms = ms(t_iter0, t_iter1)

        # ---- final summary ----
        t_all1 = time.perf_counter()
        final_summary = {
            "instance": instance_name,
            "input_vrp": str(vrp_path),
            "depot": {"x": depot_x, "y": depot_y},
            "capacity": capacity,
            "K_init": int(len(cluster_nums)),
            "K_final": int(len(clusters)),
            "perms": perms,
            "params": {
                "anneal_ms": int(args.anneal_ms),
                "nt": int(args.nt),
                "p": float(args.p),
                "q": float(args.q),
                "lam": float(args.lam),
                "alpha": float(args.alpha),
                "stage2_mode": args.stage2_mode,
                "max_iter": int(args.max_iter),
            },
            "time_sweep_ms": float(time_sweep_ms),
            "time_perms_ms": float(time_perms_ms),
            "time_iteration_ms": float(time_iteration_ms),
            "time_total_ms": float(ms(t_all0, t_all1)),
            "last_iteration": int(iteration),
        }
        with (save_dir / "final_summary.json").open("w") as f:
            json.dump(final_summary, f, indent=2, default=to_native)

        print(f"\n✅ Done: {instance_name}")
        print(f"📂 Output: {save_dir}")
        print(f"   sweep_ms={time_sweep_ms:.3f}  perms_ms={time_perms_ms:.3f}  iter_ms={time_iteration_ms:.3f}")
        print(f"   wrote iteration_0..iteration_{iteration}.json (Concorde routes)")

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
