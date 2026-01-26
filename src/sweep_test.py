#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sweep-only experiment core (baseline):

目的:
- before_data.json を読み込む
- Sweep 法で「初期クラスタ」を作る（距離+QUBO改善はしない）
- （任意）各クラスタで TSP を解いて結果を保存する（ortools / concorde / amplify）
- ログを out/ 以下に保存

使い方例:
  python core_sweep.py -j /path/to/before_data.json -sp ./out --tsp_solver ortools --tsp_time_limit_ms 2000
  python core_sweep.py -j ... -sp ./out --tsp_solver concorde
  python core_sweep.py -j ... -sp ./out --tsp_solver none   # クラスタだけ作る
"""

import os
import sys
import json
import math
import time
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional

import numpy as np

# Make project root importable if running from scripts/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.vrpfactory import vrpfactory

# optional TSP solvers (only imported when used)
# from src.tsp_ortools import solve_tsp_ortools
# from src.tsp_concorde import solve_tsp_concorde


# ---------- utils ----------
def to_native(o: Any):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    return o


def euclid_xy(ax: float, ay: float, bx: float, by: float) -> float:
    return math.hypot(ax - bx, ay - by)


def centroid(xs: List[float], ys: List[float]) -> Tuple[float, float]:
    if not xs:
        return 0.0, 0.0
    return float(np.mean(xs)), float(np.mean(ys))


# ---------- sweep preprocessing ----------
def sweep_make_initial_clusters(
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
    Sweep法で「初期クラスタ」を作るだけ（TSPはここでは解かない）
    """
    n = len(city_ids)
    assert len(xs) == n and len(ys) == n and len(demands) == n

    items = []
    for cid, x, y, d in zip(city_ids, xs, ys, demands):
        ang = math.atan2(float(y) - float(depot_y), float(x) - float(depot_x))
        ang = (ang - float(start_angle)) % (2 * math.pi)
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

    for cid, x, y, d, _ang in items:
        if d > capacity:
            raise ValueError(f"demand of city {cid} ({d}) exceeds capacity ({capacity})")

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

    gra_clusters_coordx: List[float] = []
    gra_clusters_coordy: List[float] = []
    for xs_i, ys_i in zip(clusters_coordx, clusters_coordy):
        cx, cy = centroid(xs_i, ys_i)
        gra_clusters_coordx.append(cx)
        gra_clusters_coordy.append(cy)

    k = len(clusters)
    gra_distances = np.zeros((k, k), dtype=float)
    for i in range(k):
        for j in range(k):
            gra_distances[i, j] = euclid_xy(
                gra_clusters_coordx[i], gra_clusters_coordy[i],
                gra_clusters_coordx[j], gra_clusters_coordy[j],
            )

    return {
        "cluster_nums": list(range(k)),
        "clusters": clusters,
        "clusters_coordx": clusters_coordx,
        "clusters_coordy": clusters_coordy,
        "cluster_demands": cluster_demands,
        "gra_clusters_coordx": gra_clusters_coordx,
        "gra_clusters_coordy": gra_clusters_coordy,
        "gra_distances": gra_distances,
    }


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="Sweep-only baseline runner")
    ap.add_argument("-j", required=True, type=str, help="Path to before_data.json")
    ap.add_argument("-sp", required=True, type=str, help="Base output directory (e.g. ./out)")
    ap.add_argument("--start_angle", type=float, default=0.0, help="Sweep start angle (rad)")
    ap.add_argument(
        "--tsp_solver",
        choices=["none", "ortools", "concorde", "amplify"],
        default="ortools",
        help="Solve TSP per cluster (or skip).",
    )
    ap.add_argument("--tsp_time_limit_ms", type=int, default=2000, help="OR-Tools time limit per cluster (ms)")

    # amplify only if you pick tsp_solver=amplify
    ap.add_argument("--t", type=int, default=3000, help="Annealing time (ms) for amplify tsp")
    ap.add_argument("-nt", type=int, default=3, help="num_solve for amplify tsp")
    ap.add_argument("--p", type=float, default=1.0, help="amplify tsp parameter p")
    ap.add_argument("--q", type=float, default=1.0, help="amplify tsp parameter q")

    args = ap.parse_args()

    before_path = Path(args.j).resolve()
    instance_name = before_path.stem.replace("_before_data", "")
    if not instance_name:
        instance_name = before_path.stem

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path(args.sp) / timestamp / f"{instance_name}_sweep_only"
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n🚀 Sweep-only baseline: {instance_name}")
    print(f"📂 Output: {save_dir}")
    print(f"🧭 start_angle(rad): {args.start_angle}")
    print(f"🧩 tsp_solver: {args.tsp_solver}")

    VRPfactory = vrpfactory()

    # load raw info from before_data.json using existing pipeline
    (
        cluster_nums, grax, gray, gra_distances,
        x, y, distances, demands, capacity,
        clusters, clusters_coordx, clusters_coordy, cluster_demands,
        gra_clusters_coordx, gra_clusters_coordy, depo_x, depo_y
    ) = VRPfactory.get_gluster_gravity_info(str(before_path))

    # normalize depot
    depo_x0, depo_y0 = float(depo_x[0]), float(depo_y[0])
    cap = float(capacity)

    # IMPORTANT:
    # This assumes x,y,demands are arrays INCLUDING depot at index 0.
    # If your VRPfactory already excludes depot, change [1:] to [:].
    city_ids = list(range(1, len(demands)))
    xs_city = [float(v) for v in x[1:]]
    ys_city = [float(v) for v in y[1:]]
    ds_city = [float(v) for v in demands[1:]]

    # --- Sweep clustering ---
    t0 = time.perf_counter()
    sweep_state = sweep_make_initial_clusters(
        city_ids=city_ids,
        xs=xs_city,
        ys=ys_city,
        demands=ds_city,
        capacity=cap,
        depot_x=depo_x0,
        depot_y=depo_y0,
        start_angle=float(args.start_angle),
    )
    t1 = time.perf_counter()

    clusters = sweep_state["clusters"]
    clusters_coordx = sweep_state["clusters_coordx"]
    clusters_coordy = sweep_state["clusters_coordy"]
    cluster_demands = sweep_state["cluster_demands"]
    gra_clusters_coordx = sweep_state["gra_clusters_coordx"]
    gra_clusters_coordy = sweep_state["gra_clusters_coordy"]
    gra_distances = sweep_state["gra_distances"]

    print(f"✅ Sweep clusters: K={len(clusters)}  (time={(t1 - t0)*1000:.1f} ms)")

    # save sweep init
    sweep_payload = {
        "instance": instance_name,
        "before_path": str(before_path),
        "depot": {"x": depo_x0, "y": depo_y0},
        "capacity": cap,
        "start_angle_rad": float(args.start_angle),
        "K": int(len(clusters)),
        "clusters": to_native(clusters),
        "cluster_sizes": [int(len(c)) for c in clusters],
        "cluster_demands_sum": [float(sum(ds)) for ds in cluster_demands],
        "centroids": {"x": to_native(np.array(gra_clusters_coordx)), "y": to_native(np.array(gra_clusters_coordy))},
        "sweep_time_ms": float((t1 - t0) * 1000.0),
    }
    with open(save_dir / "sweep_init.json", "w") as f:
        json.dump(sweep_payload, f, indent=2, default=to_native)
    print(f"💾 Saved: {save_dir / 'sweep_init.json'}")

    # --- optionally solve TSP per cluster ---
    tsp_routes: List[Dict[str, Any]] = []
    total_distance = 0.0

    if args.tsp_solver == "none":
        print("⏭️ Skip TSP (tsp_solver=none). Done.")
        return

    if args.tsp_solver == "ortools":
        from src.tsp_ortools import solve_tsp_ortools

        print(f"🔄 Solving TSP for all clusters with OR-Tools (time_limit={args.tsp_time_limit_ms} ms/cluster)")
        for cluster_id in range(len(clusters)):
            coordx = [depo_x0] + clusters_coordx[cluster_id]
            coordy = [depo_y0] + clusters_coordy[cluster_id]
            cluster_distance = vrpfactory.make_cluster_distance_matrix(coordx, coordy)

            res = solve_tsp_ortools(cluster_distance, time_limit_ms=args.tsp_time_limit_ms)
            if isinstance(res, dict):
                route_local = res.get("route", [])
                dist_val = res.get("total_distance")
                status = res.get("solver_status", "")
                solve_time_ms = res.get("solve_time_ms", None)
            else:
                route_local = res
                dist_val, status, solve_time_ms = None, "", None

            # local (0=depot) -> global city id
            global_ids = clusters[cluster_id]
            route_global = []
            for node in route_local:
                if node == 0:
                    route_global.append(0)
                else:
                    idx = node - 1
                    route_global.append(int(global_ids[idx]) if 0 <= idx < len(global_ids) else int(node))

            tsp_routes.append({
                "cluster_id": int(cluster_id),
                "route_local": route_local,
                "route_global": route_global,
                "total_distance": dist_val,
                "solver": "ortools",
                "solver_status": status,
                "solve_time_ms": solve_time_ms,
            })
            if dist_val is not None:
                total_distance += float(dist_val)

    elif args.tsp_solver == "concorde":
        from src.tsp_concorde import solve_tsp_concorde

        work_dir = save_dir / "concorde_work"
        print(f"🔄 Solving TSP for all clusters with Concorde (work_dir={work_dir})")
        for cluster_id in range(len(clusters)):
            coordx = [depo_x0] + clusters_coordx[cluster_id]
            coordy = [depo_y0] + clusters_coordy[cluster_id]
            cluster_distance = vrpfactory.make_cluster_distance_matrix(coordx, coordy)

            res = solve_tsp_concorde(cluster_distance, work_dir=work_dir)

            route_local = res.get("route") or []
            dist_val = res.get("total_distance")
            status = res.get("solver_status", "")
            solve_time_ms = res.get("solve_time_ms", None)

            global_ids = clusters[cluster_id]
            route_global = []
            for node in route_local:
                if node == 0:
                    route_global.append(0)
                else:
                    idx = node - 1
                    route_global.append(int(global_ids[idx]) if 0 <= idx < len(global_ids) else int(node))

            tsp_routes.append({
                "cluster_id": int(cluster_id),
                "route_local": route_local,
                "route_global": route_global,
                "total_distance": dist_val,
                "solver": "concorde",
                "solver_status": status,
                "solve_time_ms": solve_time_ms,
                "optimal_value_stdout": res.get("optimal_value_stdout"),
                "cost_from_stdout": res.get("cost_from_stdout"),
                "cost_from_route": res.get("cost_from_route"),
                "cost_diff": res.get("cost_diff"),
            })
            if dist_val is not None and status == "SUCCESS":
                total_distance += float(dist_val)

    else:  # amplify
        try:
            from amplify import FixstarsClient
        except Exception as e:
            raise RuntimeError("Failed to import 'amplify'. Install Fixstars Amplify SDK.") from e

        from TSP import TSP  # your existing QUBO TSP class

        client = FixstarsClient()
        token = os.environ.get("AMPLIFY_TOKEN")
        if token:
            client.token = token
            print("🔑 FixstarsClient token loaded from AMPLIFY_TOKEN.")
        else:
            print("⚠️ AMPLIFY_TOKEN not set. (Using default client config)")
        client.parameters.timeout = timedelta(milliseconds=args.t)

        print("🔄 Solving TSP for all clusters with Amplify(QUBO)")
        for cluster_id in range(len(clusters)):
            coordx = [depo_x0] + clusters_coordx[cluster_id]
            coordy = [depo_y0] + clusters_coordy[cluster_id]
            cluster_distance = vrpfactory.make_cluster_distance_matrix(coordx, coordy)

            # demands list for TSP class (0 + cluster demands)
            cluster_demand = [0.0] + [float(d) for d in cluster_demands[cluster_id]]
            city_list = [0] + [int(v) for v in clusters[cluster_id]]  # depot=0 then global ids

            tsp_solver = TSP(
                client,
                cluster_distance,
                cluster_demand,
                cap,
                1,
                args.nt,
                city_list,
                str(save_dir),
                coordx,
                coordy,
                str(before_path),
            )
            res = tsp_solver.solve_TSP(args.p, args.q)
            dist_val = float(res.get("total_distances", 0.0))
            route_global = to_native(res.get("route", []))

            tsp_routes.append({
                "cluster_id": int(cluster_id),
                "route_local": to_native(res.get("route", [])),
                "route_global": route_global,
                "total_distance": dist_val,
                "solver": "amplify",
                "solver_status": "SUCCESS",
                "solve_time_ms": None,
            })
            total_distance += dist_val

    print(f"📏 Total distance (sum of cluster TSP): {total_distance:.6f}")
    tsp_out = {
        "instance": instance_name,
        "tsp_solver": args.tsp_solver,
        "total_distance": float(total_distance),
        "routes": tsp_routes,
    }
    with open(save_dir / "tsp_routes.json", "w") as f:
        json.dump(tsp_out, f, indent=2, default=to_native)
    print(f"💾 Saved: {save_dir / 'tsp_routes.json'}")
    print("\n✅ Done.")


if __name__ == "__main__":
    main()
