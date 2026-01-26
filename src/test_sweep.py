#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_sweep.py
- .vrp を vrplib で読む
- Sweepでクラスタ作る
- （任意）Concordeで各クラスタのTSPを解く
- node_coord無し / MemoryError などはスキップ（バッチが止まらない）
"""

import os
import json
import math
import argparse
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import vrplib

from src.vrpfactory import vrpfactory
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


def centroid(xs: List[float], ys: List[float]) -> Tuple[float, float]:
    if not xs:
        return 0.0, 0.0
    return float(np.mean(xs)), float(np.mean(ys))


# ---------- VRP loader (.vrp via vrplib) ----------
def load_vrp_instance(vrp_path: str) -> Dict[str, Any]:
    """
    Reads .vrp using vrplib.
    Assumption:
      - node 0 is depot
      - customers are 1..n
    """
    inst = vrplib.read_instance(vrp_path)

    coord = inst.get("node_coord")
    if coord is None:
        raise ValueError("This .vrp has no node_coord. Sweep needs coordinates.")

    demand_all = inst.get("demand")
    if demand_all is None:
        raise ValueError("This .vrp has no demand vector.")
    if len(demand_all) != len(coord):
        raise ValueError(f"Mismatch: len(demand)={len(demand_all)} vs len(node_coord)={len(coord)}")

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
def sweep_clusters(
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
    Single Sweep:
      - sort by polar angle around depot
      - pack sequentially with capacity constraint
    """
    n = len(city_ids)
    assert len(xs) == n and len(ys) == n and len(demands) == n

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

    cent_x, cent_y = [], []
    for xs_i, ys_i in zip(clusters_coordx, clusters_coordy):
        cx, cy = centroid(xs_i, ys_i)
        cent_x.append(cx)
        cent_y.append(cy)

    return {
        "clusters": clusters,
        "clusters_coordx": clusters_coordx,
        "clusters_coordy": clusters_coordy,
        "cluster_demands": cluster_demands,
        "centroids": {"x": cent_x, "y": cent_y},
    }


def main():
    ap = argparse.ArgumentParser(description="Sweep-only baseline runner (input .vrp)")
    ap.add_argument("-i", "--input_vrp", required=True, type=str, help="Path to .vrp (e.g. data/raw/E-n101-k14.vrp)")
    ap.add_argument("-sp", required=True, type=str, help="Base output directory (e.g. ./out_sweep)")
    ap.add_argument("--start_angle", type=float, default=0.0, help="Sweep start angle (rad)")

    # Concorde TSP
    ap.add_argument("--solve_tsp", action="store_true", help="Solve per-cluster TSP with Concorde")
    ap.add_argument("--concorde_work", type=str, default="", help="Work dir for Concorde (default: <save_dir>/concorde_work)")
    ap.add_argument("--concorde_bin", type=str, default="", help="Path to concorde binary (or set CONCORDE_BIN env)")
    ap.add_argument("--seed", type=int, default=0, help="Concorde seed (0 disables -s)")

    args = ap.parse_args()

    base_out = Path(args.sp)
    base_out.mkdir(parents=True, exist_ok=True)
    skipped_log = base_out / "skipped.log"

    vrp_path = Path(args.input_vrp).resolve()
    instance_name = vrp_path.stem

    try:
        if not vrp_path.exists():
            raise FileNotFoundError(vrp_path)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = base_out / timestamp / f"{instance_name}_sweep_only"
        save_dir.mkdir(parents=True, exist_ok=True)

        # ---- load ----
        info = load_vrp_instance(str(vrp_path))
        depot_x, depot_y = info["depot"]
        capacity = info["capacity"]

        # ---- sweep ----
        res = sweep_clusters(
            city_ids=info["city_ids"],
            xs=info["xs"],
            ys=info["ys"],
            demands=info["demands"],
            capacity=capacity,
            depot_x=depot_x,
            depot_y=depot_y,
            start_angle=float(args.start_angle),
        )

        clusters = res["clusters"]
        clusters_coordx = res["clusters_coordx"]
        clusters_coordy = res["clusters_coordy"]
        cluster_demands = res["cluster_demands"]

        payload: Dict[str, Any] = {
            "instance": instance_name,
            "input_vrp": str(vrp_path),
            "depot": {"x": depot_x, "y": depot_y},
            "capacity": float(capacity),
            "start_angle_rad": float(args.start_angle),
            "K": int(len(clusters)),
            "cluster_sizes": [int(len(c)) for c in clusters],
            "cluster_demands_sum": [float(sum(ds)) for ds in cluster_demands],
            "clusters": clusters,  # global city ids (1..n)
            "centroids": res["centroids"],
        }

        # ---- (optional) Concorde per-cluster TSP ----
        if args.solve_tsp:
            work_dir = args.concorde_work.strip() or str(save_dir / "concorde_work")
            seed = None if args.seed == 0 else int(args.seed)
            concorde_bin = args.concorde_bin.strip() or None

            tsp_routes = []
            total_distance = 0

            for cluster_id, (xs_i, ys_i, cities_i) in enumerate(zip(clusters_coordx, clusters_coordy, clusters)):
                coordx = [depot_x] + xs_i
                coordy = [depot_y] + ys_i
                dist = vrpfactory.make_cluster_distance_matrix(coordx, coordy)

                cres = solve_tsp_concorde(
                    dist_matrix=dist,
                    work_dir=work_dir,
                    seed=seed,
                    concorde_bin=concorde_bin,
                )

                route_local = cres.get("route")
                td = cres.get("total_distance")
                status = cres.get("solver_status")
                solve_time_ms = cres.get("solve_time_ms")

                if route_local is None:
                    tsp_routes.append({
                        "cluster_id": cluster_id,
                        "solver": "concorde",
                        "solver_status": status,
                        "solve_time_ms": solve_time_ms,
                        "total_distance": None,
                        "route_local": None,
                        "route_global": None,
                    })
                    continue

                route_global = []
                for node in route_local:
                    if node == 0:
                        route_global.append(0)
                    else:
                        route_global.append(int(cities_i[node - 1]))

                if td is not None:
                    total_distance += int(td)

                tsp_routes.append({
                    "cluster_id": cluster_id,
                    "solver": "concorde",
                    "solver_status": status,
                    "solve_time_ms": solve_time_ms,
                    "total_distance": td,
                    "route_local": route_local,
                    "route_global": route_global,
                })

            payload["tsp_solver"] = "concorde"
            payload["tsp_total_distance"] = int(total_distance)
            payload["tsp_routes"] = tsp_routes

        out_path = save_dir / "sweep_only.json"
        with out_path.open("w") as f:
            json.dump(payload, f, indent=2, default=to_native)

        print(f"\n✅ Done: {instance_name}")
        print(f"📂 Output dir: {save_dir}")
        print(f"💾 Saved: {out_path}")

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
