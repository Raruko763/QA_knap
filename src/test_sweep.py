#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_sweep.py  (Sweep-only baseline, input = .vrp)

目的:
- data/raw/*.vrp を直接読んで（vrplib）
- Sweep法で初期クラスタを作る（角度ソート＋容量で詰める）
- 経路長(TSP)は計算しない（あなたの要望）
- 結果を JSON に保存（clusters, demand sums, depot, capacity など）

実行例:
  python core_sweep_vrp.py -i data/raw/E-n101-k14.vrp -sp ./out_sweep
  python core_sweep_vrp.py -i data/raw/X-n1001-k43.vrp -sp ./out_sweep --start_angle 0.0
"""

import json
import math
import argparse
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Tuple

import numpy as np
import vrplib


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


# ---------- VRP loader (vrplib) ----------
def load_vrp_instance(vrp_path: str) -> Dict[str, Any]:
    """
    Reads .vrp using vrplib and returns:
      depot: (x0,y0)
      customers: ids 1..n with coords and demands (depot excluded)
      capacity: Q
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

    # depot is assumed to be node 0
    depot_x, depot_y = float(coord[0][0]), float(coord[0][1])

    # customers are 1..n
    n_customers = len(coord) - 1
    city_ids = list(range(1, n_customers + 1))
    xs = [float(coord[i][0]) for i in range(1, n_customers + 1)]
    ys = [float(coord[i][1]) for i in range(1, n_customers + 1)]
    ds = [float(demand_all[i]) for i in range(1, n_customers + 1)]  # depot excluded

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
    Returns clusters + per-cluster coords/demands + centroids
    """
    n = len(city_ids)
    assert len(xs) == n and len(ys) == n and len(demands) == n

    items = []
    for cid, x, y, d in zip(city_ids, xs, ys, demands):
        if d > capacity:
            raise ValueError(f"demand of city {cid} ({d}) exceeds capacity ({capacity})")
        ang = math.atan2(y - depot_y, x - depot_x)
        ang = (ang - start_angle) % (2 * math.pi)
        items.append((cid, x, y, d, ang))

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
            cur_ids.append(int(cid)); cur_xs.append(float(x)); cur_ys.append(float(y)); cur_ds.append(float(d))
            load += d
        else:
            clusters.append(cur_ids)
            clusters_coordx.append(cur_xs)
            clusters_coordy.append(cur_ys)
            cluster_demands.append(cur_ds)

            cur_ids, cur_xs, cur_ys, cur_ds = [int(cid)], [float(x)], [float(y)], [float(d)]
            load = float(d)

    if cur_ids:
        clusters.append(cur_ids)
        clusters_coordx.append(cur_xs)
        clusters_coordy.append(cur_ys)
        cluster_demands.append(cur_ds)

    # centroids
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


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="Sweep-only baseline runner (input .vrp)")
    ap.add_argument("-i", "--input_vrp", required=True, type=str, help="Path to .vrp (e.g. data/raw/E-n101-k14.vrp)")
    ap.add_argument("-sp", required=True, type=str, help="Base output directory (e.g. ./out_sweep)")
    ap.add_argument("--start_angle", type=float, default=0.0, help="Sweep start angle (rad)")
    args = ap.parse_args()

    vrp_path = Path(args.input_vrp).resolve()
    if not vrp_path.exists():
        raise FileNotFoundError(vrp_path)

    instance_name = vrp_path.stem  # E-n101-k14
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path(args.sp) / timestamp / f"{instance_name}_sweep_only"
    save_dir.mkdir(parents=True, exist_ok=True)

    # load
    info = load_vrp_instance(str(vrp_path))
    depot_x, depot_y = info["depot"]
    capacity = info["capacity"]

    # sweep
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
    cluster_demands = res["cluster_demands"]

    payload = {
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

    out_path = save_dir / "sweep_only.json"
    with out_path.open("w") as f:
        json.dump(payload, f, indent=2, default=to_native)

    print(f"\n✅ Sweep-only done: {instance_name}")
    print(f"📂 Output dir: {save_dir}")
    print(f"💾 Saved: {out_path}")
    print(f"K={payload['K']}  capacity={capacity}  start_angle={args.start_angle}")


if __name__ == "__main__":
    main()
