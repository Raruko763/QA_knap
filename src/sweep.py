# src/sweep.py
import math
from typing import Dict, List, Tuple
import numpy as np


def _centroid(xs: List[float], ys: List[float]) -> Tuple[float, float]:
    if len(xs) == 0:
        return 0.0, 0.0
    return float(np.mean(xs)), float(np.mean(ys))


def sweep_make_initial_clusters(
    city_ids: List[int],
    xs: List[float],
    ys: List[float],
    demands: List[float],
    capacity: float,
    depot_x: float,
    depot_y: float,
    start_angle: float = 0.0,
) -> Dict:
    """
    Sweep前処理で初期クラスタを作る。

    Inputs:
      - city_ids: デポを除いた「グローバル都市ID」の配列 (例: [1..n])
      - xs, ys:   city_ids と同じ順序の座標
      - demands:  city_ids と同じ順序の需要
      - capacity: 車両容量
      - depot_x, depot_y: デポ座標
      - start_angle: Sweep開始角（rad）

    Returns:
      clusters: List[List[int]]  (各クラスタの都市ID)
      clusters_coordx/y: List[List[float]]
      cluster_demands: List[List[float]]
      gra_clusters_coordx/y: List[float] (各クラスタ重心)
      grax/gray: 上と同じ（互換のため）
      cluster_nums: [0..k-1]
      gra_distances: 重心間距離行列
    """

    n = len(city_ids)
    assert len(xs) == n and len(ys) == n and len(demands) == n

    # --- 角度でソート（customersを並べる） ---
    items = []
    for cid, x, y, d in zip(city_ids, xs, ys, demands):
        ang = math.atan2(float(y) - float(depot_y), float(x) - float(depot_x))
        ang = (ang - float(start_angle)) % (2 * math.pi)
        items.append((int(cid), float(x), float(y), float(d), ang))

    items.sort(key=lambda t: t[4])

    # --- 容量で詰める（クラスタ分割） ---
    clusters: List[List[int]] = []
    clusters_coordx: List[List[float]] = []
    clusters_coordy: List[List[float]] = []
    cluster_demands: List[List[float]] = []

    cur_ids, cur_xs, cur_ys, cur_ds = [], [], [], []
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

    # --- 重心計算 ---
    gra_clusters_coordx = []
    gra_clusters_coordy = []
    for xs_i, ys_i in zip(clusters_coordx, clusters_coordy):
        cx, cy = _centroid(xs_i, ys_i)
        gra_clusters_coordx.append(cx)
        gra_clusters_coordy.append(cy)

    # --- 重心間距離行列 ---
    k = len(clusters)
    gra_distances = np.zeros((k, k), dtype=float)
    for i in range(k):
        for j in range(k):
            dx = gra_clusters_coordx[i] - gra_clusters_coordx[j]
            dy = gra_clusters_coordy[i] - gra_clusters_coordy[j]
            gra_distances[i, j] = math.hypot(dx, dy)

    cluster_nums = list(range(k))

    return {
        "cluster_nums": cluster_nums,
        "clusters": clusters,
        "clusters_coordx": clusters_coordx,
        "clusters_coordy": clusters_coordy,
        "cluster_demands": cluster_demands,
        "gra_clusters_coordx": gra_clusters_coordx,
        "gra_clusters_coordy": gra_clusters_coordy,
        "grax": gra_clusters_coordx,   # 互換
        "gray": gra_clusters_coordy,   # 互換
        "gra_distances": gra_distances.tolist(),  # JSON化しやすい形
    }