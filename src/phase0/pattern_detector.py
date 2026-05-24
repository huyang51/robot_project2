"""
Phase 0 模式检测器

检测场景中的几何模式：
- 等距线性排列（如一排柱子、一排窗户）
- 对称对（如门两侧的窗户）
- 密集簇（如堆积的家具）

全部使用空间哈希优化，避免 O(n²) 暴力比较。
"""

import logging
import math
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

from ..core.types import PrimRecord, BBox, Vec3

logger = logging.getLogger(__name__)


@dataclass
class PatternMatch:
    """检测到的模式"""
    pattern_type: str  # "linear_array" | "symmetric_pair" | "dense_cluster"
    cube_ids: List[str]
    spacing: Optional[float] = None
    direction: Optional[str] = None
    confidence: float = 0.0


# ── 空间哈希工具 ──────────────────────────────────────────

def _grid_key(center: Vec3, cell_size: float) -> Tuple[int, int, int]:
    """将 3D 坐标映射到网格单元"""
    return (
        int(center.x / cell_size),
        int(center.y / cell_size),
        int(center.z / cell_size),
    )


def _build_grid(
    items: List[Tuple[str, Vec3]], cell_size: float
) -> Dict[Tuple[int, int, int], List[Tuple[str, Vec3]]]:
    """构建 3D 空间哈希网格"""
    grid = defaultdict(list)
    for item_id, center in items:
        grid[_grid_key(center, cell_size)].append((item_id, center))
    return grid


def _neighbor_keys(key: Tuple[int, int, int]) -> List[Tuple[int, int, int]]:
    """返回一个网格单元及其 26 个邻居的 key 列表"""
    kx, ky, kz = key
    neighbors = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                neighbors.append((kx + dx, ky + dy, kz + dz))
    return neighbors


# ── 等距线性排列 ──────────────────────────────────────────

def detect_linear_arrays(
    cube_prims: List[PrimRecord],
    max_spacing_deviation: float = 0.1,
    min_count: int = 3,
) -> List[PatternMatch]:
    """检测等距线性排列

    沿各轴全局排序后滑窗检测，不按类型分组（纯几何模式）。
    复杂度 O(n log n)。
    """
    results = []

    valid = [p for p in cube_prims if p.world_bbox is not None]
    if len(valid) < min_count:
        return results

    for axis in ["x", "y", "z"]:
        sorted_prims = sorted(valid, key=lambda p: getattr(p.world_bbox.center, axis))

        start = 0
        while start <= len(sorted_prims) - min_count:
            window = sorted_prims[start:start + min_count]
            centers = [getattr(p.world_bbox.center, axis) for p in window]
            gaps = [centers[i + 1] - centers[i] for i in range(len(centers) - 1)]

            if not gaps:
                start += 1
                continue
            mean_gap = sum(gaps) / len(gaps)
            if mean_gap <= 0:
                start += 1
                continue
            max_dev = max(abs(g - mean_gap) for g in gaps)

            if max_dev <= max_spacing_deviation:
                extended = list(window)
                for p in sorted_prims[start + min_count:]:
                    next_center = getattr(p.world_bbox.center, axis)
                    expected = centers[-1] + mean_gap
                    if abs(next_center - expected) <= max_spacing_deviation:
                        extended.append(p)
                        centers.append(next_center)
                    else:
                        break

                if len(extended) >= min_count:
                    results.append(PatternMatch(
                        pattern_type="linear_array",
                        cube_ids=[p.prim_path for p in extended],
                        spacing=round(mean_gap, 3),
                        direction=axis,
                        confidence=min(1.0, len(extended) / 10),
                    ))
                start += len(extended)
            else:
                start += 1

    return results


# ── 对称对检测（空间哈希优化）────────────────────────────

def detect_symmetric_pairs(
    cube_prims: List[PrimRecord],
    axis: str = "x",
    tolerance: float = 0.3,
) -> List[PatternMatch]:
    """检测沿某轴对称的 Prim 对

    使用空间哈希：按非对称轴坐标建网格 → 只比较同网格内的镜像候选。
    复杂度 O(n) 建表 + O(n * k) 查询，k = 每单元平均候选数。
    """
    if len(cube_prims) < 2:
        return []

    # 收集有 bbox 的 prim
    items = [(p.prim_path, p.world_bbox.center) for p in cube_prims if p.world_bbox]
    if len(items) < 2:
        return []

    # 对称轴参考中心（均值）
    axis_idx = {"x": 0, "y": 1, "z": 2}[axis]
    other_axes = [a for a in ["x", "y", "z"] if a != axis]
    axis_values = [getattr(c, axis) for _, c in items]
    ref_center_val = sum(axis_values) / len(axis_values)

    # 按非对称轴坐标建网格（cell_size = tolerance on other axes）
    cell_size = 0.6  # 必须 >= other_close 容差 0.5，否则有效对称对被漏掉
    grid = defaultdict(list)
    for pid, center in items:
        kx = int(getattr(center, other_axes[0]) / cell_size)
        ky = int(getattr(center, other_axes[1]) / cell_size)
        grid[(kx, ky)].append((pid, center))

    # 寻找对称对
    results = []
    paired = set()

    # 按沿对称轴的位置排序，只处理一侧
    side_items = [(pid, c) for pid, c in items if getattr(c, axis) < ref_center_val]

    for pid, center in side_items:
        if pid in paired:
            continue

        # 计算镜像位置
        mirror_val = 2 * ref_center_val - getattr(center, axis)
        # 查找同网格单元内的候选
        kx = int(getattr(center, other_axes[0]) / cell_size)
        ky = int(getattr(center, other_axes[1]) / cell_size)
        candidates = grid.get((kx, ky), [])

        for other_id, other_center in candidates:
            if other_id in paired or other_id == pid:
                continue

            # 检查：对方在对称轴另一侧，且位置接近镜像
            if getattr(other_center, axis) <= ref_center_val:
                continue
            if abs(getattr(other_center, axis) - mirror_val) > tolerance:
                continue

            # 检查其他两轴位置接近
            other_close = all(
                abs(getattr(center, a) - getattr(other_center, a)) <= 0.5
                for a in other_axes
            )
            if other_close:
                results.append(PatternMatch(
                    pattern_type="symmetric_pair",
                    cube_ids=[pid, other_id],
                    direction=axis,
                    confidence=1.0 - abs(getattr(other_center, axis) - mirror_val) / tolerance,
                ))
                paired.add(pid)
                paired.add(other_id)
                break

    return results


# ── 密集簇检测（空间哈希优化）────────────────────────────

def detect_dense_clusters(
    cube_prims: List[PrimRecord],
    distance_threshold: float = 0.5,
    min_cluster_size: int = 4,
) -> List[PatternMatch]:
    """检测密集簇（DBSCAN-like，使用 3D 空间哈希）

    将空间划分为 distance_threshold 大小的网格单元，
    仅在同一单元及 26 个邻居内建立邻接关系。
    复杂度 O(n) 建表 + O(n * k) 邻接，k = 每 27-cell 邻域内的平均 prim 数。
    """
    items_with_bbox = [(p.prim_path, p.world_bbox) for p in cube_prims if p.world_bbox]
    n = len(items_with_bbox)
    if n < min_cluster_size:
        return []

    # 构建网格：key → [index in items_with_bbox]
    grid = defaultdict(list)
    for idx, (_, bbox) in enumerate(items_with_bbox):
        grid[_grid_key(bbox.center, distance_threshold)].append(idx)

    # 构建邻接表（每个 prim 对只检查一次：以 prim i 为起点，查其邻域内 index > i 的 prim）
    adjacency = [[] for _ in range(n)]
    dist_sq = distance_threshold ** 2

    for i, (_, bbox_i) in enumerate(items_with_bbox):
        ci = bbox_i.center
        # 查找 i 的网格单元及其 26 邻居内的候选
        seen = set()
        for nk in _neighbor_keys(_grid_key(ci, distance_threshold)):
            for j in grid.get(nk, []):
                if j <= i or j in seen:
                    continue
                seen.add(j)
                cj = items_with_bbox[j][1].center
                dx, dy, dz = ci.x - cj.x, ci.y - cj.y, ci.z - cj.z
                if dx * dx + dy * dy + dz * dz <= dist_sq:
                    adjacency[i].append(j)
                    adjacency[j].append(i)

    # BFS 找连通分量
    visited = [False] * n
    results = []
    for i in range(n):
        if visited[i]:
            continue
        component = []
        queue = deque([i])
        visited[i] = True
        while queue:
            v = queue.popleft()
            component.append(v)
            for nb in adjacency[v]:
                if not visited[nb]:
                    visited[nb] = True
                    queue.append(nb)

        if len(component) >= min_cluster_size:
            results.append(PatternMatch(
                pattern_type="dense_cluster",
                cube_ids=[items_with_bbox[idx][0] for idx in component],
                confidence=min(1.0, len(component) / 20),
            ))

    return results


# ── 全模式检测 ────────────────────────────────────────────

def detect_all_patterns(
    cube_prims: List[PrimRecord],
    max_spacing_deviation: float = 0.1,
    up_axis: str = "z",
) -> Dict[str, List[PatternMatch]]:
    """执行所有模式检测（纯几何，不依赖 tactical_type）

    Args:
        up_axis: USDA 文件声明的垂直轴 ("y" 或 "z")，用于确定对称检测的水平面轴

    Returns:
        {
            "linear_arrays": [...],
            "symmetric_pairs": [...],
            "dense_clusters": [...],
        }
    """
    logger.info(f"  线性排列检测 ({len(cube_prims)} prims)...")
    linear_results = detect_linear_arrays(cube_prims, max_spacing_deviation)

    logger.info(f"  对称对检测 ...")
    symmetric_results = []
    horiz_axes = [a for a in ("x", "y", "z") if a != up_axis]
    for axis in horiz_axes:
        symmetric_results.extend(detect_symmetric_pairs(cube_prims, axis))

    logger.info(f"  密集簇检测 ({len(cube_prims)} prims)...")
    dense_results = detect_dense_clusters(cube_prims)

    return {
        "linear_arrays": linear_results,
        "symmetric_pairs": symmetric_results,
        "dense_clusters": dense_results,
    }
