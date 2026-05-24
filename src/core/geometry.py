"""
几何计算工具函数

包含 BBox 交叉、体积计算、Z 直方图、距离计算等。
"""

import math
from typing import Any, List, Dict, Tuple, Optional
from .types import BBox, Vec3, PrimRecord


def bbox_intersection(a: BBox, b: BBox) -> Optional[BBox]:
    """计算两个包围盒的交集"""
    if not a.intersects(b):
        return None
    return BBox(
        x_min=max(a.x_min, b.x_min),
        x_max=min(a.x_max, b.x_max),
        y_min=max(a.y_min, b.y_min),
        y_max=min(a.y_max, b.y_max),
        z_min=max(a.z_min, b.z_min),
        z_max=min(a.z_max, b.z_max),
    )


def xy_overlap_ratio(a: BBox, b: BBox, up_axis: str = "z") -> float:
    """计算两个包围盒在水平面上的重叠比例（相对较小者）

    根据 up_axis 选择水平面：up_axis="z" → XY 水平面；up_axis="y" → XZ 水平面。
    up_axis 来自 USDA 文件头部的 upAxis 声明。
    """
    if up_axis == "y":
        # Y 垂直 → 水平面 = XZ
        h1_overlap = max(0.0, min(a.x_max, b.x_max) - max(a.x_min, b.x_min))
        h2_overlap = max(0.0, min(a.z_max, b.z_max) - max(a.z_min, b.z_min))
        area_a = a.width * a.height
        area_b = b.width * b.height
    else:
        # Z 垂直（默认）→ 水平面 = XY
        h1_overlap = max(0.0, min(a.x_max, b.x_max) - max(a.x_min, b.x_min))
        h2_overlap = max(0.0, min(a.y_max, b.y_max) - max(a.y_min, b.y_min))
        area_a = a.width * a.depth
        area_b = b.width * b.depth

    overlap_area = h1_overlap * h2_overlap
    min_area = min(area_a, area_b)
    if min_area <= 0:
        return 0.0
    return overlap_area / min_area


def centroid_distance(a: BBox, b: BBox) -> float:
    """计算两个包围盒中心之间的欧氏距离"""
    ca = a.center
    cb = b.center
    return ((ca.x - cb.x) ** 2 + (ca.y - cb.y) ** 2 + (ca.z - cb.z) ** 2) ** 0.5


def centroid_xy_distance(a: BBox, b: BBox) -> float:
    """计算两个包围盒中心在 XY 平面上的距离"""
    ca = a.center
    cb = b.center
    return ((ca.x - cb.x) ** 2 + (ca.y - cb.y) ** 2) ** 0.5


def world_bounds(prims: List[PrimRecord]) -> BBox:
    """计算所有 Prim 的世界空间包围盒"""
    if not prims:
        return BBox()
    result = None
    for p in prims:
        if p.world_bbox:
            if result is None:
                result = BBox(
                    p.world_bbox.x_min, p.world_bbox.x_max,
                    p.world_bbox.y_min, p.world_bbox.y_max,
                    p.world_bbox.z_min, p.world_bbox.z_max,
                )
            else:
                result = result.union(p.world_bbox)
    return result or BBox()


def axis_histogram(prims: List[PrimRecord], axis: str = "z", bins: int = 20) -> Dict[str, Any]:
    """计算指定轴的直方图，用于楼层推断

    Args:
        prims: Prim 列表（需已计算 world_bbox）
        axis: 要分析的轴 ("x" | "y" | "z")，默认 z
        bins: 直方图分箱数

    Returns:
        {"histogram": List[int], "bin_edges": List[float], "peaks": List[float], ...}
    """
    if not prims:
        return {"histogram": [], "bin_edges": [], "peaks": [], "floor_count": 0}

    bounds = world_bounds(prims)
    axis_idx = {"x": 0, "y": 1, "z": 2}[axis]
    axis_min = [bounds.x_min, bounds.y_min, bounds.z_min][axis_idx]
    axis_max = [bounds.x_max, bounds.y_max, bounds.z_max][axis_idx]
    if axis_max <= axis_min:
        return {"histogram": [], "bin_edges": [], "peaks": [], "floor_count": 0}

    bin_width = (axis_max - axis_min) / bins
    histogram = [0] * bins
    bin_edges = [axis_min + i * bin_width for i in range(bins + 1)]

    # 按体积加权统计：大体积结构元素（楼板/墙体）的贡献远超小碎片
    # 使用实际体积（非 log 压缩），因为建筑结构元素（楼板 10-100m³）
    # 与三角化碎屑（0.0001m³）之间体积差达 4-6 个数量级，
    # 实际体积加权可有效区分结构性元素与碎屑。
    for p in prims:
        if not p.world_bbox or p.world_bbox.volume <= 0:
            continue
        val = [p.world_bbox.center.x, p.world_bbox.center.y, p.world_bbox.center.z][axis_idx]
        bin_idx = min(int((val - axis_min) / bin_width), bins - 1)
        if bin_idx >= 0:
            histogram[bin_idx] += p.world_bbox.volume

    # 找峰值（局部最大值）
    peaks = []
    for i in range(1, bins - 1):
        if histogram[i] > histogram[i - 1] and histogram[i] > histogram[i + 1]:
            peaks.append(bin_edges[i] + bin_width / 2)

    floor_count = max(1, len(peaks))

    return {
        "histogram": histogram,
        "bin_edges": bin_edges,
        "peaks": peaks,
        "floor_count": floor_count,
        "axis": axis,
        "axis_range": [axis_min, axis_max],
    }


def z_histogram(prims: List[PrimRecord], bins: int = 20) -> Dict[str, Any]:
    """向后兼容：默认 Z 轴直方图。推荐使用 axis_histogram(prims, axis='y') 支持自定义纵轴。"""
    return axis_histogram(prims, axis="z", bins=bins)


def is_horizontal_plane(bbox: BBox, max_height_ratio: float = 0.3,
                        up_axis: str = "z") -> bool:
    """判断包围盒是否为水平面（垂直轴厚度远小于水平面尺寸）

    Args:
        up_axis: USDA 文件声明的垂直轴 ("z" 或 "y")
    """
    # 获取垂直轴和水平轴的尺寸
    w, d, h = bbox.width, bbox.depth, bbox.height
    dims = {"x": w, "y": d, "z": h}
    vert_thickness = dims[up_axis]
    horiz_dims = [v for k, v in dims.items() if k != up_axis]
    if min(horiz_dims) <= 0:
        return False
    return vert_thickness / min(horiz_dims) < max_height_ratio


def is_vertical_plane(bbox: BBox, max_width_ratio: float = 0.3,
                      up_axis: str = "z") -> bool:
    """判断包围盒是否为垂直面（水平面某方向厚度远小于垂直轴高度）

    Args:
        up_axis: USDA 文件声明的垂直轴 ("z" 或 "y")
    """
    w, d, h = bbox.width, bbox.depth, bbox.height
    dims = {"x": w, "y": d, "z": h}
    vert_height = dims[up_axis]
    if vert_height <= 0:
        return False
    horiz_dims = {k: v for k, v in dims.items() if k != up_axis}
    # 任何一个水平方向极薄（厚度 << 垂直高度）
    for horiz_axis, horiz_val in horiz_dims.items():
        if horiz_val / vert_height < max_width_ratio:
            return True
    return False
