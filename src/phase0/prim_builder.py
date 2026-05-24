"""
Phase 0 Prim 构建器

基于 stream_parser 的解析结果，构建完整的 PrimRecord，
包括提取 extent、合并 transform、计算初步包围盒。

注意: compute_world_transforms() 与 TransformAccumulator.compute_all() 功能重复，
当前流水线使用 TransformAccumulator。此函数保留用于独立测试和简单场景。
"""

import logging
from typing import List, Dict, Optional

from ..core.types import PrimRecord, BBox, Vec3
from ..core.usda_utils import multiply_matrices, transform_point

logger = logging.getLogger(__name__)


def build_prim_hierarchy(prims: List[PrimRecord]) -> Dict[str, List[PrimRecord]]:
    """构建 Prim 父子关系映射

    Returns:
        {parent_path: [child_prim, ...]}
    """
    hierarchy: Dict[str, List[PrimRecord]] = {}
    for p in prims:
        parent = p.parent_path
        if parent not in hierarchy:
            hierarchy[parent] = []
        hierarchy[parent].append(p)
    return hierarchy


def compute_world_transforms(prims: List[PrimRecord]) -> None:
    """计算每个 Prim 的世界空间变换矩阵（原地修改 prim.world_bbox）

    算法：
    1. 按深度排序（确保父节点先处理）
    2. 累乘变换矩阵：world_transform = parent_world_transform * local_transform
    3. 计算世界空间包围盒
    """
    # 按深度排序
    sorted_prims = sorted(prims, key=lambda p: p.depth)

    # 构建路径 → prim 映射
    path_to_prim: Dict[str, PrimRecord] = {p.prim_path: p for p in prims}
    # 构建路径 → world_transform 映射
    path_to_world_transform: Dict[str, List[float]] = {}

    identity = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]

    for p in sorted_prims:
        parent_transform = path_to_world_transform.get(p.parent_path, identity)
        world_tf = multiply_matrices(parent_transform, p.transform)
        path_to_world_transform[p.prim_path] = world_tf

        # 计算世界空间包围盒（Cube 或 Mesh 几何体）
        if p.prim_type in ("Cube", "Mesh") and p.local_bbox:
            local_bbox = p.local_bbox
            # 变换 8 个角点
            corners = [
                (local_bbox.x_min, local_bbox.y_min, local_bbox.z_min),
                (local_bbox.x_min, local_bbox.y_min, local_bbox.z_max),
                (local_bbox.x_min, local_bbox.y_max, local_bbox.z_min),
                (local_bbox.x_min, local_bbox.y_max, local_bbox.z_max),
                (local_bbox.x_max, local_bbox.y_min, local_bbox.z_min),
                (local_bbox.x_max, local_bbox.y_min, local_bbox.z_max),
                (local_bbox.x_max, local_bbox.y_max, local_bbox.z_min),
                (local_bbox.x_max, local_bbox.y_max, local_bbox.z_max),
            ]
            world_corners = [transform_point(world_tf, c) for c in corners]

            xs = [c[0] for c in world_corners]
            ys = [c[1] for c in world_corners]
            zs = [c[2] for c in world_corners]

            p.world_bbox = BBox(
                x_min=min(xs), x_max=max(xs),
                y_min=min(ys), y_max=max(ys),
                z_min=min(zs), z_max=max(zs),
            )


def filter_cubes(prims: List[PrimRecord]) -> List[PrimRecord]:
    """过滤出几何体 Prim（Cube 或 Mesh）且已计算 world_bbox 的"""
    return [p for p in prims if p.prim_type in ("Cube", "Mesh") and p.world_bbox is not None]


def extract_cube_infos(cube_prims: List[PrimRecord]) -> List:
    """从 Cube Prim 列表提取 CubeInfo 列表（用于后续阶段）"""
    from ..core.types import CubeInfo

    results = []
    for i, p in enumerate(cube_prims):
        if p.world_bbox is None:
            continue
        center = p.world_bbox.center
        size = p.world_bbox.size
        color = None
        if p.material_color:
            color = (p.material_color.x, p.material_color.y, p.material_color.z)
        results.append(CubeInfo(
            cube_id=p.prim_path,
            center=center.to_tuple(),
            size=size.to_tuple(),
            tactical_type=p.tactical_type,
            material=p.material_binding or "",
            material_color=color,
        ))
    return results
