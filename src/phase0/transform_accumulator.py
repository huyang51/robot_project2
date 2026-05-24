"""
Phase 0 变换累乘器

计算世界空间包围盒（world_bbox），通过沿 Prim 层级树自顶向下累乘变换矩阵。
支持自适应场景尺度（根据 world_bounds 分位数动态调整分类阈值）。
"""

import logging
from typing import List, Dict, Optional, Tuple

from ..core.types import PrimRecord, BBox, Vec3
from ..core.usda_utils import multiply_matrices, transform_point

logger = logging.getLogger(__name__)


class TransformAccumulator:
    """变换累乘器：管理 Prim 层级树中的世界变换"""

    def __init__(self):
        self._world_transforms: Dict[str, List[float]] = {}
        self._identity = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]

    def compute_all(self, prims: List[PrimRecord]) -> None:
        """为所有 Prim 计算世界变换和 world_bbox（原地修改）"""
        sorted_prims = sorted(prims, key=lambda p: p.depth)
        path_to_prim = {p.prim_path: p for p in prims}

        for p in sorted_prims:
            parent_world = self._world_transforms.get(p.parent_path, self._identity)
            world_tf = multiply_matrices(parent_world, p.transform)
            self._world_transforms[p.prim_path] = world_tf

            if p.prim_type in ("Cube", "Mesh") and p.local_bbox:
                p.world_bbox = self._compute_world_bbox(p.local_bbox, world_tf)

    def _compute_world_bbox(self, local_bbox: BBox, world_tf: List[float]) -> BBox:
        """计算局部包围盒在世界空间中的轴对齐包围盒"""
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

        return BBox(
            x_min=min(xs), x_max=max(xs),
            y_min=min(ys), y_max=max(ys),
            z_min=min(zs), z_max=max(zs),
        )

    def get_world_transform(self, prim_path: str) -> List[float]:
        return self._world_transforms.get(prim_path, self._identity)


def compute_adaptive_thresholds(
    cube_prims: List[PrimRecord],
    base_params: Dict,
) -> Dict:
    """根据 world_bounds 动态调整阈值

    Args:
        cube_prims: 已计算 world_bbox 的 Cube Prim 列表
        base_params: 基础参数字典

    Returns:
        调整后的参数字典
    """
    if not base_params.get("adaptive_scale_enabled", True):
        return base_params

    params = dict(base_params)
    bboxes = [p.world_bbox for p in cube_prims if p.world_bbox]
    if not bboxes:
        return params

    # 计算世界包围盒
    all_bounds = bboxes[0]
    for bb in bboxes[1:]:
        all_bounds = all_bounds.union(bb)

    # 场景跨度
    span = max(all_bounds.width, all_bounds.depth, all_bounds.height)
    if span <= 0:
        return params

    # 动态调整：门尺寸、墙体尺寸等按场景比例缩放
    scale_factor = span / 10.0  # 假设 10m 为标准场景

    # 门尺寸
    params["door_height_min"] = base_params.get("door_height_min", 1.8) * scale_factor
    params["door_height_max"] = base_params.get("door_height_max", 2.5) * scale_factor
    params["door_width_min"] = base_params.get("door_width_min", 0.6) * scale_factor
    params["door_width_max"] = base_params.get("door_width_max", 1.5) * scale_factor

    # 掩体体积
    params["cover_volume_min"] = base_params.get("cover_volume_min", 0.02) * (scale_factor ** 3)
    params["cover_volume_max"] = base_params.get("cover_volume_max", 3.0) * (scale_factor ** 3)

    # 装饰物体积（调高以过滤更多）
    params["decor_volume_max"] = base_params.get("decor_volume_max", 0.005) * (scale_factor ** 3)

    logger.info(f"自适应阈值缩放: scale_factor={scale_factor:.3f}, scene_span={span:.2f}m")
    return params
