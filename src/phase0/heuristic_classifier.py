"""
Phase 0 启发式分类器

9 规则优先级分类器，为每个 Cube Prim 分配 TacticalType。
规则按优先级从高到低执行：STAIR > DOOR > WINDOW > WALL > FLOOR > PILLAR > COVER > DECOR > UNKNOWN

注意: 当前流水线（phase0/pipeline.py）不调用此分类器，语义分类由 Phase 3c LLM 完成。
此模块保留用于：(1) 未来混合分类策略 (2) 单元测试 (3) 快速几何预筛。
如需启用启发式预分类，在 phase0/pipeline.py 的 _build_metadata 调用 classify_all()。
"""

import logging
from typing import List, Dict, Optional, Tuple

from ..core.types import PrimRecord, BBox, TacticalType
from ..core.geometry import is_horizontal_plane, is_vertical_plane

logger = logging.getLogger(__name__)


class HeuristicClassifier:
    """9 规则优先级分类器"""

    def __init__(self, params: Optional[Dict] = None, up_axis: str = "z"):
        """初始化分类器

        Args:
            params: 参数字典，覆盖默认值（来自 PHASE0_HEURISTIC_PARAMS）
            up_axis: USDA 文件声明的垂直轴 ("z" 或 "y")
        """
        self.params = params or {}
        self.up_axis = up_axis
        # 墙体
        self.wall_height_min = self.params.get("wall_height_min", 2.0)
        self.wall_width_min = self.params.get("wall_width_min", 0.5)
        self.wall_thickness_max = self.params.get("wall_thickness_max", 0.5)
        # 楼板
        self.floor_thickness_max = self.params.get("floor_thickness_max", 0.4)
        self.floor_area_min = self.params.get("floor_area_min", 4.0)
        # 门
        self.door_height_min = self.params.get("door_height_min", 1.8)
        self.door_height_max = self.params.get("door_height_max", 2.5)
        self.door_width_min = self.params.get("door_width_min", 0.6)
        self.door_width_max = self.params.get("door_width_max", 1.5)
        self.door_z_bottom_margin = self.params.get("door_z_bottom_margin", 0.3)
        # 掩体
        self.cover_volume_min = self.params.get("cover_volume_min", 0.02)
        self.cover_volume_max = self.params.get("cover_volume_max", 3.0)
        self.cover_height_min = self.params.get("cover_height_min", 0.4)
        self.cover_height_max = self.params.get("cover_height_max", 1.5)
        self.cover_wall_distance_min = self.params.get("cover_wall_distance_min", 0.3)
        # 柱状
        self.pillar_height_min = self.params.get("pillar_height_min", 2.0)
        self.pillar_width_max = self.params.get("pillar_width_max", 0.6)
        self.pillar_depth_max = self.params.get("pillar_depth_max", 0.6)
        # 装饰
        self.decor_volume_max = self.params.get("decor_volume_max", 0.005)
        self.decor_dimension_min = self.params.get("decor_dimension_min", 0.03)

        # 楼梯集（由 staircase_detector 预填充）
        self.stair_cube_ids: set = set()
        # 墙体集（用于计算"距最近墙体距离"）
        self.wall_bboxes: List[BBox] = []
        # 楼板 Z 层级（用于门底高度检查）
        self.floor_z_levels: List[float] = []

    def set_stair_cubes(self, stair_cube_ids: List[str]) -> None:
        """设置已知楼梯 Cube ID 集合"""
        self.stair_cube_ids = set(stair_cube_ids)

    def classify_all(self, cube_prims: List[PrimRecord]) -> None:
        """对所有 Cube Prim 进行分类（原地修改 tactical_type）

        两遍算法：
        Pass 1: 检测 WALL/FLOOR（结构元素，用于后续规则判断）
        Pass 2: 检测其他类型
        """
        # Pass 1: 检测结构和墙体
        for p in cube_prims:
            if p.world_bbox is None:
                continue
            if self._is_wall(p):
                p.tactical_type = TacticalType.WALL
                self.wall_bboxes.append(p.world_bbox)
            elif self._is_floor(p):
                p.tactical_type = TacticalType.FLOOR
                self.floor_z_levels.append(p.world_bbox.z_min)

        # 提取楼层 Z 层级（聚类以处理多层建筑）
        self._cluster_floor_levels()

        # Pass 2: 按优先级分类
        for p in cube_prims:
            if p.world_bbox is None or p.tactical_type != TacticalType.UNKNOWN:
                continue

            # R5: Stair (最高优先级)
            if p.prim_path in self.stair_cube_ids:
                p.tactical_type = TacticalType.STAIR
                continue

            # R3: Door
            if self._is_door(p):
                p.tactical_type = TacticalType.DOOR
                continue

            # R4: Window
            if self._is_window(p):
                p.tactical_type = TacticalType.WINDOW
                continue

            # R6: Cover
            if self._is_cover(p):
                p.tactical_type = TacticalType.COVER
                continue

            # R7: Pillar
            if self._is_pillar(p):
                p.tactical_type = TacticalType.PILLAR
                continue

            # R8: Decor
            if self._is_decor(p):
                p.tactical_type = TacticalType.DECOR
                continue

            # R9: 保持 UNKNOWN

    def _is_wall(self, p: PrimRecord) -> bool:
        """R1: 墙体检测"""
        bbox = p.world_bbox
        if bbox is None:
            return False
        h = bbox.height
        w, d = bbox.width, bbox.depth
        thickness = min(w, d)
        width = max(w, d)
        return (
            is_vertical_plane(bbox, 0.4, up_axis=self.up_axis) and
            h >= self.wall_height_min and
            width >= self.wall_width_min and
            thickness <= self.wall_thickness_max
        )

    def _is_floor(self, p: PrimRecord) -> bool:
        """R2: 楼板检测"""
        bbox = p.world_bbox
        if bbox is None:
            return False
        area = bbox.width * bbox.depth
        return (
            is_horizontal_plane(bbox, 0.2, up_axis=self.up_axis) and
            bbox.height <= self.floor_thickness_max and
            area >= self.floor_area_min
        )

    def _is_door(self, p: PrimRecord) -> bool:
        """R3: 门检测"""
        bbox = p.world_bbox
        if bbox is None:
            return False
        h = bbox.height
        w, d = bbox.width, bbox.depth
        width = max(w, d)
        thickness = min(w, d)

        if not (self.door_height_min <= h <= self.door_height_max):
            return False
        if not (self.door_width_min <= width <= self.door_width_max):
            return False
        if thickness > self.wall_thickness_max:
            return False

        # 门底应靠近某个楼层 Z 层级
        door_z_bottom = bbox.z_min
        for floor_z in self.floor_z_levels:
            if abs(door_z_bottom - floor_z) <= self.door_z_bottom_margin:
                return True
        return False

    def _is_window(self, p: PrimRecord) -> bool:
        """R4: 窗检测"""
        bbox = p.world_bbox
        if bbox is None:
            return False
        h = bbox.height
        w, d = bbox.width, bbox.depth
        width = max(w, d)
        thickness = min(w, d)
        return (
            is_vertical_plane(bbox, 0.4, up_axis=self.up_axis) and
            0.8 <= h <= 2.0 and
            0.4 <= width <= 2.0 and
            thickness <= 0.3
        )

    def _is_cover(self, p: PrimRecord) -> bool:
        """R6: 大型掩体检测"""
        bbox = p.world_bbox
        if bbox is None:
            return False
        vol = bbox.volume
        h = bbox.height

        if not (self.cover_volume_min <= vol <= self.cover_volume_max):
            return False
        if not (self.cover_height_min <= h <= self.cover_height_max):
            return False

        # 与最近墙体的距离
        min_dist = self._min_distance_to_walls(bbox)
        return min_dist >= self.cover_wall_distance_min

    def _is_pillar(self, p: PrimRecord) -> bool:
        """R7: 柱状结构检测"""
        bbox = p.world_bbox
        if bbox is None:
            return False
        h = bbox.height
        w, d = bbox.width, bbox.depth
        return (
            is_vertical_plane(bbox, 0.5, up_axis=self.up_axis) and
            h >= self.pillar_height_min and
            w <= self.pillar_width_max and
            d <= self.pillar_depth_max
        )

    def _is_decor(self, p: PrimRecord) -> bool:
        """R8: 建筑装饰过滤"""
        bbox = p.world_bbox
        if bbox is None:
            return False

        if bbox.volume <= self.decor_volume_max:
            return True
        if min(bbox.width, bbox.depth, bbox.height) < self.decor_dimension_min:
            return True
        return False

    def _min_distance_to_walls(self, bbox: BBox) -> float:
        """计算包围盒中心到最近墙体的距离"""
        if not self.wall_bboxes:
            return float('inf')
        center = bbox.center
        min_dist = float('inf')
        for wb in self.wall_bboxes:
            # 包围盒到包围盒最近距离
            dx = max(0, max(wb.x_min - center.x, center.x - wb.x_max))
            dy = max(0, max(wb.y_min - center.y, center.y - wb.y_max))
            dz = max(0, max(wb.z_min - center.z, center.z - wb.z_max))
            dist = (dx * dx + dy * dy + dz * dz) ** 0.5
            if dist < min_dist:
                min_dist = dist
        return min_dist

    def _cluster_floor_levels(self, tolerance: float = 0.5) -> None:
        """将地板 Z 层级聚类为楼层"""
        if not self.floor_z_levels:
            return
        levels = sorted(set(self.floor_z_levels))
        clusters = []
        current = [levels[0]]
        for z in levels[1:]:
            if z - current[-1] <= tolerance:
                current.append(z)
            else:
                clusters.append(sum(current) / len(current))
                current = [z]
        if current:
            clusters.append(sum(current) / len(current))
        self.floor_z_levels = clusters

    def get_statistics(self, cube_prims: List[PrimRecord]) -> Dict:
        """获取分类统计"""
        counts = {t: 0 for t in TacticalType}
        for p in cube_prims:
            if p.world_bbox is not None:
                counts[p.tactical_type] = counts.get(p.tactical_type, 0) + 1
        return {
            "total_cubes": len(cube_prims),
            "by_type": {t.value: c for t, c in counts.items()},
            "walls_detected": counts.get(TacticalType.WALL, 0),
            "floors_detected": counts.get(TacticalType.FLOOR, 0),
            "floor_levels": len(self.floor_z_levels),
        }
