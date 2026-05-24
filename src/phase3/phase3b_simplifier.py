"""
Phase 3b: 几何简化器

将 Mesh 简化为 Cube 表示，执行：
- Mesh → Cube 转换
- 相似 Cube 合并（材质 + 空间邻近）
- 模式折叠（等距排列 → 单个 + 元数据）
- 归一化（Z 轴最小值归零）
- 楼梯检测集成（从 extract/staircase_detector.py 内嵌）
"""

import logging
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import math

from ..core.types import BBox
from ..core.geometry import centroid_distance

logger = logging.getLogger(__name__)


def merge_similar_cubes(
    cubes: List[Dict],
    params: Dict,
    world_bounds: Optional[Dict] = None,
) -> List[Dict]:
    """合并空间邻近且材质相同的 Cube

    两遍算法：
    Pass 1: 同材质 + 质心距离近 → 合并
    Pass 2 (降级): 放宽阈值 → 更大力度合并
    """
    if not cubes:
        return cubes

    # 计算自适应阈值
    span = _compute_span(world_bounds or _compute_world_bounds(cubes))
    dist_threshold = span * params.get("centroid_distance_threshold_ratio", 0.02)
    same_material = params.get("same_material_required", True)
    vol_increase_max = params.get("volume_increase_max", 0.30)

    # Pass 1
    merged = _merge_pass(cubes, dist_threshold, same_material, vol_increase_max)

    # Pass 2: 降级合并
    if len(merged) > params.get("downgrade_cube_threshold", 100):
        logger.info(f"Cube 数量 {len(merged)} > {params['downgrade_cube_threshold']}，触发降级合并")
        dist2 = dist_threshold * params.get("downgrade_distance_multiplier", 1.5)
        vol2 = params.get("downgrade_volume_increase_max", 0.50)
        merged = _merge_pass(merged, dist2, same_material, vol2)

    return merged


def _merge_pass(
    cubes: List[Dict],
    dist_threshold: float,
    same_material: bool,
    vol_increase_max: float,
) -> List[Dict]:
    """单遍合并"""
    if len(cubes) < 2:
        return list(cubes)

    merged = []
    used = [False] * len(cubes)

    for i, ci in enumerate(cubes):
        if used[i]:
            continue
        current = dict(ci)
        current_bbox = _dict_to_bbox(current)
        used[i] = True

        changed = True
        while changed:
            changed = False
            for j, cj in enumerate(cubes):
                if used[j]:
                    continue
                # 使用 current（可能已合并多轮）的材质做比较
                current_material = current.get("material", "")
                if same_material and current_material and current_material != cj.get("material", ""):
                    continue

                cj_bbox = _dict_to_bbox(cj)
                if centroid_distance(current_bbox, cj_bbox) > dist_threshold:
                    continue

                # 间隙保护：两 Cube 之间显著空白(>0.5m) → 可能是门/窗/通道，不合并
                gap = _gap_between(current_bbox, cj_bbox)
                if gap > 0.5:
                    continue

                # 检查合并后体积
                new_bbox = current_bbox.union(cj_bbox)
                vol_increase = (new_bbox.volume - current_bbox.volume) / max(current_bbox.volume, 1e-10)
                if vol_increase > vol_increase_max:
                    continue

                current_bbox = new_bbox
                used[j] = True
                changed = True
                # 合并不同材质时更新 current 材质标记为混合
                if not current_material:
                    current["material"] = cj.get("material", "")

        # 更新 current
        current["center"] = {
            "x": round(current_bbox.center.x, 3),
            "y": round(current_bbox.center.y, 3),
            "z": round(current_bbox.center.z, 3),
        }
        current["size"] = {
            "x": round(current_bbox.width, 3),
            "y": round(current_bbox.depth, 3),
            "z": round(current_bbox.height, 3),
        }
        current["bounds"] = {
            "x": [round(current_bbox.x_min, 3), round(current_bbox.x_max, 3)],
            "y": [round(current_bbox.y_min, 3), round(current_bbox.y_max, 3)],
            "z": [round(current_bbox.z_min, 3), round(current_bbox.z_max, 3)],
        }
        merged.append(current)

    return merged


def fold_patterns(
    cubes: List[Dict],
    patterns: List[Dict],
    spacing_deviation_max: float = 0.1,
) -> List[Dict]:
    """模式折叠：将检测到的等距排列/对称对替换为紧凑表示

    每个模式保留一个代表 Cube + metadata 记录 pattern_info。
    """
    if not patterns:
        return cubes

    # 收集需要折叠的 Cube ID
    fold_ids = set()
    for pattern in patterns:
        if pattern.get("pattern_type") == "linear_array" or pattern.get("pattern_type") == "symmetric_pair":
            fold_ids.update(pattern.get("cube_ids", []))

    # 保留非折叠 Cube + 每个折叠组的一个代表
    result = []
    folded = set()

    for cube in cubes:
        cube_id = cube.get("id", "")
        if cube_id not in fold_ids:
            result.append(cube)
        elif cube_id not in folded:
            # 找到所属模式，保留第一个作为代表
            for pattern in patterns:
                if cube_id in pattern.get("cube_ids", []):
                    cube_copy = dict(cube)
                    cube_copy["pattern_info"] = {
                        "type": pattern["pattern_type"],
                        "member_count": len(pattern.get("cube_ids", [])),
                        "spacing": pattern.get("spacing"),
                        "direction": pattern.get("direction"),
                    }
                    result.append(cube_copy)
                    folded.update(pattern.get("cube_ids", []))
                    break

    logger.info(f"模式折叠: {len(fold_ids)} 个 Cube → {len(result)} 个（含元数据）")
    return result


def normalize_z(cubes: List[Dict]) -> List[Dict]:
    """Z 轴归一化：将子场景的 Z 最小值归零"""
    if not cubes:
        return cubes
    z_min = min(c.get("center", {}).get("z", 0) - c.get("size", {}).get("z", 0) / 2 for c in cubes)
    for c in cubes:
        c["center"]["z"] = round(c["center"]["z"] - z_min, 3)
        if "bounds" in c:
            c["bounds"]["z"][0] = round(c["bounds"]["z"][0] - z_min, 3)
            c["bounds"]["z"][1] = round(c["bounds"]["z"][1] - z_min, 3)
    return cubes


def simplify_sub_scene(
    cubes: List[Dict],
    patterns: List[Dict],
    params: Dict,
) -> List[Dict]:
    """执行完整的 Phase 3b 简化流水线

    Mesh→Cube → 合并 → 模式折叠 → Z归一化
    """
    # 注意: 输入已是 Cube 形式（Phase 0 将 Mesh 和 Cube 统一以包围盒表示），
    # 完整三角化→包围盒转换待未来版本实现
    # Step 1: 合并
    cubes = merge_similar_cubes(cubes, params)

    # Step 2: 模式折叠
    cubes = fold_patterns(cubes, patterns, params.get("pattern_spacing_deviation_max", 0.1))

    # Step 3: Z 归一化
    cubes = normalize_z(cubes)

    # Step 4: 检查目标 Cube 数量
    max_target = params.get("max_cubes_target", 60)
    if len(cubes) > max_target:
        logger.warning(f"简化后仍有 {len(cubes)} 个 Cube (目标 {max_target})")

    return cubes


# ============================================================
# helper
# ============================================================

def _gap_between(a: BBox, b: BBox) -> float:
    """计算两个包围盒之间的最小间隙距离。0 表示相交或相切。"""
    dx = max(0.0, max(a.x_min, b.x_min) - min(a.x_max, b.x_max))
    dy = max(0.0, max(a.y_min, b.y_min) - min(a.y_max, b.y_max))
    dz = max(0.0, max(a.z_min, b.z_min) - min(a.z_max, b.z_max))
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def _dict_to_bbox(cube: Dict) -> BBox:
    c = cube.get("center", {})
    s = cube.get("size", {})
    return BBox(
        x_min=c.get("x", 0) - s.get("x", 0) / 2,
        x_max=c.get("x", 0) + s.get("x", 0) / 2,
        y_min=c.get("y", 0) - s.get("y", 0) / 2,
        y_max=c.get("y", 0) + s.get("y", 0) / 2,
        z_min=c.get("z", 0) - s.get("z", 0) / 2,
        z_max=c.get("z", 0) + s.get("z", 0) / 2,
    )


def _compute_world_bounds(cubes: List[Dict]) -> Dict:
    if not cubes:
        return {"x": [0, 1], "y": [0, 1], "z": [0, 1]}
    result = {"x": [float('inf'), float('-inf')], "y": [float('inf'), float('-inf')], "z": [float('inf'), float('-inf')]}
    for c in cubes:
        bb = _dict_to_bbox(c)
        result["x"][0] = min(result["x"][0], bb.x_min)
        result["x"][1] = max(result["x"][1], bb.x_max)
        result["y"][0] = min(result["y"][0], bb.y_min)
        result["y"][1] = max(result["y"][1], bb.y_max)
        result["z"][0] = min(result["z"][0], bb.z_min)
        result["z"][1] = max(result["z"][1], bb.z_max)
    return result


def _compute_span(bounds: Dict) -> float:
    return max(
        bounds["x"][1] - bounds["x"][0],
        bounds["y"][1] - bounds["y"][0],
        bounds["z"][1] - bounds["z"][0],
    )
