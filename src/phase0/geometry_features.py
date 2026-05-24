"""
几何特征提取器

从三角化碎片中提取建筑结构元素（墙面线、楼板面、开口、房间候选）。
不依赖语义标签——纯几何算法。

原理：
- 三角化碎片中的"薄而大"碎片保留了原始建筑元素的几何特征
- 墙面：薄在一个水平方向，大在另一个水平方向和垂直轴（up_axis）的碎片
- 楼板：薄在垂直方向（up_axis），大在两个水平方向的碎片
- 将同轴共线的碎片聚合为"墙面段"，从墙面段拓扑中检测房间和开口

up_axis 来自 USDA 文件头部的 upAxis 声明（"y" 或 "z"）。
"""

import logging
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)


# ── 碎片分类 ──────────────────────────────────────────────


def classify_fragments(cubes: List[Dict], up_axis: str = "y") -> Dict[str, List[Dict]]:
    """将碎片按几何特征分类

    Args:
        cubes: Cube 列表
        up_axis: USDA 文件声明的垂直轴 ("y" 或 "z")

    Returns:
        {"vertical_walls": [...], "horizontal_planes": [...], "other": [...]}
    """
    vertical_walls = []
    horizontal_planes = []
    other = []

    for c in cubes:
        s = c.get("size", {})
        dims = sorted([abs(s.get("x", 0)), abs(s.get("y", 0)), abs(s.get("z", 0))], reverse=True)
        if dims[0] <= 0:
            other.append(c)
            continue

        aspect_ratio = dims[0] / max(dims[2], 0.001)

        # 薄而大：一面远大于另一面（阈值适配米制数据，1m=100cm）
        if dims[0] > 0.005 and dims[2] < 0.05 and aspect_ratio > 5:
            # 区分垂直墙体和水平面：以 up_axis 为垂直参考轴
            vert_size = abs(s.get(up_axis, 0))
            horiz_axes = [a for a in ("x", "y", "z") if a != up_axis]
            max_horiz = max(abs(s.get(h, 0)) for h in horiz_axes)

            if vert_size > max_horiz:
                # 垂直方向尺寸最大 → 墙体（沿 up_axis 方向高耸）
                vertical_walls.append(c)
            elif vert_size < 0.05:
                # 垂直方向极薄 → 楼板/水平面
                horizontal_planes.append(c)
            else:
                other.append(c)
        else:
            other.append(c)

    return {
        "vertical_walls": vertical_walls,
        "horizontal_planes": horizontal_planes,
        "other": other,
    }


# ── 墙面线提取 ────────────────────────────────────────────


def extract_wall_lines(
    vertical_walls: List[Dict],
    axis: str = "x",
    up_axis: str = "y",
    min_fragments: int = 3,          # 最少碎片数
    position_tolerance: float = 0.5,  # 同一墙面线允许的位置偏差（m）
) -> List[Dict]:
    """从垂直墙面碎片中提取墙面线

    将沿同一 axis 坐标的墙面碎片聚合为墙面线。axis 必须是水平轴。

    Args:
        vertical_walls: 垂直墙面碎片列表
        axis: 墙面法线方向，必须是两个水平轴之一
        up_axis: USDA 文件声明的垂直轴 ("y" 或 "z")
        min_fragments: 最少碎片数才算有效墙面线
        position_tolerance: 同一墙面线上允许的位置偏差

    Returns:
        [{"position": float, "axis": str, "fragment_count": int,
          "{up_axis}_range": [min, max], "{perp_axis}_range": [min, max],
          "gaps": [{"perp_start": float, "perp_end": float, "gap_size": float}],
          "density": float}, ...]
    """
    horiz_axes = [a for a in ("x", "y", "z") if a != up_axis]
    if axis not in horiz_axes:
        raise ValueError(f"wall axis 必须是水平轴 {horiz_axes}，收到: {axis}")
    perp_axis = horiz_axes[1] if axis == horiz_axes[0] else horiz_axes[0]
    position_key = axis  # 墙面线的位置坐标（水平面第一轴）

    # 按墙面位置分组
    by_position = defaultdict(list)
    for w in vertical_walls:
        pos = round(w.get("center", {}).get(position_key, 0) / position_tolerance) * position_tolerance
        by_position[pos].append(w)

    # 聚合为墙面线
    wall_lines = []
    for pos, frags in list(by_position.items()):
        if len(frags) < min_fragments:
            continue

        # 沿墙面线方向排序
        frags_sorted = sorted(frags, key=lambda w: w.get("center", {}).get(perp_axis, 0))

        vert_vals = [w.get("center", {}).get(up_axis, 0) for w in frags]
        perp_vals = [w.get("center", {}).get(perp_axis, 0) for w in frags]

        vert_min, vert_max = min(vert_vals), max(vert_vals)
        perp_min, perp_max = min(perp_vals), max(perp_vals)

        # 检测间隙（可能的门/窗）
        gaps = []
        for i in range(len(frags_sorted) - 1):
            curr_end = frags_sorted[i].get("center", {}).get(perp_axis, 0) + \
                       abs(frags_sorted[i].get("size", {}).get(perp_axis, 0)) / 2
            next_start = frags_sorted[i + 1].get("center", {}).get(perp_axis, 0) - \
                         abs(frags_sorted[i + 1].get("size", {}).get(perp_axis, 0)) / 2
            gap = next_start - curr_end
            if gap > 0.5:  # > 0.5m 的间隙可能是开口（门/窗）
                gaps.append({
                    "perp_start": round(curr_end, 1),
                    "perp_end": round(next_start, 1),
                    "gap_size": round(gap, 1),
                })

        # 密度：碎片数 / 沿墙跨度
        perp_span = perp_max - perp_min
        density = len(frags) / max(perp_span, 0.01)

        wall_lines.append({
            "position": pos,
            "axis": axis,
            "fragment_count": len(frags),
            f"{up_axis}_range": [round(vert_min, 1), round(vert_max, 1)],
            f"{perp_axis}_range": [round(perp_min, 1), round(perp_max, 1)],
            "gaps": gaps,
            "density": round(density, 3),
        })

    wall_lines.sort(key=lambda w: -w["fragment_count"])
    return wall_lines


# ── 房间候选检测 ──────────────────────────────────────────


def detect_room_candidates(
    wall_a: List[Dict],
    wall_b: List[Dict],
    axis_a: str = "x",
    axis_b: str = "z",
    min_room_area: float = 1.0,      # 最小房间面积（m²）
    max_room_area: float = 500.0,    # 最大房间面积（m²），过滤超大假房间
    up_range: Optional[Tuple[float, float]] = None,  # 楼层 Z 范围，用于过滤墙面线
    up_axis: str = "z",
) -> List[Dict]:
    """从两组正交墙面线的交点网格中检测房间候选

    房间 = 四条墙面线围合的矩形区域。
    wall_a 和 wall_b 分别对应两个水平轴方向的墙面线。

    Args:
        wall_a: 水平轴 A 方向的墙面线列表
        wall_b: 水平轴 B 方向的墙面线列表
        axis_a: 水平轴 A 的名称 (e.g. "x")
        axis_b: 水平轴 B 的名称 (e.g. "z" 或 "y")
        min_room_area: 最小房间面积（单位²）
        max_room_area: 最大房间面积
        up_range: 可选，楼层 Z 范围 (lo, hi)，传入则只使用垂直覆盖此范围的墙面线
        up_axis: 垂直轴名称
    """
    # 若提供了楼层范围，先过滤垂直方向上不覆盖此范围的墙面线
    if up_range is not None:
        up_key = f"{up_axis}_range"
        wall_a = [w for w in wall_a if _check_vertical_overlap(w, up_range, up_key, min_coverage=0.2)]
        wall_b = [w for w in wall_b if _check_vertical_overlap(w, up_range, up_key, min_coverage=0.2)]

    a_positions = sorted(set(w["position"] for w in wall_a))
    b_positions = sorted(set(w["position"] for w in wall_b))

    if len(a_positions) < 2 or len(b_positions) < 2:
        return []

    rooms = []
    for i in range(len(a_positions) - 1):
        for j in range(len(b_positions) - 1):
            a0, a1 = a_positions[i], a_positions[i + 1]
            b0, b1 = b_positions[j], b_positions[j + 1]

            # 矩形面积
            area = abs(a1 - a0) * abs(b1 - b0)
            if area < min_room_area or area > max_room_area:
                continue

            # 检查四个边是否有墙面覆盖（含垂直方向过滤）
            has_left = _has_wall_coverage(wall_a, a0, axis_a, b0, b1, axis_b,
                                          up_range=up_range, up_axis=up_axis)
            has_right = _has_wall_coverage(wall_a, a1, axis_a, b0, b1, axis_b,
                                           up_range=up_range, up_axis=up_axis)
            has_bottom = _has_wall_coverage(wall_b, b0, axis_b, a0, a1, axis_a,
                                            up_range=up_range, up_axis=up_axis)
            has_top = _has_wall_coverage(wall_b, b1, axis_b, a0, a1, axis_a,
                                         up_range=up_range, up_axis=up_axis)

            covered_sides = sum([has_left, has_right, has_bottom, has_top])
            if covered_sides >= 3:  # 至少三面有墙 → 可能是房间
                rooms.append({
                    "bounds": {
                        axis_a: [a0, a1],
                        axis_b: [b0, b1],
                    },
                    "area": round(area, 1),
                    "covered_sides": covered_sides,
                    "open_sides": 4 - covered_sides,
                    "walls": {
                        "left": has_left,
                        "right": has_right,
                        "bottom": has_bottom,
                        "top": has_top,
                    },
                })

    rooms.sort(key=lambda r: -r["area"])
    return rooms


def _has_wall_coverage(
    walls: List[Dict],
    position: float,
    position_axis: str,
    range_start: float,
    range_end: float,
    range_axis: str,
    min_coverage: float = 0.4,
    up_range: Optional[Tuple[float, float]] = None,
    up_axis: str = "z",
    vert_min_coverage: float = 0.3,
) -> bool:
    """检查指定位置是否有墙面线覆盖给定的水平和垂直范围"""
    for w in walls:
        wp = w["position"]
        if abs(wp - position) > 0.5:  # 位置偏差容限（m）
            continue
        wr = w.get(f"{range_axis}_range", [0, 0])
        if not wr or wr[1] <= wr[0]:
            continue
        # 水平覆盖比例
        overlap_start = max(wr[0], range_start)
        overlap_end = min(wr[1], range_end)
        overlap = max(0, overlap_end - overlap_start)
        coverage = overlap / max(range_end - range_start, 0.01)
        if coverage < min_coverage:
            continue
        # 垂直覆盖检查（story-aware）
        if up_range is not None:
            up_key = f"{up_axis}_range"
            if not _check_vertical_overlap(w, up_range, up_key, vert_min_coverage):
                continue
        return True
    return False


def _check_vertical_overlap(
    wall: Dict,
    up_range: Tuple[float, float],
    up_key: str,
    min_coverage: float = 0.3,
) -> bool:
    """检查墙面的垂直范围与楼层 Z 范围是否有足够重叠"""
    w_up = wall.get(up_key, [0, 0])
    if not w_up or w_up[1] <= w_up[0]:
        return True  # 无垂直信息，保守保留
    overlap = max(0.0, min(w_up[1], up_range[1]) - max(w_up[0], up_range[0]))
    coverage = overlap / max(up_range[1] - up_range[0], 0.01)
    return coverage >= min_coverage


# ── 按楼层分组检测房间 ─────────────────────────────────────

def detect_rooms_by_story(
    wall_a: List[Dict],
    wall_b: List[Dict],
    stories: List[Dict],
    up_axis: str,
    axis_a: str,
    axis_b: str,
    min_room_area: float = 1.0,
    max_room_area: float = 500.0,
) -> Dict[str, List[Dict]]:
    """按楼层独立检测房间候选

    对每个 story（楼层空间），筛选垂直方向覆盖该楼层的墙面线后，
    单独运行房间检测。这消除了"假房间"——即墙面线在 2D 投影上围出
    矩形，但实际上这些墙面在不同的 Z 高度，根本围不出一个空间。

    Returns:
        {"F0": [{room with story_id, story_z_range}, ...], "F1": [...]}
    """
    result = {}
    horiz_axes = [a for a in ("x", "y", "z") if a != up_axis]
    for story in stories:
        z_range = story.get("z_range")
        if not z_range or len(z_range) != 2 or z_range[1] <= z_range[0]:
            continue
        up_range = (float(z_range[0]), float(z_range[1]))
        rooms = detect_room_candidates(
            wall_a=wall_a, wall_b=wall_b,
            axis_a=axis_a, axis_b=axis_b,
            min_room_area=min_room_area, max_room_area=max_room_area,
            up_range=up_range, up_axis=up_axis,
        )
        for room in rooms:
            room["story_id"] = story.get("story_id", "unknown")
            room["story_z_range"] = list(up_range)
            room["story_number"] = story.get("story_number")
        result[story.get("story_id", f"unknown")] = rooms
        logger.debug("Story %s (Z=%s): %d rooms",
                     story.get("story_id"), z_range, len(rooms))
    return result


# ── 走廊候选检测 ──────────────────────────────────────────

def detect_corridor_candidates(
    wall_lines_h1: List[Dict],
    wall_lines_h2: List[Dict],
    cubes: List[Dict],
    up_axis: str = "z",
    stories: Optional[List[Dict]] = None,
    room_candidates: Optional[List[Dict]] = None,
    min_length: float = 3.0,
    min_width: float = 1.0,
    max_width: float = 5.0,
    max_density: float = 1.0,
) -> List[Dict]:
    """从平行墙面线之间的空隙中检测走廊候选

    走廊特征：两条平行墙之间的长条形低密度区域。
    - 宽度在合理走廊范围（1~5m）
    - 长度至少 3m
    - 区域内 cube 密度低（可通行空间）
    - 长宽比 >= 2（长条形）
    """
    horiz_axes = [a for a in ("x", "y", "z") if a != up_axis]
    h1, h2 = horiz_axes[0], horiz_axes[1]
    corridor_candidates = []
    corridor_idx = 0

    def _process_wall_pairs(
        wall_lines: List[Dict],
        axis_a: str,        # 宽度方向轴
        axis_b: str,        # 长度方向轴
    ):
        nonlocal corridor_idx
        if len(wall_lines) < 2:
            return
        sorted_walls = sorted(wall_lines, key=lambda w: w["position"])
        for i in range(len(sorted_walls)):
            for j in range(i + 1, len(sorted_walls)):
                wa = sorted_walls[i]
                wb = sorted_walls[j]
                width = abs(wb["position"] - wa["position"])
                if width < min_width or width > max_width:
                    continue
                wa_b = wa.get(f"{axis_b}_range", [0, 0])
                wb_b = wb.get(f"{axis_b}_range", [0, 0])
                if wa_b[1] <= wa_b[0] or wb_b[1] <= wb_b[0]:
                    continue
                b_start = max(wa_b[0], wb_b[0])
                b_end = min(wa_b[1], wb_b[1])
                length = b_end - b_start
                if length < min_length:
                    continue
                # 垂直范围：两墙的 Z 重叠区
                wa_z = wa.get(f"{up_axis}_range", [0, 0])
                wb_z = wb.get(f"{up_axis}_range", [0, 0])
                z_lo = max(wa_z[0], wb_z[0])
                z_hi = min(wa_z[1], wb_z[1])
                if z_hi - z_lo < 1.5:
                    continue
                # 区域内 cube 密度
                pos_lo = min(wa["position"], wb["position"])
                pos_hi = max(wa["position"], wb["position"])
                corridor_cubes = [
                    c for c in cubes
                    if pos_lo <= c["center"].get(axis_a, 1e9) <= pos_hi
                    and b_start <= c["center"].get(axis_b, 1e9) <= b_end
                    and z_lo <= c["center"].get(up_axis, 1e9) <= z_hi
                ]
                density = len(corridor_cubes) / max(length, 0.1)
                if density > max_density:
                    continue
                # 分配楼层
                mid_z = (z_lo + z_hi) / 2
                story_number = None
                story_id = None
                if stories:
                    for s in stories:
                        zr = s.get("z_range", [0, 0])
                        if zr[0] <= mid_z <= zr[1]:
                            story_number = s.get("story_number")
                            story_id = s.get("story_id")
                            break
                # 连接房间
                connected_rooms = []
                if room_candidates:
                    bounds = {axis_a: [pos_lo, pos_hi], axis_b: [b_start, b_end]}
                    for ri, room in enumerate(room_candidates):
                        rb = room.get("bounds", {})
                        rh1 = rb.get(axis_a, [float('inf'), float('-inf')])
                        rh2 = rb.get(axis_b, [float('inf'), float('-inf')])
                        o1 = min(rh1[1], bounds[axis_a][1]) - max(rh1[0], bounds[axis_a][0])
                        o2 = min(rh2[1], bounds[axis_b][1]) - max(rh2[0], bounds[axis_b][0])
                        if o1 > 0 and o2 > 1.0:
                            connected_rooms.append(ri)
                        elif o2 > 0 and o1 > 1.0:
                            connected_rooms.append(ri)
                # 置信度
                aspect = length / max(width, 0.1)
                confidence = 0.20
                if aspect > 3:
                    confidence += 0.25
                elif aspect > 2:
                    confidence += 0.15
                if density < 0.5:
                    confidence += 0.20
                if story_number is not None:
                    confidence += 0.15
                if connected_rooms:
                    confidence += 0.10
                avg_frags = (wa.get("fragment_count", 0) + wb.get("fragment_count", 0)) / 2
                if avg_frags > 20:
                    confidence += 0.10
                confidence = min(1.0, confidence)

                bounds = {h1: [0.0, 0.0], h2: [0.0, 0.0]}
                bounds[axis_a] = [round(pos_lo, 1), round(pos_hi, 1)]
                bounds[axis_b] = [round(b_start, 1), round(b_end, 1)]
                corridor_candidates.append({
                    "corridor_id": f"COR_{corridor_idx:03d}",
                    "bounds": bounds,
                    "story_number": story_number,
                    "story_id": story_id,
                    "connected_room_ids": connected_rooms,
                    "width": round(width, 1),
                    "length": round(length, 1),
                    "area": round(width * length, 1),
                    "aspect_ratio": round(aspect, 1),
                    "density": round(density, 2),
                    "confidence": round(confidence, 2),
                    "z_range": [round(z_lo, 1), round(z_hi, 1)],
                    "axis": axis_a,
                    "direction": axis_b,
                })
                corridor_idx += 1

    _process_wall_pairs(wall_lines_h1, axis_a=h1, axis_b=h2)
    _process_wall_pairs(wall_lines_h2, axis_a=h2, axis_b=h1)
    corridor_candidates.sort(key=lambda c: -c["confidence"])
    return corridor_candidates


# ── 楼板面提取 ────────────────────────────────────────────


def extract_floor_planes(
    horizontal_planes: List[Dict],
    up_axis: str = "y",
    tolerance: float = 1.0,
    min_fragments: int = 5,
) -> List[Dict]:
    """从水平面碎片中提取楼板面

    将 up_axis 坐标相近的水平面碎片聚合为楼板面候选。
    """
    horiz_axes = [a for a in ("x", "y", "z") if a != up_axis]

    # 按 up_axis 坐标分组
    by_vert = defaultdict(list)
    for h in horizontal_planes:
        v_round = round(h.get("center", {}).get(up_axis, 0))
        by_vert[v_round].append(h)

    # 合并相邻 up_axis 组
    v_groups = sorted(by_vert.items())
    merged = []
    if v_groups:
        current_v = v_groups[0][0]
        current_frags = list(v_groups[0][1])
        for i in range(1, len(v_groups)):
            if v_groups[i][0] - v_groups[i - 1][0] <= tolerance:
                current_frags.extend(v_groups[i][1])
            else:
                if len(current_frags) >= min_fragments:
                    merged.append((current_v, current_frags))
                current_v = v_groups[i][0]
                current_frags = list(v_groups[i][1])
        if len(current_frags) >= min_fragments:
            merged.append((current_v, current_frags))

    planes = []
    for approx_v, frags in merged:
        v_vals = [w.get("center", {}).get(up_axis, 0) for w in frags]
        h1_vals = [w.get("center", {}).get(horiz_axes[0], 0) for w in frags]
        h2_vals = [w.get("center", {}).get(horiz_axes[1], 0) for w in frags]

        planes.append({
            f"{up_axis}_level": round(sum(v_vals) / len(v_vals), 1),
            "fragment_count": len(frags),
            f"{horiz_axes[0]}_range": [round(min(h1_vals), 1), round(max(h1_vals), 1)],
            f"{horiz_axes[1]}_range": [round(min(h2_vals), 1), round(max(h2_vals), 1)],
        })

    planes.sort(key=lambda p: p[f"{up_axis}_level"])
    return planes


# ── 主入口 ────────────────────────────────────────────────


def extract_all_features(cubes: List[Dict], unit_scale: float = 1.0,
                         up_axis: str = "y") -> Dict:
    """从碎片中提取所有几何特征

    Args:
        cubes: Cube 列表（坐标已转换为米制）
        unit_scale: 原始 USDA 的 metersPerUnit 值（1=米, 0.01=厘米），
                    用于日志记录。所有阈值基于米制数据。
        up_axis: USDA 文件声明的垂直轴 ("y" 或 "z")

    Returns:
        {
            "wall_lines_h1": [...],     # 水平轴 A 向墙面线
            "wall_lines_h2": [...],     # 水平轴 B 向墙面线
            "wall_line_count": int,
            "floor_planes": [...],      # 楼板面候选
            "room_candidates": [...],   # 房间候选
            "fragment_stats": {...},    # 碎片分类统计
        }
    """
    horiz_axes = [a for a in ("x", "y", "z") if a != up_axis]
    h1, h2 = horiz_axes[0], horiz_axes[1]

    classified = classify_fragments(cubes, up_axis=up_axis)

    walls_h1 = extract_wall_lines(classified["vertical_walls"], axis=h1, up_axis=up_axis)
    walls_h2 = extract_wall_lines(classified["vertical_walls"], axis=h2, up_axis=up_axis)
    floor_planes = extract_floor_planes(classified["horizontal_planes"], up_axis=up_axis)
    rooms = detect_room_candidates(walls_h1, walls_h2, axis_a=h1, axis_b=h2)

    # 走廊候选（初步几何检测，不含楼层信息；后续在 pipeline 中用 stories 增强）
    corridors = detect_corridor_candidates(
        walls_h1, walls_h2, cubes,
        up_axis=up_axis,
        stories=None,
        room_candidates=rooms,
    )

    logger.info(
        "几何特征提取 (up=%s): %s墙面线=%d, %s墙面线=%d, 楼板面=%d, 房间候选=%d, 走廊候选=%d",
        up_axis, h1, len(walls_h1), h2, len(walls_h2),
        len(floor_planes), len(rooms), len(corridors),
    )

    # 户外地形特征
    terrain = extract_terrain_features(cubes, classified["other"], up_axis=up_axis)

    return {
        f"wall_lines_{h1}": walls_h1,
        f"wall_lines_{h2}": walls_h2,
        "wall_line_count": len(walls_h1) + len(walls_h2),
        "floor_planes": floor_planes,
        "room_candidates": rooms,
        "corridor_candidates": corridors,
        "terrain_features": terrain,
        "fragment_stats": {
            "vertical_walls": len(classified["vertical_walls"]),
            "horizontal_planes": len(classified["horizontal_planes"]),
            "other": len(classified["other"]),
        },
    }


# ── 户外地形特征提取 ─────────────────────────────────────

def extract_terrain_features(cubes: List[Dict], other_fragments: List[Dict],
                             up_axis: str = "z") -> Dict:
    """从非建筑碎片中提取户外地形特征

    分析场景中的开放空间、地形起伏和外部结构，为战术分析提供
    外场环境信息（开阔地、掩体、制高点、坡度变化等）。

    Args:
        cubes: 全部 cube 列表
        other_fragments: 未被分类为墙体/楼板的碎片
        up_axis: 垂直轴
    """
    horiz_axes = [a for a in ("x", "y", "z") if a != up_axis]
    h1, h2 = horiz_axes[0], horiz_axes[1]

    result = {
        "open_areas": [],
        "elevation_profile": {},
        "terrain_type": "unknown",
        "exterior_cube_count": len(other_fragments),
    }

    if not cubes:
        return result

    # 1. 开放区域检测: 在水平面上找大范围低密度的连通区域
    open_areas = _detect_open_areas(cubes, up_axis=up_axis)
    result["open_areas"] = open_areas

    # 2. 高程剖面: 垂直轴上的分布统计
    up_vals = [c.get("center", {}).get(up_axis, 0) for c in other_fragments]
    if up_vals:
        up_sorted = sorted(up_vals)
        n = len(up_sorted)
        result["elevation_profile"] = {
            "min": round(up_sorted[0], 1),
            "max": round(up_sorted[-1], 1),
            "p25": round(up_sorted[n // 4], 1),
            "p50": round(up_sorted[n // 2], 1),
            "p75": round(up_sorted[3 * n // 4], 1),
            "total_range": round(up_sorted[-1] - up_sorted[0], 1),
        }

    # 3. 地形类型判定
    if open_areas:
        total_open_area = sum(a["area"] for a in open_areas)
        scene_area = _scene_horiz_area(cubes, h1, h2)
        if scene_area > 0 and total_open_area / scene_area > 0.5:
            result["terrain_type"] = "open_field"
        elif open_areas and max(a["area"] for a in open_areas) > 100:
            result["terrain_type"] = "mixed_terrain"
        else:
            result["terrain_type"] = "built_up"
    else:
        result["terrain_type"] = "dense_structure"

    # 4. 水平面密度概要
    h1_vals = [c.get("center", {}).get(h1, 0) for c in other_fragments]
    h2_vals = [c.get("center", {}).get(h2, 0) for c in other_fragments]
    if h1_vals and h2_vals:
        result["exterior_bounds"] = {
            h1: [round(min(h1_vals), 1), round(max(h1_vals), 1)],
            h2: [round(min(h2_vals), 1), round(max(h2_vals), 1)],
        }

    return result


def _detect_open_areas(cubes: List[Dict], up_axis: str = "z",
                       grid_cells: int = 20, min_open_cells: int = 4) -> List[Dict]:
    """在水平面上检测大范围低密度开放区域

    将水平面网格化，标记低密度连通区域为"开放区域"，
    代表外场的开阔地、广场、停车场等。
    """
    horiz_axes = [a for a in ("x", "y", "z") if a != up_axis]
    h1, h2 = horiz_axes[0], horiz_axes[1]

    h1_vals = [c.get("center", {}).get(h1, 0) for c in cubes]
    h2_vals = [c.get("center", {}).get(h2, 0) for c in cubes]
    if not h1_vals or not h2_vals:
        return []

    h1_min, h1_max = min(h1_vals), max(h1_vals)
    h2_min, h2_max = min(h2_vals), max(h2_vals)
    if h1_max <= h1_min or h2_max <= h2_min:
        return []

    h1_step = (h1_max - h1_min) / grid_cells
    h2_step = (h2_max - h2_min) / grid_cells

    # 每格 cube 计数
    density = [[0] * grid_cells for _ in range(grid_cells)]
    for c in cubes:
        g1 = min(int((c.get("center", {}).get(h1, 0) - h1_min) / h1_step), grid_cells - 1)
        g2 = min(int((c.get("center", {}).get(h2, 0) - h2_min) / h2_step), grid_cells - 1)
        density[g1][g2] += 1

    # 阈值: 低于中位密度的 25% 视为"开放"
    all_counts = [density[g1][g2] for g1 in range(grid_cells) for g2 in range(grid_cells) if density[g1][g2] > 0]
    if not all_counts:
        return []
    all_counts.sort()
    median_density = all_counts[len(all_counts) // 2]
    open_threshold = max(1, int(median_density * 0.25))

    # BFS 找低密度连通区域
    visited = [[False] * grid_cells for _ in range(grid_cells)]
    open_areas = []

    for g1 in range(grid_cells):
        for g2 in range(grid_cells):
            if visited[g1][g2] or density[g1][g2] > open_threshold:
                continue
            # BFS
            stack = [(g1, g2)]
            visited[g1][g2] = True
            cells = []
            total_cubes = 0
            while stack:
                c1, c2 = stack.pop()
                cells.append((c1, c2))
                total_cubes += density[c1][c2]
                for d1, d2 in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    n1, n2 = c1 + d1, c2 + d2
                    if 0 <= n1 < grid_cells and 0 <= n2 < grid_cells:
                        if not visited[n1][n2] and density[n1][n2] <= open_threshold:
                            visited[n1][n2] = True
                            stack.append((n1, n2))

            if len(cells) >= min_open_cells:
                cell_area = h1_step * h2_step
                area = len(cells) * cell_area
                center_h1 = h1_min + (sum(c[0] for c in cells) / len(cells)) * h1_step + h1_step / 2
                center_h2 = h2_min + (sum(c[1] for c in cells) / len(cells)) * h2_step + h2_step / 2

                open_areas.append({
                    "center": {h1: round(center_h1, 1), h2: round(center_h2, 1)},
                    "area": round(area, 1),
                    "cell_count": len(cells),
                    "avg_density": round(total_cubes / len(cells), 1),
                    "bounds": {
                        h1: [round(h1_min + min(c[0] for c in cells) * h1_step, 1),
                             round(h1_min + (max(c[0] for c in cells) + 1) * h1_step, 1)],
                        h2: [round(h2_min + min(c[1] for c in cells) * h2_step, 1),
                             round(h2_min + (max(c[1] for c in cells) + 1) * h2_step, 1)],
                    },
                })

    open_areas.sort(key=lambda a: -a["area"])
    return open_areas


def _scene_horiz_area(cubes: List[Dict], h1: str, h2: str) -> float:
    """估算场景水平面总面积"""
    if not cubes:
        return 0.0
    h1_vals = [c.get("center", {}).get(h1, 0) for c in cubes]
    h2_vals = [c.get("center", {}).get(h2, 0) for c in cubes]
    return (max(h1_vals) - min(h1_vals)) * (max(h2_vals) - min(h2_vals))
