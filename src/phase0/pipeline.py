"""
Phase 0 协调器

对外暴露单一入口函数 run_phase0()，
协调 USDA 流式解析 → Prim 构建 → 变换累乘 → Z 层分析 →
模式检测 → 几何特征提取 → 楼板面证据构建 → 材质分组。
"""

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any, List, Optional

from ..config import PHASE0_DIR, STAIRCASE_DETECTION_PARAMS, FLOOR_DETECTION_PARAMS
from ..core.geometry import world_bounds as compute_world_bounds
from ..core.geometry import is_horizontal_plane, is_vertical_plane
from ..core.usda_utils import read_usda_header
from .stream_parser import parse_usda_stream
from .prim_builder import build_prim_hierarchy, filter_cubes
from .transform_accumulator import TransformAccumulator
from .pattern_detector import detect_all_patterns
from .z_layer_analyzer import ZLayerAnalyzer
from .geometry_features import extract_all_features, detect_rooms_by_story, detect_corridor_candidates

logger = logging.getLogger(__name__)


def run_phase0(usda_path: str, output_dir: Optional[str] = None) -> str:
    """Phase 0 主入口

    从 USDA 文件头部读取 metersPerUnit 和 upAxis，动态确定
    单位缩放和垂直轴方向。
    """
    output_dir = Path(output_dir) if output_dir else PHASE0_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 0: 读取 USDA 头部元数据
    header = read_usda_header(usda_path)
    unit_scale = header["metersPerUnit"]
    usda_up_axis = header["upAxis"]
    logger.info("Phase 0: metersPerUnit=%.4f, usda_upAxis=%s (from file)",
                unit_scale, usda_up_axis)

    # Step 1: 流式解析
    logger.info("Step 1/8: 流式解析 USDA ...")
    prims = parse_usda_stream(usda_path)

    # Step 2: 构建层级 + 计算世界变换
    logger.info("Step 2/8: 计算世界变换与包围盒 ...")
    build_prim_hierarchy(prims)
    accumulator = TransformAccumulator()
    accumulator.compute_all(prims)

    # Step 3: 筛选几何体 + 单位标准化
    cube_prims = filter_cubes(prims)
    logger.info(f"Step 3/8: 筛选出 {len(cube_prims)} 个几何体 Prim (Cube/Mesh)")

    if abs(unit_scale - 1.0) > 0.0001:
        logger.info("坐标标准化 (metersPerUnit=%.4f) ...", unit_scale)
        for p in cube_prims:
            if p.world_bbox is not None:
                bbox = p.world_bbox
                bbox.x_min *= unit_scale; bbox.x_max *= unit_scale
                bbox.y_min *= unit_scale; bbox.y_max *= unit_scale
                bbox.z_min *= unit_scale; bbox.z_max *= unit_scale

    vertical_axis = usda_up_axis.lower()

    # Step 4: Z 层分析（空间聚类 + 楼梯检测，不再用直方图峰值做楼层检测）
    logger.info("Step 4/8: 垂直层分析 (upAxis=%s) ...", vertical_axis)
    z_analyzer = ZLayerAnalyzer(STAIRCASE_DETECTION_PARAMS, up_axis=vertical_axis,
                                floor_params=FLOOR_DETECTION_PARAMS)
    z_analysis = z_analyzer.analyze(cube_prims)

    # Step 5: 模式检测
    logger.info("Step 5/8: 模式检测 ...")
    patterns = detect_all_patterns(cube_prims, up_axis=vertical_axis)
    patterns["tentative_stairs"] = z_analysis.get("tentative_stairs", [])

    # Step 6: 几何特征提取（墙面线/楼板面/房间候选）
    logger.info("Step 6/8: 几何特征提取 ...")
    cubes_summary = _build_cubes_summary(cube_prims)
    geo_features = extract_all_features(cubes_summary, unit_scale=unit_scale,
                                        up_axis=vertical_axis)

    # Step 7: 楼板面证据构建（主证据：从 floor_planes 推断楼层）
    logger.info("Step 7/8: 楼板面证据构建 ...")
    floor_evidence = _build_floor_evidence(
        cubes_summary=cubes_summary,
        floor_planes=geo_features.get("floor_planes", []),
        wall_lines_x=geo_features.get(f"wall_lines_{'x' if vertical_axis != 'x' else 'y'}", []),
        wall_lines_y=geo_features.get(f"wall_lines_{'y' if vertical_axis != 'y' else 'z'}", []),
        up_axis=vertical_axis,
        params=FLOOR_DETECTION_PARAMS,
    )

    # ── Step 7b: 按楼层重新检测房间（利用 stories 过滤墙面线）──
    stories = floor_evidence.get("stories", [])
    if stories:
        logger.info("Step 7b: 按楼层检测房间 (story-aware) ...")
        horiz_p0 = [a for a in ("x", "y", "z") if a != vertical_axis]
        h1_p0, h2_p0 = horiz_p0[0], horiz_p0[1]
        walls_h1 = geo_features.get(f"wall_lines_{h1_p0}", [])
        walls_h2 = geo_features.get(f"wall_lines_{h2_p0}", [])
        story_rooms = detect_rooms_by_story(
            wall_a=walls_h1, wall_b=walls_h2,
            stories=stories, up_axis=vertical_axis,
            axis_a=h1_p0, axis_b=h2_p0,
        )
        geo_features["story_rooms"] = story_rooms
        total_rooms = sum(len(r) for r in story_rooms.values())
        logger.info("  %d 个房间分配到 %d 个楼层", total_rooms, len(stories))

        # ── Step 7c: 按楼层增强走廊检测 ──
        logger.info("Step 7c: 按楼层增强走廊检测 ...")
        enhanced_corridors = detect_corridor_candidates(
            wall_lines_h1=walls_h1,
            wall_lines_h2=walls_h2,
            cubes=cubes_summary,
            up_axis=vertical_axis,
            stories=stories,
            room_candidates=geo_features.get("room_candidates", []),
        )
        geo_features["corridor_candidates"] = enhanced_corridors
        logger.info("  %d 个走廊候选", len(enhanced_corridors))

        # ── Step 7d: 跨层密度楼梯检测 ──
        logger.info("Step 7d: 跨层密度楼梯检测 ...")
        density_stairs = z_analyzer.detect_stairwells_by_density(cube_prims, stories)
        if density_stairs:
            logger.info("  %d 个楼梯间 (密度法)", len(density_stairs))
            patterns["density_stairs"] = density_stairs
        else:
            patterns["density_stairs"] = []

    # Step 8: 材质分组（几何预分类，不让 LLM 猜材质功能）
    logger.info("Step 8/8: 材质几何预分类 ...")
    material_groups = _build_material_groups(cube_prims, up_axis=vertical_axis)

    # 组装输出
    wb = compute_world_bounds(cube_prims)
    metadata = _build_metadata(
        usda_path=usda_path,
        cube_prims=cube_prims,
        world_bounds=wb,
        z_analysis=z_analysis,
        patterns=patterns,
        floor_evidence=floor_evidence,
        material_groups=material_groups,
        geo_features=geo_features,
        up_axis=vertical_axis,
    )
    metadata["usda_config"] = {
        "meters_per_unit": unit_scale,
        "usda_up_axis": usda_up_axis,
        "world_vertical_axis": vertical_axis,
    }

    output_path = output_dir / "scene_metadata.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    logger.info("Phase 0 完成: %s (%d 几何体, %d 楼板面候选, %d 楼板层级, %d 可居住楼层)",
                output_path, len(cube_prims),
                len(floor_evidence.get("floor_candidates", [])),
                floor_evidence.get("logical_slab_count", 0),
                floor_evidence.get("story_count", 0))
    return str(output_path)


# ── Cube 摘要 ──────────────────────────────────────────────

def _build_cubes_summary(cube_prims: list) -> List[Dict]:
    """从 PrimRecord 列表构建 cube 摘要（保持 Phase 3 兼容格式）"""
    result = []
    for p in cube_prims:
        if p.world_bbox is None:
            continue
        result.append({
            "id": p.prim_path,
            "parent": p.parent_path,
            "bounds": {
                "x": [round(p.world_bbox.x_min, 3), round(p.world_bbox.x_max, 3)],
                "y": [round(p.world_bbox.y_min, 3), round(p.world_bbox.y_max, 3)],
                "z": [round(p.world_bbox.z_min, 3), round(p.world_bbox.z_max, 3)],
            },
            "center": {
                "x": round(p.world_bbox.center.x, 3),
                "y": round(p.world_bbox.center.y, 3),
                "z": round(p.world_bbox.center.z, 3),
            },
            "size": {
                "x": round(p.world_bbox.width, 3),
                "y": round(p.world_bbox.depth, 3),
                "z": round(p.world_bbox.height, 3),
            },
            "material": p.material_binding or "unknown",
        })
    return result


# ── 楼板面证据构建 ──────────────────────────────────────────

def _build_floor_evidence(
    cubes_summary: List[Dict],
    floor_planes: List[Dict],
    wall_lines_x: List[Dict],
    wall_lines_y: List[Dict],
    up_axis: str,
    params: Dict,
) -> Dict[str, Any]:
    """从 floor_planes（薄水平面碎片聚合）构建结构化楼层候选

    这是替代直方图峰值法的核心：floor_planes 是算法从实际几何中
    检测到的水平面层级，比直方图统计峰值更可靠。

    Returns:
        {
            "floor_candidates": [...],   # 按 Z 排序的候选楼层
            "inter_floor_gaps": [...],   # 楼层间空白区间
            "primary_floor_count": int,  # 基于 STRONG+MEDIUM 证据的初步推断
        }
    """
    all_wall_lines = wall_lines_x + wall_lines_y
    horiz_axes = [a for a in ("x", "y", "z") if a != up_axis]
    h1, h2 = horiz_axes[0], horiz_axes[1]
    up_level_key = f"{up_axis}_level"
    z_tol = params.get("floor_candidate_z_tolerance", 2.0)

    # 计算建筑参考面积（从所有 cube 的水平面投影）
    h1_vals = [c["center"][h1] for c in cubes_summary]
    h2_vals = [c["center"][h2] for c in cubes_summary]
    building_h1_span = max(h1_vals) - min(h1_vals) if h1_vals else 1
    building_h2_span = max(h2_vals) - min(h2_vals) if h2_vals else 1
    building_ref_area = building_h1_span * building_h2_span

    # 阈值
    strong_frags = params.get("strong_evidence_min_fragments", 1000)
    strong_walls = params.get("strong_evidence_min_wall_intersections", 5)
    strong_cov = params.get("strong_evidence_min_horiz_coverage", 0.40)
    medium_frags = params.get("medium_evidence_min_fragments", 200)
    medium_walls = params.get("medium_evidence_min_wall_intersections", 2)
    medium_cov = params.get("medium_evidence_min_horiz_coverage", 0.20)

    floor_candidates = []

    for fp in floor_planes:
        z_level = fp.get(up_level_key, 0)
        frag_count = fp.get("fragment_count", 0)
        h1_range = fp.get(f"{h1}_range", [0, 0])
        h2_range = fp.get(f"{h2}_range", [0, 0])

        # 围绕该 Z 层级收集 cube 证据
        z_band_cubes = [
            c for c in cubes_summary
            if abs(c["center"][up_axis] - z_level) <= z_tol
        ]

        # 水平覆盖率
        if building_ref_area > 0:
            horiz_area = abs(h1_range[1] - h1_range[0]) * abs(h2_range[1] - h2_range[0])
            horiz_coverage = min(1.0, horiz_area / building_ref_area)
        else:
            horiz_coverage = 0.0

        # 墙面线交叉：有多少墙面线的垂直范围包含此 Z 层级
        intersecting_walls = sum(
            1 for w in all_wall_lines
            if w.get(f"{up_axis}_range", [0, 0])[0] <= z_level <= w.get(f"{up_axis}_range", [0, 0])[1]
        )

        # 该 Z 层级的碎片材质统计
        mat_counts = defaultdict(int)
        for c in z_band_cubes:
            mat_counts[c.get("material", "unknown")] += 1
        top_materials = [
            {"name": m, "count": cnt}
            for m, cnt in sorted(mat_counts.items(), key=lambda x: -x[1])[:5]
        ]

        # 证据强度判定
        if (frag_count >= strong_frags and intersecting_walls >= strong_walls
                and horiz_coverage >= strong_cov):
            evidence = "strong"
        elif (frag_count >= medium_frags or intersecting_walls >= medium_walls
                or horiz_coverage >= medium_cov):
            evidence = "medium"
        else:
            evidence = "weak"

        floor_candidates.append({
            "z_level": round(z_level, 1),
            "fragment_count": frag_count,
            "cube_count_in_band": len(z_band_cubes),
            "horizontal_span": {
                h1: [round(h1_range[0], 1), round(h1_range[1], 1)],
                h2: [round(h2_range[0], 1), round(h2_range[1], 1)],
            },
            "horizontal_coverage": round(horiz_coverage, 3),
            "intersecting_wall_lines": intersecting_walls,
            "supporting_materials": top_materials,
            "evidence_strength": evidence,
        })

    # 按 Z 排序
    floor_candidates.sort(key=lambda fc: fc["z_level"])

    # 计算楼层间空白区间
    inter_floor_gaps = []
    gap_min = params.get("inter_floor_gap_min", 2.5)
    gap_max = params.get("inter_floor_gap_max", 8.0)
    density_warn = params.get("inter_floor_density_warning", 100)

    for i in range(len(floor_candidates) - 1):
        z_lo = floor_candidates[i]["z_level"]
        z_hi = floor_candidates[i + 1]["z_level"]
        gap = z_hi - z_lo

        # 统计空白区间内的 cube 数量
        gap_cubes = sum(
            1 for c in cubes_summary
            if z_lo < c["center"][up_axis] < z_hi
        )
        density = gap_cubes / max(gap, 0.1)

        status = "normal"
        if gap < gap_min:
            status = "too_narrow"
        elif gap > gap_max:
            status = "too_wide_possible_missing_floor"

        inter_floor_gaps.append({
            "between": [round(z_lo, 2), round(z_hi, 2)],
            "gap_meters": round(gap, 2),
            "cube_count": gap_cubes,
            "cube_density": round(density, 1),
            "status": status,
        })

    # ── 二级合并：将间距 < merge_threshold 的候选合并为逻辑楼板层级 ──
    merge_threshold = params.get("logical_floor_merge_threshold", 6.0)
    logical_slabs = _merge_adjacent_candidates(
        floor_candidates, merge_threshold, up_axis,
    )
    logical_slab_count = len(logical_slabs)

    # ── 三级：楼板层级 → 可居住楼层（N块板 = N-1层楼） ──
    # 每对相邻楼板之间构成一个可使用楼层空间。
    # 最低板 = 地面层地板，最高板 = 顶层天花板/屋顶，
    # 中间的板 = 下层天花板 + 上层地板（同一块板的两面）。
    stories = _build_stories_from_slabs(logical_slabs, cubes_summary, all_wall_lines, up_axis)
    story_count = len(stories)

    return {
        "floor_candidates": floor_candidates,
        "inter_floor_gaps": inter_floor_gaps,
        "logical_slab_count": logical_slab_count,
        "logical_slabs": logical_slabs,
        "story_count": story_count,
        "stories": stories,
        "merge_threshold_used": merge_threshold,
        "source": "floor_planes_geometry",
    }


# ── 逻辑楼层合并 ──────────────────────────────────────────

def _merge_adjacent_candidates(
    floor_candidates: List[Dict],
    merge_threshold: float,
    up_axis: str,
) -> List[Dict]:
    """将 Z 间距 < merge_threshold 的相邻楼板面候选合并为逻辑楼层

    合并规则：
    - 按 Z 排序后，相邻间距 < merge_threshold → 视为同一逻辑楼层
    - 合并后的 Z 取值：优先取 STRONG 证据的 Z 层级；都是同一等级则取碎片数加权平均
    - 碎片数和 cube_count 取组内总和
    - 水平覆盖范围取组内所有候选的并集
    - 证据强度取组内最高等级
    """
    if not floor_candidates:
        return []

    gap_min = merge_threshold
    groups = []
    current_group = [floor_candidates[0]]

    for i in range(1, len(floor_candidates)):
        gap = floor_candidates[i]["z_level"] - floor_candidates[i - 1]["z_level"]
        if gap < gap_min:
            current_group.append(floor_candidates[i])
        else:
            groups.append(current_group)
            current_group = [floor_candidates[i]]
    groups.append(current_group)

    logical_floors = []
    for g_idx, group in enumerate(groups):
        # 选主候选：STRONG 优先，否则碎片数最多
        strong = [c for c in group if c["evidence_strength"] == "strong"]
        medium = [c for c in group if c["evidence_strength"] == "medium"]
        primary_pool = strong if strong else (medium if medium else group)
        primary = max(primary_pool, key=lambda c: c["fragment_count"])

        # Z 层级：碎片数加权平均
        total_frags = sum(c["fragment_count"] for c in group)
        if total_frags > 0:
            z_weighted = sum(c["z_level"] * c["fragment_count"] for c in group) / total_frags
        else:
            z_weighted = primary["z_level"]

        # Z 范围：组内 min/max 各外扩 2m（覆盖整个楼层高度）
        z_vals = [c["z_level"] for c in group]
        z_range = [round(min(z_vals) - 2.0, 1), round(max(z_vals) + 2.0, 1)]

        # 证据等级：取组内最高
        if strong:
            evidence = "strong"
        elif medium:
            evidence = "medium"
        else:
            evidence = "weak"

        # 合并覆盖范围
        merged = {
            "floor_id": f"F{g_idx}",
            "floor_number": g_idx,
            "z_level_weighted": round(z_weighted, 1),
            "z_range": z_range,
            "fragment_count": sum(c["fragment_count"] for c in group),
            "cube_count_in_band": sum(c["cube_count_in_band"] for c in group),
            "horizontal_coverage": max(c["horizontal_coverage"] for c in group),
            "intersecting_wall_lines": max(c["intersecting_wall_lines"] for c in group),
            "evidence_strength": evidence,
            "source_candidates": len(group),
            "source_z_levels": [c["z_level"] for c in group],
        }
        logical_floors.append(merged)

    return logical_floors


# ── 楼层空间构建（楼板层级 → 可居住楼层） ──────────────────

def _build_stories_from_slabs(
    logical_slabs: List[Dict],
    cubes_summary: List[Dict],
    all_wall_lines: List[Dict],
    up_axis: str,
) -> List[Dict]:
    """从楼板层级构建可居住楼层空间

    核心逻辑：N 块水平楼板 = N-1 个可居住楼层。
    - 最低板 → 地面层地板（F0 踩在上面）
    - 中间板 → 下层天花板 + 上层地板（同一块板的双面）
    - 最高板 → 顶层天花板 / 屋顶

    每个楼层空间 = 相邻两块板之间的垂直区间。
    """
    if len(logical_slabs) < 1:
        return []
    if len(logical_slabs) == 1:
        # 只有一块板 → 可能是单层建筑（地面板以上算一层 + 无屋顶板检出）
        slab = logical_slabs[0]
        z_lo = slab["z_level_weighted"]
        z_hi = z_lo + 4.0
        return [{
            "story_id": "F0",
            "story_number": 0,
            "floor_slab_z": z_lo,
            "ceiling_slab_z": None,
            "z_range": [z_lo, z_hi],
            "height_meters": 4.0,
            "evidence": slab["evidence_strength"],
            "cube_count": sum(
                1 for c in cubes_summary
                if z_lo <= c["center"][up_axis] <= z_hi
            ),
            "wall_line_count": sum(
                1 for w in all_wall_lines
                if w.get(f"{up_axis}_range", [0, 0])[0] <= z_hi
                and w.get(f"{up_axis}_range", [0, 0])[1] >= z_lo
            ),
            "floor_slab_source": slab.get("source_z_levels", [z_lo]),
            "ceiling_slab_source": [],
        }]

    stories = []
    for i in range(len(logical_slabs) - 1):
        lower = logical_slabs[i]       # 地板板
        upper = logical_slabs[i + 1]   # 天花板板 (= 上层地板板)

        z_floor = lower["z_level_weighted"]
        z_ceiling = upper["z_level_weighted"]
        story_height = z_ceiling - z_floor

        # 统计此楼层空间内的 cube 数量和墙面线
        story_cubes = sum(
            1 for c in cubes_summary
            if z_floor <= c["center"][up_axis] <= z_ceiling
        )
        story_walls = sum(
            1 for w in all_wall_lines
            if w.get(f"{up_axis}_range", [0, 0])[0] <= z_ceiling
            and w.get(f"{up_axis}_range", [0, 0])[1] >= z_floor
        )

        # 证据：地板证据优先，辅以天花板
        if lower["evidence_strength"] == "strong" or upper["evidence_strength"] == "strong":
            evidence = "strong"
        elif lower["evidence_strength"] == "medium" or upper["evidence_strength"] == "medium":
            evidence = "medium"
        else:
            evidence = "weak"

        stories.append({
            "story_id": f"F{i}",
            "story_number": i,
            "floor_slab_z": round(z_floor, 1),
            "ceiling_slab_z": round(z_ceiling, 1),
            "z_range": [round(z_floor, 1), round(z_ceiling, 1)],
            "height_meters": round(story_height, 1),
            "evidence": evidence,
            "cube_count": story_cubes,
            "wall_line_count": story_walls,
            "floor_slab_source": lower.get("source_z_levels", []),
            "ceiling_slab_source": upper.get("source_z_levels", []),
        })

    return stories


# ── 材质几何预分类 ──────────────────────────────────────────

def _build_material_groups(cube_prims: list, up_axis: str = "z") -> List[Dict]:
    """按材质分组，并用几何特征预分类每个材质的角色

    不让 LLM 从材质名猜功能——用 is_horizontal_plane / is_vertical_plane
    判定碎片几何角色，然后给材质打标签。
    """
    # 收集: {material_name: [bbox, ...]}
    mat_bboxes = defaultdict(list)
    mat_z_values = defaultdict(list)
    for p in cube_prims:
        if p.world_bbox is None:
            continue
        mat = p.material_binding or "unknown"
        mat_bboxes[mat].append(p.world_bbox)
        mat_z_values[mat].append(getattr(p.world_bbox.center, up_axis))

    groups = []
    for mat, bboxes in mat_bboxes.items():
        n = len(bboxes)
        if n < 5:
            continue  # 少于 5 个碎片的材质忽略

        # 几何角色分类
        horiz_count = sum(1 for b in bboxes if is_horizontal_plane(b, up_axis=up_axis))
        vert_count = sum(1 for b in bboxes if is_vertical_plane(b, up_axis=up_axis))
        horiz_ratio = horiz_count / n
        vert_ratio = vert_count / n

        if horiz_ratio >= 0.6:
            geometric_role = "horizontal_plane"
        elif vert_ratio >= 0.6:
            geometric_role = "vertical_plane"
        else:
            geometric_role = "mixed"

        # Z 分布模式: discrete（集中在离散层级）或 continuous（连续跨度）
        z_vals = mat_z_values[mat]
        z_range = max(z_vals) - min(z_vals) if z_vals else 1
        z_mean_density = n / max(z_range, 0.1)

        if z_range > 20 and z_mean_density < 50:
            z_distribution = "continuous"
        elif n >= 50 and z_mean_density >= 50:
            z_distribution = "discrete_concentrated"
        else:
            z_distribution = "sparse"

        groups.append({
            "material": mat,
            "cube_count": n,
            "geometric_role": geometric_role,
            "horiz_plane_ratio": round(horiz_ratio, 2),
            "vert_plane_ratio": round(vert_ratio, 2),
            f"{up_axis}_range": [round(min(z_vals), 1), round(max(z_vals), 1)],
            "z_distribution": z_distribution,
            "z_density": round(z_mean_density, 1),
        })

    groups.sort(key=lambda g: -g["cube_count"])
    return groups


# ── 元数据组装 ──────────────────────────────────────────────

def _build_metadata(
    usda_path: str,
    cube_prims: list,
    world_bounds,
    z_analysis: Dict,
    patterns: Dict,
    floor_evidence: Dict,
    material_groups: List[Dict],
    geo_features: Dict,
    up_axis: str = "z",
) -> Dict[str, Any]:
    """构建 scene_metadata.json"""

    cubes_summary = _build_cubes_summary(cube_prims)

    return {
        "source_file": usda_path,
        "phase": 0,
        "world_bounds": world_bounds.to_dict() if world_bounds else {},
        "total_prims": len(cube_prims),

        # 垂直分析：楼层 = 楼板层级间的空间，即 N块板 = N-1层楼
        "vertical_analysis": {
            "analyzed_axis": up_axis,
            "floor_count": floor_evidence.get("story_count", 0),
            "floor_boundaries": [
                s["z_range"]
                for s in floor_evidence.get("stories", [])
            ],
            "floor_slab_count": floor_evidence.get("logical_slab_count",
                                                   len(floor_evidence.get("logical_slabs", []))),
            "vertical_complexity": _classify_complexity(
                floor_evidence.get("story_count", 0),
                len(z_analysis.get("spatial_clusters", [])),
            ),
            "_histogram_raw": z_analysis.get("vertical_histogram", {}),
            "spatial_clusters": z_analysis.get("spatial_clusters", []),
        },

        # 楼板面证据（替代直方图峰值作为主要楼层信号）
        "floor_evidence": floor_evidence,

        # 材质分组（几何预分类）
        "material_groups": material_groups,

        # 模式检测
        "patterns": {
            "linear_array_count": len(patterns.get("linear_arrays", [])),
            "linear_arrays": [
                {
                    "pattern_type": p.pattern_type,
                    "cube_ids": p.cube_ids,
                    "spacing": p.spacing,
                    "direction": p.direction,
                    "confidence": p.confidence,
                }
                for p in patterns.get("linear_arrays", [])
            ],
            "symmetric_pair_count": len(patterns.get("symmetric_pairs", [])),
            "symmetric_pairs": [
                {
                    "pattern_type": p.pattern_type,
                    "cube_ids": p.cube_ids,
                    "direction": p.direction,
                    "confidence": p.confidence,
                }
                for p in patterns.get("symmetric_pairs", [])
            ],
            "dense_cluster_count": len(patterns.get("dense_clusters", [])),
            "dense_clusters": [
                {
                    "pattern_type": p.pattern_type,
                    "cube_ids": p.cube_ids,
                    "confidence": p.confidence,
                }
                for p in patterns.get("dense_clusters", [])
            ],
            "tentative_stairs": patterns.get("tentative_stairs", []),
            "density_stairs": patterns.get("density_stairs", []),
        },

        # 几何特征
        "geometry_features": geo_features,

        # Cube 摘要（Phase 3 兼容格式）
        "cubes": cubes_summary,
    }


def _classify_complexity(floor_count: int, cluster_count: int) -> str:
    if floor_count >= 3 or cluster_count > 2:
        return "complex"
    if floor_count >= 2:
        return "moderate"
    return "simple"


# ── CLI 入口 ────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Phase 0: 流式 USDA 解析 → scene_metadata.json"
    )
    parser.add_argument("usda_path", help="原始 USDA 场景文件路径")
    parser.add_argument(
        "--output-dir", "-o",
        default=None,
        help=f"输出目录 (默认: {PHASE0_DIR})",
    )
    parser.add_argument(
        "--log-level", "-l",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if not Path(args.usda_path).exists():
        print(f"错误: 文件不存在: {args.usda_path}", file=sys.stderr)
        sys.exit(1)

    output = run_phase0(args.usda_path, output_dir=args.output_dir)
    print(f"Phase 0 完成: {output}")
