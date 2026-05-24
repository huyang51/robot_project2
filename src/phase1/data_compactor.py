"""
Phase 1 数据精简器

将 Phase 0 的纯几何 scene_metadata.json 精简为适合 LLM 输入的紧凑格式。
输入数据不含 semantic type —— 仅从空间分布和尺寸统计中提取结构信号。
"""

import json
from typing import Dict, Any, List
from collections import Counter


def compact_scene_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """将 Phase 0 元数据精简为 LLM 友好的紧凑格式

    策略:
    - 若 Phase 0 检测到多个 XZ 空间聚类（如分离的地形和建筑），
      则按 (空间聚类, 楼层) 分组，每组独立统计
    - 否则按楼层分组（回退行为）
    - 每层提供体积/尺寸统计、密度热力图、空间密度
    - 选择代表性几何体
    - 保留模式检测和楼梯候选
    """
    v_analysis = metadata.get("vertical_analysis", metadata.get("z_analysis", {}))
    spatial_clusters = v_analysis.get("spatial_clusters", [])

    compact = {
        "source_file": metadata.get("source_file", ""),
        "world_bounds": metadata.get("world_bounds", {}),
        "total_prims": metadata.get("total_prims", 0),
        "floor_info": _compact_floor_info(metadata, spatial_clusters),
        "patterns_summary": metadata.get("patterns", {}),
        "vertical_analysis": v_analysis,
        "analyzed_axis": v_analysis.get("analyzed_axis", "z"),
        "geometry_features": metadata.get("geometry_features", {}),
        "floor_evidence": metadata.get("floor_evidence", {}),
        "material_groups": metadata.get("material_groups", []),
    }

    if spatial_clusters:
        compact["spatial_clusters"] = spatial_clusters

    return compact


def _compact_floor_info(metadata: Dict[str, Any],
                        spatial_clusters: List[Dict] = None) -> List[Dict[str, Any]]:
    """按 (空间聚类, 楼层) 分组几何体，提供尺寸统计和代表性样本

    若 Phase 0 检测到多个水平面空间聚类（如分离的地形和建筑），
    则按聚类分组后再按楼层分组，每组独立统计。
    这确保户外地形和室内建筑的统计数据不会混合。
    """
    v_analysis = metadata.get("vertical_analysis", metadata.get("z_analysis", {}))
    floor_boundaries = v_analysis.get("floor_boundaries", [])
    cubes = metadata.get("cubes", [])
    analyzed_axis = v_analysis.get("analyzed_axis", "z")
    horiz_axes = [a for a in ("x", "y", "z") if a != analyzed_axis]
    h1, h2 = horiz_axes[0], horiz_axes[1]

    if not floor_boundaries:
        # 无楼层边界时，使用全量数据
        return [{
            "floor_number": 0,
            "axis_range": [
                metadata.get("world_bounds", {}).get(analyzed_axis, [0, 0])[0],
                metadata.get("world_bounds", {}).get(analyzed_axis, [0, 0])[1],
            ],
            "cube_count": len(cubes),
            "size_stats": _compute_size_stats(cubes),
            "material_summary": _material_summary(cubes),
            "representative_cubes": _select_representative(cubes, analyzed_axis=analyzed_axis),
        }]

    # 若有多空间聚类，按聚类分组后再按楼层分组
    if spatial_clusters and len(spatial_clusters) > 1:
        floor_infos = []
        for cl in spatial_clusters:
            cl_label = cl.get("label", f"region_{cl.get('cluster_id', '?')}")
            cl_h1 = cl.get(f"{h1}_range", [float('-inf'), float('inf')])
            cl_h2 = cl.get(f"{h2}_range", [float('-inf'), float('inf')])
            # 筛选属于此空间聚类的 Cube
            cluster_cubes = [
                c for c in cubes
                if cl_h1[0] <= c.get("center", {}).get(h1, 0) <= cl_h1[1]
                and cl_h2[0] <= c.get("center", {}).get(h2, 0) <= cl_h2[1]
            ]
            # 在此聚类内按楼层分组
            for fi, fb in enumerate(cl.get("floor_boundaries", floor_boundaries)):
                v_min, v_max = fb[0], fb[1]
                floor_cubes = [
                    c for c in cluster_cubes
                    if v_min <= c.get("center", {}).get(analyzed_axis, 0) <= v_max
                ]
                if not floor_cubes:
                    continue
                floor_infos.append({
                    "floor_number": len(floor_infos),
                    "cluster_id": cl.get("cluster_id", 0),
                    "cluster_label": cl_label,
                    "vertical_range": [v_min, v_max],
                    "analyzed_axis": analyzed_axis,
                    "cube_count": len(floor_cubes),
                    "size_stats": _compute_size_stats(floor_cubes),
                    "material_summary": _material_summary(floor_cubes),
                    "representative_cubes": _select_representative(floor_cubes, analyzed_axis=analyzed_axis),
                    "spatial_density": _describe_spatial_density(floor_cubes, analyzed_axis=analyzed_axis),
                    "density_heatmap": _compute_density_heatmap(floor_cubes, analyzed_axis=analyzed_axis),
                })
        return floor_infos

    # 无多聚类：按楼层分组（回退行为）
    floor_infos = []
    for i, (v_min, v_max) in enumerate(floor_boundaries):
        floor_cubes = [
            c for c in cubes
            if v_min <= c.get("center", {}).get(analyzed_axis, 0) <= v_max
        ]

        floor_infos.append({
            "floor_number": i,
            "vertical_range": [v_min, v_max],
            "analyzed_axis": analyzed_axis,
            "cube_count": len(floor_cubes),
            "size_stats": _compute_size_stats(floor_cubes),
            "material_summary": _material_summary(floor_cubes),
            "representative_cubes": _select_representative(floor_cubes, analyzed_axis=analyzed_axis),
            "spatial_density": _describe_spatial_density(floor_cubes, analyzed_axis=analyzed_axis),
            "density_heatmap": _compute_density_heatmap(floor_cubes, analyzed_axis=analyzed_axis),
        })

    return floor_infos


def _compute_size_stats(cubes: List[Dict]) -> Dict[str, Any]:
    """计算一层内几何体的尺寸统计"""
    if not cubes:
        return {"count": 0}

    volumes = []
    widths = []
    depths = []
    heights = []
    for c in cubes:
        s = c.get("size", {})
        v = abs(s.get("x", 0) * s.get("y", 0) * s.get("z", 0))
        volumes.append(v)
        widths.append(abs(s.get("x", 0)))
        depths.append(abs(s.get("y", 0)))
        heights.append(abs(s.get("z", 0)))

    volumes.sort()
    heights.sort()

    n = len(volumes)
    return {
        "count": n,
        "volume": {
            "min": round(volumes[0], 4),
            "median": round(volumes[n // 2], 4),
            "max": round(volumes[-1], 4),
        },
        "width_range": [round(min(widths), 3), round(max(widths), 3)],
        "depth_range": [round(min(depths), 3), round(max(depths), 3)],
        "height_range": [round(min(heights), 3), round(max(heights), 3)],
    }


def _material_summary(cubes: List[Dict], top_n: int = 5) -> List[Dict]:
    """材质统计摘要"""
    mat_counts = Counter(c.get("material", "unknown") for c in cubes)
    return [
        {"material": m, "count": c}
        for m, c in mat_counts.most_common(top_n)
    ]


def _select_representative(cubes: List[Dict], count: int = 12,
                          analyzed_axis: str = "z") -> List[Dict]:
    """选择代表性几何体：体积 + 空间象限 + 极端尺寸 + 中位

    新策略（count=12）:
    - 2 个体积最大的（代表结构元素：楼板/大墙体）
    - 4 个空间象限代表（确保水平面全面覆盖）
    - 2 个纵轴极端位置（楼层顶部和底部的代表性物体）
    - 2 个体积中位附近的（代表"典型"碎片）
    - 2 个不同材质的（确保材质多样性）
    """
    if not cubes:
        return []

    selected = []
    seen_ids = set()
    horiz_axes = [a for a in ("x", "y", "z") if a != analyzed_axis]
    h1, h2 = horiz_axes[0], horiz_axes[1]

    def _add(cube):
        if cube["id"] not in seen_ids:
            selected.append(_compact_cube(cube))
            seen_ids.add(cube["id"])

    by_volume = sorted(cubes, key=_cube_volume, reverse=True)

    # 1. 体积最大的 2 个
    for c in by_volume[:2]:
        _add(c)

    # 2. 水平面四象限采样（确保空间覆盖）
    if cubes:
        ch1 = sum(c.get("center", {}).get(h1, 0) for c in cubes) / len(cubes)
        ch2 = sum(c.get("center", {}).get(h2, 0) for c in cubes) / len(cubes)
        quadrants = {"Q1": [], "Q2": [], "Q3": [], "Q4": []}
        for c in cubes:
            d1 = c.get("center", {}).get(h1, 0) - ch1
            d2 = c.get("center", {}).get(h2, 0) - ch2
            if d1 >= 0 and d2 >= 0:
                quadrants["Q1"].append(c)
            elif d1 < 0 and d2 >= 0:
                quadrants["Q2"].append(c)
            elif d1 < 0 and d2 < 0:
                quadrants["Q3"].append(c)
            else:
                quadrants["Q4"].append(c)
        for q_cubes in quadrants.values():
            if q_cubes:
                _add(max(q_cubes, key=_cube_volume))

    # 3. 纵轴极端位置各一个
    axis = analyzed_axis
    for extreme_key in [min, max]:
        ext = extreme_key(cubes, key=lambda c: c.get("center", {}).get(axis, 0))
        _add(ext)

    # 4. 体积中位附近的 2 个
    mid_idx = len(by_volume) // 2
    for c in by_volume[mid_idx:mid_idx + 4]:
        if sum(1 for _ in [s for s in selected if _cube_volume(s) > 0]) >= count - 2:
            break
        _add(c)

    # 5. 不同材质各一个补齐
    seen_materials = {s.get("material", "") for s in selected}
    for c in by_volume:
        if len(selected) >= count:
            break
        mat = c.get("material", "unknown")
        if mat not in seen_materials:
            _add(c)
            seen_materials.add(mat)

    return selected


def _compute_density_heatmap(cubes: List[Dict], grid_size: int = 50,
                            analyzed_axis: str = "z") -> Dict:
    """生成本平面的密度热力图

    将楼层内的 Cube 投影到水平面（根据 up_axis 确定），
    生成 grid_size × grid_size 的网格，每格记录 Cube 数量和总体积。
    用于让 LLM "看到"空间拓扑：
    - 高密度带 = 墙体/结构
    - 低密度带 = 走廊/房间内部
    - 密度断点 = 可能的门/开口

    Returns:
        {
            "grid_size": 50,
            "h1_range": [min, max],
            "h2_range": [min, max],
            "non_empty_cells": int,       # 非空格数
            "max_count_per_cell": int,
            "high_density_regions": [...],  # 高密度区域描述
            "low_density_regions": [...],   # 低密度区域（可能是走廊/房间）
            "density_matrix": "文本矩阵 (每格用 █▓▒░· 表示密度等级)",
        }
    """
    if not cubes:
        return {"grid_size": grid_size, "non_empty_cells": 0}

    horiz_axes = [a for a in ("x", "y", "z") if a != analyzed_axis]
    h1, h2 = horiz_axes[0], horiz_axes[1]

    h1_vals = [c.get("center", {}).get(h1, 0) for c in cubes]
    h2_vals = [c.get("center", {}).get(h2, 0) for c in cubes]
    h1_min, h1_max = min(h1_vals), max(h1_vals)
    h2_min, h2_max = min(h2_vals), max(h2_vals)

    if h1_max <= h1_min or h2_max <= h2_min:
        h1_max = max(h1_min + 1, h1_max)
        h2_max = max(h2_min + 1, h2_max)

    h1_step = (h1_max - h1_min) / grid_size
    h2_step = (h2_max - h2_min) / grid_size

    # count + volume grid
    count_grid = [[0] * grid_size for _ in range(grid_size)]
    volume_grid = [[0.0] * grid_size for _ in range(grid_size)]

    for c in cubes:
        g1 = min(int((c.get("center", {}).get(h1, 0) - h1_min) / h1_step), grid_size - 1)
        g2 = min(int((c.get("center", {}).get(h2, 0) - h2_min) / h2_step), grid_size - 1)
        s = c.get("size", {})
        vol = abs(s.get("x", 0) * s.get("y", 0) * s.get("z", 0))
        count_grid[g1][g2] += 1
        volume_grid[g1][g2] += vol

    non_empty = sum(1 for row in count_grid for cell in row if cell > 0)
    flat_counts = [cell for row in count_grid for cell in row if cell > 0]
    max_count = max(flat_counts) if flat_counts else 0

    if max_count == 0:
        return {"grid_size": grid_size, "non_empty_cells": 0,
                f"{h1}_range": [round(h1_min, 1), round(h1_max, 1)],
                f"{h2}_range": [round(h2_min, 1), round(h2_max, 1)]}

    # Density levels for text matrix: █ >75% ▓ 50-75% ▒ 25-50% ░ 5-25% · <5%
    def _level(count):
        ratio = count / max_count
        if ratio > 0.75: return "█"
        if ratio > 0.50: return "▓"
        if ratio > 0.25: return "▒"
        if ratio > 0.05: return "░"
        return "·"

    # Build a compact text matrix (subsample if grid too large for prompt)
    display_size = min(grid_size, 50)
    step = grid_size // display_size if grid_size > display_size else 1
    lines = []
    for g1 in range(0, grid_size, step):
        line = "".join(_level(count_grid[g1][g2]) for g2 in range(0, grid_size, step))
        lines.append(line)

    # Identify high-density regions (walls/structures) and low-density regions (voids)
    high_regions = []
    low_regions = []
    visited = [[False] * grid_size for _ in range(grid_size)]

    for g1 in range(grid_size):
        for g2 in range(grid_size):
            if visited[g1][g2] or count_grid[g1][g2] == 0:
                continue
            # BFS to find connected region
            stack = [(g1, g2)]
            visited[g1][g2] = True
            region_cells = []
            region_volume = 0.0
            while stack:
                c1, c2 = stack.pop()
                region_cells.append((c1, c2))
                region_volume += volume_grid[c1][c2]
                for d1, d2 in [(-1,0),(1,0),(0,-1),(0,1)]:
                    n1, n2 = c1 + d1, c2 + d2
                    if 0 <= n1 < grid_size and 0 <= n2 < grid_size:
                        if not visited[n1][n2] and count_grid[n1][n2] > 0:
                            visited[n1][n2] = True
                            stack.append((n1, n2))

            avg_count = sum(count_grid[c1][c2] for c1, c2 in region_cells) / len(region_cells)
            region_center_h1 = h1_min + (sum(c1 for c1, _ in region_cells) / len(region_cells)) * h1_step
            region_center_h2 = h2_min + (sum(c2 for _, c2 in region_cells) / len(region_cells)) * h2_step

            if avg_count > max_count * 0.5:
                high_regions.append({
                    "center": [round(region_center_h1, 1), round(region_center_h2, 1)],
                    "cell_count": len(region_cells),
                    "avg_density": round(avg_count / max(max_count, 1), 2),
                    "total_volume": round(region_volume, 1),
                })
            elif avg_count < max_count * 0.15 and len(region_cells) >= 3:
                low_regions.append({
                    "center": [round(region_center_h1, 1), round(region_center_h2, 1)],
                    "cell_count": len(region_cells),
                    "avg_density": round(avg_count / max(max_count, 1), 2),
                })

    high_regions.sort(key=lambda r: -r["cell_count"])
    low_regions.sort(key=lambda r: -r["cell_count"])

    return {
        "grid_size": display_size,
        f"{h1}_range": [round(h1_min, 1), round(h1_max, 1)],
        f"{h2}_range": [round(h2_min, 1), round(h2_max, 1)],
        "non_empty_cells": non_empty,
        "max_count_per_cell": max_count,
        "high_density_regions": high_regions[:10],
        "low_density_regions": low_regions[:8],
        "density_matrix": "\n".join(lines),
    }


def _cube_volume(cube: Dict) -> float:
    s = cube.get("size", {})
    return abs(s.get("x", 0) * s.get("y", 0) * s.get("z", 0))


def _describe_spatial_density(cubes: List[Dict],
                            analyzed_axis: str = "z") -> Dict:
    """描述一层内 Cube 在水平面的空间分布密度

    使用网格法：将水平面（根据 up_axis 确定）分为 4×4 网格，统计每格 Cube 数量，
    描述为分布模式（均匀/中心集中/边缘分散/多点聚集）。
    """
    if not cubes:
        return {"pattern": "empty", "grid_counts": []}

    horiz_axes = [a for a in ("x", "y", "z") if a != analyzed_axis]
    h1, h2 = horiz_axes[0], horiz_axes[1]

    h1_vals = [c.get("center", {}).get(h1, 0) for c in cubes]
    h2_vals = [c.get("center", {}).get(h2, 0) for c in cubes]
    h1_min, h1_max = min(h1_vals), max(h1_vals)
    h2_min, h2_max = min(h2_vals), max(h2_vals)

    if h1_max <= h1_min or h2_max <= h2_min:
        return {"pattern": "linear", "grid_counts": [len(cubes)]}

    grid_size = 4
    h1_step = (h1_max - h1_min) / grid_size
    h2_step = (h2_max - h2_min) / grid_size
    grid = [[0] * grid_size for _ in range(grid_size)]

    for c in cubes:
        g1 = min(int((c.get("center", {}).get(h1, 0) - h1_min) / h1_step), grid_size - 1)
        g2 = min(int((c.get("center", {}).get(h2, 0) - h2_min) / h2_step), grid_size - 1)
        grid[g1][g2] += 1

    flat = [cell for row in grid for cell in row]
    non_empty = sum(1 for c in flat if c > 0)
    max_cell = max(flat) if flat else 0
    total = sum(flat)
    avg_cell = total / (grid_size * grid_size) if total > 0 else 0

    plane_label = f"{h1.upper()}{h2.upper()}"

    if non_empty <= 2:
        pattern = "sparse_clusters"
    elif max_cell > total * 0.6:
        pattern = "center_concentrated"
    elif non_empty >= 12:
        pattern = "uniformly_distributed"
    else:
        pattern = "multi_cluster"

    return {
        "pattern": pattern,
        "non_empty_cells": non_empty,
        "max_cell_count": max_cell,
        "avg_cell_count": round(avg_cell, 1),
        "description": (
            f"Cube 在 {plane_label} 平面呈{pattern}分布"
            f"（{grid_size}×{grid_size} 网格，{non_empty}/{grid_size*grid_size} 格非空，"
            f"最密格 {max_cell} 个，均值 {avg_cell:.1f} 个/格）"
        ),
    }


def _compact_cube(cube: Dict) -> Dict:
    """精简单个几何体"""
    full_id = cube.get("id", "")
    short_id = full_id.split("/")[-1] if "/" in full_id else full_id
    return {
        "id": short_id,
        "center": cube.get("center", {}),
        "size": cube.get("size", {}),
        "material": cube.get("material", "unknown"),
    }


def format_for_llm(compact_data: Dict[str, Any]) -> str:
    """将精简数据格式化为 LLM prompt 文本"""
    analyzed_axis = compact_data.get("analyzed_axis", "z")
    v_analysis = compact_data.get("vertical_analysis", {})
    horiz_axes = [a for a in ("x", "y", "z") if a != analyzed_axis]
    h1, h2 = horiz_axes[0], horiz_axes[1]

    sections = [
        "# Phase 1 全局场景理解",
        "",
        "## 场景基本信息",
        f"- 文件: {compact_data.get('source_file', 'N/A')}",
        f"- 世界包围盒: x={_fmt_range(compact_data, 'x')}, "
        f"y={_fmt_range(compact_data, 'y')}, z={_fmt_range(compact_data, 'z')}",
        f"- 几何体总数: {compact_data.get('total_prims', 0)}",
        f"- **垂直轴**: {analyzed_axis}（楼层边界沿此轴分布）",
        f"- **水平面**: {h1.upper()}{h2.upper()}（密度热力图投影平面）",
        "",
        "## 数据解读指南",
        "输入数据为纯几何信息（无语义标签），你需要从空间分布模式中推断建筑结构。",
        "",
        f"### {h1.upper()}{h2.upper()} 密度热力图解读",
        f"每层提供 {h1.upper()}{h2.upper()} 平面密度热力图。密度等级：█>75% ▓50-75% ▒25-50% ░5-25% ·<5%",
        "从热力图中识别空间拓扑：",
        "- 两条平行高密度带(█)之间的长条形低密度带(·) = **走廊/通道**",
        "- 被高密度边框包围的低密度矩形区域 = **房间**",
        "- 高密度墙线中的低密度断点 = **入口/门/窗**",
        "- 大面积极低密度(·)区域 = **开阔空间/外场**",
        "- 多个高密度斑块密集分布 = **设备区/结构密集区**",
        "",
        "### 空间聚类信息",
        f"若提供了 {h1.upper()}{h2.upper()} 空间聚类，表示场景中存在空间上不相连的区域",
        "（如分离的户外地形和建筑主体）。请独立分析每个聚类的空间特征。",
        "",
        "### 其他推断规则",
        "- 某高度附近大量共面碎片 → 可能是地面/楼板",
        "- 某平面位置一条垂直向密集碎片带 → 可能是墙",
        f"- 楼层边界基于 {analyzed_axis} 轴直方图峰值检测，需结合材质名辅助判断",
        "",
        "## 楼层信息",
    ]

    for floor in compact_data.get("floor_info", []):
        ax = floor.get("analyzed_axis", analyzed_axis)
        vr = floor.get("vertical_range", floor.get("z_range", [0, 0]))
        sections.append(f"\n### 第 {floor['floor_number']} 层 ({ax}: {vr})")
        sections.append(f"- 几何体数量: {floor['cube_count']}")

        ss = floor.get("size_stats", {})
        if ss:
            vol = ss.get("volume", {})
            sections.append(f"- 体积范围: {vol.get('min')} ~ {vol.get('max')} (中位 {vol.get('median')})")
            sections.append(f"- 宽度范围: {ss.get('width_range')}")
            sections.append(f"- 高度范围: {ss.get('height_range')}")

        sd = floor.get("spatial_density", {})
        if sd:
            sections.append(f"- 空间分布: {sd.get('description', sd.get('pattern', 'unknown'))}")

        mats = floor.get("material_summary", [])
        if mats:
            mat_str = ", ".join(f"{m['material']}({m['count']})" for m in mats)
            sections.append(f"- 主要材质: {mat_str}")

        sections.append("- 代表性几何体:")
        for rc in floor.get("representative_cubes", []):
            sections.append(f"  - {rc['id']}: center={rc['center']}, size={rc['size']}, material={rc.get('material', '?')}")

        # 密度热力图（水平面投影）
        heatmap = floor.get("density_heatmap", {})
        if heatmap and heatmap.get("density_matrix"):
            # 动态读取水平面两轴 range key
            h1_range_key = f"{h1}_range"
            h2_range_key = f"{h2}_range"
            h1_r = heatmap.get(h1_range_key, [0, 1])
            h2_r = heatmap.get(h2_range_key, [0, 1])
            sections.append(f"\n- {h1.upper()}{h2.upper()} 密度热力图 ({heatmap.get('grid_size')}x{heatmap.get('grid_size')} 网格, "
                          f"{h1.upper()}=[{h1_r[0]:.1f},{h1_r[1]:.1f}], "
                          f"{h2.upper()}=[{h2_r[0]:.1f},{h2_r[1]:.1f}])")
            sections.append(f"  密度等级: █>75% ▓50-75% ▒25-50% ░5-25% ·<5%")
            sections.append(f"  {heatmap['density_matrix']}")
            sections.append(f"  (非空格: {heatmap.get('non_empty_cells')}, 最密格: {heatmap.get('max_count_per_cell')} cubes)")

            # 高密度区域
            hd = heatmap.get("high_density_regions", [])
            if hd:
                sections.append(f"  高密度聚集区 ({len(hd)} 个):")
                for r in hd[:5]:
                    sections.append(f"    - 中心=({r['center'][0]:.0f},{r['center'][1]:.0f}), "
                                  f"格数={r['cell_count']}, 密度={r['avg_density']:.2f}")

            # 低密度区域（潜在的走廊/房间内部）
            ld = heatmap.get("low_density_regions", [])
            if ld:
                sections.append(f"  低密度空旷区 ({len(ld)} 个, 可能是走廊、房间内部):")
                for r in ld[:5]:
                    sections.append(f"    - 中心=({r['center'][0]:.0f},{r['center'][1]:.0f}), "
                                  f"格数={r['cell_count']}, 密度={r['avg_density']:.2f}")

    # 模式信号
    patterns = compact_data.get("patterns_summary", {})
    sections.append(f"\n## 几何模式")
    sections.append(f"- 等距线性排列: {patterns.get('linear_array_count', 0)} 组")
    sections.append(f"- 对称对: {patterns.get('symmetric_pair_count', 0)} 对")
    sections.append(f"- 密集簇: {patterns.get('dense_cluster_count', 0)} 个")

    # 楼梯候选（启发式 + 跨层密度）
    stairs_heuristic = patterns.get("tentative_stairs", [])
    stairs_density = patterns.get("density_stairs", [])
    if stairs_heuristic or stairs_density:
        sections.append("\n## 疑似楼梯（算法检测）")
        if stairs_heuristic:
            sections.append(f"\n### 台阶序列启发式 ({len(stairs_heuristic)} 处)")
            for s in stairs_heuristic[:8]:
                sections.append(f"- {s.get('stair_id', '?')}: {s.get('step_count', 0)}级, "
                              f"总高{s.get('total_height', 0)}m, 置信度{s.get('confidence', 0)}")
        if stairs_density:
            sections.append(f"\n### 跨层密度检测 ({len(stairs_density)} 处)")
            for s in stairs_density[:8]:
                stories_str = " ↔ ".join(s.get('connected_stories', ['?', '?']))
                sections.append(f"- {s.get('stair_id', '?')}: 连接{stories_str}, "
                              f"面积={s.get('footprint_area_m2', 0):.1f}m², "
                              f"中心=({s.get('center', {}).get('x',0):.0f},{s.get('center', {}).get('y',0):.0f}), "
                              f"置信度={s.get('confidence', 0):.2f}")

    # 空间聚类概述（Phase 0 算法检测，用于内外分离）
    sc = compact_data.get("spatial_clusters", [])
    if sc:
        sections.append(f"\n## {h1.upper()}{h2.upper()} 空间聚类 ({len(sc)} 个) —— 内外分离依据")
        sections.append("Phase 0 在水平面上检测到多个空间上不相连的区域。")
        sections.append("每个聚类的 label 由算法根据 cube 密度和占比自动标注，请据此判断内外：")
        sections.append("")
        # 按 label 分组显示
        from collections import defaultdict
        by_label = defaultdict(list)
        for cl in sc:
            by_label[cl.get("label", "unknown")].append(cl)

        label_descriptions = {
            "building": "【建筑主体】cube 密度最高、占比最大的聚类 → 对应 building_interior",
            "exterior_structure": "【外部结构】中等密度、可能为独立附属建筑 → 对应 exterior_open 或独立建筑",
            "outdoor_terrain": "【户外地形】低密度、大范围 → 对应 outdoor_terrain",
            "background": "【背景噪声】极稀疏、可忽略",
        }

        for label, clusters in by_label.items():
            desc = label_descriptions.get(label, "")
            sections.append(f"### {label} ({len(clusters)} 个聚类) {desc}")
            for cl in clusters:
                zr = cl.get(f"{analyzed_axis}_range", [0, 0])
                sections.append(
                    f"- **聚类 {cl.get('cluster_id')}**: "
                    f"{cl.get('cube_count')} cubes "
                    f"({cl.get('cube_ratio', 0):.1%} 总占比), "
                    f"z 密度={cl.get('z_density', 0):.0f} cubes/m, "
                    f"检测到 {cl.get('floor_count')} 个楼层候选"
                )
                sections.append(
                    f"  水平范围: {h1.upper()}={cl.get(f'{h1}_range', [0,0])}, "
                    f"{h2.upper()}={cl.get(f'{h2}_range', [0,0])}, "
                    f"垂直范围: {analyzed_axis.upper()}={zr}"
                )
                fb = cl.get("floor_boundaries", [])
                if fb:
                    sections.append(f"  楼层边界: {fb}")
        sections.append("")
        sections.append("**重要**: building 聚类是场景分析的主体，请基于其楼层边界划分 zones；")
        sections.append("exterior_structure / outdoor_terrain 聚类应标注为 exterior 或 transition zone。")

    # 几何特征（墙面线、楼板面、房间候选）
    geo = compact_data.get("geometry_features", {})
    if geo:
        walls_h1 = geo.get(f"wall_lines_{h1}", [])
        walls_h2 = geo.get(f"wall_lines_{h2}", [])
        floors = geo.get("floor_planes", [])
        rooms = geo.get("room_candidates", [])
        frag_stats = geo.get("fragment_stats", {})

        sections.append(f"\n## 几何特征提取（算法检测，非 LLM 推断）")
        sections.append(f"- 碎片分类: 垂直墙面{frag_stats.get('vertical_walls',0)}个, "
                      f"水平面{frag_stats.get('horizontal_planes',0)}个, "
                      f"其他{frag_stats.get('other',0)}个")
        sections.append(f"- 坐标系: 垂直轴={analyzed_axis}, 水平面={h1}{h2}")

        if walls_h1:
            h2_range_key = f"{h2}_range"
            sections.append(f"\n### {h1.upper()}向墙面线 ({len(walls_h1)} 条, 法线沿{h1}轴, 墙体沿{h2}轴延伸)")
            sections.append(f"(每条线代表一面潜在的墙体，position={h1}坐标, {h2}_range=墙体沿{h2}的延伸范围)")
            for w in walls_h1[:15]:
                gaps_str = f", 间隙{len(w['gaps'])}处" if w['gaps'] else ""
                sections.append(
                    f"  {h1.upper()}={w['position']:5.1f}: {w['fragment_count']:4d}碎片, "
                    f"{analyzed_axis.upper()}=[{w[f'{analyzed_axis}_range'][0]:.0f},{w[f'{analyzed_axis}_range'][1]:.0f}], "
                    f"{h2.upper()}=[{w.get(h2_range_key,[0,0])[0]:.0f},{w.get(h2_range_key,[0,0])[1]:.0f}], "
                    f"密度={w['density']:.2f}{gaps_str}"
                )
                for g in w['gaps'][:3]:
                    sections.append(f"    └ 间隙: {h2.upper()}=[{g['perp_start']:.0f},{g['perp_end']:.0f}], 宽{g['gap_size']:.1f} (可能的门/窗)")

        if walls_h2:
            h1_range_key = f"{h1}_range"
            sections.append(f"\n### {h2.upper()}向墙面线 ({len(walls_h2)} 条, 法线沿{h2}轴, 墙体沿{h1}轴延伸)")
            for w in walls_h2[:15]:
                gaps_str = f", 间隙{len(w['gaps'])}处" if w['gaps'] else ""
                sections.append(
                    f"  {h2.upper()}={w['position']:5.1f}: {w['fragment_count']:4d}碎片, "
                    f"{analyzed_axis.upper()}=[{w[f'{analyzed_axis}_range'][0]:.0f},{w[f'{analyzed_axis}_range'][1]:.0f}], "
                    f"{h1.upper()}=[{w.get(h1_range_key,[0,0])[0]:.0f},{w.get(h1_range_key,[0,0])[1]:.0f}], "
                    f"密度={w['density']:.2f}{gaps_str}"
                )
                for g in w['gaps'][:3]:
                    sections.append(f"    └ 间隙: {h1.upper()}=[{g['perp_start']:.0f},{g['perp_end']:.0f}], 宽{g['gap_size']:.1f} (可能的门/窗)")

        if floors:
            up_level_key = f"{analyzed_axis}_level"
            sections.append(f"\n### 楼板/天花板面 ({len(floors)} 个水平面)")
            for f in floors[:8]:
                sections.append(
                    f"  {analyzed_axis.upper()}={f[up_level_key]:6.1f}: {f['fragment_count']:4d}碎片, "
                    f"{h1.upper()}=[{f[f'{h1}_range'][0]:.0f},{f[f'{h1}_range'][1]:.0f}], "
                    f"{h2.upper()}=[{f[f'{h2}_range'][0]:.0f},{f[f'{h2}_range'][1]:.0f}]"
                )

        if rooms:
            sections.append(f"\n### 房间候选 ({len(rooms)} 个, 从{h1.upper()}/{h2.upper()}墙面线交点网格检测)")
            sections.append("(covered_sides: 有几面被墙体覆盖, open_sides: 有几面无墙=可能的入口)")
            for r in rooms[:12]:
                b = r['bounds']
                w = r['walls']
                sections.append(
                    f"  {h1.upper()}=[{b[h1][0]:.0f},{b[h1][1]:.0f}] {h2.upper()}=[{b[h2][0]:.0f},{b[h2][1]:.0f}], "
                    f"面积={r['area']:.0f}, 墙面={r['covered_sides']}/4 "
                    f"(左{'✓' if w['left'] else '✗'}右{'✓' if w['right'] else '✗'}下{'✓' if w['bottom'] else '✗'}上{'✓' if w['top'] else '✗'})"
                )

        # 走廊候选
        corridors = geo.get("corridor_candidates", [])
        if corridors:
            sections.append(f"\n### 走廊候选 ({len(corridors)} 个, 从平行墙面线间隙检测)")
            sections.append("走廊 = 两条平行墙之间的长条形低密度区域，连接多个房间。")
            for c in corridors[:12]:
                story_str = f"F{c['story_number']}" if c.get('story_number') is not None else "未知"
                b = c.get('bounds', {})
                b_keys = list(b.keys())
                if len(b_keys) >= 2:
                    sections.append(
                        f"  {c['corridor_id']}: {c['direction'].upper()}向, "
                        f"宽{c['width']:.1f}m x 长{c['length']:.1f}m, "
                        f"面积={c['area']:.1f}m², 长宽比={c['aspect_ratio']:.1f}, "
                        f"密度={c['density']:.2f}, 置信度={c['confidence']:.2f}, "
                        f"楼层={story_str}, Z=[{c['z_range'][0]:.1f},{c['z_range'][1]:.1f}]"
                    )
                    sections.append(
                        f"    范围: {b_keys[0].upper()}=[{b[b_keys[0]][0]:.0f},{b[b_keys[0]][1]:.0f}], "
                        f"{b_keys[1].upper()}=[{b[b_keys[1]][0]:.0f},{b[b_keys[1]][1]:.0f}]"
                    )
            sections.append("")
            sections.append("**走廊战术价值**: 走廊是关键的室内移动通道和火力轴线。")
            sections.append("- 长直走廊 → 潜在的致命漏斗/杀伤区")
            sections.append("- 连接多个房间的走廊 → 控制此走廊即可控制多个房间的进出")

        # 按楼层分组的房间候选
        story_rooms = geo.get("story_rooms", {})
        if story_rooms:
            total_sr = sum(len(r) for r in story_rooms.values())
            sections.append(f"\n### 按楼层分组房间 ({total_sr} 个房间, {len(story_rooms)} 个楼层)")
            for story_id, srooms in story_rooms.items():
                sections.append(f"  {story_id}: {len(srooms)} 个房间")

    # 户外地形特征
    terrain = geo.get("terrain_features", {}) if geo else {}
    if terrain:
        sections.append(f"\n## 户外地形特征（算法检测）")
        sections.append(f"- 地形类型: **{terrain.get('terrain_type', 'unknown')}**")
        sections.append(f"- 外部碎片数量: {terrain.get('exterior_cube_count', 0)}")

        ep = terrain.get("elevation_profile", {})
        if ep:
            sections.append(
                f"- 高程剖面: min={ep.get('min')}m, median={ep.get('p50')}m, "
                f"max={ep.get('max')}m, 总起伏={ep.get('total_range')}m"
            )

        oa = terrain.get("open_areas", [])
        if oa:
            sections.append(f"\n### 开放区域 ({len(oa)} 个, 可能是广场/停车场/开阔地/射击阵地)")
            sections.append(f"(面积单位: m², 密度单位: cubes/格)")
            for a in oa[:8]:
                center = a.get("center", {})
                bounds = a.get("bounds", {})
                sections.append(
                    f"  - 面积={a['area']:.0f}m², 中心=({list(center.values())[0]:.0f},{list(center.values())[1]:.0f}), "
                    f"密度={a['avg_density']:.1f}"
                )
                h1k, h2k = list(center.keys())[0], list(center.keys())[1]
                b1 = bounds.get(h1k, [0, 0])
                b2 = bounds.get(h2k, [0, 0])
                sections.append(
                    f"    范围: {h1k.upper()}=[{b1[0]:.0f},{b1[1]:.0f}], "
                    f"{h2k.upper()}=[{b2[0]:.0f},{b2[1]:.0f}]"
                )
            sections.append("")
            sections.append("**战术提示**: 开放区域代表外场的开阔地，可用于：")
            sections.append("- 面积>200m² 的大开放区域 → 潜在火力扇区/杀伤区")
            sections.append("- 建筑附近的开放区域 → 建筑物周围的接近路/缓冲区")
            sections.append("- 多块不连续开放区域 → 可提供分段掩护的跃进路线")

    # ── 楼板面证据（主楼层信号）──────────────────────────
    floor_ev = compact_data.get("floor_evidence", {})
    if floor_ev:
        candidates = floor_ev.get("floor_candidates", [])
        gaps = floor_ev.get("inter_floor_gaps", [])
        source = floor_ev.get("source", "unknown")

        sections.append(f"\n## 楼层候选（主证据来源: {source}）")
        sections.append("以下楼层候选是从场景中实际检测到的**薄水平面碎片**聚合而来，")
        sections.append("比统计直方图峰值更可靠。请基于此表判断真实楼层数。")
        sections.append("")

        if candidates:
            # 表头
            h1_label = h1.upper()
            h2_label = h2.upper()
            sections.append(f"| Z层级 | 碎片数 | 带内Cube | {h1_label}范围 | {h2_label}范围 | 水平覆盖 | 墙面线 | 主要材质(前3) | 证据强度 |")
            sections.append(f"|-------|--------|----------|-----------|-----------|---------|--------|--------------|---------|")

            for fc in candidates:
                zl = fc.get("z_level", 0)
                fc_cnt = fc.get("fragment_count", 0)
                cb_cnt = fc.get("cube_count_in_band", 0)
                span = fc.get("horizontal_span", {})
                h1s = span.get(h1, [0, 0])
                h2s = span.get(h2, [0, 0])
                cov = fc.get("horizontal_coverage", 0)
                walls = fc.get("intersecting_wall_lines", 0)
                mats = fc.get("supporting_materials", [])[:3]
                mat_str = ", ".join(f"{m['name'][:20]}({m['count']})" for m in mats)
                ev = fc.get("evidence_strength", "weak")

                ev_mark = {"strong": "STRONG", "medium": "MEDIUM", "weak": "WEAK"}.get(ev, ev)
                sections.append(
                    f"| {zl:5.1f} | {fc_cnt:6d} | {cb_cnt:8d} | "
                    f"[{h1s[0]:.0f},{h1s[1]:.0f}] | [{h2s[0]:.0f},{h2s[1]:.0f}] | "
                    f"{cov:.0%} | {walls:4d} | {mat_str[:40]} | **{ev_mark}** |"
                )
            sections.append("")

        if gaps:
            sections.append(f"### 楼层间空白区间")
            sections.append(f"| 区间 | 跨度(m) | Cube数 | 密度 | 状态 |")
            sections.append(f"|------|---------|--------|------|------|")
            for g in gaps:
                between = g.get("between", [0, 0])
                gap_m = g.get("gap_meters", 0)
                cnt = g.get("cube_count", 0)
                dens = g.get("cube_density", 0)
                status = g.get("status", "normal")
                status_mark = {
                    "normal": "正常",
                    "too_narrow": "过窄(可能是同一层的子结构)",
                    "too_wide_possible_missing_floor": "过宽(可能漏检中间层!)",
                }.get(status, status)
                sections.append(
                    f"| [{between[0]:.1f}, {between[1]:.1f}] | {gap_m:.1f} | {cnt} | {dens:.0f}/m | {status_mark} |"
                )
            sections.append("")

        sections.append("**楼层判定规则**:")
        sections.append("- STRONG 证据的 Z 层级极可能是真实楼层——请务必将其纳入 building_structure.floors")
        sections.append("- MEDIUM 证据可能是真实楼层或大型平台/夹层——需结合墙面线和房间候选判断")
        sections.append("- WEAK 证据可以忽略（稀疏碎片、地面杂物、屋顶装饰）")
        sections.append("- 若某空白区间标记为\"过宽(可能漏检中间层!)\"，请检查墙面线在该区间是否有密集分布")
        sections.append("- 同材质在多个楼层出现是正常的（地板材料每层都用），不表示它们是同一层")

    # ── 材质几何预分类 ──────────────────────────────────
    mat_groups = compact_data.get("material_groups", [])
    if mat_groups:
        # 只展示有意义的材质（排除纯垃圾名 + 混合型）
        display_mats = [
            m for m in mat_groups
            if m.get("geometric_role") != "mixed" and m.get("cube_count", 0) >= 10
        ][:20]

        if display_mats:
            sections.append(f"\n## 材质几何预分类（算法判定，非 LLM 推断）")
            sections.append("每种材质的几何角色由碎片形状判定（水平薄面→地板类，垂直薄面→墙体类），不依赖材质名称。")
            sections.append("")
            sections.append(f"| 材质 | Cube数 | 几何角色 | 水平面比 | 垂直面比 | Z范围 | Z分布 |")
            sections.append(f"|------|--------|---------|---------|---------|-------|------|")

            for mg in display_mats:
                name = mg.get("material", "?")[:25]
                cnt = mg.get("cube_count", 0)
                role = mg.get("geometric_role", "?")
                role_cn = {"horizontal_plane": "地板/楼板类", "vertical_plane": "墙体/立面类"}.get(role, role)
                hr = mg.get("horiz_plane_ratio", 0)
                vr = mg.get("vert_plane_ratio", 0)
                zr = mg.get(f"{analyzed_axis}_range", [0, 0])
                zd = mg.get("z_distribution", "?")
                zd_cn = {"discrete_concentrated": "离散集中", "continuous": "连续跨度", "sparse": "稀疏"}.get(zd, zd)

                sections.append(
                    f"| {name} | {cnt} | **{role_cn}** | {hr:.0%} | {vr:.0%} | "
                    f"[{zr[0]:.0f},{zr[1]:.0f}] | {zd_cn} |"
                )
            sections.append("")
            sections.append("**解读**: 标记为\"地板/楼板类\"且 Z 分布为\"离散集中\"的材质，其集中层级很可能就是楼层位置。")

    sections.append(f"\n## 垂直结构 ({analyzed_axis}轴)")
    story_count = floor_ev.get('story_count', v_analysis.get('floor_count', 1))
    sections.append(f"- **story_count (N-1 可居住楼层数): {story_count}** ← building_structure.total_floors 必须等于此值")
    sections.append(f"- 楼板层级数 (logical_slab_count): {floor_ev.get('logical_slab_count', '?')}")
    sections.append(f"- 垂直复杂度: {v_analysis.get('vertical_complexity', 'simple')}")
    sections.append(f"- 楼层来源: {floor_ev.get('source', 'histogram') if floor_ev else 'histogram'}")

    # 逐聚类楼层摘要
    sc_list = compact_data.get("spatial_clusters", [])
    if sc_list:
        building_cl = [c for c in sc_list if c.get("label") == "building"]
        for bc in building_cl:
            sections.append(
                f"- **建筑聚类 {bc.get('cluster_id')}**: "
                f"{bc.get('floor_count')} 层, "
                f"cube 占比 {bc.get('cube_ratio', 0):.0%}, "
                f"z 密度 {bc.get('z_density', 0):.0f} cubes/m"
            )
        other_cl = [c for c in sc_list if c.get("label") != "building"]
        if other_cl:
            sections.append(f"- 其他聚类 ({len(other_cl)} 个): " +
                          ", ".join(f"{c.get('label')}({c.get('floor_count')}层)" for c in other_cl))
    sections.append(f"- 全局直方图峰值: {len(v_analysis.get('_histogram_raw', {}).get('peaks', []))} 个")

    sections.append("\n## 任务")
    sections.append("请分析以上场景信息，输出完整的 GlobalUnderstanding JSON。")
    sections.append("注意从空间分布中推理建筑结构，而非期待语义标签。")

    return "\n".join(sections)


def _fmt_range(data: Dict, axis: str) -> str:
    wb = data.get("world_bounds", {})
    r = wb.get(axis, [0, 0])
    return f"[{r[0]:.1f}, {r[1]:.1f}]"
