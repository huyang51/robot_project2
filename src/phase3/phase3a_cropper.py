"""
Phase 3a: 几何裁剪器

根据 sub_scene_definitions 中的 spatial_bounds，
从原始 USDA 中提取属于每个子场景的 Cube。
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

from ..core.types import BBox

logger = logging.getLogger(__name__)


def crop_cubes(
    cubes: List[Dict],
    bounds: Dict[str, List[float]],
    overlap_threshold: float = 0.3,
) -> List[Dict]:
    """从 Cube 列表中裁剪出与子场景边界有交集的 Cube

    按概念模型 §7.1 规定，overlap_threshold 默认为 0.3（30% 交集比）。

    Args:
        cubes: 来自 Phase 0 的 cubes 数组
        bounds: 子场景边界 {"x": [min,max], "y": [min,max], "z": [min,max]}
                应使用 overlap_bounds（精确边界外扩 5m），若不可用则用 spatial_bounds。
        overlap_threshold: Cube 与 bounds 的包含比例阈值（默认 0.3）

    Returns:
        属于该子场景的 Cube 列表
    """
    if not bounds or "x" not in bounds:
        return []
    crop_bbox = BBox(
        x_min=bounds["x"][0], x_max=bounds["x"][1],
        y_min=bounds["y"][0], y_max=bounds["y"][1],
        z_min=bounds["z"][0], z_max=bounds["z"][1],
    )

    selected = []
    for cube in cubes:
        # 构建 Cube 包围盒
        c = cube.get("center", {})
        s = cube.get("size", {})
        cube_bbox = BBox(
            x_min=c["x"] - s["x"] / 2, x_max=c["x"] + s["x"] / 2,
            y_min=c["y"] - s["y"] / 2, y_max=c["y"] + s["y"] / 2,
            z_min=c["z"] - s["z"] / 2, z_max=c["z"] + s["z"] / 2,
        )

        if crop_bbox.intersects(cube_bbox):
            # 计算相交比例（相对 Cube 体积）
            intersection = cube_bbox.intersection_volume(crop_bbox)
            ratio = intersection / cube_bbox.volume if cube_bbox.volume > 0 else 0
            if ratio >= overlap_threshold:
                selected.append(cube)

    return selected


def expand_bounds(
    bounds: Dict[str, List[float]],
    expansion: float = 5.0,
) -> Dict[str, List[float]]:
    """将 spatial_bounds 每侧外扩 expansion 米，生成 overlap_bounds

    符合概念模型 §6.2: overlap_bounds = bounds 每侧外扩 5m，
    确保 Phase 3 几何裁切时边界处几何完整，不因恰好跨边界而截断。
    """
    return {
        axis: [v[0] - expansion, v[1] + expansion]
        for axis, v in bounds.items()
    }


def write_sub_scene_usda(
    crops: List[Dict],
    sub_scene_id: str,
    output_dir: Path,
) -> str:
    """为子场景写入简化版 USDA 文件

    当前版本输出 JSON 格式的 Cube 列表而非完整 USDA。
    完整 USDA 输出需要从原始文件提取 Xform 层级并拼接。

    Args:
        crops: 裁剪后的 Cube 列表
        sub_scene_id: 子场景 ID
        output_dir: 子场景输出目录

    Returns:
        输出的 scene.usda 路径（当前为 scene_cubes.json）
    """
    ss_dir = output_dir / sub_scene_id
    ss_dir.mkdir(parents=True, exist_ok=True)

    output_path = ss_dir / "scene_cubes.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "sub_scene_id": sub_scene_id,
            "cube_count": len(crops),
            "cubes": crops,
        }, f, ensure_ascii=False, indent=2)

    return str(output_path)
