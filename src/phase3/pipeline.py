"""
Phase 3 协调器

协调 Phase 3a/3b/3c 完整流水线：
1. 加载 sub_scene_definitions
2. 对每个子场景: 裁剪 → 简化 → LLM 语义标注 → 校验
3. 输出 scene.usda + desc.json
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

from ..core.llm_client import MiniMaxClient
from ..core.exceptions import LLMError
from ..config import SUB_SCENES_DIR, PHASE3B_AGGREGATION_PARAMS
from ..phase2.dedup_scenes import deduplicate_sub_scenes
from .phase3a_cropper import crop_cubes, expand_bounds, write_sub_scene_usda
from .phase3b_simplifier import simplify_sub_scene
from .phase3c_annotator import annotate_sub_scene
from .validator import validate_sub_scene_completeness

logger = logging.getLogger(__name__)


def run_phase3(
    scene_metadata_path: str,
    sub_scene_definitions_path: str,
    client: Optional[MiniMaxClient] = None,
    output_dir: Optional[str] = None,
) -> List[Dict[str, str]]:
    """Phase 3 主入口

    对每个子场景执行裁剪→简化→标注→校验。

    Returns:
        [{"sub_scene_id": "...", "usda_path": "...", "desc_path": "..."}, ...]
    """
    output_dir = Path(output_dir) if output_dir else SUB_SCENES_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 加载数据
    logger.info("Phase 3 开始")
    with open(scene_metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    with open(sub_scene_definitions_path, "r", encoding="utf-8") as f:
        definitions = json.load(f)

    cubes = metadata.get("cubes", [])
    sub_scenes = definitions.get("sub_scenes", [])
    patterns = _extract_patterns(metadata)

    if client is None:
        client = MiniMaxClient()

    # 去重：移除战术等价的冗余子场景（安全网，Phase 2 通常已完成去重）
    if len(sub_scenes) > 1:
        sub_scenes = deduplicate_sub_scenes(sub_scenes, client)
        definitions["sub_scenes"] = sub_scenes

    # 2b. 构建 scene_cubes 索引（用于校验）
    scene_cubes_index = {c["id"]: c for c in cubes}

    # 构建子场景索引（用于生成相邻场景上下文）
    ss_index = {ss.get("sub_scene_id"): ss for ss in sub_scenes}

    results = []
    for i, ss_def in enumerate(sub_scenes):
        ss_id = ss_def.get("sub_scene_id", f"SS_{i + 1:02d}")
        logger.info(f"Phase 3 处理 {ss_id} ({i + 1}/{len(sub_scenes)})")

        # 3a: 裁剪（优先使用 overlap_bounds，无则从 spatial_bounds 外扩 5m）
        crop_bounds = ss_def.get("overlap_bounds")
        if crop_bounds is None:
            spatial = ss_def.get("spatial_bounds", {})
            if spatial:
                crop_bounds = expand_bounds(spatial, expansion=5.0)
                logger.debug(f"{ss_id}: 无 overlap_bounds，从 spatial_bounds 外扩 5m")
            else:
                crop_bounds = {}
        ss_cubes = crop_cubes(cubes, crop_bounds)

        if len(ss_cubes) == 0:
            logger.warning(
                f"{ss_id}: 裁切后为 0 个 Cube！crop_bounds={crop_bounds}。"
                f"可能原因: spatial_bounds 坐标轴与 cube 数据不匹配"
            )
            # 空子场景：跳过 3b/3c，直接标记失败
            results.append({
                "sub_scene_id": ss_id,
                "usda_path": str(output_dir / ss_id / "scene_cubes.json"),
                "desc_path": str(output_dir / ss_id / "desc.json"),
                "validation_passed": False,
                "cube_count": 0,
            })
            # 仍写入空 cube JSON 以便调试
            write_sub_scene_usda([], ss_id, output_dir)
            continue

        # 3b: 简化
        ss_cubes = simplify_sub_scene(ss_cubes, patterns, PHASE3B_AGGREGATION_PARAMS)

        # 写入简化后的 Cube JSON
        usda_path = write_sub_scene_usda(ss_cubes, ss_id, output_dir)

        # 3c: LLM 语义标注（传入全局上下文、Phase 2 权威数据、相邻子场景信息）
        global_ctx = _build_global_context(ss_def, sub_scenes, ss_index)
        adj_ctx = _build_adjacent_context(ss_def, ss_index)
        phase2_ctx = _build_phase2_context(ss_def, definitions, ss_index)
        try:
            desc = annotate_sub_scene(
                cubes=ss_cubes,
                sub_scene_id=ss_id,
                tactical_role=ss_def.get("primary_role", ss_def.get("tactical_role", "")),
                task_hint=ss_def.get("task_hint", ""),
                client=client,
                global_context=global_ctx,
                adjacent_scenes=adj_ctx,
                phase2_context=phase2_ctx,
            )
        except LLMError as e:
            logger.error(f"{ss_id} LLM 标注失败: {e}，跳过此子场景")
            results.append({
                "sub_scene_id": ss_id,
                "usda_path": str(usda_path),
                "desc_path": str(output_dir / ss_id / "desc.json"),
                "validation_passed": False,
                "cube_count": len(ss_cubes),
                "error": str(e),
            })
            continue

        # 校验
        validation = validate_sub_scene_completeness(desc, scene_cubes_index)
        if not validation["passed"]:
            logger.warning(f"{ss_id} 校验失败: {validation['errors']}")

        # 写入 desc.json
        desc_path = output_dir / ss_id / "desc.json"
        with open(desc_path, "w", encoding="utf-8") as f:
            json.dump(desc, f, ensure_ascii=False, indent=2)

        results.append({
            "sub_scene_id": ss_id,
            "usda_path": str(usda_path),
            "desc_path": str(desc_path),
            "validation_passed": validation["passed"],
            "cube_count": len(ss_cubes),
        })

        logger.info(f"Phase 3 {ss_id} 完成: {len(ss_cubes)} Cube, 校验={'通过' if validation['passed'] else '未通过'}")

    # 3. 输出总览
    summary_path = output_dir / "phase3_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"sub_scenes": results, "total": len(results)}, f, ensure_ascii=False, indent=2)

    logger.info("Phase 3 完成: %d 个子场景", len(results))
    return results


def _extract_patterns(metadata: Dict) -> List[Dict]:
    """从 Phase 0 元数据中提取模式信息"""
    patterns_data = metadata.get("patterns", {})
    all_patterns = []
    for key in ["linear_arrays", "symmetric_pairs", "dense_clusters"]:
        all_patterns.extend(patterns_data.get(key, []))
    return all_patterns


def _build_global_context(
    ss_def: Dict,
    all_sub_scenes: List[Dict],
    ss_index: Dict[str, Dict],
) -> str:
    """为 Phase 3c 构建全局上下文摘要

    从子场景定义的字段中提取该子场景在建筑中的位置信息。
    """
    lines = []
    floor = ss_def.get("floor", "未知")
    sp = ss_def.get("space_profile", {})
    shape = sp.get("shape", "?")
    enclosure = sp.get("enclosure", "?")
    vert_pos = sp.get("vertical_position", "?")

    lines.append(f"- 楼层: F{floor}")
    lines.append(f"- 空间形态: {shape}")
    lines.append(f"- 围合程度: {enclosure}")
    lines.append(f"- 垂直位置: {vert_pos}")
    lines.append(f"- 所属 zones: {', '.join(ss_def.get('zone_ids', []))}")

    # 该子场景在整个连通图中的角色
    connected = ss_def.get("connected_sub_scenes", [])
    if connected:
        lines.append(f"- 连通子场景 ({len(connected)} 个): {', '.join(connected)}")

    # entry_options 中与该子场景相关的入口
    lines.append(f"- 建议角色: {ss_def.get('primary_role', ss_def.get('tactical_role', '?'))}")
    suggested = ss_def.get('suggested_roles', [])
    if suggested:
        lines.append(f"- 可选角色: {', '.join(suggested)}")

    return '\n'.join(lines)


def _build_adjacent_context(
    ss_def: Dict,
    ss_index: Dict[str, Dict],
) -> str:
    """为 Phase 3c 构建相邻子场景摘要

    提取与该子场景有连通关系的其他子场景的关键信息：
    空间类型、开口位置、关键威胁（如果有的话）。
    """
    connected_ids = ss_def.get("connected_sub_scenes", [])
    if not connected_ids:
        return "（无相邻子场景）"

    lines = []
    for adj_id in connected_ids[:8]:  # 最多 8 个相邻场景
        adj = ss_index.get(adj_id)
        if not adj:
            continue
        sp = adj.get("space_profile", {})
        shape = sp.get("shape", "?")
        floor = adj.get("floor", "?")
        role = adj.get("primary_role", adj.get("tactical_role", "?"))
        desc = adj.get("description", "")[:100]

        # 直接用子场景自己的描述
        lines.append(
            f"- {adj_id}: F{floor} {shape}, role={role}, "
            f"desc=\"{desc}\""
        )

        # 标注该相邻场景的关键出入口
        entries = sp.get("entries_exits", [])
        if entries:
            entry_strs = []
            for e in entries[:3]:
                entry_strs.append(f"{e.get('type')}(朝向{e.get('facing','?')})")
            lines.append(f"  出入口: {', '.join(entry_strs)}")

    return '\n'.join(lines) if lines else "（无相邻子场景信息）"


def _build_phase2_context(
    ss_def: Dict,
    definitions: Dict,
    ss_index: Dict[str, Dict],
) -> str:
    """为 Phase 3c 构建 Phase 2 权威数据上下文

    包含该子场景的空间描述（含开口尺寸）、space_profile 中的已知出入口、
    以及 entry_options 中对应的入口选项。
    这些数据是 Phase 1/2 多源几何分析的产物，标注时优先采用其尺寸。
    """
    lines = []

    # 1. 空间描述（含开口尺寸等关键数据）
    desc = ss_def.get("description", "")
    if desc:
        lines.append(f"空间描述: {desc}")

    # 2. space_profile 中的已知出入口
    sp = ss_def.get("space_profile", {})
    entries_exits = sp.get("entries_exits", [])
    if entries_exits:
        lines.append(f"已知开口 ({len(entries_exits)} 个):")
        for ee in entries_exits:
            etype = ee.get("type", "?")
            facing = ee.get("facing", "?")
            width_cat = ee.get("width_category", "?")
            access = ee.get("accessibility", "?")
            lines.append(f"  - {etype}: 朝向{facing}, 宽度类别={width_cat}, 可达性={access}")

    # 3. 与该子场景相关的 entry_options
    entry_options = definitions.get("entry_options", [])
    related_entries = [e for e in entry_options if e.get("sub_scene_id") == ss_def.get("sub_scene_id")]
    if related_entries:
        lines.append(f"入口选项 ({len(related_entries)} 个):")
        for e in related_entries:
            lines.append(
                f"  - {e.get('entry_id')}: type={e.get('type')}, "
                f"floor={e.get('floor')}, access={e.get('access_means')}, "
                f"desc=\"{e.get('description', '')}\""
            )

    # 4. 楼层信息
    floor = ss_def.get("floor", "?")
    lines.append(f"楼层: F{floor}" + (
        " (地面层，只存在向上楼梯)" if floor == 0 else
        " (顶层，只存在向下楼梯)" if floor == 2 else
        " (中间层，存在双向楼梯)"
    ))

    # 5. connected_sub_scenes 及连接类型
    connected = ss_def.get("connected_sub_scenes", [])
    if connected:
        lines.append(f"连通子场景: {', '.join(connected)}")

    return '\n'.join(lines)


# ── CLI 入口 ────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Phase 3: 几何处理 + 语义标注 → per-SS scene_cubes.json + desc.json"
    )
    parser.add_argument(
        "scene_metadata_path",
        help="Phase 0 输出的 scene_metadata.json 路径",
    )
    parser.add_argument(
        "sub_scene_definitions_path",
        help="Phase 2 输出的 sub_scene_definitions.json 路径",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default=None,
        help=f"子场景输出目录 (默认: {SUB_SCENES_DIR})",
    )
    args = parser.parse_args()

    for p in [args.scene_metadata_path, args.sub_scene_definitions_path]:
        if not Path(p).exists():
            print(f"错误: 文件不存在: {p}", file=sys.stderr)
            sys.exit(1)

    results = run_phase3(
        args.scene_metadata_path,
        args.sub_scene_definitions_path,
        output_dir=args.output_dir,
    )
    print(f"Phase 3 完成: {len(results)} 个子场景")
    for r in results:
        status = "PASS" if r["validation_passed"] else "FAIL"
        print(f"  {status} {r['sub_scene_id']}: {r['desc_path']}")
