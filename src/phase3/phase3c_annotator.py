"""
Phase 3c: LLM 语义标注器

调用 LLM 对子场景进行语义标注：zones/openings/cover/threats/movement。
"""

import json
import logging
from typing import Dict, List, Optional, Any

from ..core.llm_client import MiniMaxClient
from .phase3c_prompts import PHASE3C_SYSTEM_PROMPT, build_phase3c_user_prompt

logger = logging.getLogger(__name__)


def annotate_sub_scene(
    cubes: List[Dict],
    sub_scene_id: str,
    tactical_role: str,
    task_hint: str,
    client: Optional[MiniMaxClient] = None,
    global_context: str = "",
    adjacent_scenes: str = "",
    phase2_context: str = "",
) -> Dict[str, Any]:
    """对单个子场景进行 LLM 语义标注

    Args:
        cubes: Phase 3b 简化后的 Cube 列表
        sub_scene_id: 子场景 ID
        tactical_role: 战术角色
        task_hint: 阶段目标
        client: MiniMaxClient 实例
        global_context: 全局上下文（该子场景在建筑中的位置/楼层/相邻zone）
        adjacent_scenes: 相邻子场景摘要
        phase2_context: Phase 2 场景定义（权威开口尺寸和空间描述）

    Returns:
        desc.json 内容字典
    """
    if client is None:
        client = MiniMaxClient()

    # 构建精简输入
    cubes_text = json.dumps(cubes, ensure_ascii=False, indent=2)
    user_prompt = build_phase3c_user_prompt(
        cubes_text, task_hint, tactical_role,
        global_context=global_context,
        adjacent_scenes=adjacent_scenes,
        phase2_context=phase2_context,
    )

    logger.info("Phase 3c LLM 语义标注: %s (%d Cube)", sub_scene_id, len(cubes))

    result = client.generate_json(
        system_prompt=PHASE3C_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        temperature=0.3,
    )
    # generate_json 可能返回 list（Extra data 修复路径）
    if isinstance(result, list):
        result = result[0] if len(result) > 0 and isinstance(result[0], dict) else {}
    if not isinstance(result, dict):
        result = {}

    # 确保 sub_scene_id 一致
    result["sub_scene_id"] = sub_scene_id
    result["tactical_role"] = tactical_role
    result["task_hint"] = task_hint

    return result


def batch_annotate(
    sub_scene_cubes_map: Dict[str, Dict],
    client: Optional[MiniMaxClient] = None,
    max_parallel: int = 1,
) -> Dict[str, Dict]:
    """批量标注多个子场景

    Args:
        sub_scene_cubes_map: {sub_scene_id: {"cubes": [...], "tactical_role": "...", "task_hint": "..."}}
        client: MiniMaxClient
        max_parallel: 最大并行数（当前仅支持 1，未来可扩展）

    Returns:
        {sub_scene_id: desc_json}
    """
    results = {}
    for ss_id, info in sub_scene_cubes_map.items():
        results[ss_id] = annotate_sub_scene(
            cubes=info["cubes"],
            sub_scene_id=ss_id,
            tactical_role=info.get("tactical_role", info.get("primary_role", "")),
            task_hint=info.get("task_hint", ""),
            client=client,
            global_context=info.get("global_context", ""),
            adjacent_scenes=info.get("adjacent_scenes", ""),
            phase2_context=info.get("phase2_context", ""),
        )
    return results
