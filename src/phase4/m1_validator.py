"""
M1: desc.json 一致性验证

在进入 LLM 战术生成前，验证 desc.json 与 scene_cubes 的一致性。
"""

import logging
from typing import Any, Dict

from ..phase3.validator import validate_sub_scene_completeness

logger = logging.getLogger(__name__)


def run_m1(
    desc_json: Dict[str, Any],
    scene_cubes: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """M1 主入口

    Returns:
        {"validation": {...}, "ready_for_phase4": bool}
    """
    validation = validate_sub_scene_completeness(desc_json, scene_cubes)

    return {
        "validation": validation,
        "ready_for_phase4": validation["passed"],
    }
