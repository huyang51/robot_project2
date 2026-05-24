"""
M1: desc.json 一致性验证 + 战术标注

在进入 LLM 战术生成前：
1. 验证 desc.json 与 scene_cubes 的一致性
2. 提取战术标注（空间挑战类型、任务提示）供 M2 使用
"""

import logging
from typing import Dict, List, Any

from ..phase3.validator import validate_sub_scene_completeness

logger = logging.getLogger(__name__)


def run_m1(
    desc_json: Dict[str, Any],
    scene_cubes: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """M1 主入口

    Returns:
        {
            "validation": {...},
            "tactical_annotations": {...},
            "ready_for_phase4": bool,
        }
    """
    # 1. 一致性验证
    validation = validate_sub_scene_completeness(desc_json, scene_cubes)

    # 2. 战术标注提取
    annotations = extract_tactical_annotations(desc_json)

    ready = validation["passed"]

    return {
        "validation": validation,
        "tactical_annotations": annotations,
        "ready_for_phase4": ready,
    }


def extract_tactical_annotations(desc_json: Dict[str, Any]) -> Dict[str, Any]:
    """从 desc.json 提取战术标注

    Returns:
        {
            "spatial_challenge_types": [...],
            "threat_summary": [...],
            "cover_summary": {...},
            "opening_summary": [...],
            "zone_graph": {...},
        }
    """
    # 空间挑战类型
    challenges = _extract_spatial_challenges(desc_json)

    # 威胁摘要
    threats = desc_json.get("inferred_threats", [])
    threat_summary = [
        {
            "type": t.get("type"),
            "severity": t.get("severity"),
            "zone": t.get("location_zone"),
        }
        for t in threats
    ]

    # 掩体摘要
    covers = desc_json.get("cover_assessment", [])
    cover_summary = {
        "total": len(covers),
        "standing": sum(1 for c in covers if c.get("quality") == "standing"),
        "crouching": sum(1 for c in covers if c.get("quality") == "crouching"),
        "concealment": sum(1 for c in covers if c.get("quality") == "concealment_only"),
    }

    # 开口摘要
    openings = desc_json.get("openings", [])
    opening_summary = [
        {"id": op.get("id"), "type": op.get("type"), "connects": op.get("connects")}
        for op in openings
    ]

    # 区域连通图
    zones = desc_json.get("zones", [])
    zone_graph = {
        z.get("zone_id"): z.get("connected_to", [])
        for z in zones
    }

    return {
        "spatial_challenge_types": challenges,
        "threat_summary": threat_summary,
        "cover_summary": cover_summary,
        "opening_summary": opening_summary,
        "zone_graph": zone_graph,
    }


def _extract_spatial_challenges(desc_json: Dict[str, Any]) -> List[str]:
    """提取空间挑战类型"""
    challenges = set()
    zone_types = {z.get("type", "") for z in desc_json.get("zones", [])}
    threat_types = {t.get("type", "") for t in desc_json.get("inferred_threats", [])}
    tags = set(desc_json.get("inferred_tags", []))

    zone_map = {
        "corridor": "走廊推进",
        "corner": "转角处理",
        "room": "房间突入",
        "stairwell": "楼梯推进",
        "open_area": "开阔地机动",
        "entry": "入口突破",
    }
    for zt in zone_types:
        if zt in zone_map:
            challenges.add(zone_map[zt])

    if "blind_corner" in threat_types:
        challenges.add("转角处理")
    if "vertical_threat" in threat_types:
        challenges.add("垂直空间控制")
    if "choke_point" in threat_types:
        challenges.add("瓶颈通过")

    return sorted(challenges)
