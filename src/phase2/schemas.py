"""
Phase 2 子场景定义 Schema

定义 SubSceneDef 数据结构。
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class SubSceneDef:
    """子场景定义

    核心设计：空间与战术解耦。space_profile 描述空间物理特征，
    suggested_roles 提供战术建议而非约束。Phase 4 决定具体战术。
    """
    sub_scene_id: str  # 如 "SS_01"
    parent_scene_id: str
    primary_role: str = ""  # 初步建议的主要战术角色
    suggested_roles: List[str] = field(default_factory=list)  # 多种可行战术方案
    space_profile: Dict = field(default_factory=dict)  # 空间物理特征
    spatial_bounds: Dict[str, List[float]] = field(default_factory=dict)  # {"x": [min, max], ...}
    zone_ids: List[str] = field(default_factory=list)
    floor: int = 0
    task_hint: str = ""
    priority: str = "medium"
    description: str = ""
    connected_sub_scenes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "sub_scene_id": self.sub_scene_id,
            "parent_scene_id": self.parent_scene_id,
            "space_profile": self.space_profile,
            "suggested_roles": self.suggested_roles,
            "primary_role": self.primary_role,
            "spatial_bounds": self.spatial_bounds,
            "zone_ids": self.zone_ids,
            "floor": self.floor,
            "task_hint": self.task_hint,
            "priority": self.priority,
            "description": self.description,
            "connected_sub_scenes": self.connected_sub_scenes,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "SubSceneDef":
        return cls(
            sub_scene_id=data.get("sub_scene_id", ""),
            parent_scene_id=data.get("parent_scene_id", ""),
            space_profile=data.get("space_profile", {}),
            suggested_roles=data.get("suggested_roles", []),
            primary_role=data.get("primary_role", data.get("tactical_role", "")),
            spatial_bounds=data.get("spatial_bounds", {}),
            zone_ids=data.get("zone_ids", []),
            floor=data.get("floor", 0),
            task_hint=data.get("task_hint", ""),
            priority=data.get("priority", "medium"),
            description=data.get("description", ""),
            connected_sub_scenes=data.get("connected_sub_scenes", []),
        )
