"""
Phase 3 desc.json Schema 类型定义

从 robot_project 的 core/phase3c_schema.py 适配。
包含 ZoneDef, OpeningDef, CoverAssessment, InferredThreat, DescJson 等核心结构。
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any

# ============================================================
# cover_assessment.quality 枚举
# ============================================================
COVER_QUALITY_ENUM = {
    "standing": {
        "label": "立姿掩体",
        "height_range": "> 1.5m",
        "description": "提供站立状态的掩护，可掩护立姿射击和观察",
        "ballistic_protection": True,
    },
    "crouching": {
        "label": "蹲姿掩体",
        "height_range": "0.7 - 1.5m",
        "description": "需蹲姿或跪姿才能获得掩护",
        "ballistic_protection": True,
    },
    "concealment_only": {
        "label": "仅隐蔽",
        "height_range": "任意高度",
        "description": "仅阻隔视线，无弹道防护能力",
        "ballistic_protection": False,
    },
    "obstacle": {
        "label": "障碍物",
        "height_range": "任意高度",
        "description": "阻挡移动路线，不作掩体用",
        "ballistic_protection": False,
    },
    "none": {
        "label": "无掩护价值",
        "height_range": "-",
        "description": "无掩护价值，仅用于标记评估过的物体",
        "ballistic_protection": False,
    },
}

COVER_HEIGHT_THRESHOLDS = {
    "standing_min": 1.5,
    "crouching_min": 0.7,
}

# ============================================================
# inferred_threats 枚举
# ============================================================
THREAT_SEVERITY_GRADES = {
    "critical": {"label": "致命威胁", "score_range": "9-10"},
    "high": {"label": "高威胁", "score_range": "7-8"},
    "medium": {"label": "中等威胁", "score_range": "5-6"},
    "low": {"label": "低威胁", "score_range": "3-4"},
}


# ============================================================
# desc.json 核心 dataclass
# ============================================================

@dataclass
class ZoneDef:
    zone_id: str
    type: str
    bounds_rel: Dict[str, List[float]]
    description: str
    connected_to: List[str] = field(default_factory=list)
    contained_cube_ids: List[str] = field(default_factory=list)


@dataclass
class OpeningDef:
    id: str
    type: str  # door / window / balcony_opening
    connects: List[str]
    width: float
    height: float
    position_rel: Dict[str, float]


@dataclass
class CoverAssessment:
    cube_id: str
    quality: str
    height: float
    coverage_direction: List[str]
    notes: str
    effective_positions: List[Dict] = field(default_factory=list)
    # effective_positions: [{"behind": "south", "covers_from": ["north"], "exposed_to": ["east"]}]
    # 描述"蹲在这个物体的哪个方位，能掩护哪个方向的火力，暴露于哪个方向"


@dataclass
class InferredThreat:
    type: str
    severity: str
    location_zone: str
    description: str


@dataclass
class KeyControlPoint:
    id: str
    position_rel: Dict[str, float]
    description: str
    controls: List[str]
    tactical_value: str


@dataclass
class MovementAnalysis:
    available_paths: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    key_control_points: List[KeyControlPoint] = field(default_factory=list)


@dataclass
class MovementPath:
    """机动路径片段"""
    from_position: str
    to_position: str
    exposed_to: List[str] = field(default_factory=list)
    exposure_time_category: str = "brief"  # brief/medium/prolonged
    cover_available_during_movement: bool = False


@dataclass
class TacticalBoundary:
    """战术前置/后置状态，定义子场景的起止条件"""
    entry_points: List[Dict] = field(default_factory=list)
    # [{"id": "EP_01", "zone_id": "Z1", "opening_id": "DOOR_01", "approach_from": "corridor_south"}]
    objective_criteria: List[str] = field(default_factory=list)
    # ["all corners visually confirmed clear", "all cover positions checked"]
    completion_transitions: List[Dict] = field(default_factory=list)
    # [{"to_sub_scene": "SS_05", "via": "DOOR_02", "condition": "room secure"}]


@dataclass
class DescJson:
    """Phase 3 子场景语义标注完整结构"""
    sub_scene_id: str
    tactical_role: str
    task_hint: str
    zones: List[ZoneDef] = field(default_factory=list)
    openings: List[OpeningDef] = field(default_factory=list)
    cover_assessment: List[CoverAssessment] = field(default_factory=list)
    inferred_threats: List[InferredThreat] = field(default_factory=list)
    movement_analysis: MovementAnalysis = field(default_factory=MovementAnalysis)
    exposure_assessment: List[MovementPath] = field(default_factory=list)
    tactical_boundary: TacticalBoundary = field(default_factory=TacticalBoundary)
    spatial_description: str = ""
    inferred_tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """将 DescJson 序列化为字典"""
        return {
            "sub_scene_id": self.sub_scene_id,
            "tactical_role": self.tactical_role,
            "task_hint": self.task_hint,
            "zones": [z.__dict__ for z in self.zones],
            "openings": [o.__dict__ for o in self.openings],
            "cover_assessment": [
                {**c.__dict__, "effective_positions": c.effective_positions}
                for c in self.cover_assessment
            ],
            "inferred_threats": [t.__dict__ for t in self.inferred_threats],
            "movement_analysis": {
                "available_paths": self.movement_analysis.available_paths,
                "constraints": self.movement_analysis.constraints,
                "key_control_points": [
                    kp.__dict__ for kp in self.movement_analysis.key_control_points
                ],
            },
            "exposure_assessment": [e.__dict__ for e in self.exposure_assessment],
            "tactical_boundary": {
                "entry_points": self.tactical_boundary.entry_points,
                "objective_criteria": self.tactical_boundary.objective_criteria,
                "completion_transitions": self.tactical_boundary.completion_transitions,
            },
            "spatial_description": self.spatial_description,
            "inferred_tags": self.inferred_tags,
        }
