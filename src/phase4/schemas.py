"""
Phase 4 战术生成 Schema (deprecated 版本)

注意: 此文件中的 Violation, ReviewFeedback, EvalResult, DimensionScore
为早期 stub 版本，已被子模块中的完整版本取代：
- ReviewFeedback / Violation → phase4.m3_refinement.review_schema
- EvalResult / DimensionScore → phase4.m4_evaluation.eval_schema

此文件保留以维持向后兼容，新代码请从子模块导入。
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class Violation:
    rule_id: str
    rule_type: str  # "hard" | "soft"
    version: str  # "text_version" | "struct_version"
    location: str
    original: str
    suggestion: str
    reason: str = ""


@dataclass
class ReviewFeedback:
    """A_review 审查反馈"""
    round: int
    overall_pass: bool
    score: float
    dimension_scores: Dict[str, float]
    violations: List[Violation] = field(default_factory=list)
    improvement_suggestions: List[str] = field(default_factory=list)
    military_feasibility_notes: str = ""
    should_discard: bool = False


@dataclass
class DimensionScore:
    score: float
    checks: Dict[str, bool] = field(default_factory=dict)
    deductions: List[Dict] = field(default_factory=list)


@dataclass
class EvalResult:
    """A_eval 质量评估结果"""
    scores: Dict[str, DimensionScore]
    overall_score: float
    quality_level: str  # H/M/L
    veto_triggered: bool = False
    veto_reason: str = ""
    evaluation_summary: str = ""


@dataclass
class TacticAction:
    step: int
    intent: str  # text_version: 详细描述, struct_version: 精简名称
    visual_aids: List[str] = field(default_factory=list)
    instructions: Optional[List[str]] = None  # struct_version only


@dataclass
class Tactic:
    """战术 JSON（双版本）"""
    tactic_id: str
    tactic_name: str
    mission_phase: str
    tactic_type: str
    objective: str
    description: str
    credibility: float = 0.0      # LLM 评估 0-10 分
    applicable_environment: str = ""
    execution_time: float = 0.0   # 分钟
    parent_tactic: Optional[str] = None
    sub_tactics: List[str] = field(default_factory=list)
    semantic_tags: List[str] = field(default_factory=list)
    action_sequence: List[TacticAction] = field(default_factory=list)
    visual_aid_overall: List[str] = field(default_factory=list)

    # 元数据
    source_sub_scene: str = ""
    generation_mode: str = "GEN"
    quality_score: Optional[float] = None
    quality_level: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "Tactic_ID": self.tactic_id,
            "Tactic_Name": self.tactic_name,
            "Mission_Phase": self.mission_phase,
            "Tactic_Type": self.tactic_type,
            "objective": self.objective,
            "Description": self.description,
            "credibility": self.credibility,
            "Applicable environment": self.applicable_environment,
            "execution time": self.execution_time,
            "Parent_Tactic": self.parent_tactic,
            "Sub_Tactics": self.sub_tactics,
            "Semantic_Tags": self.semantic_tags,
            "Action_Sequence": [
                {
                    "Step": a.step,
                    "Intent": a.intent,
                    "Visual_Aids": a.visual_aids,
                    **({"Instructions": a.instructions} if a.instructions else {}),
                }
                for a in self.action_sequence
            ],
            "Visual_Aid_Overall": self.visual_aid_overall,
        }


@dataclass
class TacticPair:
    """双版本战术对（text_version + struct_version）"""
    text_version: Tactic
    struct_version: Tactic
    sub_scene_id: str
    desc_json: Dict
