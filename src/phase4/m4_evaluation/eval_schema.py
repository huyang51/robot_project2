"""
M4 评估结果 Schema

定义 EvalResult 和 DimensionScore 数据结构。
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class DimensionScore:
    score: float
    checks: Dict[str, bool] = field(default_factory=dict)
    deductions: List[Dict] = field(default_factory=list)
    sub_scores: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        result = {
            "score": self.score,
            "checks": self.checks,
            "deductions": self.deductions,
        }
        if self.sub_scores:
            result["sub_scores"] = self.sub_scores
        return result

    @classmethod
    def from_dict(cls, data: Dict) -> "DimensionScore":
        return cls(
            score=data.get("score", 0.0),
            checks=data.get("checks", {}),
            deductions=data.get("deductions", []),
            sub_scores=data.get("sub_scores", {}),
        )


@dataclass
class EvalResult:
    """A_eval 质量评估结果"""

    DIMENSION_NAMES: List[str] = field(default_factory=lambda: [
        "scene_adaptation", "execution_efficiency",
        "comprehension", "granularity_compliance",
        "text_visual_consistency", "military_feasibility",
    ], init=False, repr=False)

    scores: Dict[str, DimensionScore] = field(default_factory=dict)
    overall_score: float = 0.0
    quality_level: str = "L"  # H/M/L
    veto_triggered: bool = False
    veto_reason: str = ""
    evaluation_summary: str = ""

    @classmethod
    def from_dict(cls, data: Dict) -> "EvalResult":
        scores_raw = data.get("scores", {})
        scores = {}
        for dim, dim_data in scores_raw.items():
            if isinstance(dim_data, dict):
                scores[dim] = DimensionScore.from_dict(dim_data)
            elif isinstance(dim_data, (int, float)):
                scores[dim] = DimensionScore(score=float(dim_data))
            else:
                scores[dim] = DimensionScore(score=0.0)

        return cls(
            scores=scores,
            overall_score=data.get("overall_score", 0.0),
            quality_level=data.get("quality_level", "L"),
            veto_triggered=data.get("veto_triggered", False),
            veto_reason=data.get("veto_reason", ""),
            evaluation_summary=data.get("evaluation_summary", ""),
        )

    def to_dict(self) -> Dict:
        return {
            "scores": {k: v.to_dict() for k, v in self.scores.items()},
            "overall_score": self.overall_score,
            "quality_level": self.quality_level,
            "veto_triggered": self.veto_triggered,
            "veto_reason": self.veto_reason,
            "evaluation_summary": self.evaluation_summary,
        }

    @property
    def is_high_quality(self) -> bool:
        return self.quality_level == "H"

    @property
    def is_ingestible(self) -> bool:
        return self.quality_level in ("H", "M")
