"""
M3 审查反馈 Schema

定义 ReviewFeedback 和 Violation 的 dataclass。
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class Violation:
    rule_id: str
    rule_type: str  # "hard" | "soft"
    version: str  # "text_version" | "struct_version"
    location: str
    original: str
    suggestion: str
    reason: str = ""

    def to_dict(self) -> Dict:
        return {
            "rule_id": self.rule_id,
            "rule_type": self.rule_type,
            "version": self.version,
            "location": self.location,
            "original": self.original,
            "suggestion": self.suggestion,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "Violation":
        return cls(
            rule_id=data.get("rule_id", ""),
            rule_type=data.get("rule_type", "soft"),
            version=data.get("version", "text_version"),
            location=data.get("location", ""),
            original=data.get("original", ""),
            suggestion=data.get("suggestion", ""),
            reason=data.get("reason", ""),
        )


@dataclass
class ReviewFeedback:
    """A_review 审查反馈"""
    round: int
    overall_pass: bool
    score: float
    dimension_scores: Dict[str, float] = field(default_factory=dict)
    violations: List[Violation] = field(default_factory=list)
    improvement_suggestions: List[str] = field(default_factory=list)
    military_feasibility_notes: str = ""
    should_discard: bool = False

    @classmethod
    def from_dict(cls, data: Dict, round_number: int = 1) -> "ReviewFeedback":
        violations = [
            Violation.from_dict(v)
            for v in data.get("violations", [])
        ]
        return cls(
            round=data.get("round", round_number),
            overall_pass=data.get("overall_pass", False),
            score=data.get("score", 0.0),
            dimension_scores=data.get("dimension_scores", {}),
            violations=violations,
            improvement_suggestions=data.get("improvement_suggestions", []),
            military_feasibility_notes=data.get("military_feasibility_notes", ""),
            should_discard=data.get("should_discard", False),
        )

    @property
    def hard_violation_count(self) -> int:
        return sum(1 for v in self.violations if v.rule_type == "hard")

    @property
    def soft_violation_count(self) -> int:
        return sum(1 for v in self.violations if v.rule_type == "soft")
