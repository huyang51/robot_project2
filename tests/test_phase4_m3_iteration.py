"""
Phase 4 M3 迭代精炼测试
"""
import pytest
from unittest.mock import MagicMock, patch
from src.phase4.m3_refinement.iteration_loop import M3IterationLoop
from src.phase4.m3_refinement.review_schema import ReviewFeedback, Violation
from src.phase4.m3_refinement.precheck import extract_review_context


class TestPrecheck:
    """审查上下文提取测试"""

    def test_extract_text_fields(self):
        """验证从 text_version 提取审查文本字段"""
        tactic = {
            "text_version": {
                "Tactic_ID": "CQB_01",
                "Description": "突击手沿走廊一侧贴墙机动至转角紧邻处，实施短点射压制",
                "objective": "肃清走廊并建立火力控制",
                "Action_Sequence": [
                    {"Step": 1, "Intent": "侦察节点同步前出至高位观察点"},
                    {"Step": 2, "Intent": "突击手低姿跃进至转角紧邻处", "Visual_Aids": ["走廊俯视图"]},
                ]
            },
            "struct_version": {
                "Action_Sequence": [
                    {"Step": 1, "Intent": "侦察", "Instructions": [
                        "[Unit侦察节点] 移动至 {observation_point}",
                        "[Unit侦察节点] 扫描 {area}"
                    ]}
                ]
            }
        }
        ctx = extract_review_context(tactic)

        # text_version 应提取 Description + objective + 2 个 Intent + 1 个 Visual_Aids = 5 个字段
        assert len(ctx["text_version_fields"]) == 5
        assert ctx["text_version_fields"][0]["location"] == "Description"
        assert ctx["text_version_fields"][1]["location"] == "objective"

        # struct_version 应提取 Intent + 2 个 Instructions = 3 个字段
        assert len(ctx["struct_version_fields"]) == 3

    def test_empty_action_sequence(self):
        """空 Action_Sequence 不崩溃"""
        tactic = {
            "text_version": {
                "Description": "测试描述",
                "objective": "测试目标",
                "Action_Sequence": []
            },
            "struct_version": {"Action_Sequence": []}
        }
        ctx = extract_review_context(tactic)
        assert len(ctx["text_version_fields"]) == 2
        assert len(ctx["struct_version_fields"]) == 0

    def test_missing_fields_handled(self):
        """缺失字段不崩溃"""
        tactic = {
            "text_version": {},
            "struct_version": {}
        }
        ctx = extract_review_context(tactic)
        assert len(ctx["text_version_fields"]) == 0
        assert len(ctx["struct_version_fields"]) == 0


class TestReviewFeedback:
    """审查反馈 Schema 测试"""

    def test_from_dict(self):
        data = {
            "round": 1,
            "overall_pass": True,
            "score": 8.5,
            "dimension_scores": {
                "granularity_compliance": 9.0,
                "military_feasibility": 8.0
            },
            "violations": [],
            "improvement_suggestions": ["建议1"],
            "military_feasibility_notes": "合格",
            "should_discard": False,
        }
        fb = ReviewFeedback.from_dict(data, round_number=1)
        assert fb.overall_pass
        assert fb.score == 8.5
        assert fb.hard_violation_count == 0

    def test_violation_count(self):
        fb = ReviewFeedback(
            round=1, overall_pass=False, score=5.0,
            violations=[
                Violation(rule_id="G-T1", rule_type="hard", version="text_version",
                          location="Description", original="3米", suggestion="紧邻"),
                Violation(rule_id="G-T7", rule_type="soft", version="text_version",
                          location="Description", original="白色", suggestion="通用描述"),
            ]
        )
        assert fb.hard_violation_count == 1
        assert fb.soft_violation_count == 1

class TestM3IterationLoop:
    """M3 迭代协议测试"""

    def test_build_fix_instruction(self):
        """测试修正指令构建"""
        loop = M3IterationLoop()
        feedback = ReviewFeedback(
            round=1, overall_pass=False, score=5.0,
            violations=[
                Violation(rule_id="G-T4", rule_type="hard", version="text_version",
                          location="Description", original="距转角2米", suggestion="转角紧邻处"),
            ],
            improvement_suggestions=["使用通用化方位词替代绝对方向"],
        )

        instruction = loop._build_fix_instruction(feedback)
        assert "G-T4" in instruction
        assert "距转角2米" in instruction
        assert "转角紧邻处" in instruction
        assert "通用化方位词" in instruction

    def test_build_fix_instruction_only_hard(self):
        """修正指令仅包含硬约束违规"""
        loop = M3IterationLoop()
        feedback = ReviewFeedback(
            round=1, overall_pass=False, score=5.0,
            violations=[
                Violation(rule_id="G-T1", rule_type="hard", version="text_version",
                          location="Desc", original="5", suggestion="数步"),
                Violation(rule_id="G-T7", rule_type="soft", version="text_version",
                          location="Desc", original="混凝土", suggestion="掩体"),
            ]
        )
        instruction = loop._build_fix_instruction(feedback)
        assert "G-T1" in instruction
        assert "G-T7" not in instruction  # 软约束不放入修正指令
