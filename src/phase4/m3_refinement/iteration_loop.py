"""
M3 辩论式迭代协议

A_gen ↔ A_review 迭代循环（最多 3 轮）：
1. A_gen 生成初始战术
2. 提取审查上下文 → A_review 执行全部 14 条通用性审查 + 军事可行性审查
3. 若有硬约束违规 → A_gen 修正 → 回到 2
4. 若通过通用性审查 → A_review 执行军事可行性审查
5. 军事可行性 >= 阈值 → 输出最终战术
6. 军事可行性 < 3.0 → 早期丢弃

注意：全部 14 条通用性规则 (G-T1~G-T9, G-S1~G-S5) 由 A_review LLM 语义审查，
不依赖前置正则预检。正则已被废弃——中文文本的语义判断（区分"步骤1"与"1米"、
区分方向用字"一侧"与数量用字"三名"）无法用正则可靠完成。
"""

import json
import logging
from typing import Dict, Optional, Tuple, Any

from ...core.llm_client import MiniMaxClient
from ...core.exceptions import TacticGenerateError
from ...config import M3_ITERATION_PARAMS
from .agen_prompts import get_agen_prompt_for_mode
from .areview_prompts import AREVIEW_SYSTEM_PROMPT, build_areview_input
from .precheck import extract_review_context
from .review_schema import ReviewFeedback
from .few_shot_examples import format_examples_for_prompt

logger = logging.getLogger(__name__)


class M3IterationLoop:
    """A_gen ↔ A_review 辩论式迭代协议"""

    def __init__(
        self,
        client: Optional[MiniMaxClient] = None,
        params: Optional[Dict] = None,
    ):
        self.client = client or MiniMaxClient()
        self.params = params or M3_ITERATION_PARAMS
        self.max_rounds = self.params.get("max_rounds", 3)
        self.q_pass_threshold = self.params.get("q_pass_threshold", 7.0)
        self.early_discard_threshold = self.params.get("early_discard_threshold", 3.0)
        self.score_convergence_delta = self.params.get("score_convergence_delta", 0.5)

    def run(
        self,
        desc_json: Dict[str, Any],
        mode: str = "GEN",
        reference_content: Optional[str] = None,
        mission_phase: Optional[str] = None,
        seed_concept: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], ReviewFeedback]:
        """执行完整的 M3 迭代协议

        Args:
            desc_json: 子场景语义标注
            mode: 生成模式 "RAG"|"HYBRID"|"GEN"
            reference_content: PDF 参考资料（RAG/Hybrid 模式）
            mission_phase: 作战阶段约束，仅生成该阶段战术
            seed_concept: 可选，穷举阶段产出的战术概念。提供后 A_gen 将基于此概念
                          扩展生成完整双版本战术，而非从零设计。

        Returns:
            (final_tactic_json, last_review_feedback)
        """
        # 准备 Few-Shot 示例
        few_shot_text = format_examples_for_prompt(mode)

        # 准备 A_gen system prompt
        agen_prompt = get_agen_prompt_for_mode(
            mode=mode,
            reference_content=reference_content,
            few_shot_text=few_shot_text,
            mission_phase=mission_phase,
        )

        # 构建场景输入（保存基础版本，避免修正指令跨轮累积）
        scene_input = json.dumps(desc_json, ensure_ascii=False, indent=2)

        if seed_concept:
            seed_str = json.dumps(seed_concept, ensure_ascii=False, indent=2)
            base_user_prompt = (
                f"## 子场景语义标注\n\n{scene_input}\n\n"
                f"## 战术概念种子\n\n以下是从穷举生成阶段产出的战术概念，请基于此概念扩展为"
                f"完整的 text_version + struct_version 双版本战术方案。\n\n"
                f"```json\n{seed_str}\n```\n\n"
                f"请保持 Tactic_Name, objective, Tactic_Type, Semantic_Tags 不变，"
                f"将 Description 和 Action_Sequence 扩展为符合双版本格式的完整内容。"
            )
        else:
            base_user_prompt = (
                f"## 子场景语义标注\n\n{scene_input}\n\n"
                f"请基于以上场景信息生成通用化战术方案。"
            )

        tactic_json = None
        last_review: Optional[ReviewFeedback] = None
        prev_score: Optional[float] = None

        for round_idx in range(1, self.max_rounds + 1):
            logger.info(f"M3 第 {round_idx}/{self.max_rounds} 轮")

            # Step 1: 检查上一轮的通用性违规并修正
            agen_user_prompt = base_user_prompt
            if round_idx > 1 and last_review and last_review.hard_violation_count > 0:
                fix_instruction = self._build_fix_instruction(last_review)
                agen_user_prompt = f"{base_user_prompt}\n\n{fix_instruction}"

            # Step 2: A_gen 生成/修正
            # 触发条件：首轮无战术 / 审查未通过 / 审查通过但分数低于阈值需改进
            needs_regeneration = (
                tactic_json is None
                or (last_review and not last_review.overall_pass)
                or (last_review and last_review.score < self.q_pass_threshold)
            )
            if needs_regeneration:
                logger.info(f"  A_gen 生成战术 (mode={mode})")
                result = self.client.generate_json(
                    system_prompt=agen_prompt,
                    user_prompt=agen_user_prompt,
                    temperature=0.7,
                )
                # generate_json 在 "Extra data" 修复路径中可能返回列表
                if isinstance(result, list):
                    tactic_json = result[0] if len(result) > 0 and isinstance(result[0], dict) else {}
                elif isinstance(result, dict):
                    tactic_json = result
                else:
                    tactic_json = None

            if tactic_json is None:
                raise TacticGenerateError("A_gen 生成失败")

            # Step 3: 提取审查上下文 → A_review 审查全部 14 条规则
            precheck_context = extract_review_context(tactic_json)
            review_input = build_areview_input(
                tactic_json=tactic_json,
                precheck_context=precheck_context,
                round_number=round_idx,
            )

            logger.info(f"  A_review 审查中 (14条通用性规则 + 军事可行性)...")
            review_raw = self.client.generate_json(
                system_prompt=AREVIEW_SYSTEM_PROMPT,
                user_prompt=review_input,
                temperature=0.3,
            )
            # generate_json 可能返回列表
            if isinstance(review_raw, list):
                review_raw = review_raw[0] if len(review_raw) > 0 and isinstance(review_raw[0], dict) else {}
            elif not isinstance(review_raw, dict):
                review_raw = {}
            last_review = ReviewFeedback.from_dict(review_raw, round_idx)

            # Step 5: 早期丢弃检查
            if last_review.should_discard:
                logger.warning(f"  M3 第{round_idx}轮: 军事可行性 {last_review.score:.1f} < {self.early_discard_threshold}, 早期丢弃")
                return tactic_json, last_review

            # Step 6: 收敛检查
            if last_review.overall_pass:
                if last_review.score >= self.q_pass_threshold:
                    logger.info(f"  M3 第{round_idx}轮: 审核通过 (score={last_review.score})")
                    return tactic_json, last_review
                elif prev_score is not None and abs(last_review.score - prev_score) < self.score_convergence_delta:
                    logger.info(f"  M3 第{round_idx}轮: 收敛 (delta={abs(last_review.score - prev_score):.2f} < {self.score_convergence_delta})")
                    return tactic_json, last_review

            prev_score = last_review.score

        # 达到最大轮次
        logger.info(f"M3 达到最大轮次 {self.max_rounds}, 返回当前最优")
        if tactic_json is None:
            raise TacticGenerateError(f"M3: 经过 {self.max_rounds} 轮迭代仍未能生成有效战术")
        return tactic_json, last_review or ReviewFeedback(
            round=self.max_rounds, overall_pass=False, score=0.0
        )

    def _build_fix_instruction(self, review: ReviewFeedback) -> str:
        """构建修正指令（仅包含硬约束违规）"""
        hard_violations = [v for v in review.violations if v.rule_type == "hard"]
        if not hard_violations:
            return ""

        lines = ["【重要】上一轮审查发现以下必须修正的违规：", ""]
        for v in hard_violations:
            lines.append(f"- [{v.rule_id}] {v.location}: \"{v.original}\" → {v.suggestion}")

        if review.improvement_suggestions:
            lines.append("")
            lines.append("改进建议：")
            for s in review.improvement_suggestions:
                lines.append(f"- {s}")

        lines.append("")
        lines.append("请修正以上违规后重新生成战术JSON。")
        return "\n".join(lines)
