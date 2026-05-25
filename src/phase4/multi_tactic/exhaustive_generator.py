"""
穷举战术生成器

参考 robot_project TacticGeneratorV2.enumerate_tactics()，
实现参考资料分批注入、迭代生成、去重、终止控制。
"""

import json
import logging
from typing import Dict, List, Optional, Any

from ...core.llm_client import MiniMaxClient
from .exhaustive_prompts import build_exhaustive_prompts

logger = logging.getLogger(__name__)

# ── 常量 ──

REF_BATCH_SIZE = 2000       # 参考资料每批字数
MAX_TACTICS = 15            # 每个子场景最多生成战术数
MAX_NO_REF_ROUNDS = 2       # 无参考阶段最多生成轮数


class ExhaustiveTacticGenerator:
    """穷举式多战术生成器

    Stage 0（在 M3 之前）:
    1. 把参考资料按 REF_BATCH_SIZE 分批
    2. 每轮注入一批参考 + 已生成战术列表 → LLM 发现新战术方向
    3. Python 层基于名称 + Jaccard 词重叠去重
    4. 终止: MAX_TACTICS 个 或 MAX_NO_REF_ROUNDS 轮无新战术
    """

    def __init__(self, client: Optional[MiniMaxClient] = None):
        self.client = client or MiniMaxClient()

    # ── Public API ──────────────────────────────────────

    def enumerate(
        self,
        desc_json: Dict[str, Any],
        reference_content: str = "",
        mission_phase: str = "",
    ) -> List[Dict[str, Any]]:
        """穷举生成所有候选战术概念

        Args:
            desc_json: 子场景语义标注
            reference_content: M2 聚合的参考资料（可能很长）
            mission_phase: 作战阶段约束

        Returns:
            去重后的战术概念列表（每个含 Tactic_Name, objective, Description 等）
        """
        sub_scene_id = desc_json.get("sub_scene_id", "unknown")
        scene_str = json.dumps(desc_json, ensure_ascii=False, indent=2)

        # 1. 分割参考资料
        batches = self._split_reference(reference_content)
        total_batches = len(batches)
        batch_idx = 0

        all_tactics: List[Dict] = []
        no_ref_rounds = 0

        logger.info(
            "穷举生成开始: %s, 参考 %d 批, 上限 %d 个",
            sub_scene_id, total_batches, MAX_TACTICS,
        )

        while True:
            # 确定本轮注入的参考批次
            if batches and batch_idx < total_batches:
                batch_content = batches[batch_idx]
                logger.info(
                    "  批次 %d/%d (%d 字)",
                    batch_idx + 1, total_batches, len(batch_content),
                )
            else:
                batch_content = "无更多参考资料"
                logger.info("  参考耗尽，无参考生成轮 %d/%d",
                            no_ref_rounds + 1, MAX_NO_REF_ROUNDS)

            no_ref_this_round = (batch_idx >= total_batches) or (not batches)

            # 2. 构建 prompt
            system_prompt, user_prompt = build_exhaustive_prompts(
                scene_json=scene_str,
                reference_content=batch_content,
                existing_tactics=all_tactics,
                mission_phase=mission_phase,
            )

            # 3. 调用 LLM
            result = self._safe_generate(system_prompt, user_prompt)

            # 4. 去重并添加新战术
            new_count = 0
            if isinstance(result, list):
                for tactic in result:
                    if not isinstance(tactic, dict):
                        continue
                    if self._is_duplicate(tactic, all_tactics):
                        continue
                    # 注入来源信息（后续 M3 会用）
                    tactic["_source_sub_scene"] = sub_scene_id
                    all_tactics.append(tactic)
                    new_count += 1

            if new_count > 0:
                logger.info("  本轮生成 %d 个新战术，累计 %d", new_count, len(all_tactics))
            else:
                if no_ref_this_round:
                    no_ref_rounds += 1
                else:
                    logger.info("  本轮无新战术，仍有参考批待注入")

            # 5. 推进批次
            if batches and batch_idx < total_batches:
                batch_idx += 1

            # 6. 终止检查
            if no_ref_rounds >= MAX_NO_REF_ROUNDS:
                logger.info("穷举终止: 无参考阶段 %d 轮", no_ref_rounds)
                break

            if len(all_tactics) >= MAX_TACTICS:
                logger.info("穷举终止: 达到上限 %d", MAX_TACTICS)
                break

        logger.info("穷举完成: %s 共 %d 个战术", sub_scene_id, len(all_tactics))
        return all_tactics

    # ── 去重 ──────────────────────────────────────────

    def _is_duplicate(self, tactic: Dict, existing: List[Dict]) -> bool:
        """检查战术是否与已有列表重复

        1. Tactic_Name 完全相同 → 重复
        2. Description Jaccard 词重叠 > 0.8 → 重复
        """
        name = (tactic.get("Tactic_Name") or "").strip().lower()
        desc = (tactic.get("Description") or "").strip().lower()

        for ex in existing:
            ex_name = (ex.get("Tactic_Name") or "").strip().lower()
            ex_desc = (ex.get("Description") or "").strip().lower()

            if name and ex_name and name == ex_name:
                return True

            if desc and ex_desc:
                if self._jaccard(desc, ex_desc) > 0.8:
                    return True

        return False

    @staticmethod
    def _jaccard(text1: str, text2: str) -> float:
        """Jaccard 词重叠系数"""
        for sep in [",", "。", "、", "；", ";", ".", "\n"]:
            text1 = text1.replace(sep, " ")
            text2 = text2.replace(sep, " ")
        words1 = set(w for w in text1.split() if len(w) > 1)
        words2 = set(w for w in text2.split() if len(w) > 1)
        if not words1 or not words2:
            return 0.0
        return len(words1 & words2) / len(words1 | words2)

    # ── 内部工具 ──────────────────────────────────────

    @staticmethod
    def _split_reference(content: str) -> List[str]:
        """将参考资料按字数分批"""
        if not content or not content.strip():
            return []
        batches = []
        ref = content.strip()
        for i in range(0, len(ref), REF_BATCH_SIZE):
            batches.append(ref[i:i + REF_BATCH_SIZE])
        return batches

    def _safe_generate(self, system_prompt: str, user_prompt: str) -> Any:
        """安全的 LLM 调用，失败返回 []"""
        try:
            result = self.client.generate_json(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.7,
            )
            # generate_json 可能返回 list 或 dict（Extra data 修复后取首元素）
            if isinstance(result, dict):
                # 单对象包裹为列表
                return [result] if result else []
            return result if isinstance(result, list) else []
        except Exception as e:
            logger.warning("穷举生成 LLM 调用失败: %s", e)
            return []
