"""
M4 六维度质量评估器

调用 A_eval LLM 对战术进行六维度全项评分。
"""

import json
import logging
from typing import Dict, Any, Optional

from ...core.llm_client import MiniMaxClient
from ...config import M4_EVALUATION_PARAMS
from .eval_schema import EvalResult
from .rubric import FULL_RUBRIC

logger = logging.getLogger(__name__)


AEVAL_FULL_SYSTEM_PROMPT = f"""
# Role: 质量评估专家 Agent ($A_{{eval}}$)

## Profile
- language: 中文
- description: 专为战术方案质量评价设计的评估专家。对战术JSON进行六维度全项评分，判定质量等级（H/M/L）。
- background: 精通军事战术评估方法论、机器人编队能力模型、通用性合规标准。
- expertise: 多维度战术质量评估、Checklist驱动评分、否决条件判定、质量分级。

## 评估框架

{FULL_RUBRIC}

## 输出格式
```json
{{
  "scores": {{
    "scene_adaptation": {{"score": <float>, "checks": {{"S1": <bool>, ..., "S8": <bool>}}, "deductions": []}},
    "execution_efficiency": {{"score": <float>, "checks": {{"E1": <bool>, ..., "E6": <bool>}}, "deductions": []}},
    "comprehension": {{"score": <float>, "checks": {{"C1": <bool>, ..., "C7": <bool>}}, "deductions": []}},
    "granularity_compliance": {{"score": <float>, "checks": {{"G1": <bool>, ..., "G8": <bool>}}, "deductions": []}},
    "text_visual_consistency": {{"score": <float>, "checks": {{"V1": <bool>, ..., "V7": <bool>}}, "deductions": []}},
    "military_feasibility": {{
      "score": <float>,
      "checks": {{"M1": <bool>, ..., "M18": <bool>}},
      "sub_scores": {{
        "mutual_support_relations": <float>,
        "spatial_control": <float>,
        "tempo_timing": <float>,
        "adversarial_realism": <float>,
        "principles_compliance": <float>
      }},
      "deductions": []
    }}
  }},
  "overall_score": <float>,
  "quality_level": "H|M|L",
  "veto_triggered": <bool>,
  "veto_reason": "<string>",
  "evaluation_summary": "<50-150字>"
}}
```

> **sub_scores 说明**：military_feasibility 的 sub_scores 包含 5 个子维度评分，分别对应：
> - mutual_support_relations（角色互助关系，M1-M3）
> - spatial_control（空间控制，M4-M7）
> - tempo_timing（时间与节奏，M8-M10）
> - adversarial_realism（对抗合理性，M11-M13）
> - principles_compliance（战术原则遵循，M14-M18）
> 各子维度评分按 0-10 制，子维度维度评分锚点见军事可行性各子维度 Checklist 内的核查方法。维度总分 q6 由 5 个子维度分值的加权平均换算（参考 Q 公式的形式），直接给出最终的综合维度分。

## 评分操作流程
对每个维度，请按以下步骤进行：
1. 逐项判定：对 Checklist 中每一项，判定 通过/不通过/部分通过
2. 统计：统计不通过和部分通过的数量，评估严重程度（核心项 vs 边缘项）
3. 对照锚点表：根据不通过数量和严重程度，对照该维度的评分锚点表确定分数区间
4. 精细调分：在确定的分数区间内，根据细节表现上下微调 0.5-1 分
5. 扣分记录：每个不通过项必须记录到 deductions 中，注明 check_id、severity、location、detail

## 行为准则
- 逐项核查，不跳过任何Checklist项
- 每个扣分必须有具体位置和理由
- 严格遵循各维度的评分锚点表
- 否决条件必须显式检查并记录
- military_feasibility 必须输出 sub_scores（5 个子维度评分）

## JSON 格式硬约束（极重要）
- **所有字符串值内的双引号必须转义为 \\\"**。引用原文时使用「」角括号代替ASCII双引号
- **字符串值内的换行必须用 \\n 表示**，禁止在 JSON 字符串中直接换行
- **输出纯 JSON，不要包裹在 ```json 代码块中**
"""


def evaluate_tactic(
    tactic_json: Dict[str, Any],
    desc_json: Dict[str, Any],
    client: Optional[MiniMaxClient] = None,
) -> EvalResult:
    """对战术进行六维度全项评分

    Args:
        tactic_json: 待评估的战术JSON
        desc_json: 对应的子场景语义标注
        client: MiniMaxClient

    Returns:
        EvalResult
    """
    if client is None:
        client = MiniMaxClient()

    user_prompt = "\n".join([
        "## 待评估战术JSON（双版本结构）",
        "",
        "战术输出包含 text_version（文字描述版，侧重可读性）和 "
        "struct_version（结构化描述版，侧重可执行性）两个版本。",
        "评估时请综合考量两个版本：",
        "- 场景适配度/执行效率/军事可行性：以 text_version 的 Description/objective 为主",
        "- 粒度合规性：同时检查两个版本的 Action_Sequence",
        "- 可理解性：以 text_version 的 Intent 字段为主",
        "- 图文一致性：综合两个版本的 Visual_Aids/Instructions",
        "",
        json.dumps(tactic_json, ensure_ascii=False, indent=2),
        "",
        "## 子场景语义标注（desc.json）",
        json.dumps(desc_json, ensure_ascii=False, indent=2),
        "",
        "## 任务",
        "请按照A_eval评估框架对以上战术进行六维度全项评分。",
        "逐维度逐项核查Checklist，给出评分、判定等级。",
    ])

    result = client.generate_json(
        system_prompt=AEVAL_FULL_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        temperature=0.3,
    )

    # generate_json 可能返回 list（Extra data 修复路径）
    if isinstance(result, list):
        result = result[0] if len(result) > 0 and isinstance(result[0], dict) else {}
    if not isinstance(result, dict):
        logger.warning("M4: generate_json 返回非字典类型 %s", type(result))
        result = {}

    eval_result = EvalResult.from_dict(result)

    # 算术校验：6 维分数均值应与 overall_score 一致
    # LLM 自行计算 Q = (q1+...+q6)/6 可能出错，程序化校核确保正确
    if eval_result.scores:
        dim_scores = [s.score for s in eval_result.scores.values()]
        if dim_scores:
            if len(dim_scores) != 6:
                logger.warning(
                    "M4: 期望 6 个维度评分，实际收到 %d 个，可能缺少维度",
                    len(dim_scores)
                )
            calculated_q = sum(dim_scores) / 6
            if abs(calculated_q - eval_result.overall_score) > 0.5:
                logger.warning(
                    "M4: LLM reported overall_score=%.1f but calculated Q=%.1f "
                    "from %d dimension scores, using calculated value",
                    eval_result.overall_score, calculated_q, len(dim_scores)
                )
                eval_result.overall_score = round(calculated_q, 1)

    logger.info(
        f"M4 评估: overall={eval_result.overall_score:.1f}, "
        f"level={eval_result.quality_level}, "
        f"veto={eval_result.veto_triggered}"
    )
    return eval_result


def batch_evaluate(
    tactic_pairs: Dict[str, Dict[str, Any]],
    desc_jsons: Dict[str, Dict[str, Any]],
    client: Optional[MiniMaxClient] = None,
) -> Dict[str, EvalResult]:
    """批量评估多个战术"""
    results = {}
    for ss_id, tactic_json in tactic_pairs.items():
        desc = desc_jsons.get(ss_id, {})
        results[ss_id] = evaluate_tactic(tactic_json, desc, client)
    return results
