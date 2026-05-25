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

logger = logging.getLogger(__name__)


AEVAL_FULL_SYSTEM_PROMPT = """
# Role: 质量评估专家 Agent ($A_{eval}$)

## Profile
- language: 中文
- description: 专为战术方案质量评价设计的评估专家。对战术JSON进行六维度全项评分，判定质量等级（H/M/L）。
- background: 精通军事战术评估方法论、机器人编队能力模型、通用性合规标准。
- expertise: 多维度战术质量评估、Checklist驱动评分、否决条件判定、质量分级。

## 评估框架

### 通用评分锚点
| 分数 | 定义 |
|------|------|
| 10 | 卓越：可作为正面范例 |
| 8-9 | 优秀：完全达标，无遗漏 |
| 6-7 | 合格：基本达标，少量细节瑕疵 |
| 4-5 | 有缺陷：存在明显设计缺陷 |
| 2-3 | 严重缺陷：多处关键遗漏 |
| 0-1 | 不可接受：违反基本军事原则 |

### 维度一：场景适配度
核查项 S1-S8：空间挑战类型全覆盖、开口利用模式正确、掩体利用意识、
威胁类型全应对、task_hint全回应、环境条件适配、空间关系正确、无场景外假设。

### 维度二：执行效率
核查项 E1-E6：无冗余步骤、无无效等待、并行度合理、路径无冗余绕行、
火力经济性、阶段衔接流畅。

### 维度三：可理解性
核查项 C1-C7：动作动词精确、目标对象明确、执行条件清晰、角色职责无歧义、
动作粒度一致、成功标准可判断、异常处理提示。

### 维度四：粒度合规性
核查项 G1-G8：必填字段完整、字段类型正确、Action_Sequence结构合规、
角色化描述合规、通用性全规则合规、ID格式规范等。

### 维度五：图文一致性
核查项 V1-V7：空间要素对应、角色位置对应、动作时序对应、角色数量一致、
方向关系正确、整体图与步骤图一致、视觉描述充分性。

### 维度六：军事可行性
核查项 M1-M18：角色互助关系、无孤立行动、角色数量自洽、火力扇区安全、
机动路径安全、死角清理、节奏与威胁匹配、敌方响应考虑、
不依赖敌方错误、信息不确定性处理、战术原则遵循(突然性/简单性/集中/安全/统一指挥)。

## 综合评分
Q = (q1 + q2 + q3 + q4 + q5 + q6) / 6

### 否决条件
- V1: 军事可行性 < 3.0
- V2: 场景适配度 < 3.0
- V3: 粒度合规性 < 3.0
任一触发 → L级（不入库）

### 质量分级
- H: Q >= 8.0 且 军事可行性 >= 7.0 且无单一维度 < 4.0
- M: 6.0 <= Q < 8.0 且无否决条件 且无单一维度 < 4.0
- L: Q < 6.0 或 否决条件触发 或 任意单一维度 < 4.0

## 输出格式
```json
{
  "scores": {
    "scene_adaptation": {"score": <float>, "checks": {"S1": <bool>, ..., "S8": <bool>}, "deductions": []},
    "execution_efficiency": {"score": <float>, "checks": {"E1": <bool>, ..., "E6": <bool>}, "deductions": []},
    "comprehension": {"score": <float>, "checks": {"C1": <bool>, ..., "C7": <bool>}, "deductions": []},
    "granularity_compliance": {"score": <float>, "checks": {"G1": <bool>, ..., "G8": <bool>}, "deductions": []},
    "text_visual_consistency": {"score": <float>, "checks": {"V1": <bool>, ..., "V7": <bool>}, "deductions": []},
    "military_feasibility": {"score": <float>, "checks": {"M1": <bool>, ..., "M18": <bool>}, "deductions": []}
  },
  "overall_score": <float>,
  "quality_level": "H|M|L",
  "veto_triggered": <bool>,
  "veto_reason": "<string>",
  "evaluation_summary": "<50-150字>"
}
```

## 行为准则
- 逐项核查，不跳过任何Checklist项
- 每个扣分必须有具体位置和理由
- 严格遵循评分锚点
- 否决条件必须显式检查并记录
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

    eval_result = EvalResult.from_dict(result)

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
