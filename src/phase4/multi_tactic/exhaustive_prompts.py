"""
穷举生成 Prompt 模板

从 robot_project TACTIC_ITERATION_PROMPTS 适配到 robot_project2
的双版本（text_version + struct_version）JSON Schema。
"""

import json

# ── 战术 JSON Schema（简化版，只要求核心字段的 concept）──

TACTIC_CONCEPT_SCHEMA = """```json
{
  "Tactic_Name": "string - 战术名称，简洁有区分度",
  "Tactic_Type": "string - 如 室内作战/班组级、走廊推进/班组级",
  "objective": "string - 战术核心目的，一句话概括",
  "Description": "string - 80-200字战术执行方式完整描述",
  "Action_Sequence": [
    {
      "Step": 1,
      "Intent": "该步骤的战术意图，编组化描述",
      "key_roles": ["突击组", "掩护组"]
    }
  ],
  "Semantic_Tags": ["标签1", "标签2"]
}
```
"""

# ── 阶段多样性维度映射 ──

PHASE_DIVERSITY_DIMENSIONS = {
    "侦察阶段": (
        "- **不同侦察方式**: 隐蔽观察、试探性前出、高空俯瞰、多角度交叉侦察\n"
        "- **不同观察-报告流程**: 逐扇区扫描→编队汇总、持续监视→即时通报、先侦察后行动\n"
        "- **不同隐蔽转移模式**: 利用掩体跃进、低姿贴墙移动、分段推进、交替掩护侦察\n"
        "- **不同信息交接方式**: 侦察组→突击组态势移交、多编组信息交叉验证、逐步确认推进"
    ),
    "进攻阶段": (
        "- **不同突入策略**: 正面突入、迂回包抄、多点同步突入、分进合击\n"
        "- **不同编组协同模式**: 前后交替掩护、交叉火力覆盖、扇区分工警戒\n"
        "- **不同节奏**: 快速突袭、稳扎稳打、先侦察后行动\n"
        "- **不同火力配置**: 近距突击、中距火力掩护、远距压制\n"
        "- **不同空间利用**: 沿墙壁推进、利用掩体跳跃、高低位立体推进\n"
        "- **不同处置方式**: 歼灭、驱离、绕行、固守、抓捕"
    ),
    "防御阶段": (
        "- **不同阵地配置**: 纵深梯次配置、环形防御、侧翼布防、要点固守\n"
        "- **不同火力扇区划分**: 交叉覆盖、重点扇区重叠、正面阻截+侧翼牵制\n"
        "- **不同梯队配置**: 前沿警戒+纵深预备、一线展开+机动反击组\n"
        "- **不同反击时机**: 主动诱敌后反击、待敌方展开后反击、敌撤退时追击\n"
        "- **不同阵地轮换**: 前沿与预备队交替接敌、各编组轮换休整"
    ),
    "撤退与脱离阶段": (
        "- **不同断后编组模式**: 单一断后组逐段掩护、双断后组交替掩护、诱敌组+断后组协同\n"
        "- **不同交替脱离节奏**: 逐段撤离（每到一个掩体线换手）、一次性脱离（佯攻后同时撤出）、诱敌后脱离（以攻为退创造窗口）\n"
        "- **不同迟滞方式**: 烟幕遮蔽、压制射击迟滞、设置障碍/诱饵、佯攻迷惑\n"
        "- **不同撤退路线选择**: 沿来时路径、绕行备用路线、分散多路撤退、利用遮蔽地形隐蔽撤退\n"
        "- **不同集结与清点方式**: 逐段集结清点、指定集结区域同步汇合、分散撤退后逐批归建"
    ),
}

PHASE_TASK_QUESTIONS = {
    "侦察阶段": (
        "1. **场景还有哪些侦察角度未被利用？**\n"
        "   - 是否有不同的观察位置（高位/低位/侧翼）可覆盖更多扇区？\n"
        "   - 是否有不同的侦察推进模式（逐段侦察/全向扫描/先远后近）未覆盖？\n"
        "   - 是否有不同的信息传递与确认流程可组合？\n"
        "   - 是否有不同的隐蔽侦察路线未考虑？\n"
        "\n"
        "2. **参考资料中还有哪些侦察战术知识尚未被应用？**\n"
        "\n"
        "3. **生成独特的侦察战术**，确保与已有战术不重复。"
    ),
    "进攻阶段": (
        "1. **场景还有哪些战术角度未被利用？**\n"
        "   - 是否有不同的突入方向/入口点可用？\n"
        "   - 是否有不同的编队配置可尝试？\n"
        "   - 是否有不同的节奏策略（快速/稳健）未覆盖？\n"
        "\n"
        "2. **参考资料中还有哪些战术知识尚未被应用？**\n"
        "\n"
        "3. **生成独特的新战术**，确保与已有战术不重复。"
    ),
    "防御阶段": (
        "1. **场景还有哪些防御角度未被利用？**\n"
        "   - 是否有不同的阵地配置（纵深/环形/要点固守）可尝试？\n"
        "   - 是否有不同的火力扇区划分方案未覆盖？\n"
        "   - 是否有不同的反击时机与触发条件可组合？\n"
        "   - 是否有不同的阵地轮换节奏未考虑？\n"
        "\n"
        "2. **参考资料中还有哪些防御战术知识尚未被应用？**\n"
        "\n"
        "3. **生成独特的防御战术**，确保与已有战术不重复。"
    ),
    "撤退与脱离阶段": (
        "1. **场景还有哪些撤退角度未被利用？**\n"
        "   - 是否有不同的断后掩护模式可尝试？\n"
        "   - 是否有不同的脱离节奏（逐段撤离/一次性脱离/诱敌后脱离）未覆盖？\n"
        "   - 是否有不同的迟滞方式（烟幕/压制射击/诱饵）可组合？\n"
        "   - 是否有不同的撤退路线/集结方式未考虑？\n"
        "\n"
        "2. **参考资料中还有哪些撤退战术知识尚未被应用？**\n"
        "\n"
        "3. **生成独特的撤退战术**，确保与已有战术不重复。"
    ),
}

# Fallback: generic dimensions and questions when no phase or unknown phase
_FALLBACK_DIMENSIONS = (
    "- **不同编组协同模式**: 前后交替掩护、交叉火力覆盖、扇区分工警戒\n"
    "- **不同节奏策略**: 快速行动、稳扎稳打、先侦察后行动\n"
    "- **不同空间利用**: 沿墙壁推进、利用掩体跃进、高低位协同\n"
    "- **不同阶段对应**: 侦察/进攻/防御/撤退的差异化处置"
)

_FALLBACK_TASK_QUESTIONS = (
    "1. **场景还有哪些战术角度未被利用？**\n"
    "   - 是否有不同的编组配置可尝试？\n"
    "   - 是否有不同的节奏策略未覆盖？\n"
    "   - 是否有不同的空间利用方式可组合？\n"
    "\n"
    "2. **参考资料中还有哪些战术知识尚未被应用？**\n"
    "\n"
    "3. **生成独特的新战术**，确保与已有战术不重复。"
)


# ── 穷举生成 System Prompt ──

EXHAUSTIVE_SYSTEM_PROMPT = f"""
# Role：机器人协同战术创新生成专家

## Background
在机器人协同作战领域，战术设计的完整性和多样性直接影响任务成功率。
然而，单一战术无法覆盖场景的所有可能性——不同的突入角度、不同的编队配置、
不同的节奏策略都可能适用于同一场景。你的任务是基于已有战术集合和外部参考资料，
系统性地发现缺失的战术方向，并生成结构化的独特新战术。

## Attention
充分发挥穷尽意识，追求每一个新战术的独一无二与实战价值。
不要让参考资料中的战术灵感流失，始终保持对已生成战术的严格去重。

## 核心原则
1. **信息损失最小化**: 充分利用参考资料中的所有战术知识
2. **战术独特性**: 只生成与已有战术不重复的新战术
3. **穷尽意识**: 思考"还能生成什么"而非简单重复

## 战术重复判断标准

以下维度中任意3项或以上雷同即视为重复:
1. **Tactic_Name**: 完全相同或高度相似
2. **战术目标（objective）**: 核心目的相同
3. **核心动作链**: Action_Sequence 中关键动作的重合度 >= 80%
4. **编组配置与分工**: 编组构成和任务分配一致
5. **适用环境**: 环境类型完全相同
6. **关键执行条件**: 战术的核心前提依赖一致

## 战术多样性维度

请从以下角度思考还有哪些战术未覆盖:
{{diversity_dimensions}}

## 输出格式

只输出一个纯 JSON 数组，每个元素为一个战术概念对象（只含核心字段）。
当无新战术时返回空数组 `[]`。

不要包含其他说明文字或代码块标记。

## JSON 格式约束
- 字符串值内的双引号必须转义为 `\\\"`，引用原文时使用「」角括号
- 字符串值内禁止直接换行，换行用 `\\n` 表示

## 战术概念 Schema
{TACTIC_CONCEPT_SCHEMA}

## 输出约束
- 直接输出 JSON 数组，不要输出推理过程
- 每个战术概念只输出一次，不要在回复中重复
- 如果无法找到新的战术角度，输出 `[]` 并在下一个 token 立即停止
"""

# ── 穷举生成 User Prompt 模板 ──

EXHAUSTIVE_USER_PROMPT_TEMPLATE = """## 当前场景

{scene}

## 作战阶段约束

{mission_phase_text}

## 参考资料（本批次）

{reference_content}

## 已生成的战术（共 {existing_count} 个）

{existing_tactics}

## 任务

请分析场景特征、参考资料和已生成战术，识别仍可生成的新战术方向：

{task_questions}

请只输出新战术的 JSON 数组，格式为 [{{Tactic_Name: ..., ...}}, ...]。
如果已穷尽所有可能，则输出空数组：[]"""


def build_exhaustive_prompts(
    scene_json: str,
    reference_content: str,
    existing_tactics: list,
    mission_phase: str = "",
) -> tuple:
    """构建穷举生成的 system/user prompt

    Args:
        scene_json: 场景 desc.json 序列化字符串
        reference_content: 当前批次参考资料
        existing_tactics: 已生成的战术列表
        mission_phase: 作战阶段约束文本

    Returns:
        (system_prompt, user_prompt)
    """
    mission_phase_text = (
        f"当前作战阶段为 **{mission_phase}**，只生成此阶段的战术。"
        if mission_phase else "无特定阶段约束，可覆盖所有阶段。"
    )

    # 根据 mission_phase 选择多样性维度和任务问题
    diversity_dimensions = PHASE_DIVERSITY_DIMENSIONS.get(
        mission_phase, _FALLBACK_DIMENSIONS
    )
    task_questions = PHASE_TASK_QUESTIONS.get(
        mission_phase, _FALLBACK_TASK_QUESTIONS
    )

    system_prompt = EXHAUSTIVE_SYSTEM_PROMPT.format(
        diversity_dimensions=diversity_dimensions,
    )

    user_prompt = EXHAUSTIVE_USER_PROMPT_TEMPLATE.format(
        scene=scene_json,
        reference_content=reference_content,
        existing_tactics=json.dumps(existing_tactics, ensure_ascii=False, indent=2),
        existing_count=len(existing_tactics),
        mission_phase_text=mission_phase_text,
        task_questions=task_questions,
    )

    return system_prompt, user_prompt
