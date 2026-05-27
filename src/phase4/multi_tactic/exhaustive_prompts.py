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
      "Intent": "该步骤的战术意图，角色化描述",
      "key_roles": ["突击手", "掩护手"]
    }
  ],
  "Semantic_Tags": ["标签1", "标签2"]
}
```
"""

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
3. **核心动作链**: Action_Sequence 中关键动作的重合度 ≥ 80%
4. **人员配置与分工**: 兵力构成和任务分配一致
5. **适用环境**: 环境类型完全相同
6. **关键执行条件**: 战术的核心前提依赖一致

## 战术多样性维度

请从以下角度思考还有哪些战术未覆盖:
- **不同突入策略**: 正面突入、迂回包抄、多点同步突入、分进合击
- **不同角色协同模式**: 前后交替掩护、交叉火力覆盖、扇区分工警戒
- **不同节奏**: 快速突袭、稳扎稳打、先侦察后行动
- **不同火力配置**: 近距突击、中距火力掩护、远距压制
- **不同空间利用**: 沿墙壁推进、利用掩体跳跃、高低位立体推进
- **不同处置方式**: 歼灭、驱离、绕行、固守、抓捕

## 输出格式

只输出一个纯 JSON 数组，每个元素为一个战术概念对象（只含核心字段）。
当无新战术时返回空数组 `[]`。

不要包含其他说明文字或代码块标记。

## JSON 格式约束
- 字符串值内的双引号必须转义为 `\\\"`，引用原文时使用「」角括号
- 字符串值内禁止直接换行，换行用 `\\n` 表示

## 战术概念 Schema
{TACTIC_CONCEPT_SCHEMA}
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

1. **场景还有哪些战术角度未被利用？**
   - 是否有不同的突入方向/入口点可用？
   - 是否有不同的编队配置可尝试？
   - 是否有不同的节奏策略（快速/稳健）未覆盖？

2. **参考资料中还有哪些战术知识尚未被应用？**

3. **生成独特的新战术**，确保与已有战术不重复。

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

    user_prompt = EXHAUSTIVE_USER_PROMPT_TEMPLATE.format(
        scene=scene_json,
        reference_content=reference_content,
        existing_tactics=json.dumps(existing_tactics, ensure_ascii=False, indent=2),
        existing_count=len(existing_tactics),
        mission_phase_text=mission_phase_text,
    )

    return EXHAUSTIVE_SYSTEM_PROMPT, user_prompt
