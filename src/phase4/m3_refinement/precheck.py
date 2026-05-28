"""
M3 审查上下文提取

从战术 JSON 中提取文本字段，格式化为 A_review 的审查输入。
不做任何正则检查——全部 14 条规则的语义审查由 A_review LLM 完成。

正则预检被废弃的原因：
- G-T1~G-T6 的数值检测本质是语义判断（区分"步骤3"与"3米"），正则无法可靠区分
- G-T2 的中文数字排除依赖穷举白名单，无法覆盖所有非数字用字组合
- G-T3 的量词表无法穷举所有军事量词（挺/支/具/枚/门/多…）
- G-T4/G-T5 的单位覆盖永远不完整（cm/mm/km/min/h…）
- G-T7 的场景特定物体关键词表是无底洞
- G-T8 只匹配显式编号，漏掉字母编号和隐含指代
- G-T9 和 G-S1~G-S5 原本就由 A_review 负责（regex check 代码体是 pass）
"""

from typing import Dict, List, Tuple


def extract_review_context(tactic_json: Dict) -> Dict:
    """从战术 JSON 中提取审查所需的文本上下文

    不做任何违规判定——只提取文本字段并按版本分组，
    作为 A_review LLM 审查的参考材料。

    Args:
        tactic_json: {"text_version": {...}, "struct_version": {...}}

    Returns:
        {
            "text_version_fields": [{"location": str, "text": str}, ...],
            "struct_version_fields": [{"location": str, "text": str}, ...],
        }
    """
    text_version = tactic_json.get("text_version", {})
    struct_version = tactic_json.get("struct_version", {})

    text_fields = _extract_text_fields(text_version)
    struct_fields = _extract_instruction_fields(struct_version)

    return {
        "text_version_fields": [
            {"location": loc, "text": txt} for loc, txt in text_fields
        ],
        "struct_version_fields": [
            {"location": loc, "text": txt} for loc, txt in struct_fields
        ],
    }


def _extract_text_fields(version: Dict) -> List[Tuple[str, str]]:
    """提取 text_version 中的文本字段"""
    fields = []
    if "Description" in version:
        fields.append(("Description", version["Description"]))
    if "objective" in version:
        fields.append(("objective", version["objective"]))
    if "Tactic_Name" in version:
        fields.append(("Tactic_Name", version["Tactic_Name"]))
    for action in version.get("Action_Sequence", []):
        step = action.get("Step", "?")
        if "Intent" in action:
            fields.append((f"Action_Sequence[{step}].Intent", action["Intent"]))
        if "Visual_Aids" in action:
            va = action["Visual_Aids"]
            for i, item in enumerate(va if isinstance(va, list) else [va]):
                fields.append((f"Action_Sequence[{step}].Visual_Aids[{i}]", str(item)))
    # 提取顶层 Visual_Aid_Overall
    if "Visual_Aid_Overall" in version:
        va = version["Visual_Aid_Overall"]
        for i, item in enumerate(va if isinstance(va, list) else [va]):
            fields.append((f"Visual_Aid_Overall[{i}]", str(item)))
    # 提取 Semantic_Tags
    if "Semantic_Tags" in version:
        tags = version["Semantic_Tags"]
        if isinstance(tags, list):
            fields.append(("Semantic_Tags", ", ".join(str(t) for t in tags)))
    return fields


def _extract_instruction_fields(version: Dict) -> List[Tuple[str, str]]:
    """提取 struct_version 中的 Instructions 文本及顶层描述字段"""
    fields = []
    if "Description" in version:
        fields.append(("struct_version.Description", version["Description"]))
    if "objective" in version:
        fields.append(("struct_version.objective", version["objective"]))
    for action in version.get("Action_Sequence", []):
        step = action.get("Step", "?")
        if "Intent" in action:
            fields.append((f"Action_Sequence[{step}].Intent", action["Intent"]))
        for i, instr in enumerate(action.get("Instructions", [])):
            fields.append((f"Action_Sequence[{step}].Instructions[{i}]", instr))
    return fields
