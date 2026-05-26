"""
Direction Generalizer — desc.json 方向信息预处理模块

在 desc.json 送入 LLM 之前，将绝对方向指代（东/南/西/北）替换为
功能/关系描述（近端/远端/一侧/对侧），从源头切断方向污染。

核心思路：
1. 从 desc.json 中提取任务轴线（entry → objective）
2. 沿轴线方向 → "近端/远端"（以 entry 为"近"，objective 为"远"）
3. 垂直于轴线 → "一侧/对侧"（以编队推进面向为参照）
4. 垂直方向 → "上层/下层"

注意：这是一个语义增强模块，不追求完美 NLP，而是提供方向泛化的
强信号给 LLM。即使某些映射不完全精确，也比原始绝对方向更通用。
"""

from __future__ import annotations

import re
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

# ============================================================
# 方向替换规则表
# ============================================================

# X轴方向词（沿走廊/通道方向，通常映射为"远端/近端"）
DIRECTION_X_WEST: List[str] = ["西", "west", "West", "WEST"]
DIRECTION_X_EAST: List[str] = ["东", "east", "East", "EAST"]

# Y轴方向词（垂直于走廊，通常映射为"一侧/对侧"）
DIRECTION_Y_NORTH: List[str] = ["北", "north", "North", "NORTH"]
DIRECTION_Y_SOUTH: List[str] = ["南", "south", "South", "SOUTH"]

# 复合方向（通常包含多个轴向成分）
DIRECTION_COMPOUND: Dict[str, str] = {
    "东北": "远端一侧",
    "西北": "远端对侧",
    "东南": "近端一侧",
    "西南": "近端对侧",
    "northeast": "far side left",
    "northwest": "far side right",
    "southeast": "near side left",
    "southwest": "near side right",
}

# 方向后缀模式
DIRECTION_SUFFIX_PATTERNS: Dict[str, str] = {
    # 中文
    "东端": "近端",
    "西端": "远端",
    "东段": "入口方向段（近端）",
    "西段": "目标方向段（远端）",
    "北侧": "一侧",  # 默认一侧，有更多上下文时可能为对侧
    "南侧": "对侧",  # 默认为对侧
    "北端": "近侧端",
    "南端": "远侧端",
    "东面": "入口方向面",
    "西面": "目标方向面",
    "北面": "侧向面",
    "南面": "对向面",
    "朝东": "朝向入口方向",
    "朝西": "朝向目标方向",
    "朝南": "朝向对侧方向",
    "朝北": "朝向侧向",
    "东翼": "近端翼侧",
    "西翼": "远端翼侧",
    "北翼": "侧翼（走廊一侧）",
    "南翼": "侧翼（走廊对侧）",
    # 英文
    " east": " (entry side)",
    " west": " (objective side)",
    " north": " (left flank)",
    " south": " (right flank)",
    "-east": "-entry_side",
    "-west": "-objective_side",
    "-north": "-left_flank",
    "-south": "-right_flank",
}

# 具体的常见描述模式（优先级高于上述通用规则）
SPECIFIC_PATTERNS: Dict[str, str] = {
    # 走廊场景
    "走廊西端尽头": "走廊远端尽头",
    "走廊西端楼梯井": "走廊远端楼梯井入口",
    "走廊西侧墙壁": "走廊目标侧墙壁",
    "走廊东段": "走廊近端（入口方向段）",
    "走廊东侧": "走廊入口侧",
    "走廊西侧": "走廊目标侧",
    "走廊北侧": "走廊一侧",
    "走廊南侧": "走廊对侧",
    "北侧开口": "走廊中部侧向开口",
    "南侧入口": "走廊对侧入口",
    "东侧入口": "走廊近端入口",
    "西侧入口": "走廊远端入口",
    # 建筑场景
    "建筑东侧": "建筑入口侧",
    "建筑西侧": "建筑远端侧",
    "建筑北侧": "建筑一侧",
    "建筑南侧": "建筑对侧",
    "朝南开向": "朝向对侧开向",
    "朝北开向": "朝向一侧开向",
    "朝东开向": "朝向入口方向开向",
    "朝西开向": "朝向目标方向开向",
    # 垂直方向
    "F1层": "上层",
    "F2层": "上层",
    "F1/F2层": "垂直贯通上层",
    # 室外场景
    "东侧树林边缘": "入口侧植被遮蔽线",
    "西侧树林": "目标侧植被区域",
    # 坐标替换
    "X=-26.5墙面线位置": "走廊中段分隔墙位置",
    "X=0墙面线位置": "走廊近端边界墙位置",
    "X=-60区域": "走廊中段区域",
}

# 可重用的 compiled regex（方向检测用）
_DIRECTION_CHECK_RE = re.compile(
    r"[东南西北](?:侧|端|段|面|翼|向|部|方|区|角|头)"
    r"|朝[东南西北]"
    r"|[东南西北]侧[东南西北]?"
    r"|(?:east|west|north|south)(?:\s|-|$)"
    r"|F[0-9]+层",
    re.IGNORECASE,
)


# ============================================================
# 核心函数
# ============================================================

def generalize_directions(desc: Dict[str, Any]) -> Dict[str, Any]:
    """将 desc.json 中的所有绝对方向替换为功能/关系描述。

    Args:
        desc: 原始 desc.json dict

    Returns:
        方向泛化后的 desc.json dict（深拷贝，不修改原始数据）
    """
    result = deepcopy(desc)

    # 递归遍历所有字符串字段，执行替换
    result = _generalize_dict(result)

    return result


def _generalize_dict(obj: Any) -> Any:
    """递归遍历 dict/list/str，替换所有字符串中的方向词。"""
    if isinstance(obj, dict):
        return {k: _generalize_dict(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_generalize_dict(item) for item in obj]
    elif isinstance(obj, str):
        return _generalize_string(obj)
    else:
        return obj


def _generalize_string(text: str) -> str:
    """对单个字符串执行方向泛化替换。"""
    result = text

    # 第一步：精确匹配特定模式（优先级最高）
    for pattern, replacement in SPECIFIC_PATTERNS.items():
        result = result.replace(pattern, replacement)

    # 第二步：后缀模式替换
    for pattern, replacement in DIRECTION_SUFFIX_PATTERNS.items():
        result = result.replace(pattern, replacement)

    # 第三步：复合方向替换
    for pattern, replacement in DIRECTION_COMPOUND.items():
        result = result.replace(pattern, replacement)

    # 第四步：清理残留的绝对方向标志
    # 将 zone_id 中的方向标志替换为功能标志
    result = result.replace("-EAST", "-NEAR")
    result = result.replace("-WEST", "-FAR")
    result = result.replace("-NORTH", "-FLANK")
    result = result.replace("-SOUTH", "-OPPOSITE")

    return result


# ============================================================
# 方向检测辅助函数
# ============================================================

def has_absolute_directions(text: str) -> bool:
    """检测文本中是否包含绝对方向指代。

    用于后处理验证——确认泛化后无残留方向词。
    """
    if not text:
        return False
    return bool(_DIRECTION_CHECK_RE.search(text))


def find_absolute_directions(text: str) -> List[str]:
    """返回文本中所有匹配的绝对方向指代。"""
    if not text:
        return []
    return _DIRECTION_CHECK_RE.findall(text)


def verify_generalization(original: Dict[str, Any], generalized: Dict[str, Any]) -> Dict[str, Any]:
    """验证泛化效果，返回报告。

    Returns:
        {
            "total_replacements": int,
            "residual_issues": List[str],  # 仍有绝对方向的文本
            "generalized_preview": str,    # 关键字段的泛化前后对比
        }
    """
    report = {
        "total_replacements": 0,
        "residual_issues": [],
        "generalized_preview": "",
    }

    # 统计替换数量（通过比较字符串差异）
    orig_text = _extract_all_text(original)
    gen_text = _extract_all_text(generalized)

    # 简单统计：计算方向词出现次数的变化
    orig_directions = len(find_absolute_directions(orig_text))
    gen_directions = len(find_absolute_directions(gen_text))
    report["total_replacements"] = orig_directions - gen_directions

    # 检测残留
    if gen_directions > 0:
        residual = find_absolute_directions(gen_text)
        report["residual_issues"] = list(set(residual))

    # 生成预览
    if "spatial_description" in original and "spatial_description" in generalized:
        orig_spatial = original["spatial_description"]
        gen_spatial = generalized["spatial_description"]
        report["generalized_preview"] = (
            f"原始: {orig_spatial}\n泛化: {gen_spatial}"
        )

    return report


def _extract_all_text(obj: Any) -> str:
    """递归提取所有字符串，拼接为单个文本用于检测。"""
    if isinstance(obj, dict):
        return " ".join(_extract_all_text(v) for v in obj.values())
    elif isinstance(obj, list):
        return " ".join(_extract_all_text(item) for item in obj)
    elif isinstance(obj, str):
        return obj
    else:
        return ""
