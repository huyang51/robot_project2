"""
Desc Sanitizer — desc.json 系统枚举值消毒模块

将 desc.json 中作为独立字段值出现的英文 zone type 枚举值转为中文。
这是纯翻译，不涉及任何方向假设或场景特定信息替换。
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable, Dict, List

# ============================================================
# zone type 枚举 → 中文（纯翻译，无假设）
# ============================================================

_SYSTEM_ENUM_ZH: Dict[str, str] = {
    "corridor": "走廊",
    "entry": "入口",
    "room": "房间",
    "stairwell": "楼梯井",
    "open_area": "开阔地",
    "exterior": "外部",
}


# ============================================================
# 递归遍历 + 消毒逻辑
# ============================================================

def _walk(obj: Any, transform: Callable[[str], str]) -> Any:
    """递归遍历 dict/list/str，对每个字符串值应用 transform。"""
    if isinstance(obj, dict):
        return {k: _walk(v, transform) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_walk(item, transform) for item in obj]
    elif isinstance(obj, str):
        return transform(obj)
    else:
        return obj


def _sanitize_string(text: str) -> str:
    """若整个字符串是已知英文枚举值，返回中文；否则原样返回。"""
    key = text.lower()
    return _SYSTEM_ENUM_ZH.get(key, text)


def sanitize_desc(desc: Dict[str, Any]) -> Dict[str, Any]:
    """消毒 desc.json：将独立字段值中的英文枚举转为中文。

    Args:
        desc: desc.json dict

    Returns:
        消毒后的 dict（深拷贝，不修改原始数据）
    """
    return _walk(deepcopy(desc), _sanitize_string)
