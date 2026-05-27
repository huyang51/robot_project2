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
    "北墙": "走廊一侧墙壁",
    "南墙": "走廊对侧墙壁",
    "东墙": "走廊近端墙壁",
    "西墙": "走廊远端墙壁",
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
    # 北墙相关
    "北墙尽头": "走廊一侧墙壁尽头",
    "北墙墙角": "走廊一侧墙壁转角",
    "北墙沿线": "走廊一侧墙壁沿线",
    # 坐标替换
    "X=-26.5墙面线位置": "走廊中段分隔墙位置",
    "X=0墙面线位置": "走廊近端边界墙位置",
    "X=-60区域": "走廊中段区域",
}

# 可重用的 compiled regex（方向检测用）
_DIRECTION_CHECK_RE = re.compile(
    r"[东南西北](?:侧|端|段|面|翼|向|部|方|区|角|头|墙|缘)"
    r"|朝[东南西北]"
    r"|[东南西北]侧[东南西北]?"
    r"|(?:east|west|north|south)(?:\s|-|$)"
    r"|F[0-9]+层",
    re.IGNORECASE,
)

# ============================================================
# 物体身份→功能类别映射表（安全网）
# ============================================================

# 此模块定位为安全网（与方向泛化相同定位），提供强信号但不求完美 NLP。
# 核心防线是 A_gen 的双重不变性哲学教学。以下映射表覆盖常见场景特定物体→战术功能类别。

OBJECT_IDENTITY_MAP: Dict[str, str] = {
    # ── 低矮掩体类 ──
    "低矮长桌": "低矮掩体",
    "低矮木质长桌": "低矮掩体",
    "矮桌": "低矮掩体",
    "茶几": "低矮掩体",
    "矮柜": "低矮掩体",
    "低矮矮柜": "低矮掩体",
    "樱桃木矮柜": "低矮掩体",
    "橡木矮柜": "低矮掩体",
    "矮茶几": "低矮掩体",
    "花坛": "低矮掩体",
    "低矮花坛": "低矮掩体",
    "低矮方形花坛": "低矮掩体",
    "低矮石台": "低矮掩体",
    "石台": "低矮掩体",
    # ── 低矮大型遮挡物类 ──
    "大型沙发": "低矮大型遮挡物",
    "L形沙发": "低矮大型遮挡物",
    "大型L形沙发": "低矮大型遮挡物",
    "组合沙发": "低矮大型遮挡物",
    "长沙发": "低矮大型遮挡物",
    "大沙发": "低矮大型遮挡物",
    # ── 中型掩体类 ──
    "办公桌": "中型掩体",
    "方桌": "中型掩体",
    "长桌": "中型掩体",
    "金属办公桌": "中型掩体",
    "木桌": "中型掩体",
    "书桌": "中型掩体",
    "餐桌": "中型掩体",
    # ── 柱状掩体类 ──
    "混凝土立柱": "柱状掩体",
    "混凝土圆柱立柱": "柱状掩体",
    "铁质支柱": "柱状掩体",
    "圆形立柱": "柱状掩体",
    "水泥柱": "柱状掩体",
    "方柱": "柱状掩体",
    "承重柱": "柱状掩体",
    "立柱": "柱状掩体",
    # ── 透明隔断类 ──
    "玻璃幕墙": "透明隔断",
    "玻璃隔断": "透明隔断",
    "落地玻璃窗": "透明隔断",
    "玻璃墙": "透明隔断",
    # ── 低矮栅栏状障碍类 ──
    "不锈钢栏杆": "低矮栅栏状障碍",
    "木质扶手": "低矮栅栏状障碍",
    "铁栏杆": "低矮栅栏状障碍",
    "金属栏杆": "低矮栅栏状障碍",
    "护栏": "低矮栅栏状障碍",
    # ── 门框类 ──
    "木门框": "门框",
    "金属门框": "门框",
    "铁门框": "门框",
    # ── 车辆掩体类 ──
    "白色厢式货车": "车辆掩体",
    "红色轿车": "车辆掩体",
    "废弃车辆": "车辆掩体",
    "货车": "车辆掩体",
    "轿车": "车辆掩体",
    # ── 植被遮蔽类 ──
    "灌木丛": "植被遮蔽",
    "树丛": "植被遮蔽",
    "花丛": "植被遮蔽",
    # ── 可提供隐蔽的障碍物类 ──
    "空调外机": "可提供隐蔽的障碍物",
    "配电箱": "可提供隐蔽的障碍物",
    "通风管道": "可提供隐蔽的障碍物",
    # ── 无战术价值物体类 ──
    "踢脚板": "无战术价值物体",
    "装饰条": "无战术价值物体",
    "窗帘": "无战术价值物体",
}

# ============================================================
# 编队规模→编队/角色描述映射表（安全网）
# ============================================================

# 将编队规模限定词替换为编队/角色通用描述。
# 核心原则：战术描述角色交互模式，而非兵力组成。
# 此映射表与 TEAM_SIZE_VOCABULARY（agen_prompts.py 中的 LLM 教学）对齐。

TEAM_SIZE_MAP: Dict[str, str] = {
    "双人编队": "编队",
    "两人编队": "编队",
    "三人编队": "编队",
    "四人编队": "编队",
    "两名突击手": "突击手",
    "两名掩护手": "掩护手",
    "三名突击手": "突击手",
    "两名警戒手": "警戒手",
    "三人战斗小组": "战斗编组",
    "双组协同": "多组协同",
    "两人交替掩护": "编队交替掩护",
    "1名掩护手": "掩护手",
    "1名突击手": "突击手",
    "2名突击手": "突击手",
    "2名掩护手": "掩护手",
    "3单元": "编队各单元",
    "双组": "多组",
    "双人": "编队",
    "两人": "编队",
}


# ============================================================
# 物体泛化辅助函数
# ============================================================

def generalize_objects(desc: Dict[str, Any]) -> Dict[str, Any]:
    """将 desc.json 中所有场景特定物体身份替换为战术功能类别。

    作为安全网运行（与 generalize_directions 相同定位），
    提供强信号但不求完美 NLP。核心防线是 A_gen 的双重不变性哲学教学。

    Args:
        desc: 已通过 generalize_directions 处理后的 desc.json dict

    Returns:
        物体身份泛化后的 desc.json dict（深拷贝，不修改原始数据）
    """
    result = deepcopy(desc)
    result = _generalize_objects_in_dict(result)
    return result


def _generalize_objects_in_dict(obj: Any) -> Any:
    """递归遍历 dict/list/str，替换所有字符串中的物体身份词。"""
    if isinstance(obj, dict):
        return {k: _generalize_objects_in_dict(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_generalize_objects_in_dict(item) for item in obj]
    elif isinstance(obj, str):
        return _generalize_objects_in_string(obj)
    else:
        return obj


def _generalize_objects_in_string(text: str) -> str:
    """对单个字符串执行物体身份→功能类别替换。"""
    result = text
    # 按映射表键的长度降序排列，优先匹配长串（"低矮木质长桌"先于"长桌"）
    sorted_keys = sorted(OBJECT_IDENTITY_MAP.keys(), key=len, reverse=True)
    for identity in sorted_keys:
        replacement = OBJECT_IDENTITY_MAP[identity]
        result = result.replace(identity, replacement)
    return result


# ============================================================
# 编队规模泛化辅助函数
# ============================================================

def generalize_team_size(desc: Dict[str, Any]) -> Dict[str, Any]:
    """将 desc.json 中所有编队规模限定词替换为编队/角色通用描述。

    作为安全网运行（与 generalize_directions/generalize_objects 相同定位），
    提供强信号但不求完美 NLP。核心防线是 A_gen 的编队规模不变性哲学教学。

    Args:
        desc: 已通过 generalize_objects 处理后的 desc.json dict

    Returns:
        编队规模泛化后的 desc.json dict（深拷贝，不修改原始数据）
    """
    result = deepcopy(desc)
    result = _generalize_team_size_in_dict(result)
    return result


def _generalize_team_size_in_dict(obj: Any) -> Any:
    """递归遍历 dict/list/str，替换所有字符串中的编队规模限定词。"""
    if isinstance(obj, dict):
        return {k: _generalize_team_size_in_dict(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_generalize_team_size_in_dict(item) for item in obj]
    elif isinstance(obj, str):
        return _generalize_team_size_in_string(obj)
    else:
        return obj


def _generalize_team_size_in_string(text: str) -> str:
    """对单个字符串执行编队规模限定词→编队/角色描述替换。"""
    result = text
    # 按映射表键的长度降序排列，优先匹配长串（"双人编队"先于"双人"）
    sorted_keys = sorted(TEAM_SIZE_MAP.keys(), key=len, reverse=True)
    for identity in sorted_keys:
        replacement = TEAM_SIZE_MAP[identity]
        result = result.replace(identity, replacement)
    return result


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
