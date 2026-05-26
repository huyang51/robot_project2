"""
Phase 2 子场景去重 — LLM 驱动

判断战术等价性，删除冗余子场景，只保留代表性场景。
一次 LLM 调用处理全部子场景。
"""

import logging
from typing import Dict, List

from ..core.llm_client import MiniMaxClient

logger = logging.getLogger(__name__)

# ── System Prompt ──────────────────────────────────────────────

DEDUP_SYSTEM_PROMPT = """# Role: 战术场景去重专家

## Profile
- language: 中文
- description: 专为战术行动场景去重设计。分析多个子场景的战术等价性，识别可以用同一套战术方案应对的冗余场景。
- background: 精通 CQB 战术、室内近战、建筑肃清行动。深刻理解空间物理特征如何影响战术选择。
- expertise: 战术等价性判断、空间特征对比、冗余场景识别

## 核心任务
判断输入的子场景中哪些是"战术等价"的，将等价的场景分组，每组只保留一个代表。

## "战术等价"的唯一标准
如果为场景 A 写一份 Phase 4 战术方案（包含突入方式、清房流程、火力配置、队形选择），
不加任何修改直接用于场景 B 也**完全合理**，则 A 和 B 是战术等价的。

## 影响等价的关键维度

### 1. 空间形态 (shape)
- room vs corridor vs stairwell vs open_area 完全不同，绝对不等价
- 同为 room 但面积差异极大（如 10m² vs 50m²）不等价
- 同为 corridor 但宽度类别不同（窄廊 vs 宽廊）不等价

### 2. 围合程度 (enclosure)
- fully_enclosed vs partially_open vs fully_open 不等价
- 围合程度直接影响突入方式、掩护需求、撤离路线

### 3. 门的数量与宽度类别 (entries_exits)
- 单门房间 vs 多门房间不等价（门的数量改变切角和清房流程）
- 门的宽度类别 (narrow/standard/wide) 不同不等价
- 门的朝向分布不同可能导致不等价

### 4. 房间尺度 (spatial_bounds)
- 面积量级不同不等价（如 10m² vs 25m² vs 50m²）
- 长宽比极端差异可能不等价（如正方形房间 vs 狭长房间）

### 5. 垂直位置 (vertical_position / floor)
- ground vs mid vs top 不等价
- 不同楼层的房间通常不等价（即使空间形态相似），垂直位置改变了接近方式和撤离路线

### 6. 连通场景类型 (connected_sub_scenes)
- 连通到走廊 vs 连通到楼梯间 vs 连通到开放区域 不等价
- 连通场景的类型分布影响清房后的下一步行动

### 7. 特殊几何特征 (key_features)
- 有致命漏斗 (fatal funnel) vs 无
- 栏杆 vs 实体墙
- 有盲角 vs 开阔视线
- 有掩体可用 vs 空旷无掩体

## 分组规则
1. 将战术等价的子场景归为一组
2. 每组只保留一个代表
3. 不等价于任何其他场景的子场景单独成组（只有 1 个成员）
4. 输出所有组的信息，不要遗漏任何子场景

## 选择代表规则
每组选择代表的优先级（从高到低）：
1. 位于进攻流更上游的（更靠近建筑入口/外场接近点）
2. 空间结构更典型的（更能代表该组的通用特征）
3. sub_scene_id 字母序更小的（作为最终 tie-breaker）

## 特别注意
- **走廊 (corridor)** 和 **楼梯井 (stairwell)** 通常不应被合并——它们的连通拓扑位置几乎总是独特的
- **入口/外场 (entry_point)** 场景各有独特的接近路径，不应合并
- 不同楼层的场景通常不等价
- **宁可保守保留，不要激进删除**。如果不确定两个场景是否等价，保留两者

## 输出格式
严格输出以下 JSON（只返回 JSON，不要包含 ```json 标记或其他内容）：

{
  "kept_sub_scene_ids": ["SS_01", "SS_03", "SS_05", ...],
  "groups": [
    {
      "representative": "SS_01",
      "members": ["SS_01", "SS_02"],
      "removed": ["SS_02"],
      "reason": "两者均为F0标准尺寸房间，单门fully_enclosed，空间尺寸和连通特征接近，可用同一套房间肃清战术应对"
    },
    {
      "representative": "SS_03",
      "members": ["SS_03"],
      "removed": [],
      "reason": "唯一楼梯井场景，连通拓扑位置独特，无法与其他场景合并"
    }
  ]
}
"""
# ── User Prompt Builder ────────────────────────────────────────


def _build_dedup_user_prompt(sub_scenes: List[Dict]) -> str:
    """构建去重用户 prompt，列出所有子场景的摘要信息"""
    lines = ["# 待去重子场景列表", ""]
    lines.append(f"共 {len(sub_scenes)} 个子场景：")
    lines.append("")

    for ss in sub_scenes:
        ss_id = ss.get("sub_scene_id", "?")
        role = ss.get("primary_role", ss.get("tactical_role", "?"))
        floor = ss.get("floor", "?")
        sp = ss.get("space_profile", {})
        shape = sp.get("shape", "?")
        enclosure = sp.get("enclosure", "?")
        vert = sp.get("vertical_position", "?")
        features = sp.get("key_features", [])
        entries = sp.get("entries_exits", [])
        bounds = ss.get("spatial_bounds", {})
        connected = ss.get("connected_sub_scenes", [])
        desc = ss.get("description", "")[:200]
        task_hint = ss.get("task_hint", "")
        zones = ss.get("zone_ids", [])

        # 从 spatial_bounds 估算面积
        area_str = "?"
        if bounds:
            try:
                x_span = bounds.get("x", [0, 0])
                y_span = bounds.get("y", [0, 0])
                if len(x_span) >= 2 and len(y_span) >= 2:
                    area = (x_span[1] - x_span[0]) * (y_span[1] - y_span[0])
                    area_str = f"{area:.0f} m²"
            except Exception:
                pass

        # 出入口摘要
        entry_strs = []
        for e in entries:
            entry_strs.append(
                f"{e.get('type', '?')}"
                f"(朝向{e.get('facing', '?')}"
                f"/宽度{e.get('width_category', '?')})"
            )
        entry_summary = ", ".join(entry_strs) if entry_strs else "无"

        lines.append(f"## {ss_id}")
        lines.append(f"- 角色: {role}")
        lines.append(f"- 楼层: F{floor}")
        lines.append(f"- 空间形态: {shape}")
        lines.append(f"- 围合程度: {enclosure}")
        lines.append(f"- 垂直位置: {vert}")
        lines.append(f"- 估算面积: {area_str}")
        lines.append(f"- 空间范围: x{bounds.get('x', [])}, y{bounds.get('y', [])}, z{bounds.get('z', [])}")
        lines.append(f"- 出入口 ({len(entries)}个): {entry_summary}")
        lines.append(f"- 关键特征: {', '.join(features) if features else '无'}")
        lines.append(f"- 连通子场景: {', '.join(connected) if connected else '无'}")
        lines.append(f"- 所属 zones: {', '.join(zones) if zones else '无'}")
        lines.append(f"- 描述: {desc}")
        if task_hint:
            lines.append(f"- 任务提示: {task_hint}")
        lines.append("")

    lines.append("## 任务")
    lines.append("请分析以上子场景的战术等价性，将可以用同一套战术应对的场景分组，")
    lines.append("每组只保留一个代表场景。输出保留的子场景 ID 列表，并说明每组的删除理由。")
    lines.append("注意：走廊、楼梯井、入口/外场场景通常各有独特性，不要轻易合并。")

    return "\n".join(lines)


# ── 主函数 ─────────────────────────────────────────────────────

def deduplicate_sub_scenes(
    sub_scenes: List[Dict],
    client: MiniMaxClient,
) -> List[Dict]:
    """对子场景列表进行战术等价去重

    一次 LLM 调用处理全部子场景，识别并移除战术等价的冗余场景。

    Args:
        sub_scenes: Phase 2 产出的子场景列表
        client: MiniMaxClient 实例

    Returns:
        去重后的子场景列表
    """
    if len(sub_scenes) <= 1:
        logger.info("子场景数量 ≤1，跳过去重")
        return sub_scenes

    logger.info("开始 LLM 去重: %d 个子场景", len(sub_scenes))

    user_prompt = _build_dedup_user_prompt(sub_scenes)

    try:
        result = client.generate_json(
            system_prompt=DEDUP_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=0.3,
        )
    except Exception as e:
        logger.warning("LLM 去重调用失败: %s，保留全部子场景", e)
        return sub_scenes

    # generate_json 可能返回 list（Extra data 修复路径）
    if isinstance(result, list):
        result = result[0] if len(result) > 0 and isinstance(result[0], dict) else {}
    if not isinstance(result, dict):
        logger.warning("LLM 去重返回了非预期格式: %s，保留全部子场景", type(result).__name__)
        return sub_scenes

    kept_ids = set(result.get("kept_sub_scene_ids", []))
    groups = result.get("groups", [])

    if not kept_ids:
        logger.warning("LLM 去重返回的 kept_sub_scene_ids 为空，保留全部子场景")
        return sub_scenes

    # 记录去重决策
    for group in groups:
        removed = group.get("removed", [])
        if removed:
            logger.info(
                "去重: [%s] → 保留 %s, 移除 %s | 理由: %s",
                group.get("representative", "?"),
                group.get("representative", "?"),
                removed,
                group.get("reason", "未说明"),
            )

    # 构建 ID → 子场景映射
    ss_map = {ss.get("sub_scene_id"): ss for ss in sub_scenes}

    filtered = []
    for ss_id in kept_ids:
        if ss_id in ss_map:
            filtered.append(ss_map[ss_id])
        else:
            logger.warning("LLM 返回了未知的 sub_scene_id: %s，跳过", ss_id)

    removed_count = len(sub_scenes) - len(filtered)
    logger.info(
        "去重完成: %d → %d 个子场景（移除 %d 个）",
        len(sub_scenes), len(filtered), removed_count,
    )

    return filtered


def cleanup_definitions(definitions: Dict, kept_ids: set) -> Dict:
    """从 definitions 字典中移除去重掉的子场景的关联引用

    清理 sub_scene_graph 的 nodes/edges 和 entry_options。

    Args:
        definitions: Phase 2 输出的完整定义字典
        kept_ids: 保留的子场景 ID 集合

    Returns:
        清理后的定义字典（原地修改 + 返回）
    """
    # 清理 sub_scene_graph
    graph = definitions.get("sub_scene_graph", {})
    if graph:
        graph["nodes"] = [n for n in graph.get("nodes", []) if n in kept_ids]
        graph["edges"] = [
            e for e in graph.get("edges", [])
            if e.get("from") in kept_ids and e.get("to") in kept_ids
        ]

    # 清理 entry_options
    definitions["entry_options"] = [
        e for e in definitions.get("entry_options", [])
        if e.get("sub_scene_id") in kept_ids
    ]

    # 清理每个子场景的 connected_sub_scenes（移除指向已删场景的引用）
    for ss in definitions.get("sub_scenes", []):
        connected = ss.get("connected_sub_scenes", [])
        if connected:
            ss["connected_sub_scenes"] = [c for c in connected if c in kept_ids]

    return definitions
