"""
正向 Few-Shot 示例库

从 robot_project core/few_shot_examples.py 直接迁移。
5条固定对比示例，用于 A_gen 的 system prompt 注入。
"""

from typing import Dict, List, TypedDict


class ConversionPair(TypedDict):
    violation: str
    correction: str
    conversion_type: str


class FewShotExample(TypedDict):
    example_id: str
    scene_type: str
    phase: str
    primary_conversion_types: List[str]
    scene_context: str
    violation_version: str
    corrected_version: str
    conversion_table: List[ConversionPair]


EX_01: FewShotExample = {
    "example_id": "EX-01",
    "scene_type": "走廊转角",
    "phase": "进攻—肃清",
    "primary_conversion_types": ["角色化", "方位泛化", "空间泛化", "火力泛化"],
    "scene_context": (
        "狭长走廊，北端呈 L 形向东转角。任务：沿走廊向北推进，肃清转角。"
    ),
    "violation_version": (
        "人形机器人1号在3秒内沿走廊北侧墙壁机动至距L形转角2米处，"
        "对转角后方实施5发短点射压制。无人机2号同步从走廊上空悬停，"
        "以俯视视角确认转角后方敌情。"
    ),
    "corrected_version": (
        "突击手沿走廊一侧贴墙机动至转角紧邻处，对转角后方实施短点射试探性"
        "压制。侦察节点同步前出至转角上方高位观察点，利用俯视视角确认转角"
        "后方态势。确认安全后，突击手低姿跃进至转角后方就近掩体。"
    ),
    "conversion_table": [
        {"violation": "人形机器人1号", "correction": "突击手", "conversion_type": "角色化"},
        {"violation": "3秒内", "correction": "（删除）", "conversion_type": "时间泛化"},
        {"violation": "走廊北侧墙壁", "correction": "走廊一侧", "conversion_type": "方位泛化"},
        {"violation": "距L形转角2米处", "correction": "转角紧邻处", "conversion_type": "空间泛化"},
        {"violation": "5发短点射", "correction": "短点射", "conversion_type": "火力泛化"},
        {"violation": "无人机2号", "correction": "侦察节点", "conversion_type": "角色化"},
    ],
}

EX_02: FewShotExample = {
    "example_id": "EX-02",
    "scene_type": "开阔地接近",
    "phase": "进攻—接近",
    "primary_conversion_types": ["物体泛化", "距离泛化", "时间泛化"],
    "scene_context": (
        "建筑东侧为开阔草地，分布有废弃车辆和灌木丛。"
        "任务：从东侧树林边缘出发，穿越开阔地接近建筑正门。"
    ),
    "violation_version": (
        "人形机器人1号从东侧树林边缘出发，以15km/h速度沿废弃白色厢式货车方向"
        "机动至距建筑正门10米处的货车引擎盖后方。2秒后，机器狗从北侧灌木丛后方"
        "跃进至第二辆红色轿车后方。"
    ),
    "corrected_version": (
        "突击手从外场植被遮蔽线出发，沿车辆掩体连线方向分段跃进，"
        "依次占据距建筑入口最近的大型车辆掩体。警戒手同步从侧翼植被遮蔽线"
        "跃进至相邻车辆掩体后方，与突击手形成对建筑正面的交叉掩护扇区。"
    ),
    "conversion_table": [
        {"violation": "人形机器人1号", "correction": "突击手", "conversion_type": "角色化"},
        {"violation": "15km/h速度", "correction": "（删除）", "conversion_type": "数值泛化"},
        {"violation": "白色厢式货车", "correction": "车辆掩体", "conversion_type": "物体泛化"},
        {"violation": "距建筑正门10米处", "correction": "距建筑入口最近处", "conversion_type": "距离泛化"},
        {"violation": "机器狗", "correction": "警戒手", "conversion_type": "角色化"},
        {"violation": "红色轿车后方", "correction": "相邻车辆掩体后方", "conversion_type": "物体泛化"},
    ],
}

EX_03: FewShotExample = {
    "example_id": "EX-03",
    "scene_type": "房间突入",
    "phase": "进攻—清剿",
    "primary_conversion_types": ["角色化", "空间泛化", "战术动作泛化"],
    "scene_context": (
        "走廊西侧有一扇门通往房间R102。任务：从走廊进入房间，肃清室内威胁。"
    ),
    "violation_version": (
        "人形机器人1号在3秒内完成门框切片侦察——利用门框边缘以最小暴露面"
        "逐象限扫描室内。与机器人2号以双路突入方式进入房间。"
        "全程耗时不超过8秒。"
    ),
    "corrected_version": (
        "突击手在门框处执行切片侦察——利用门框边缘以最小暴露面逐象限扫描"
        "室内，优先确认门后死角及大型掩体后方区域。侦察节点在门框对侧提供"
        "交叉观察。确认可见区域安全后，突击手与警戒手以动态突入方式进入。"
    ),
    "conversion_table": [
        {"violation": "人形机器人1号", "correction": "突击手", "conversion_type": "角色化"},
        {"violation": "机器人2号", "correction": "警戒手", "conversion_type": "角色化"},
        {"violation": "3秒内完成", "correction": "（删除）", "conversion_type": "时间泛化"},
        {"violation": "全程耗时不超过8秒", "correction": "突入节奏以侦察确认速度为准", "conversion_type": "时间泛化"},
    ],
}

EX_04: FewShotExample = {
    "example_id": "EX-04",
    "scene_type": "楼梯推进",
    "phase": "进攻—肃清",
    "primary_conversion_types": ["方位泛化", "火力泛化", "协同时序泛化"],
    "scene_context": (
        "建筑内部楼梯间呈U形转折。任务：从一层楼梯口出发，沿楼梯向上推进至二层。"
    ),
    "violation_version": (
        "人形机器人1号低姿沿楼梯右侧扶手推进。无人机2号悬停在楼梯井中央距一层地面4米高度。"
        "随后1号向二层方向发射3发短点射试探压制，以20km/h冲刺速度占领二层楼梯口。"
    ),
    "corrected_version": (
        "突击手低姿沿楼梯扶手一侧推进，枪口指向楼梯上方。"
        "侦察节点悬停于楼梯井高位，利用垂直视线提供楼梯上方态势感知。"
        "突击手推进至台阶中段时短暂停顿，对楼梯上段实施短点射试探性压制。"
        "确认无反击后，突击手以可控节奏占领上层楼梯口。"
    ),
    "conversion_table": [
        {"violation": "人形机器人1号", "correction": "突击手", "conversion_type": "角色化"},
        {"violation": "无人机2号", "correction": "侦察节点", "conversion_type": "角色化"},
        {"violation": "距一层地面4米高度", "correction": "楼梯井高位", "conversion_type": "空间泛化"},
        {"violation": "3发短点射", "correction": "短点射", "conversion_type": "火力泛化"},
        {"violation": "20km/h冲刺速度", "correction": "可控节奏", "conversion_type": "数值泛化"},
    ],
}

EX_05: FewShotExample = {
    "example_id": "EX-05",
    "scene_type": "撤退掩护",
    "phase": "撤离",
    "primary_conversion_types": ["角色化", "物体泛化", "距离泛化", "时间泛化"],
    "scene_context": (
        "完成人质解救后需从建筑二层经楼梯撤至一层并退出建筑。"
        "任务：掩护人质从二层撤至建筑外车辆撤离点。"
    ),
    "violation_version": (
        "人形机器人1号在二层楼梯口以3发短点射压制追兵，人形机器人2号"
        "护送人质沿楼梯以每秒2米速度下撤。撤至30米外的白色货车后方集结。"
    ),
    "corrected_version": (
        "断后掩护手在撤离起始层楼梯口建立火力封锁，压制追兵可能出现的"
        "方向。护卫手护送人质沿楼梯以可控速度下撤。两角色交替掩护撤出建筑入口，"
        "逐段跃进至外场车辆撤离点。"
    ),
    "conversion_table": [
        {"violation": "人形机器人1号", "correction": "断后掩护手", "conversion_type": "角色化"},
        {"violation": "人形机器人2号", "correction": "护卫手", "conversion_type": "角色化"},
        {"violation": "3发短点射", "correction": "火力封锁", "conversion_type": "火力泛化"},
        {"violation": "每秒2米速度", "correction": "可控速度", "conversion_type": "数值泛化"},
        {"violation": "30米外的白色货车后方", "correction": "外场车辆撤离点", "conversion_type": "物体泛化"},
    ],
}

FIXED_EXAMPLES: Dict[str, FewShotExample] = {
    "EX-01": EX_01, "EX-02": EX_02, "EX-03": EX_03,
    "EX-04": EX_04, "EX-05": EX_05,
}

EXAMPLES_BY_MODE: Dict[str, List[str]] = {
    "RAG": ["EX-01", "EX-03"],
    "HYBRID": ["EX-01", "EX-02", "EX-03"],
    "GEN": ["EX-01", "EX-02", "EX-03", "EX-04", "EX-05"],
}


def format_example_for_prompt(example: FewShotExample) -> str:
    lines = [
        f"【示例 {example['example_id']}：{example['scene_type']}】",
        "",
        "【场景上下文】", example["scene_context"],
        "",
        "【违规版】", example["violation_version"],
        "",
        "【修正版】", example["corrected_version"],
        "",
        "【转换对照】",
    ]
    for pair in example["conversion_table"]:
        lines.append(f"  {pair['violation']} → {pair['correction']} [{pair['conversion_type']}]")
    return "\n".join(lines)


def get_examples_by_mode(mode: str) -> List[FewShotExample]:
    example_ids = EXAMPLES_BY_MODE.get(mode, EXAMPLES_BY_MODE["GEN"])
    return [FIXED_EXAMPLES[eid] for eid in example_ids]


def format_examples_for_prompt(mode: str) -> str:
    examples = get_examples_by_mode(mode)
    return "\n\n".join(format_example_for_prompt(ex) for ex in examples)
