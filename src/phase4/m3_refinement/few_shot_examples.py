"""
正向 Few-Shot 示例库

从 robot_project core/few_shot_examples.py 直接迁移。
8条固定对比示例，用于 A_gen 的 system prompt 注入。
"""

from typing import Dict, List, Optional, TypedDict


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
    "primary_conversion_types": ["编组化", "方位泛化", "空间泛化", "火力泛化"],
    "scene_context": (
        "狭长走廊，远端呈 L 形转角。任务：沿走廊向远端推进，肃清转角。"
    ),
    "violation_version": (
        "人形机器人1号在3秒内沿走廊北侧墙壁机动至距L形转角2米处，"
        "对转角后方实施5发短点射压制。无人机2号同步从走廊上空悬停，"
        "以俯视视角确认转角后方敌情。"
    ),
    "corrected_version": (
        "突击组沿走廊一侧贴墙机动至L形转角紧邻处，对转角后方实施短点射试探性"
        "压制。侦察组同步前出至转角上方高位观察点，利用俯视视角确认转角"
        "后方态势。确认安全后，突击组低姿跃进至转角后方就近掩体。"
    ),
    "conversion_table": [
        {"violation": "人形机器人1号", "correction": "突击组", "conversion_type": "编组化"},
        {"violation": "3秒内", "correction": "（删除）", "conversion_type": "时间泛化"},
        {"violation": "走廊北侧墙壁", "correction": "走廊一侧", "conversion_type": "方位泛化"},
        {"violation": "距L形转角2米处", "correction": "L形转角紧邻处", "conversion_type": "空间泛化"},
        {"violation": "5发短点射", "correction": "短点射", "conversion_type": "火力泛化"},
        {"violation": "无人机2号", "correction": "侦察组", "conversion_type": "编组化"},
    ],
}

EX_02: FewShotExample = {
    "example_id": "EX-02",
    "scene_type": "开阔地接近",
    "phase": "进攻—接近",
    "primary_conversion_types": ["物体泛化", "距离泛化", "时间泛化"],
    "scene_context": (
        "建筑一侧为开阔草地，分布有废弃车辆和灌木丛。"
        "任务：从外场树林边缘出发，穿越开阔地接近建筑正门。"
    ),
    "violation_version": (
        "人形机器人1号从东侧树林边缘出发，以15km/h速度沿废弃白色厢式货车方向"
        "机动至距建筑正门10米处的货车引擎盖后方。2秒后，机器狗从北侧灌木丛后方"
        "跃进至第二辆红色轿车后方。"
    ),
    "corrected_version": (
        "突击组从外场植被遮蔽线出发，沿车辆掩体连线方向分段跃进，"
        "依次占据距建筑入口最近的大型车辆掩体。侧翼警戒组同步从侧翼植被遮蔽线"
        "跃进至相邻车辆掩体后方，与突击组形成对建筑正面的交叉掩护扇区。"
    ),
    "conversion_table": [
        {"violation": "人形机器人1号", "correction": "突击组", "conversion_type": "编组化"},
        {"violation": "15km/h速度", "correction": "（删除）", "conversion_type": "数值泛化"},
        {"violation": "白色厢式货车", "correction": "车辆掩体", "conversion_type": "物体泛化"},
        {"violation": "距建筑正门10米处", "correction": "距建筑入口最近处", "conversion_type": "距离泛化"},
        {"violation": "机器狗", "correction": "侧翼警戒组", "conversion_type": "编组化"},
        {"violation": "红色轿车后方", "correction": "相邻车辆掩体后方", "conversion_type": "物体泛化"},
    ],
}

EX_03: FewShotExample = {
    "example_id": "EX-03",
    "scene_type": "房间突入",
    "phase": "进攻—清剿",
    "primary_conversion_types": ["编组化", "空间泛化", "战术动作泛化"],
    "scene_context": (
        "走廊一侧有一扇门通往房间。任务：从走廊进入房间，肃清室内威胁。"
    ),
    "violation_version": (
        "人形机器人1号在3秒内完成门框切片侦察——利用门框边缘以最小暴露面"
        "逐象限扫描室内。与机器人2号以双路突入方式进入房间。"
        "全程耗时不超过8秒。"
    ),
    "corrected_version": (
        "突击组在门框处执行切片侦察——利用门框边缘以最小暴露面逐象限扫描"
        "室内，优先确认门后死角及大型掩体后方区域。侦察组在门框对侧提供"
        "交叉观察。确认可见区域安全后，突击组与侧翼警戒组以动态突入方式进入。"
    ),
    "conversion_table": [
        {"violation": "人形机器人1号", "correction": "突击组", "conversion_type": "编组化"},
        {"violation": "机器人2号", "correction": "侧翼警戒组", "conversion_type": "编组化"},
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
        "突击组低姿沿楼梯扶手一侧推进，枪口指向楼梯上方。"
        "侦察组悬停于楼梯井高位，利用垂直视线提供楼梯上方态势感知。"
        "突击组推进至台阶中段时短暂停顿，对楼梯上段实施短点射试探性压制。"
        "确认无反击后，突击组以可控节奏占领上层楼梯口。"
    ),
    "conversion_table": [
        {"violation": "人形机器人1号", "correction": "突击组", "conversion_type": "编组化"},
        {"violation": "无人机2号", "correction": "侦察组", "conversion_type": "编组化"},
        {"violation": "距一层地面4米高度", "correction": "楼梯井高位", "conversion_type": "空间泛化"},
        {"violation": "3发短点射", "correction": "短点射", "conversion_type": "火力泛化"},
        {"violation": "20km/h冲刺速度", "correction": "可控节奏", "conversion_type": "数值泛化"},
    ],
}

EX_05: FewShotExample = {
    "example_id": "EX-05",
    "scene_type": "撤退掩护",
    "phase": "撤离",
    "primary_conversion_types": ["编组化", "物体泛化", "距离泛化", "时间泛化"],
    "scene_context": (
        "完成人质解救后需从建筑二层经楼梯撤至一层并退出建筑。"
        "任务：掩护人质从二层撤至建筑外车辆撤离点。"
    ),
    "violation_version": (
        "人形机器人1号在二层楼梯口以3发短点射压制追兵，人形机器人2号"
        "护送人质沿楼梯以每秒2米速度下撤。撤至30米外的白色货车后方集结。"
    ),
    "corrected_version": (
        "断后掩护组在撤离起始层楼梯口建立火力封锁，压制追兵可能出现的方向。"
        "护卫组护送人质沿楼梯以可控速度下撤。编队交替掩护撤出建筑入口，"
        "逐段跃进至外场车辆撤离点。"
    ),
    "conversion_table": [
        {"violation": "人形机器人1号", "correction": "断后掩护组", "conversion_type": "编组化"},
        {"violation": "人形机器人2号", "correction": "护卫组", "conversion_type": "编组化"},
        {"violation": "3发短点射", "correction": "火力封锁", "conversion_type": "火力泛化"},
        {"violation": "每秒2米速度", "correction": "可控速度", "conversion_type": "数值泛化"},
        {"violation": "30米外的白色货车后方", "correction": "外场车辆撤离点", "conversion_type": "物体泛化"},
    ],
}

EX_06: FewShotExample = {
    "example_id": "EX-06",
    "scene_type": "房间肃清",
    "phase": "进攻—肃清",
    "primary_conversion_types": ["方位泛化", "物体泛化", "编组化", "空间泛化"],
    "scene_context": (
        "室内房间，散布低矮木质长桌和大型L形沙发。"
        "任务：从走廊进入房间，肃清室内威胁。"
    ),
    "violation_version": (
        "突击手沿走廊北侧贴墙机动至木门框处，执行切片侦察确认门后死角无威胁。"
        "利用走廊内低矮长桌作为掩护，逐段推进至大型L形沙发后方。"
        "枪口指向混凝土立柱方向，防止柱后伏击。"
    ),
    "corrected_version": (
        "突击组沿走廊入口侧贴墙机动至门框处，执行切片侦察确认门后死角无威胁。"
        "利用走廊内低矮掩体作为掩护，逐段推进至近侧大型遮挡物后方。"
        "枪口指向柱状掩体方向，防止掩体后伏击。"
    ),
    "conversion_table": [
        {"violation": "突击手", "correction": "突击组", "conversion_type": "编组化"},
        {"violation": "走廊北侧", "correction": "走廊入口侧", "conversion_type": "方位泛化"},
        {"violation": "木门框", "correction": "门框", "conversion_type": "物体泛化"},
        {"violation": "低矮长桌", "correction": "低矮掩体", "conversion_type": "物体泛化"},
        {"violation": "大型L形沙发", "correction": "近侧大型遮挡物", "conversion_type": "物体泛化"},
        {"violation": "混凝土立柱", "correction": "柱状掩体", "conversion_type": "物体泛化"},
    ],
}

EX_07: FewShotExample = {
    "example_id": "EX-07",
    "scene_type": "走廊肃清",
    "phase": "进攻—肃清",
    "primary_conversion_types": ["编队规模泛化", "编组化"],
    "scene_context": (
        "狭长走廊，编队从入口向远端推进肃清。"
        "任务：沿走廊推进，肃清沿途威胁。"
    ),
    "violation_version": (
        "双人编队沿走廊推进，前方掩护组贴靠走廊一侧墙壁前进，"
        "后方掩护组保持对入口方向警戒。两人交替掩护至走廊中段。"
    ),
    "corrected_version": (
        "编队沿走廊推进，前方掩护组贴靠走廊一侧墙壁前进，"
        "后方掩护组保持对入口方向警戒。编队交替掩护至走廊中段。"
    ),
    "conversion_table": [
        {"violation": "双人编队", "correction": "编队", "conversion_type": "编队规模泛化"},
        {"violation": "两人交替掩护", "correction": "编队交替掩护", "conversion_type": "编队规模泛化"},
    ],
}

EX_08: FewShotExample = {
    "example_id": "EX-08",
    "scene_type": "建筑外围两翼包抄",
    "phase": "进攻—接近",
    "primary_conversion_types": ["编组化", "方位泛化", "编队规模泛化", "空间泛化"],
    "scene_context": (
        "独立建筑位于开放区域，四周有低矮围墙。"
        "任务：从外围安全距离接近建筑，形成控制态势后突入。"
    ),
    "violation_version": (
        "无人机2号升空侦察确认建筑外围无威胁后，人形机器人1号沿建筑东侧围墙"
        "接近正门，人形机器人3号从建筑西侧围墙迂回至建筑北面进行包抄。"
        "两台机器人在距建筑正门10米处汇合，准备突入。"
    ),
    "corrected_version": (
        "侦察组升空对建筑外围进行全向侦察，确认态势后编队展开。"
        "突击组沿建筑正面方向控制接近，利用沿途掩体分段跃进至入口紧邻处。"
        "侧翼警戒组同步从建筑一侧外围迂回至对侧方向——"
        "与突击组形成两翼包抄的控制态势。"
        "编队在建筑入口紧邻处汇合，确认突入条件后发起协同突入。"
    ),
    "conversion_table": [
        {"violation": "无人机2号", "correction": "侦察组", "conversion_type": "编组化"},
        {"violation": "人形机器人1号", "correction": "突击组", "conversion_type": "编组化"},
        {"violation": "人形机器人3号", "correction": "侧翼警戒组", "conversion_type": "编组化"},
        {"violation": "沿建筑东侧围墙接近正门", "correction": "沿建筑正面方向控制接近", "conversion_type": "方位泛化"},
        {"violation": "从建筑西侧围墙迂回至建筑北面进行包抄", "correction": "从建筑一侧外围迂回至对侧方向", "conversion_type": "方位泛化"},
        {"violation": "两台机器人", "correction": "编队", "conversion_type": "编队规模泛化"},
        {"violation": "距建筑正门10米处", "correction": "建筑入口紧邻处", "conversion_type": "空间泛化"},
    ],
}


EX_09: FewShotExample = {
    "example_id": "EX-09",
    "scene_type": "房间突入（struct_version 多编组协同）",
    "phase": "进攻—突入",
    "primary_conversion_types": ["多编组协同", "触发条件", "交接逻辑"],
    "scene_context": (
        "走廊中部一侧有门通往目标房间。侦察组已确认门后为开阔房间，散布有低矮掩体。"
        "任务：编队从走廊突入房间，肃清后控制出口。"
    ),
    "violation_version": (
        "struct_version Instructions 单一编组版（违规）：\n"
        "Step 1:\n"
        "  [Unit突击组] → 切片侦察门后区域\n"
        "  [Unit突击组] → 突入房间至近角\n"
        "Step 2:\n"
        "  [Unit突击组] → 逐掩体推进至房间远角\n"
        "  [Unit突击组] → 确认出口方向安全\n"
        "\n"
        "问题：全部步骤由单一编组执行，这不是战术——是单个编组的行动清单。"
        "缺少侦察组的态势支撑和掩护组的火力覆盖。"
    ),
    "corrected_version": (
        "struct_version Instructions 多编组协同版（正确）：\n"
        "Step 1:\n"
        "  [Unit侦察组] → [动作: 高位侦察], [目标: {{门后扇区}}],"
        " [报告: 掩体分布+威胁位置]\n"
        "  [Unit突击组] → [动作: 门框切片侦察],"
        " [触发条件: 侦察组报告无威胁后], [确认: 门后死角安全]\n"
        "  [Unit掩护组] → [就位: {{门框对侧}}], [武器姿态: 枪口指向室内远角]\n"
        "Step 2:\n"
        "  [Unit突击组] → [触发条件: 掩护组就位+侦察组持续监视],"
        " [动作: 穿过门框], [路径: 沿最近墙壁至{{房间近角}}]\n"
        "  [Unit掩护组] → [同步跟进: 门框内侧],"
        " [武器姿态: 覆盖远角+突击组未覆盖扇区]\n"
        "  [Unit侦察组] → [持续监视: {{门后区域}}],"
        " [交接信号: 报告掩护组已覆盖全部扇区]\n"
        "  [编队确认] → [房间安全]"
    ),
    "conversion_table": [
        {"violation": "全部步骤由单一编组执行", "correction": "三个不同编组协同：侦察组态势支撑→突击组突入→掩护组火力覆盖", "conversion_type": "多编组协同"},
        {"violation": "缺少触发条件", "correction": "侦察组报告无威胁后→突击组行动；掩护组就位后→突击组突入", "conversion_type": "触发条件"},
        {"violation": "编组间无交接逻辑", "correction": "侦察组监视→确认掩护组覆盖→报告安全→编队确认", "conversion_type": "交接逻辑"},
    ],
}


FIXED_EXAMPLES: Dict[str, FewShotExample] = {
    "EX-01": EX_01, "EX-02": EX_02, "EX-03": EX_03,
    "EX-04": EX_04, "EX-05": EX_05, "EX-06": EX_06,
    "EX-07": EX_07, "EX-08": EX_08, "EX-09": EX_09,
}

EXAMPLES_BY_MODE: Dict[str, List[str]] = {
    "RAG": ["EX-01", "EX-03", "EX-06", "EX-07", "EX-08", "EX-09"],
    "HYBRID": ["EX-01", "EX-02", "EX-03", "EX-06", "EX-07", "EX-08", "EX-09"],
    "GEN": ["EX-01", "EX-02", "EX-03", "EX-04", "EX-05", "EX-06", "EX-07", "EX-08", "EX-09"],
}

# 按作战阶段将示例分类，用于阶段感知排序
# 阶段匹配的示例前置（利用 primacy effect），不删除其他示例
EXAMPLES_BY_PHASE: Dict[str, List[str]] = {
    "侦察阶段": [],
    "进攻阶段": ["EX-01", "EX-02", "EX-03", "EX-04", "EX-06", "EX-07", "EX-08", "EX-09"],
    "防御阶段": [],
    "撤退与脱离阶段": ["EX-05"],
}

# 阶段关键词 → 标准阶段名（用于模糊匹配）
_PHASE_KEYWORDS: Dict[str, str] = {
    "侦察": "侦察阶段",
    "进攻": "进攻阶段",
    "突击": "进攻阶段",
    "肃清": "进攻阶段",
    "清剿": "进攻阶段",
    "突入": "进攻阶段",
    "接近": "进攻阶段",
    "防御": "防御阶段",
    "固守": "防御阶段",
    "撤退": "撤退与脱离阶段",
    "脱离": "撤退与脱离阶段",
    "撤离": "撤退与脱离阶段",
    "断后": "撤退与脱离阶段",
    "掩护撤退": "撤退与脱离阶段",
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


def format_examples_for_prompt(mode: str, mission_phase: Optional[str] = None) -> str:
    """格式化 Few-Shot 示例为 prompt 文本

    将阶段匹配的示例前置（利用 primacy effect），不删除其他示例。
    当指定阶段无专属示例时，注入跨阶段引导说明。
    """
    examples = get_examples_by_mode(mode)
    if mission_phase and mission_phase in EXAMPLES_BY_PHASE:
        phase_example_ids = EXAMPLES_BY_PHASE[mission_phase]
        # 将阶段匹配的示例前置
        phase_matched = [e for e in examples if e["example_id"] in phase_example_ids]
        other = [e for e in examples if e["example_id"] not in phase_example_ids]
        examples = phase_matched + other

    parts = []
    for ex in examples:
        parts.append(format_example_for_prompt(ex))

    # 当指定阶段无专属示例时，注入跨阶段引导说明
    if mission_phase and mission_phase in EXAMPLES_BY_PHASE and not EXAMPLES_BY_PHASE[mission_phase]:
        cross_phase_note = (
            f"注意：当前作战阶段为「{mission_phase}」，该阶段暂无专属示例。"
            f"以下示例来自其他作战阶段，请将示例中的进攻/防御意图替换为当前"
            f"{mission_phase}的对应意图。重点关注编组化、通用化、动作粒度等"
            f"格式模式，而非模仿示例的具体战术目的。"
        )
        parts.insert(0, cross_phase_note)

    return "\n\n".join(parts)
