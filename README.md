# robot_project2 — GTKG-CM Pipeline

生成式战法战术知识生成概念模型（Generative Tactical Knowledge Generation Conceptual Model, GTKG-CM）的完整实现。

---

## 一、项目背景

### 1.1 问题

当前战术战法知识面临三重困境：

1. **资料适配性不足**：传统战术资料聚焦人力作战条件，难以直接迁移至无人协同作战场景。
2. **知识粒度断层**：从"连排级战术"到"机器人可执行指令"之间存在显著的粒度鸿沟。
3. **场景覆盖有限**：结构化战术知识库无法穷尽所有作战场景组合，见到新场景时缺乏生成手段。

### 1.2 方案

LLM 为"从场景出发自动生成战术知识"提供了技术可行性。但纯 LLM 生成存在**知识幻觉**和**质量不可控**两个核心风险。GTKG-CM 通过**场景驱动、知识锚定、多智能体协同**三管齐下，在充分利用 LLM 生成能力的同时保证输出质量。

### 1.3 核心设计原则

| 原则 | 说明 |
|------|------|
| **角色化描述** | 执行主体用功能角色（"突击手"、"侦察节点"），不绑定机器人编号 |
| **双版本输出** | 文字描述版（VLA 理解）+ 结构化描述版（可执行指令），内容一致，形式不同 |
| **参数占位** | 结构化版中的坐标/速度/射速留为 `{占位符}`，执行阶段填入 |
| **场景无关（双重不变性）** | 战术描述行动**模式**，必须具有双重不变性——旋转不变性（场景旋转任意角度后每个字仍正确）+ 物体功能不变性（掩体替换为功能等价物体后每个字仍正确）。禁止具体数值、禁止绝对方向指代（东/南/西/北+侧/端/段/翼/面/墙/角/缘/头/区/部/方/向），禁止场景特定物体身份描述（桌子/沙发/椅子/花坛/空调外机等），必须替换为战术功能类别（低矮掩体/大型遮挡物/柱状掩体等） |
| **相对关系** | 空间用"紧邻/中距离"，时间用"即时/短暂延迟"，火力用"短点射/持续压制"，方向用"远端/近端/一侧/对侧" |
| **零具体数值** | 文字描述版完全不出现阿拉伯或中文数字（步骤编号除外） |

---

## 二、流水线架构

整个系统将"原始 USDA 大场景文件"转化为"原子级战术知识 JSON"，分五个独立阶段执行。阶段之间通过文件系统解耦——每个阶段的输出存储在固定目录下，下一阶段读取。

```
                        ┌─────────────────────────────────────────────────────┐
                        │                 GTKG-CM 五阶段流水线                  │
                        └─────────────────────────────────────────────────────┘

  data/raw/scene.usda
        │
        ▼
  ┌──────────────────────────────────────────────────────────────────────────┐
  │  Phase 0: 流式 USDA 解析（纯几何输出）                                     │
  │  ──────────────────────────────────                                       │
  │  实现方式: 纯 Python (无 LLM)                                              │
  │  功能:                                                                    │
  │    • 逐行扫描 USDA，brace counting 确定 Prim 边界                          │
  │    • 解析 extent / xformOp(translate/rotate/scale) / material 属性         │
  │    • 累乘变换矩阵计算世界空间包围盒 (world_bbox)                             │
  │    • 碎片→结构提取: 墙面线 / 楼板面 / 楼板证据 / 材质预分类                  │
  │    • 楼板面→楼层(N-1): 候选面→合并(6m)→楼板层级→可居住楼层(stories)         │
  │    • 房间检测(按楼层): 正交墙面交点网格 + 垂直范围过滤                         │
  │    • 走廊检测: 平行墙面间隙中长条形低密度区域                                  │
  │    • 楼梯检测: 台阶序列启发式 + 跨层密度检测（双方法）                         │
  │    • XZ 空间预聚类: building / exterior / terrain / background               │
  │    • 不做启发式语义分类——碎片化 Mesh 使规则不可靠，语义交 LLM                │
  │  输入: 原始 .usda 文件 (可处理 GB 级大文件)                                  │
  │  输出: data/processed/phase0/scene_metadata.json                           │
  └──────────────────────────────────────────────────────────────────────────┘
        │
        ▼
  ┌──────────────────────────────────────────────────────────────────────────┐
  │  Phase 1: LLM 全局场景理解 (A_parse)                                       │
  │  ─────────────────────────────────────                                    │
  │  实现方式: LLM (MiniMax)                                                   │
  │  功能:                                                                    │
  │    • 精简 Phase 0 纯几何元数据 (data_compactor)                              │
  │    • LLM 推断: 建筑类型、N-1楼层逻辑、空间区域划分                            │
  │    • 输入: 精简元数据 + 楼板面证据(stories) + 走廊候选 + 密度楼梯             │
  │    • 识别战术地物: 掩体位置、瓶颈点、视线走廊、盲区、致命漏斗                  │
  │    • 与 Phase 0 双方法楼梯检测结果（启发式+密度）交叉验证                    │
  │  输入: data/processed/phase0/scene_metadata.json                           │
  │  输出: data/processed/phase1/global_understanding.json                     │
  └──────────────────────────────────────────────────────────────────────────┘
        │
        ▼
  ┌──────────────────────────────────────────────────────────────────────────┐
  │  Phase 2: 子场景划分 (LLM + 人工审核) — 空间/战术解耦                       │
  │  ────────────────────────────────────────                                 │
  │  实现方式: LLM 生成 + CLI 交互式审核                                        │
  │  功能:                                                                    │
  │    • LLM 将大场景拆分为子场景 — Phase 2 定义 SPACE, Phase 4 决定 TACTICS    │
  │    • LLM 驱动子场景去重 — 战术等价性判断，移除冗余场景，只保留代表性场景    │
  │    • space_profile: 物理空间特征(shape/enclosure/vertical_position/出入口)   │
  │    • suggested_roles: 多战术建议(非唯一固定角色, Phase 4 按任务选择)         │
  │    • entry_options: 全入口清单(地面门/窗户/阳台/屋顶/可破拆墙体)             │
  │    • spatial_bounds 完整性: 包含围合结构 + 门外 ≥2m 走廊                    │
  │    • CLI 审核工具: 接受 / 修改 / 重排序 / 删除子场景                        │
  │    • 每个子场景标注 task_hint、spatial_bounds、priority、connected          │
  │  输入: data/processed/phase1/global_understanding.json                     │
  │  输出: data/processed/phase2/sub_scene_definitions.json                    │
  └──────────────────────────────────────────────────────────────────────────┘
        │
        ▼
  ┌──────────────────────────────────────────────────────────────────────────┐
  │  Phase 3: 几何处理 + 语义标注                                              │
  │  ────────────────────────────                                             │
  │  实现方式: Python + LLM                                                    │
  │  功能:                                                                    │
  │    • 子场景去重安全网: Phase 2 未去重时自动移除战术等价冗余场景             │
  │    3a. 裁剪 (crop): 按 spatial_bounds 从全局 Cube 中提取子场景              │
  │    3b. 简化 (simplify) — 间隙保护合并(>0.5m间隙→不合并，保护门/窗):        │
  │         • 相似 Cube 合并 (同材质 + 空间邻近)                                 │
  │         • 模式折叠 (等距排列 → 单代表 + 元数据)                              │
  │         • Z 轴归一化 (最小值归零)                                           │
  │         • 目标: 每个子场景 ≤ 60 个 Cube                                    │
  │    3c. LLM 语义标注 (接收全局上下文 + 相邻子场景摘要):                      │
  │         • zones: 战术区域划分 (类型/边界/连通关系)                           │
  │         • openings: 门/窗/阳台口 (类型/宽度/高度/连接)                       │
  │         • cover_assessment: 掩体评估 + effective_positions                  │
  │           (behind/covers_from/exposed_to — 按有效利用位置标注)                  │
  │         • inferred_threats: 威胁推理 (利用全局上下文+相邻场景信息)           │
  │         • tactical_boundary: 战术状态边界 (entry/objective/transition)      │
  │         • exposure_assessment: 机动暴露度 (from/to/exposed_to/time/cover)   │
  │         • movement_analysis: 机动路径 + 关键控制点                          │
  │         • Prompt 内嵌自检清单: 确保标注完整性与一致性                        │
  │         • spatial_description: 100-200 字自然语言描述                      │
  │    • 坐标一致性自动校验 (validator)                                         │
  │  输入: Phase 0 scene_metadata.json + Phase 2 sub_scene_definitions.json    │
  │  输出: data/processed/sub_scenes/SS_XX/scene_cubes.json                     │
  │        data/processed/sub_scenes/SS_XX/desc.json                            │
  └──────────────────────────────────────────────────────────────────────────┘
        │
        ▼
  ┌──────────────────────────────────────────────────────────────────────────┐
  │  Phase 4: 战术生成 (Stage 0 穷举 → M1 → M2 → M3 → M4)                │
  │  ─────────────────────────────────────────                                │
  │  实现方式: 多智能体 LLM (A_gen / A_review / A_eval)                         │
  │  功能:                                                                    │
  │    M1 验证:                                                                │
  │         • desc.json 与 scene_cubes 一致性检查                               │
  │         • 提取战术标注 (空间挑战类型 / 威胁摘要 / 掩体摘要)                   │
  │    M2 策略:                                                                │
  │         • 计算 desc.json 嵌入与 PDF 章节的余弦相似度                         │
  │         • 判定生成模式: RAG (相似≥0.7) / Hybrid (0.3~0.7) / GEN (<0.3)       │
  │    Stage 0 穷举枚举 (默认启用):                                              │
  │         • 参考资料按段落分批注入 (每批 2000 字)                               │
  │         • LLM 迭代发现新战术方向 → 生成概念列表                               │
  │         • 字符二元组 Jaccard 去重 + 名称去重                                  │
  │         • 终止: 最多 15 个 / 连续 2 轮无参考无新产出                          │
  │    M3 迭代精炼 (每概念最多 3 轮):                                            │
  │         • A_gen: 将战术概念扩展为完整双版本 JSON                              │
  │         • A_review: 16 条通用性规则 (含 G-T10 绝对方向禁止 + G-T11 场景标注标识符禁止) + 军事可行性语义审查   │
  │         • 硬约束违规 → A_gen 修正; 通过 → 收敛/输出                          │
  │    M4 评估:                                                                │
  │         • A_eval 六维度评分 (每维度 Checklist 驱动)                          │
  │         • 否决条件: 军事可行/场景适配/粒度合规 < 3.0 → L 级不入库            │
  │         • 分级: H (≥8.0) / M (≥6.0) / L (<6.0 或否决)                      │
  │  输入: Phase 3 输出的 desc.json + scene_cubes.json                          │
  │  输出: data/processed/tactics/text_version/{H,M,L}/<tactic_id>.json         │
  │        data/processed/tactics/struct_version/{H,M,L}/<tactic_id>.json       │
  └──────────────────────────────────────────────────────────────────────────┘
        │
        ▼
  ┌──────────────────────────────────────────────────────────────────────────┐
  │  知识库入库 (ChromaDB)                                                     │
  │  ──────────────────────                                                   │
  │  • 战术 JSON → 嵌入向量 (本地 BAAI/bge-large-zh-v1.5, GPU 加速)             │
  │  • 写入 ChromaDB: tactics_text / tactics_struct Collection                  │
  │  • 支持语义检索: 场景描述 → 相似战术; PDF 章节 → 相关场景                    │
  └──────────────────────────────────────────────────────────────────────────┘
```

### 阶段概览

| 阶段 | 输入 | 处理 | 输出 | LLM 调用 |
|------|------|------|------|---------|
| P0 | `raw_scene.usda` | Python 流式解析 + 纯几何输出 | `scene_metadata.json` | 0 |
| P1 | `scene_metadata.json` | LLM 全局场景理解 | `global_understanding.json` | 1 |
| P2 | `global_understanding.json` | LLM 子场景划分 + 去重 + 人工审核 | `sub_scene_definitions.json` | 2 |
| P3 | P0 + P2 输出 | 裁剪 → 简化 → LLM 语义标注 → 校验 | `SS_XX/scene_cubes.json` + `desc.json` | N (每子场景 1 次) |
| P4 | P3 输出 | M1→M2→Stage 0 穷举→M3 逐概念精炼→M4 评估 | `tactics/*/{H,M,L}/*.json` | ~3 per 概念 (多概念) |

---

## 三、Phase 4 详细流程

Phase 4 是整个流水线的核心，细分为五个子模块：

```
desc.json + scene_cubes.json
        │
        ▼
   ┌──────────────────┐
   │ 双重泛化预处理     │  desc.json 绝对方向 → 功能/关系描述
   │ direction_generalizer│  + 物体身份 → 战术功能类别（安全网）
   └────────┬─────────┘
        │
        ▼
   ┌─────────┐
   │   M1    │  desc.json 一致性验证 + 战术标注提取
   │ validator│  输出: {ready_for_phase4, tactical_annotations}
   └────┬────┘
        │
        ▼
   ┌─────────┐
   │   M2    │  嵌入相似度 → RAG / Hybrid / GEN 模式判定
   │ strategy │  输出: {mode, reference_content}
   └────┬────┘
        │
        ▼
   ┌──────────────────────────────────────────────────────┐
   │              Stage 0: 穷举枚举 (默认)                 │
   │                                                      │
   │   参考资料按段落分批 (2000字/批)                        │
   │   ┌──────────┐     ┌──────────────┐                  │
   │   │ 参考批N  │ ──► │  LLM 发现缺口 │ ──► 新战术概念   │
   │   │ 已有战术 │     │  "还能生成什么"│                  │
   │   └──────────┘     └──────────────┘                  │
   │                                                      │
   │   去重: 名称匹配 + 字符二元组 Jaccard (>0.8 丢弃)      │
   │   终止: 最多 15 个 / 连续 2 轮无产出                    │
   │   支持 --single 回退单战术模式                          │
   └──────────────────────┬───────────────────────────────┘
                          │  N 个战术概念
                          ▼
   ┌───────────────────────────────────────────────────────┐
   │              M3 迭代精炼 (每概念 ≤3 轮)                 │
   │                                                       │
   │   ┌────────┐        文本提取        ┌──────────┐      │
   │   │ A_gen  │ ──►  (precheck)  ──►  │ A_review │      │
   │   │ 扩展/  │                       │ 16条通用性│      │
   │   │ 修正   │ ◄──────────────────── │ +军事审查 │      │
   │   └────────┘    修正指令反馈       └──────────┘      │
   │                                                       │
   │   全部 16 条规则由 A_review LLM 语义审查（无正则预检）  │
   │   硬约束违规 → A_gen 修正；通用性通过 → 军事审查        │
   │   军事 < 3.0 → 早期丢弃；score ≥ 7.0 且收敛 → 通过     │
   └───────────────────────┬───────────────────────────────┘
                           │
                           ▼
   ┌─────────┐
   │   M4    │  A_eval 六维度 54 项 Checklist 全项评分
   │ evaluate│  综合分 Q = (q1+...+q6)/6 → H/M/L
   └────┬────┘
        │
        ▼
  tactics/text_version/{H,M,L}/<tactic_id>.json
  tactics/struct_version/{H,M,L}/<tactic_id>.json
```

### M4 六维度评分体系

| 维度 | 核查项 | 否决条件 |
|------|--------|---------|
| 场景适配度 | S1-S8 (8 项) | < 3.0 → 否决 |
| 执行效率 | E1-E6 (6 项) | — |
| 可理解性 | C1-C7 (7 项) | — |
| 粒度合规性 | G1-G8 (8 项) | < 3.0 → 否决 |
| 图文一致性 | V1-V7 (7 项) | — |
| 军事可行性 | M1-M18 (18 项) | < 3.0 → 否决 |

### M3 通用性审查规则 (16 条，全部由 A_review LLM 语义审查)

**文字描述版 (G-T1~G-T11)**：检查 text_version 的 Description / Objective / Action_Sequence Intent / Visual_Aids

| 规则 | 约束级 | 内容 | 审查难点（正则无法可靠处理） |
|------|--------|------|------|
| G-T1 | 硬 | 禁止阿拉伯数字 | 区分步骤编号（"步骤1"不违规）与参数绑定（"3米"违规） |
| G-T2 | 硬 | 禁止中文数字 | 区分方向/顺序用字（"一侧"、"下一"）与数量用字（"三名"、"两翼"） |
| G-T3 | 硬 | 禁止数量词后缀 | 全量词覆盖：台/个/名/辆/架/组/队/挺/支/具/枚/门/发/次/轮/波 |
| G-T4 | 硬 | 禁止具体距离+单位 | 间接表达（"约三人身高"、"一臂之长"）也是违规 |
| G-T5 | 硬 | 禁止具体时间+单位 | 间接表达（"呼吸一次的时间"）也是违规 |
| G-T6 | 硬 | 禁止具体射击数量 | 口语化表达（"一梭子"、"一个弹匣"）也是违规 |
| G-T7 | 硬 | 禁止场景特定物体身份 | 具体物体类别（桌子/沙发/椅子/床/柜子/茶几/办公桌/花坛/空调外机）、颜色/品牌/型号/特定材质；必须使用战术功能类别（低矮掩体/大型遮挡物/柱状掩体/透明隔断等） |
| G-T8 | 软 | 禁止具体机器人编号 | 数字/字母编号；隐含指代（"领头的那个"）也是违规 |
| G-T9 | 软 | 执行主体必须用功能角色名 | 有效角色：突击手/压制射手/侦察节点/侧翼警戒手/断后掩护手/护卫手/留守节点/警戒手 |
| G-T10 | 硬 | 禁止绝对方向指代 | 禁止东/南/西/北+侧/端/段/翼/面/墙/角/缘/头/区/部/方/方向（如"走廊西端"、"北侧开口"、"北墙"）；必须替换为功能/关系描述（如"走廊远端"、"走廊一侧的开口"、"走廊一侧墙壁"）；方向作为角色职能时除外（如"侧翼警戒手"） |
| G-T11 | 硬 | 禁止场景标注标识符 | 检查是否出现含下划线的英文标识符（corridor_main）、字母+数字编号（SS_07）等场景标注标识符；战术中应使用空间类型描述替代 |

**结构化描述版 (G-S1~G-S5)**：检查 struct_version 的 Action_Sequence Instructions

| 规则 | 约束级 | 内容 |
|------|--------|------|
| G-S1 | 硬 | 目标位置必须用 `{...}` 占位符 |
| G-S2 | 硬 | 距离参数必须用 `{...}` 占位符 |
| G-S3 | 硬 | 时间参数必须用 `{...}` 占位符 |
| G-S4 | 硬 | 射击数量必须用 `{...}` 占位符 |
| G-S5 | 硬 | 执行主体必须用 `[Unit角色名]` 格式 |

> **设计决策**：全部 16 条规则由 A_review LLM 做语义审查。此前尝试用 Python 正则做前置预检，但中文文本的语义判断（区分步骤编号与参数绑定、区分方向用字与数量用字、区分绝对方向词与功能角色名中的方位字）正则无法可靠完成——假阴性多（漏检）、假阳性也多（误报），白名单/黑名单无限膨胀。precheck 模块仅提取文本字段供 LLM 逐条审查，不做任何规则判定。

---

## 四、快速开始

### 4.1 硬件要求

| 组件 | 最低要求 | 推荐配置 |
|------|---------|---------|
| GPU | NVIDIA RTX 3060 (12GB) | **RTX PRO 6000 (96GB)** |
| CUDA | 12.8+ (Blackwell) / 11.8+ (Ampere) | 12.8+ |
| 系统内存 | 32 GB | 64 GB+ |
| 磁盘 | 50 GB | 200 GB+ (PDF + ChromaDB) |

### 4.2 环境准备

**前置依赖**：
- Python 3.10+（推荐 3.11）
- CUDA 12.8+ (Blackwell GPU) 或 CUDA 11.8+ (Ampere/Ada Lovelace)
- Git LFS（本地嵌入模型下载需要）

```bash
# 1. 创建并激活虚拟环境
conda create -n robot_project2 python=3.11 -y
conda activate robot_project2

# 2. 安装 PyTorch（根据 GPU 选择）
# RTX PRO 6000 / Blackwell (sm_120):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# RTX 30/40 系列 (Ampere/Ada Lovelace):
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CPU only:
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 3. 安装项目依赖
cd robot_project2
pip install -r requirements.txt

# 4. 首次运行时，本地嵌入模型会自动下载 (~1.3GB)
# 模型: BAAI/bge-large-zh-v1.5
# 缓存路径: ~/.cache/huggingface/hub/

# 5. (无网络环境) 手动下载嵌入模型
# 在有网络的机器上:
pip install huggingface_hub
huggingface-cli download BAAI/bge-large-zh-v1.5 --local-dir ./bge-large-zh-v1.5
# 将下载的目录拷贝到项目 data/models/ 下:
#   robot_project2/data/models/bge-large-zh-v1.5/
# 或放到任意位置，通过环境变量指定:
#   export BGE_LOCAL_PATH=/your/path/to/bge-large-zh-v1.5
```

**配置 .env**：

```bash
cp .env.example .env
```

编辑 `.env`，配置 API Key：

```ini
# MiniMax API (LLM 调用, 必须)
# 获取地址: https://platform.minimaxi.com
MINIMAX_API_KEY=sk-cp-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# OpenAI API (嵌入模型备选, 可选)
# 默认使用本地 BGE 模型, 无需 API Key
# 如需使用 OpenAI text-embedding-3-large, 取消注释并填入:
# OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# ChromaDB (默认路径, 无需修改)
CHROMA_PERSIST_DIR=data/processed/chroma_db
```

### 4.3 嵌入模型配置

Phase 4 的 M2 策略判定依赖嵌入向量检索。系统支持三种嵌入后端，按优先级降级：

| 后端 | 模型 | 维度 | 需要 API Key | 适用场景 |
|------|------|------|-------------|---------|
| **本地 BGE** (默认) | `BAAI/bge-large-zh-v1.5` | 1024 | 否 | 离线环境、无需网络 |
| MiniMax | `embo-01` | 1536 | 是 | MiniMax API 用户 |
| OpenAI | `text-embedding-3-large` | 1024 | 是 | OpenAI API 用户 |

**Python API 用法**：

```python
from src.kb.embedding_client import EmbeddingClient

# 默认本地 BGE 模型（推荐，无需 API Key，GPU 自动加速）
emb = EmbeddingClient()

# 强制使用 MiniMax
emb = EmbeddingClient(use_minimax=True)

# 强制使用 OpenAI
emb = EmbeddingClient(use_openai=True)

# 生成嵌入向量
vec = emb.embed("房间突入战术描述文本")
```

**降级链**：`本地 BGE → MiniMax → OpenAI → stub 零向量`

如果默认的本地模型不可用（如文件损坏），自动尝试 MiniMax（需配置 API Key），再尝试 OpenAI（需配置 API Key）。如果所有后端都不可用，使用零向量 stub —— M2 将自动退化为 GEN 模式（纯 LLM 生成，不使用 PDF 参考资料）。

**切换嵌入模型后**，必须重建 ChromaDB：
```bash
python -m src.extract.pdf_preprocessor --force
```

### 4.4 PDF 预处理（启用 RAG 模式前必须执行）

RAG 检索增强生成依赖 PDF 参考资料的嵌入向量。首次运行前需要预处理：

```bash
# 将 data/raw/ 下的 PDF 参考资料转换为 ChromaDB 嵌入向量
python -m src.extract.pdf_preprocessor

# 若需重建（切换嵌入模型后必须重建）
python -m src.extract.pdf_preprocessor --force
```

### 4.5 一键运行全流水线

```bash
# 基础用法（LLM 自行推断任务目标）
python -m src.pipeline data/raw/scene.usda

# 指定任务描述（推荐 —— 同一场景不同任务产生不同战术）
python -m src.pipeline data/raw/scene.usda --task "进攻该三层建筑，解救人质。优先保证人质安全。"

# 限定作战阶段（仅生成该阶段战术）
python -m src.pipeline data/raw/scene.usda --task "进攻该三层建筑，解救人质" --mission-phase "进攻阶段"

# 从指定阶段开始（断点续跑）
python -m src.pipeline data/raw/scene.usda --from-phase 2 --task "侦察建筑外围"
```

### 4.6 逐阶段 CLI 运行

```bash
# Phase 0: USDA 解析（纯 Python，无 LLM 调用）
python -m src.phase0.pipeline data/raw/scene.usda -o data/processed/phase0

# Phase 1: 全局场景理解（1 次 LLM 调用）
python -m src.phase1.runner data/processed/phase0/scene_metadata.json \
    -o data/processed/phase1 -t "进攻该三层建筑，解救人质"

# Phase 2: 子场景划分（1 次 LLM 调用）
python -m src.phase2.runner data/processed/phase1/global_understanding.json \
    -o data/processed/phase2 -t "进攻该三层建筑，解救人质"

### 4.7 人工审核 (Phase 2)

Phase 2 完成后，流水线会自动提示可进行人工审核。这是 Phase 3（每子场景 1 次 LLM 调用）和 Phase 4（每子场景 3-5 次 LLM 调用）之前的最后一个低成本质量关卡：

```bash
# 启动 CLI 交互审核
python -m src.phase2.reviewer data/processed/phase2/sub_scene_definitions.json
```

**操作命令**：`a`=接受保存, `m`=修改子场景, `r`=重排序, `d`=删除, `q`=放弃退出

**审查检查清单**：

| 检查项 | 怎么看 |
|--------|--------|
| 是否遗漏战术区域 | 子场景覆盖所有楼层/房间/走廊/入口了吗 |
| primary_role 是否合理 | 房间→room_entry, 楼梯→vertical_insertion, 入口→entry_breach |
| suggested_roles 是否充分 | 考虑了多种进入方式吗（索降/架梯/正门/爆破） |
| 连通性是否有断裂 | 看自动警告：有没有孤立节点或孤儿引用 |
| 入口选项是否齐全 | entry_options 覆盖窗户/阳台/屋顶等非标准入口了吗 |
| task_hint 是否服务于总任务 | 每个子场景目标和"进攻建筑，解救人质"一致吗 |
| 优先级是否正确 | 入口/楼梯/要害房间=high, 普通房间=medium |
| spatial_bounds 是否完整 | bounds 包含门洞外走廊了吗，楼梯覆盖全高度了吗 |

审核通过并保存后，从 Phase 3 继续：
```bash
python -m src.pipeline data/raw/scene.usda --from-phase 3 --task "..."
```

### 4.8 Phase 3-4 CLI

```bash
# Phase 3: 几何处理 + 语义标注（每个子场景 1 次 LLM 调用）
python -m src.phase3.pipeline \
    data/processed/phase0/scene_metadata.json \
    data/processed/phase2/sub_scene_definitions.json \
    --output-dir data/processed/sub_scenes

# Phase 4: 战术生成（单个子场景，多战术穷举）
# 默认穷举模式 — 生成多个不重复战术
python -m src.phase4.pipeline \
    data/processed/sub_scenes/SS_01/desc.json \
    data/processed/sub_scenes/SS_01/scene_cubes.json \
    --mission-phase "进攻阶段"

# 单战术模式 — 只生成 1 条战术
python -m src.phase4.pipeline \
    data/processed/sub_scenes/SS_01/desc.json \
    data/processed/sub_scenes/SS_01/scene_cubes.json \
    --mission-phase "进攻阶段" --single
```

### 4.9 自定义输入输出路径

所有阶段的文件路径均可自由指定，无需遵循默认目录结构：

```bash
# 从任意位置读取 USDA，输出到任意位置
python -m src.phase0.pipeline /path/to/any_scene.usda -o /tmp/my_metadata

# 跨阶段串联：每个阶段的输出作为下一阶段的输入
python -m src.phase1.runner /tmp/my_metadata/scene_metadata.json -o /tmp/my_understanding
```

### 4.10 Python API 调用

```python
from src.core.llm_client import MiniMaxClient
from src.kb.embedding_client import EmbeddingClient
from src.kb.vector_store import VectorStore
from src.phase0.pipeline import run_phase0
from src.phase1.runner import run_phase1
from src.phase2.runner import run_phase2
from src.phase3.pipeline import run_phase3
from src.phase4.pipeline import run_phase4

client = MiniMaxClient()
emb_client = EmbeddingClient()    # 默认本地 BGE 模型, GPU 加速
vec_store = VectorStore()         # ChromaDB

task = "进攻该三层建筑，解救人质。优先保证人质安全。"

# Phase 0: 纯 Python，无 LLM
meta = run_phase0("data/raw/scene.usda")

# Phase 1: 1 次 LLM 调用，传入任务描述引导场景理解
understand = run_phase1(meta, client=client, task=task)

# Phase 2: 1 次 LLM 调用，子场景 task_hint 由总任务驱动
defs = run_phase2(understand, client=client, task=task)

# Phase 3: N 次 LLM 调用（每个子场景 1 次）
ss_results = run_phase3(meta, defs, client=client)

# Phase 4: 多战术穷举（默认）；传入嵌入客户端启用 RAG
# run_phase4 现在返回 list，每个子场景可生成多条不重复战术
# mission_phase 可选: "侦察阶段"|"进攻阶段"|"防御阶段"|"撤退与脱离阶段"
for ss in ss_results:
    results = run_phase4(
        ss["desc_path"], ss["usda_path"],
        client=client,
        embedding_client=emb_client,
        vector_store=vec_store,
        mission_phase="进攻阶段",
    )
    for r in results:
        print(f"{ss['sub_scene_id']}: {r['seed_concept_name']} ({r['quality_level']})")

# 单战术模式
for ss in ss_results:
    results = run_phase4(
        ss["desc_path"], ss["usda_path"],
        client=client, embedding_client=emb_client, vector_store=vec_store,
        mission_phase="进攻阶段", multi_tactic=False,
    )
```
### 4.11 使用 PipelineRunner

```python
from src.pipeline import PipelineRunner
from src.kb.embedding_client import EmbeddingClient
from src.kb.vector_store import VectorStore

runner = PipelineRunner(
    embedding_client=EmbeddingClient(),
    vector_store=VectorStore(),
    task="进攻该三层建筑，解救人质。优先保证人质安全。",
    mission_phase="进攻阶段",
)

summary = runner.run_all("data/raw/scene.usda")

# 单步执行
runner.step_phase0("data/raw/scene.usda")                     # → scene_metadata.json
runner.step_phase1("data/processed/phase0/scene_metadata.json")  # → global_understanding.json
runner.step_phase2("data/processed/phase1/global_understanding.json")  # → sub_scene_definitions.json
runner.step_phase3(".../scene_metadata.json", ".../sub_scene_definitions.json")  # → per-SS
runner.step_phase4(phase3_results)                            # → per-SS tactics
```

---

## 五、项目结构

```
robot_project2/
├── .env.example
├── .gitignore
├── README.md                    # 本文档
├── CLAUDE.md                    # Claude Code 配置
├── requirements.txt
│
├── data/
│   ├── raw/                     # 原始 USDA 文件 + 任务描述
│   └── processed/
│       ├── phase0/              # scene_metadata.json
│       ├── phase1/              # global_understanding.json
│       ├── phase2/              # sub_scene_definitions.json
│       ├── sub_scenes/          # SS_01/{scene_cubes.json, desc.json}, SS_02/, ...
│       ├── tactics/
│       │   ├── text_version/    # {H,M,L}/<tactic_id>.json  (文字描述版)
│       │   └── struct_version/  # {H,M,L}/<tactic_id>.json  (结构化描述版)
│       └── chroma_db/           # ChromaDB 持久化
│
├── src/
│   ├── pipeline.py              # 全流水线编排
│   ├── config.py                # 集中配置 (API/参数/阈值)
│   │
│   ├── core/                    # 跨阶段共享基础设施
│   │   ├── exceptions.py        # GTKGError 异常层次
│   │   ├── types.py             # Vec3, BBox, PrimRecord, CubeInfo
│   │   ├── geometry.py          # BBox 交并集、楼板面检测、平面判定
│   │   ├── usda_utils.py        # Brace 计数、矩阵累乘、extent 解析
│   │   └── llm_client.py        # MiniMaxClient (JSON 解析自动修复)
│   │
│   ├── phase0/                  # Phase 0: 流式 USDA 解析 (纯 Python)
│   │   ├── stream_parser.py     # 逐行扫描 + Prim 堆栈
│   │   ├── prim_builder.py      # 层级构建 + 世界变换
│   │   ├── transform_accumulator.py  # 变换累乘 + 自适应阈值
│   │   ├── heuristic_classifier.py   # 9 规则分类器（已弃用，不参与流水线）
│   │   ├── pattern_detector.py       # 等距排列/对称对/密集簇
│   │   ├── z_layer_analyzer.py       # Z层空间预聚类 + 楼梯启发式检测
│   │   ├── geometry_features.py     # 碎片分类 + 墙面线/楼板面/房间/走廊/密度楼梯
│   │   └── pipeline.py              # Phase 0 协调器
│   │
│   ├── phase1/                  # Phase 1: LLM 全局场景理解
│   │   ├── schemas.py           # 已弃用（空占位，Schema 定义在 prompts.py 中）
│   │   ├── data_compactor.py    # 场景元数据精简
│   │   ├── prompts.py           # A_parse System/User Prompt
│   │   └── runner.py            # LLM 调用 + 输出
│   │
│   ├── phase2/                  # Phase 2: 子场景划分
│   │   ├── schemas.py           # SubSceneDef 数据结构
│   │   ├── prompts.py           # 子场景划分 Prompt
│   │   ├── runner.py            # LLM 生成子场景定义
│   │   ├── dedup_scenes.py      # LLM 驱动战术等价去重
│   │   └── reviewer.py          # CLI 人工审核工具
│   │
│   ├── phase3/                  # Phase 3: 几何处理 + 语义标注
│   │   ├── schemas.py           # ZoneDef, OpeningDef, DescJson 等
│   │   ├── phase3a_cropper.py   # overlap_bounds 裁剪
│   │   ├── phase3b_simplifier.py # 合并/模式折叠/Z 归一化
│   │   ├── phase3c_prompts.py   # 语义标注 Prompt
│   │   ├── phase3c_annotator.py # LLM 语义标注器
│   │   ├── validator.py         # 坐标一致性自动检查
│   │   └── pipeline.py          # Phase 3 协调器
│   │
│   ├── phase4/                  # Phase 4: 战术生成核心
│   │   ├── direction_generalizer.py  # 双重泛化预处理 (绝对方向→功能描述 + 物体身份→战术功能类别)
│   │   ├── m1_validator.py      # desc.json 一致性 + 战术标注
│   │   ├── m2_strategy.py       # 双层自适应策略 (RAG/Hybrid/GEN)
│   │   ├── multi_tactic/        # Stage 0: 多战术穷举枚举
│   │   │   ├── exhaustive_prompts.py  # 穷举生成 Prompt
│   │   │   └── exhaustive_generator.py # 去重/分批/迭代控制
│   │   ├── m3_refinement/       # M3 迭代精炼
│   │   │   ├── agen_prompts.py       # A_gen 三模式变体 + seed_concept
│   │   │   ├── areview_prompts.py    # A_review 审查 Prompt
│   │   │   ├── aeval_prompts.py      # 精简 A_eval (保留供扩展)
│   │   │   ├── few_shot_examples.py  # 6 条 Few-Shot 对比示例
│   │   │   ├── precheck.py           # 审查上下文提取
│   │   │   ├── review_schema.py      # ReviewFeedback 数据结构
│   │   │   └── iteration_loop.py     # 辩论式迭代协议
│   │   ├── m4_evaluation/       # M4 质量评估
│   │   │   ├── evaluator.py          # A_eval 六维度全项评分
│   │   │   ├── quality_classifier.py  # H/M/L 分级 + 否决
│   │   │   └── eval_schema.py        # EvalResult 数据结构
│   │   └── pipeline.py          # Phase 4 协调器 (M1→M2→Stage0→M3→M4)
│   │
│   ├── extract/                  # PDF 预处理 (离线)
│   │   ├── pdf_extractor.py     # PyMuPDF 文本提取
│   │   ├── text_chunker.py      # 章节+固定大小分块
│   │   └── pdf_preprocessor.py  # 编排: 提取→分块→嵌入→ChromaDB
│   │
│   └── kb/                      # 知识库 (ChromaDB)
│       ├── embedding_client.py  # 本地 BGE / MiniMax / OpenAI 嵌入
│       ├── vector_store.py      # ChromaDB 封装 (cosine 距离)
│       ├── retriever.py         # 战术 + PDF 章节检索
│       └── ingestion.py         # 战术入库
│
└── tests/
    ├── test_phase0_stream_parser.py
    ├── test_phase0_classifier.py
    ├── test_phase0_pattern_detector.py
    ├── test_phase3b_simplifier.py
    └── test_phase4_m3_iteration.py
```

---

## 六、数据目录结构

```
data/
├── raw/
│   ├── scene.usda              # 原始 Blender 导出的大场景
│   ├── *.pdf                   # 战术参考资料 PDF（Phase 4 M2 RAG 检索用）
│   └── task_description.txt    # 任务描述文本（可选）
│
└── processed/
    ├── phase0/
    │   └── scene_metadata.json  # 几何体列表 + 楼板面证据 + 房间/走廊/楼梯检测 + 模式
    ├── phase1/
    │   └── global_understanding.json  # 建筑类型/楼层/空间布局/战术地物
    ├── phase2/
    │   └── sub_scene_definitions.json  # 子场景列表 + 连通图
    ├── sub_scenes/
    │   ├── SS_01/
    │   │   ├── scene_cubes.json  # 简化后的 Cube 几何数据
    │   │   └── desc.json         # 语义标注 (zones/openings/cover/threats)
    │   ├── SS_02/
    │   │   ├── scene_cubes.json
    │   │   └── desc.json
    │   └── ...
    ├── tactics/
    │   ├── text_version/
    │   │   ├── H/               # 高质量：Q ≥ 8.0
    │   │   ├── M/               # 中质量：6.0 ≤ Q < 8.0
    │   │   └── L/               # 低质量：不入库
    │   └── struct_version/
    │       ├── H/
    │       ├── M/
    │       └── L/
    └── chroma_db/               # ChromaDB 持久化文件
```

---

## 七、配置说明

`src/config.py` 中所有可配置参数：

| 参数类别 | 说明 |
|---------|------|
| `MINIMAX_API_KEY` / `MINIMAX_MODEL` | LLM API 配置 |
| `LLM_TEMPERATURE` / `LLM_MAX_TOKENS` | 生成参数 |
| `EMBEDDING_MODEL` / `EMBEDDING_DIMENSION` | 嵌入模型配置 (默认本地 BGE, GPU 加速) |
| `EMBEDDING_MODEL_LOCAL_PATH` | BGE 模型本地路径 (默认 `data/models/bge-large-zh-v1.5/`, 可通过 `BGE_LOCAL_PATH` 环境变量覆盖) |
| `MINIMAX_EMBEDDING_MODEL` | MiniMax 嵌入备选 (需 API Key) |
| `CHROMA_PERSISTENCE_DIR` | ChromaDB 持久化路径 |
| `PHASE0_HEURISTIC_PARAMS` | Phase 0 分类阈值（已弃用，classifier 当前管线不调用） |
| `PHASE3B_AGGREGATION_PARAMS` | Phase 3b 合并/折叠参数 (含 gap_safe_merge: >0.5m 间隙保护) |
| `FLOOR_DETECTION_PARAMS` | 楼板面证据/楼层间距/逻辑楼层合并(N-1) |
| `STAIRCASE_DETECTION_PARAMS` | 楼梯检测参数（含厚度/密度阈值） |
| `STAIRCASE_DENSITY_PARAMS` | 跨层密度楼梯检测参数 |
| `M2_THETA_LOW` / `M2_THETA_HIGH` | M2 策略判定阈值 |
| `M3_ITERATION_PARAMS` | M3 最大轮次/收敛阈值/丢弃阈值 |
| `M4_EVALUATION_PARAMS` | M4 质量分级与否决条件 |
| `FEW_SHOT_CONFIG` | Few-Shot 示例注入策略 |

---

## 八、运行测试

```bash
python -m pytest tests/ -v
```

---

## 九、关键技术决策

1. **阶段间文件解耦**：每个 Phase 独立读写文件，可断点续跑，可单独调试
2. **Phase 0 纯几何输出 + 结构特征提取**：不做启发式语义分类（triangulated mesh 使规则不可靠），改为提取纯几何结构特征——楼板面证据+N-1楼层逻辑、墙面线、按楼层房间检测、走廊候选、双方法楼梯检测。LLM 从空间分布和结构特征推理建筑语义
3. **schemas 在各自 phase 内定义**：不引入跨阶段不必要的耦合
4. **双版本输出**：文字描述版面向 VLA，结构化描述版面向上层执行系统
5. **16 条规则全部 LLM 语义审查**：废止正则预检——中文文本中的"步骤1"与"3米"、方向"一侧"与数量"三名"、绝对方向"西端"与角色"侧翼警戒手"、场景标识符"corridor_main"与空间类型描述"走廊区域"只有语义判断能可靠区分
6. **双重不变性哲学**：从"禁止什么"升级为"理解为什么"——旋转不变性（空间关系层面）+ 物体功能不变性（物体描述层面），三层防线（预处理安全网 + A_gen 提示词教学 + A_review 硬约束审查）确保战术跨场景、跨朝向、跨物体身份迁移

---

## 十、许可

Internal use only.
