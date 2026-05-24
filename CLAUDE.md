# robot_project2 - GTKG-CM Pipeline

## 项目概述

- **项目路径**: D:\claude__code\robot_project2
- **项目目标**: 实现生成式战法战术知识生成概念模型 (GTKG-CM) 五阶段流水线
- **输入**: 原始 USDA 大场景文件 + 任务描述
- **输出**: 双版本战术 JSON (text + struct) + 知识库 (ChromaDB)

## 技术栈

- **编程语言**: Python 3.10+
- **LLM**: MiniMax API (MiniMax-M2.7)
- **嵌入**: OpenAI text-embedding-3-large / bge-large-zh-v1.5
- **向量数据库**: ChromaDB
- **场景格式**: USD/USDA (Universal Scene Description)

## 五阶段流水线

```
Phase 0 (Python):  raw_scene.usda → scene_metadata.json
Phase 1 (LLM):    scene_metadata.json → global_understanding.json
Phase 2 (LLM+人):  global_understanding.json → sub_scene_definitions.json
Phase 3 (Python+LLM): sub_scene_defs + raw USDA → per SS: scene_cubes.json + desc.json
Phase 4 (LLM):    desc.json + task_hint → tactic.json (text + struct) + quality → KB
```

## 文件结构

```
robot_project2/
├── src/
│   ├── config.py           # 集中配置
│   ├── core/               # 跨阶段基础设施
│   │   ├── exceptions.py   # GTKGError 层次
│   │   ├── types.py        # Vec3, BBox, PrimRecord
│   │   ├── geometry.py     # 几何工具
│   │   ├── usda_utils.py   # USDA 解析工具
│   │   └── llm_client.py   # MiniMax API 封装
│   ├── phase0/             # 流式 USDA 解析
│   ├── phase1/             # LLM 全局场景理解
│   ├── phase2/             # 子场景划分
│   ├── phase3/             # 几何处理 + 语义标注
│   ├── phase4/             # 战术生成核心
│   │   └── m3_refinement/  # 迭代精炼 (A_gen ↔ A_review)
│   │   └── m4_evaluation/  # 质量评估 (A_eval)
│   └── kb/                 # 知识库 (ChromaDB)
├── data/
├── tests/
└── requirements.txt
```

## 关键设计决策

1. **Phase 独立子包**: 清晰的阶段边界
2. **从 robot_project 适配而非复制**: 核心逻辑保留，清理遗留引用
3. **schemas 在各 phase 内定义**: 避免不必要的跨阶段耦合
4. **pipeline.py 作为协调器**: 每阶段单一入口
5. **双版本输出**: text_version (VLA) + struct_version (可执行)
