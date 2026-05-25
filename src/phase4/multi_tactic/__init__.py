"""
Phase 4 多战术穷举生成模块

参考 robot_project TacticGeneratorV2 的穷举生成策略:
1. 参考资料分批注入（每批 2000 字）
2. 迭代调用 LLM 发现新的战术方向
3. Python 层基于名称 + Jaccard 词重叠去重
4. 终止条件: 最多 15 个战术 / 连续 2 轮无新战术
"""
