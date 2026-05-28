"""
M3 A_eval 质量评估 Agent System Prompt（已弃用 / DEPRECATED）

当前实现使用 M4 evaluator 评估全部 6 维度。
此文件保留仅为历史参考。如果你在寻找活跃的 A_eval，
请查看 src/phase4/m4_evaluation/evaluator.py（M4 全维度评估器）。

M4 评估器覆盖维度一~六（场景适配度/执行效率/可理解性/粒度合规性/
图文一致性/军事可行性），使用 rubric.py 的共享评分锚点和 Checklist。
此文件的过时内容（G-T1~G-T9, M1-M8, G1-G5）不应在任何活跃代码路径中使用。
"""

# 以下过时 prompt 已替换为占位注释，防止误用。
# 原始内容仅包含维度四（G1-G5）和维度六（M1-M8）的简化评估，
# 缺少 G-T10~G-T11, G-S1~G-S5, M9-M18, G6-G8 等扩展内容。

AEVAL_SYSTEM_PROMPT = (
    "# DEPRECATED: 此 prompt 已过时。\n"
    "# M3 迭代循环当前使用 A_review（areview_prompts.py），\n"
    "# M4 质量评估使用 evaluator.py + rubric.py 的全维度评估框架。\n"
    "# 如需修改评估逻辑，请编辑 src/phase4/m4_evaluation/ 下的文件。\n"
)
