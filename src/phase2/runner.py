"""
Phase 2 执行器

调用 LLM 生成子场景定义。
"""

import json
import logging
from pathlib import Path
from typing import Optional

from ..core.llm_client import MiniMaxClient
from ..config import PHASE2_DIR
from .prompts import PHASE2_SYSTEM_PROMPT, build_phase2_user_prompt

logger = logging.getLogger(__name__)


def run_phase2(
    global_understanding_path: str,
    client: Optional[MiniMaxClient] = None,
    output_dir: Optional[str] = None,
    task: Optional[str] = None,
) -> str:
    """Phase 2 主入口

    Args:
        global_understanding_path: Phase 1 输出的 global_understanding.json
        client: MiniMaxClient 实例
        output_dir: 输出目录
        task: 任务描述（1-3句自然语言，可选）。若提供，LLM 将以此为指导
              为每个子场景分配 task_hint，而非从几何猜测。

    Returns:
        sub_scene_definitions.json 文件路径
    """
    output_dir = Path(output_dir) if output_dir else PHASE2_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Phase 2 开始: %s", global_understanding_path)

    # 1. 加载 Phase 1 输出
    with open(global_understanding_path, "r", encoding="utf-8") as f:
        understanding = json.load(f)

    # 获取 Phase 1 中存储的 task（作为 fallback）
    if not task:
        task = understanding.get("_task")

    understanding_str = json.dumps(understanding, ensure_ascii=False, indent=2)

    # 2. 调用 LLM
    logger.info("调用 LLM 划分子场景 ...")
    if client is None:
        client = MiniMaxClient()

    user_prompt = build_phase2_user_prompt(understanding_str, task=task)
    result = client.generate_json(
        system_prompt=PHASE2_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        temperature=0.3,
    )
    # generate_json 可能返回 list（Extra data 修复路径）
    if isinstance(result, list):
        result = result[0] if len(result) > 0 and isinstance(result[0], dict) else {}
    if not isinstance(result, dict):
        result = {}

    # 将全局 task 存储到输出，确保每个子场景继承任务上下文
    if task:
        result["_task"] = task
        for ss in result.get("sub_scenes", []):
            if not ss.get("task_hint"):
                ss["task_hint"] = task

    # 后处理：向后兼容——如果 LLM 用了旧字段名 tactical_role，转为 primary_role
    for ss in result.get("sub_scenes", []):
        if "tactical_role" in ss and "primary_role" not in ss:
            ss["primary_role"] = ss.pop("tactical_role")
        if not ss.get("primary_role"):
            suggested = ss.get("suggested_roles", [])
            ss["primary_role"] = suggested[0] if suggested else "unknown"

    # 3. 后处理：为每个子场景计算 overlap_bounds（spatial_bounds 每侧 +5m）
    _add_overlap_bounds(result)

    # 4. 输出
    output_path = output_dir / "sub_scene_definitions.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    logger.info("Phase 2 完成: %s (共 %d 个子场景)",
                output_path, len(result.get("sub_scenes", [])))
    return str(output_path)


def _add_overlap_bounds(result: dict, expansion: float = 5.0):
    """为每个子场景添加 overlap_bounds（spatial_bounds 每侧外扩 expansion 米）

    符合概念模型 §6.2: overlap_bounds 确保 Phase 3 裁切时边界几何完整。
    """
    for ss in result.get("sub_scenes", []):
        sb = ss.get("spatial_bounds", {})
        if sb:
            ss["overlap_bounds"] = {
                axis: [v[0] - expansion, v[1] + expansion]
                for axis, v in sb.items()
            }


# ── CLI 入口 ────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Phase 2: LLM 子场景划分 → sub_scene_definitions.json"
    )
    parser.add_argument(
        "global_understanding_path",
        help="Phase 1 输出的 global_understanding.json 路径",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default=None,
        help=f"输出目录 (默认: {PHASE2_DIR})",
    )
    parser.add_argument(
        "--task", "-t",
        default=None,
        help="任务描述（可选，也可从 Phase 1 输出中的 _task 字段读取）",
    )
    args = parser.parse_args()

    if not Path(args.global_understanding_path).exists():
        print(f"错误: 文件不存在: {args.global_understanding_path}", file=sys.stderr)
        sys.exit(1)

    output = run_phase2(args.global_understanding_path,
                        output_dir=args.output_dir, task=args.task)
    print(f"Phase 2 完成: {output}")
    print(f"[提示] 人工审核: python -m src.phase2.reviewer \"{output}\"")
