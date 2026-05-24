"""
Phase 1 执行器

调用 LLM 生成全局场景理解 JSON。
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

from ..core.llm_client import MiniMaxClient
from ..core.exceptions import PhaseError
from ..config import PHASE1_DIR
from .data_compactor import compact_scene_metadata, format_for_llm
from .prompts import PHASE1_SYSTEM_PROMPT, build_phase1_user_prompt

logger = logging.getLogger(__name__)


def run_phase1(
    scene_metadata_path: str,
    client: Optional[MiniMaxClient] = None,
    output_dir: Optional[str] = None,
    task: Optional[str] = None,
) -> str:
    """Phase 1 主入口

    Args:
        scene_metadata_path: Phase 0 输出的 scene_metadata.json 路径
        client: MiniMaxClient 实例（可选，无则自动创建）
        output_dir: 输出目录
        task: 任务描述（1-3句自然语言，可选），用于引导场景理解方向。
              若不提供，LLM 做通用场景理解。

    Returns:
        global_understanding.json 文件路径
    """
    output_dir = Path(output_dir) if output_dir else PHASE1_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Phase 1 开始: %s", scene_metadata_path)

    # 1. 加载 Phase 0 输出
    with open(scene_metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    # 2. 精简数据
    logger.info("精简场景元数据 ...")
    compact_data = compact_scene_metadata(metadata)

    # 3. 格式化 LLM 输入
    llm_input = format_for_llm(compact_data)

    # 4. 调用 LLM
    logger.info("调用 LLM 全局场景理解 ...")
    if client is None:
        client = MiniMaxClient()

    user_prompt = build_phase1_user_prompt(llm_input, task=task)
    result = client.generate_json(
        system_prompt=PHASE1_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        temperature=0.3,
    )

    # 将 task 写入输出以便追溯
    if task:
        result["_task"] = task

    # 5. 输出
    output_path = output_dir / "global_understanding.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    logger.info("Phase 1 完成: %s", output_path)
    return str(output_path)


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
        description="Phase 1: LLM 全局场景理解 → global_understanding.json"
    )
    parser.add_argument(
        "scene_metadata_path",
        help="Phase 0 输出的 scene_metadata.json 路径",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default=None,
        help=f"输出目录 (默认: {PHASE1_DIR})",
    )
    parser.add_argument(
        "--task", "-t",
        default=None,
        help="任务描述（可选），引导场景理解聚焦于任务相关特征",
    )
    args = parser.parse_args()

    if not Path(args.scene_metadata_path).exists():
        print(f"错误: 文件不存在: {args.scene_metadata_path}", file=sys.stderr)
        sys.exit(1)

    output = run_phase1(args.scene_metadata_path,
                        output_dir=args.output_dir, task=args.task)
    print(f"Phase 1 完成: {output}")
