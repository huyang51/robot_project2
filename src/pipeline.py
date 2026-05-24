"""
GTKG-CM 全流水线编排

串联 Phase 0 → Phase 1 → Phase 2 → Phase 3 → Phase 4，
将原始 USDA 文件转化为战术战法知识库。

用法:
    python -m src.pipeline data/raw/scene.usda
    python -m src.pipeline data/raw/scene.usda --task "进攻该建筑，解救人质"
    python -m src.pipeline data/raw/scene.usda --from-phase 2 --task "侦察建筑外围"
"""

import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Any

from .config import (
    PHASE0_DIR, PHASE1_DIR, PHASE2_DIR, SUB_SCENES_DIR,
    TACTICS_TEXT_DIR, TACTICS_STRUCT_DIR,
    ensure_dirs,
)
from .core.llm_client import MiniMaxClient
from .core.exceptions import GTKGError

# Phase 入口
from .phase0.pipeline import run_phase0
from .phase1.runner import run_phase1
from .phase2.runner import run_phase2
from .phase3.pipeline import run_phase3
from .phase4.pipeline import run_phase4, run_phase4_batch

logger = logging.getLogger(__name__)


class PipelineRunner:
    """全流水线编排器"""

    def __init__(
        self,
        client: Optional[MiniMaxClient] = None,
        embedding_client=None,
        vector_store=None,
        task: Optional[str] = None,
        mission_phase: Optional[str] = None,
    ):
        self.client = client or MiniMaxClient()
        self.embedding_client = embedding_client
        self.vector_store = vector_store
        self.task = task
        self.mission_phase = mission_phase  # 作战阶段约束

    def _ensure_rag(self):
        """懒加载嵌入客户端和向量存储（用于 M2 PDF 检索）"""
        if self.embedding_client is None or self.vector_store is None:
            try:
                from .kb.embedding_client import EmbeddingClient
                from .kb.vector_store import VectorStore
                self.embedding_client = self.embedding_client or EmbeddingClient()
                self.vector_store = self.vector_store or VectorStore()
            except ImportError as e:
                logger.warning("无法加载 RAG 组件 (chromadb?): %s", e)
                self.embedding_client = None
                self.vector_store = None

    # ── 单阶段运行 ──────────────────────────────────────────

    def step_phase0(self, usda_path: str) -> str:
        """Phase 0: USDA 流式解析 → scene_metadata.json"""
        logger.info("=" * 50)
        logger.info("Phase 0: 流式 USDA 解析")
        logger.info("=" * 50)
        t0 = time.time()
        path = run_phase0(usda_path)
        elapsed = time.time() - t0
        logger.info(f"Phase 0 完成 ({elapsed:.1f}s): {path}")
        return path

    def step_phase1(self, scene_metadata_path: str, task: Optional[str] = None) -> str:
        """Phase 1: scene_metadata.json → global_understanding.json"""
        logger.info("=" * 50)
        logger.info("Phase 1: LLM 全局场景理解")
        if task:
            logger.info("  任务描述: %s", task)
        logger.info("=" * 50)
        t0 = time.time()
        path = run_phase1(scene_metadata_path, client=self.client, task=task)
        elapsed = time.time() - t0
        logger.info(f"Phase 1 完成 ({elapsed:.1f}s): {path}")
        return path

    def step_phase2(self, global_understanding_path: str, task: Optional[str] = None) -> str:
        """Phase 2: global_understanding.json → sub_scene_definitions.json"""
        logger.info("=" * 50)
        logger.info("Phase 2: LLM 子场景划分")
        if task:
            logger.info("  任务描述: %s", task)
        logger.info("=" * 50)
        t0 = time.time()
        path = run_phase2(global_understanding_path, client=self.client, task=task)
        elapsed = time.time() - t0
        logger.info(f"Phase 2 完成 ({elapsed:.1f}s): {path}")
        # 提示人工审查可用
        logger.info("💡 提示: 运行 python -m src.phase2.reviewer \"%s\" 进行人工审核", path)
        return path

    def step_phase3(
        self, scene_metadata_path: str, sub_scene_definitions_path: str,
    ) -> List[Dict[str, str]]:
        """Phase 3: 裁剪 + 简化 + 语义标注 → per-SS scene_cubes.json + desc.json"""
        logger.info("=" * 50)
        logger.info("Phase 3: 几何处理 + 语义标注")
        logger.info("=" * 50)
        t0 = time.time()
        results = run_phase3(
            scene_metadata_path,
            sub_scene_definitions_path,
            client=self.client,
        )
        elapsed = time.time() - t0
        logger.info(f"Phase 3 完成 ({elapsed:.1f}s): {len(results)} 个子场景")
        return results

    def step_phase4(self, phase3_results: List[Dict[str, str]]) -> List[Dict]:
        """Phase 4: 对每个子场景生成战术 (M1→M2→M3→M4)"""
        logger.info("=" * 50)
        logger.info(f"Phase 4: 战术生成 ({len(phase3_results)} 个子场景)")
        logger.info("=" * 50)

        # 确保 RAG 组件已加载（M2 PDF 检索需要）
        self._ensure_rag()

        all_results = []
        success_count = 0
        t0 = time.time()

        for i, ss in enumerate(phase3_results):
            ss_id = ss["sub_scene_id"]
            logger.info(f"--- Phase 4 [{i + 1}/{len(phase3_results)}] {ss_id} ---")

            try:
                result = run_phase4(
                    desc_json_path=ss["desc_path"],
                    scene_cubes_path=ss["usda_path"],
                    client=self.client,
                    embedding_client=self.embedding_client,
                    vector_store=self.vector_store,
                    mission_phase=self.mission_phase,
                )
                result["sub_scene_id"] = ss_id
                all_results.append(result)
                success_count += 1
                logger.info(
                    f"  {ss_id}: quality={result['quality_level']}, "
                    f"score={result['m4_result'].get('overall_score', '?')}"
                )
            except Exception as e:
                logger.error(f"  {ss_id} 失败: {e}", exc_info=True)
                all_results.append({"sub_scene_id": ss_id, "error": str(e)})

        elapsed = time.time() - t0
        logger.info(
            f"Phase 4 完成 ({elapsed:.1f}s): "
            f"{success_count}/{len(phase3_results)} 成功"
        )
        return all_results

    # ── 全流水线 ────────────────────────────────────────────

    def run_all(self, usda_path: str, start_phase: int = 0,
                task: Optional[str] = None) -> Dict[str, Any]:
        """执行完整流水线

        Args:
            usda_path: 原始 USDA 文件路径
            start_phase: 从哪一阶段开始 (0-4)，用于断点续跑
            task: 任务描述（1-3句自然语言），传达指挥官意图和任务约束。
                  若不提供，LLM 将从场景几何自行推断（准确度受限）。

        Returns:
            包含各阶段输出路径和统计的摘要字典
        """
        # 使用传入的 task 或构造函数中的 task
        task = task or self.task
        ensure_dirs()

        if start_phase < 0 or start_phase > 4:
            raise ValueError(f"start_phase 必须在 0-4 之间，当前值: {start_phase}")

        summary: Dict[str, Any] = {
            "usda_path": usda_path,
            "task": task,
            "start_phase": start_phase,
            "phases": {},
        }

        # ── Phase 0 ──────────────────────────────────────────
        if start_phase <= 0:
            result = self.step_phase0(usda_path)
            summary["phases"]["phase0"] = {"output": result, "status": "ok"}
            scene_metadata_path = result
        else:
            scene_metadata_path = str(PHASE0_DIR / "scene_metadata.json")
            if not Path(scene_metadata_path).exists():
                raise FileNotFoundError(
                    f"断点续跑需要 Phase 0 输出，但文件不存在: {scene_metadata_path}"
                )
            logger.info(f"跳过 Phase 0，使用已有输出: {scene_metadata_path}")

        # ── Phase 1 ──────────────────────────────────────────
        if start_phase <= 1:
            result = self.step_phase1(scene_metadata_path, task=task)
            summary["phases"]["phase1"] = {"output": result, "status": "ok"}
            global_understanding_path = result
        else:
            global_understanding_path = str(PHASE1_DIR / "global_understanding.json")
            if not Path(global_understanding_path).exists():
                raise FileNotFoundError(
                    f"断点续跑需要 Phase 1 输出，但文件不存在: {global_understanding_path}"
                )
            logger.info(f"跳过 Phase 1，使用已有输出: {global_understanding_path}")

        # ── Phase 2 ──────────────────────────────────────────
        if start_phase <= 2:
            result = self.step_phase2(global_understanding_path, task=task)
            summary["phases"]["phase2"] = {"output": result, "status": "ok"}
            sub_scene_defs_path = result
        else:
            sub_scene_defs_path = str(PHASE2_DIR / "sub_scene_definitions.json")
            if not Path(sub_scene_defs_path).exists():
                raise FileNotFoundError(
                    f"断点续跑需要 Phase 2 输出，但文件不存在: {sub_scene_defs_path}"
                )
            logger.info(f"跳过 Phase 2，使用已有输出: {sub_scene_defs_path}")

        # ── Phase 3 ──────────────────────────────────────────
        if start_phase <= 3:
            phase3_results = self.step_phase3(scene_metadata_path, sub_scene_defs_path)
            summary["phases"]["phase3"] = {
                "output": str(SUB_SCENES_DIR / "phase3_summary.json"),
                "sub_scene_count": len(phase3_results),
                "status": "ok",
            }
        else:
            phase3_summary_path = SUB_SCENES_DIR / "phase3_summary.json"
            if not phase3_summary_path.exists():
                raise FileNotFoundError(
                    f"断点续跑需要 Phase 3 输出，但文件不存在: {phase3_summary_path}"
                )
            with open(phase3_summary_path, "r", encoding="utf-8") as f:
                phase3_summary = json.load(f)
            phase3_results = phase3_summary.get("sub_scenes", [])
            logger.info(f"跳过 Phase 3，使用已有输出: {len(phase3_results)} 个子场景")

        # ── Phase 4 ──────────────────────────────────────────
        if start_phase <= 4:
            phase4_results = self.step_phase4(phase3_results)
            h_count = sum(1 for r in phase4_results if r.get("quality_level") == "H")
            m_count = sum(1 for r in phase4_results if r.get("quality_level") == "M")
            l_count = sum(1 for r in phase4_results if r.get("quality_level") == "L")
            err_count = sum(1 for r in phase4_results if "error" in r)
            summary["phases"]["phase4"] = {
                "total": len(phase4_results),
                "H": h_count,
                "M": m_count,
                "L": l_count,
                "errors": err_count,
                "status": "ok",
            }
        else:
            logger.info("跳过 Phase 4")

        # ── 写入总摘要 ────────────────────────────────────────
        summary_path = Path(usda_path).parent.parent / "processed" / "pipeline_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        self._print_summary(summary)
        return summary

    # ── 辅助 ────────────────────────────────────────────────

    def _print_summary(self, summary: Dict[str, Any]) -> None:
        """输出可读摘要"""
        print("\n" + "=" * 60)
        print("GTKG-CM 流水线执行摘要")
        print("=" * 60)
        print(f"  输入文件: {summary['usda_path']}")

        for phase_name in ["phase0", "phase1", "phase2", "phase3", "phase4"]:
            info = summary["phases"].get(phase_name)
            if not info:
                print(f"  {phase_name}: 跳过")
                continue
            status = info.get("status", "?")
            if phase_name == "phase4":
                print(
                    f"  {phase_name}: {status} "
                    f"(总数={info.get('total', 0)}, "
                    f"H={info.get('H', 0)}, M={info.get('M', 0)}, "
                    f"L={info.get('L', 0)}, 错误={info.get('errors', 0)})"
                )
            elif phase_name == "phase3":
                print(
                    f"  {phase_name}: {status} "
                    f"(子场景={info.get('sub_scene_count', 0)})"
                )
            else:
                output = info.get("output", "")
                print(f"  {phase_name}: {status} → {output}")

        print("=" * 60)


# ── CLI 入口 ────────────────────────────────────────────────


def main():
    """命令行入口"""
    import argparse

    parser = argparse.ArgumentParser(
        description="GTKG-CM 全流水线：USDA → 战术知识库"
    )
    parser.add_argument(
        "usda_path",
        help="原始 USDA 场景文件路径",
    )
    parser.add_argument(
        "--from-phase", "-f",
        type=int, default=0,
        help="从指定阶段开始（0-4），用于断点续跑 (default: 0)",
    )
    parser.add_argument(
        "--task", "-t",
        default=None,
        help="任务描述（1-3句自然语言），传达指挥官意图和任务约束。"
             "例: '进攻该三层建筑，解救人质。优先保证人质安全。'",
    )
    parser.add_argument(
        "--mission-phase", "-m",
        default=None,
        choices=["侦察阶段", "进攻阶段", "防御阶段", "撤退与脱离阶段"],
        help="作战阶段约束（可选）。仅生成指定阶段的战术。"
             "不指定则由 LLM 根据子场景自行判定。",
    )
    parser.add_argument(
        "--log-level", "-l",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志级别 (default: INFO)",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    usda_path = args.usda_path
    if not Path(usda_path).exists():
        print(f"错误: 文件不存在: {usda_path}", file=sys.stderr)
        sys.exit(1)

    runner = PipelineRunner(task=args.task, mission_phase=args.mission_phase)
    try:
        runner.run_all(usda_path, start_phase=args.from_phase, task=args.task)
    except GTKGError as e:
        print(f"流水线错误: {e}", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError as e:
        print(f"断点续跑错误: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
