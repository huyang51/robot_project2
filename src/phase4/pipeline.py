"""
Phase 4 协调器

对外暴露单一入口 run_phase4()，
协调 M1→M2→M3→M4 完整战术生成流水线。

双版本输出：
- text_version (文字描述版) → tactics/text_version/{H,M,L}/
- struct_version (结构化描述版) → tactics/struct_version/{H,M,L}/
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

from ..core.llm_client import MiniMaxClient
from ..config import TACTICS_TEXT_DIR, TACTICS_STRUCT_DIR
from .m1_validator import run_m1
from .m2_strategy import run_m2
from .m3_refinement.iteration_loop import M3IterationLoop
from .m4_evaluation.evaluator import evaluate_tactic
from .m4_evaluation.quality_classifier import classify_quality

logger = logging.getLogger(__name__)


def run_phase4(
    desc_json_path: str,
    scene_cubes_path: str,
    client: Optional[MiniMaxClient] = None,
    embedding_client=None,
    vector_store=None,
    mission_phase: Optional[str] = None,
) -> Dict[str, Any]:
    """Phase 4 主入口：单子场景端到端战术生成

    Args:
        desc_json_path: Phase 3 输出的 desc.json 路径
        scene_cubes_path: Phase 3 输出的 scene_cubes.json 路径
        client: MiniMaxClient
        embedding_client: 嵌入客户端（M2 用）
        vector_store: 向量存储（M2 用）
        mission_phase: 作战阶段约束，仅生成该阶段的战术。
                       "侦察阶段"|"进攻阶段"|"防御阶段"|"撤退与脱离阶段"
                       若为 None，由 LLM 根据子场景自行判定。

    Returns:
        {
            "text_version": {...},
            "struct_version": {...},
            "m1_result": {...},
            "m2_result": {...},
            "m3_feedback": {...},
            "m4_result": {...},
            "quality_level": "H|M|L",
            "output_paths": {"text": str, "struct": str},
        }
    """
    if client is None:
        client = MiniMaxClient()

    # 1. 加载数据
    with open(desc_json_path, "r", encoding="utf-8") as f:
        desc_json = json.load(f)
    with open(scene_cubes_path, "r", encoding="utf-8") as f:
        scene_data = json.load(f)

    scene_cubes = {c["id"]: c for c in scene_data.get("cubes", [])}
    sub_scene_id = desc_json.get("sub_scene_id", "unknown")

    logger.info("Phase 4 开始: %s", sub_scene_id)

    # 2. M1: 一致性验证 + 战术标注
    logger.info("  M1: 一致性验证")
    m1_result = run_m1(desc_json, scene_cubes)
    if not m1_result["ready_for_phase4"]:
        errors = m1_result["validation"].get("errors", [])
        logger.warning("  M1 验证未通过: %s", errors)
        # 如果存在硬错误（如空子场景），中止当前子场景
        if m1_result["validation"].get("cube_count", 0) == 0:
            logger.error("  M1 致命错误：子场景无几何数据，跳过")
            from .m4_evaluation.eval_schema import EvalResult
            return {
                "text_version": {}, "struct_version": {},
                "m1_result": m1_result, "m2_result": {},
                "m3_feedback": {"round": 0, "score": 0, "overall_pass": False},
                "m4_result": EvalResult().to_dict(),
                "quality_level": "L",
                "output_paths": {"text": "", "struct": ""},
                "error": "M1 validation failed: empty sub-scene",
            }

    # 3. M2: 自适应策略
    logger.info("  M2: 策略判定")
    m2_result = run_m2(desc_json, embedding_client, vector_store)
    mode = m2_result["mode"]
    reference_content = m2_result.get("reference_content")

    # 4. M3: 迭代精炼 (A_gen ↔ A_review)
    logger.info("  M3: 迭代精炼 (mode=%s)", mode)
    loop = M3IterationLoop(client=client)
    try:
        tactic_json, review_feedback = loop.run(
            desc_json=desc_json,
            mode=mode,
            reference_content=reference_content,
            mission_phase=mission_phase,
        )
    except Exception as e:
        logger.error("  M3 迭代失败: %s", e)
        raise

    # 5. M4: 质量评估
    logger.info("  M4: 质量评估")
    try:
        eval_result = evaluate_tactic(tactic_json, desc_json, client)
    except Exception as e:
        logger.error("  M4 评估失败: %s", e)
        raise
    quality_level = classify_quality(eval_result)
    # 将 Python 分类器的结果写回 eval_result，确保 to_dict() 与顶层一致
    eval_result.quality_level = quality_level

    # 6. 分离双版本
    text_version = tactic_json.get("text_version", {})
    struct_version = tactic_json.get("struct_version", {})

    # 7. 写入结果
    level_dir_text = TACTICS_TEXT_DIR / quality_level
    level_dir_struct = TACTICS_STRUCT_DIR / quality_level
    level_dir_text.mkdir(parents=True, exist_ok=True)
    level_dir_struct.mkdir(parents=True, exist_ok=True)

    # 写入元数据
    text_version["_metadata"] = {
        "sub_scene_id": sub_scene_id,
        "generation_mode": mode,
        "quality_level": quality_level,
        "quality_score": eval_result.overall_score,
        "m3_review_score": review_feedback.score if review_feedback else None,
        "m4_scores": {k: v.score for k, v in eval_result.scores.items()},
    }
    struct_version["_metadata"] = text_version["_metadata"]

    tactic_id = text_version.get("Tactic_ID", sub_scene_id)
    text_path = level_dir_text / f"{tactic_id}.json"
    struct_path = level_dir_struct / f"{tactic_id}.json"

    with open(text_path, "w", encoding="utf-8") as f:
        json.dump(text_version, f, ensure_ascii=False, indent=2)
    with open(struct_path, "w", encoding="utf-8") as f:
        json.dump(struct_version, f, ensure_ascii=False, indent=2)

    logger.info("Phase 4 完成: %s (level=%s, score=%.1f)",
                sub_scene_id, quality_level, eval_result.overall_score)

    return {
        "text_version": text_version,
        "struct_version": struct_version,
        "m1_result": m1_result,
        "m2_result": m2_result,
        "m3_feedback": {
            "round": review_feedback.round if review_feedback else 0,
            "score": review_feedback.score if review_feedback else 0,
            "overall_pass": review_feedback.overall_pass if review_feedback else False,
        },
        "m4_result": eval_result.to_dict(),
        "quality_level": quality_level,
        "output_paths": {
            "text": str(text_path),
            "struct": str(struct_path),
        },
    }


def run_phase4_batch(
    desc_json_paths: List[str],
    scene_cubes_paths: List[str],
    client: Optional[MiniMaxClient] = None,
    embedding_client=None,
    vector_store=None,
    mission_phase: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """批量运行 Phase 4（多个子场景）"""
    results = []
    for desc_path, cubes_path in zip(desc_json_paths, scene_cubes_paths):
        try:
            result = run_phase4(desc_path, cubes_path, client,
                                embedding_client=embedding_client,
                                vector_store=vector_store,
                                mission_phase=mission_phase)
            results.append(result)
        except Exception as e:
            logger.error(f"Phase 4 子场景失败: {desc_path}, error={e}")
            results.append({"error": str(e), "desc_path": desc_path})
    return results


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
        description="Phase 4: 战术生成 (M1→M2→M3→M4) → tactics JSON"
    )
    parser.add_argument(
        "desc_json_path",
        help="Phase 3 输出的 desc.json 路径",
    )
    parser.add_argument(
        "scene_cubes_path",
        help="Phase 3 输出的 scene_cubes.json 路径",
    )
    parser.add_argument(
        "--mission-phase", "-m",
        default=None,
        choices=["侦察阶段", "进攻阶段", "防御阶段", "撤退与脱离阶段"],
        help="作战阶段约束（可选）。不指定则由 LLM 根据子场景自行判定。"
    )
    args = parser.parse_args()

    for p in [args.desc_json_path, args.scene_cubes_path]:
        if not Path(p).exists():
            print(f"错误: 文件不存在: {p}", file=sys.stderr)
            sys.exit(1)

    # Phase 4 内部使用默认目录，如需自定义可通过环境变量或修改 config
    result = run_phase4(args.desc_json_path, args.scene_cubes_path,
                        mission_phase=args.mission_phase)
    print(f"Phase 4 完成")
    print(f"  质量等级: {result['quality_level']}")
    print(f"  综合评分: {result['m4_result'].get('overall_score', 'N/A')}")
    print(f"  文字版: {result['output_paths']['text']}")
    print(f"  结构化版: {result['output_paths']['struct']}")
