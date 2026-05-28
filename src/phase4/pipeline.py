"""
Phase 4 协调器

对外暴露单一入口 run_phase4()，
协调 M1→M2→穷举枚举→M3→M4 完整战术生成流水线。

默认启用多战术穷举生成（参考 robot_project TacticGeneratorV2）：
- Stage 0: 参考资料分批注入 + LLM 迭代发现新战术方向
- 每个战术概念经过 M3 迭代精炼（A_gen ↔ A_review）
- 每个精炼后战术经过 M4 六维度质量评估

双版本输出：
- text_version (文字描述版) → tactics/text_version/{H,M,L}/
- struct_version (结构化描述版) → tactics/struct_version/{H,M,L}/
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

from ..core.llm_client import MiniMaxClient
from ..config import TACTICS_TEXT_DIR, TACTICS_STRUCT_DIR, COLLECTION_PDF_CHAPTERS
from .m1_validator import run_m1
from .m2_strategy import run_m2
from .m3_refinement.iteration_loop import M3IterationLoop
from .m4_evaluation.evaluator import evaluate_tactic
from .m4_evaluation.quality_classifier import classify_quality
from .m4_evaluation.eval_schema import EvalResult
from .multi_tactic.exhaustive_generator import ExhaustiveTacticGenerator
from .direction_generalizer import sanitize_desc

logger = logging.getLogger(__name__)


def run_phase4(
    desc_json_path: str,
    scene_cubes_path: str,
    client: Optional[MiniMaxClient] = None,
    embedding_client=None,
    vector_store=None,
    mission_phase: Optional[str] = None,
    multi_tactic: bool = True,
) -> List[Dict[str, Any]]:
    """Phase 4 主入口：单子场景端到端战术生成（支持多战术穷举）

    Args:
        desc_json_path: Phase 3 输出的 desc.json 路径
        scene_cubes_path: Phase 3 输出的 scene_cubes.json 路径
        client: MiniMaxClient
        embedding_client: 嵌入客户端（M2 用）
        vector_store: 向量存储（M2 用）
        mission_phase: 作战阶段约束，仅生成该阶段的战术。
        multi_tactic: 是否启用多战术穷举生成（默认 True）

    Returns:
        list of result dicts，每个 dict:
        {
            "text_version": {...}, "struct_version": {...},
            "m1_result": {...}, "m2_result": {...},
            "m3_feedback": {...}, "m4_result": {...},
            "quality_level": "H|M|L",
            "output_paths": {"text": str, "struct": str},
        }
    """
    if client is None:
        client = MiniMaxClient()

    # 1. 加载数据
    with open(desc_json_path, "r", encoding="utf-8") as f:
        desc_json = json.load(f)

    # 1a. 系统枚举值消毒（英文枚举字段值 → 中文）
    desc_json = sanitize_desc(desc_json)

    with open(scene_cubes_path, "r", encoding="utf-8") as f:
        scene_data = json.load(f)

    scene_cubes = {c["id"]: c for c in scene_data.get("cubes", [])}
    sub_scene_id = desc_json.get("sub_scene_id", "unknown")

    logger.info("Phase 4 开始: %s", sub_scene_id)

    # 2. M1: 一致性验证
    logger.info("  M1: 一致性验证")
    m1_result = run_m1(desc_json, scene_cubes)
    if not m1_result["ready_for_phase4"]:
        errors = m1_result["validation"].get("errors", [])
        logger.error("  M1 验证未通过，跳过: %s", errors)
        return [{
            "text_version": {}, "struct_version": {},
            "m1_result": m1_result, "m2_result": {},
            "m3_feedback": {"round": 0, "score": 0, "overall_pass": False},
            "m4_result": EvalResult().to_dict(),
            "quality_level": "L",
            "output_paths": {"text": "", "struct": ""},
            "error": f"M1 validation failed: {errors}",
        }]

    # 3. M2: 自适应策略
    logger.info("  M2: 策略判定")
    m2_result = run_m2(desc_json, embedding_client, vector_store)
    mode = m2_result["mode"]
    reference_content = m2_result.get("reference_content")

    # 4. Stage 0: 穷举枚举（多战术）或 单战术
    if multi_tactic:
        logger.info("  Stage 0: 穷举枚举 (mode=%s)", mode)
        exhaustive = ExhaustiveTacticGenerator(client=client)
        concepts = exhaustive.enumerate(
            desc_json=desc_json,
            reference_content=reference_content or "",
            mission_phase=mission_phase or "",
        )
        if not concepts:
            logger.warning("  穷举枚举未产出任何概念，回退到单战术模式")
            concepts = [{}]
    else:
        logger.info("  Stage 0: 单战术模式 (mode=%s)", mode)
        concepts = [{}]

    logger.info("  共 %d 个战术概念待精炼", len(concepts))

    # 5. M3 + M4: 逐概念迭代精炼 + 质量评估
    results = []
    loop = M3IterationLoop(client=client)
    for idx, concept in enumerate(concepts):
        seed = concept if concept else None
        concept_name = (concept or {}).get("Tactic_Name", f"auto_{idx + 1}")
        logger.info("  [%d/%d] M3: 精炼 '%s'", idx + 1, len(concepts), concept_name)

        try:
            tactic_json, review_feedback = loop.run(
                desc_json=desc_json,
                mode=mode,
                reference_content=reference_content,
                mission_phase=mission_phase,
                seed_concept=seed,
            )
        except Exception as e:
            logger.error("  [%d/%d] M3 失败: %s", idx + 1, len(concepts), e)
            continue

        # M4: 质量评估
        logger.info("  [%d/%d] M4: 质量评估", idx + 1, len(concepts))
        try:
            eval_result = evaluate_tactic(tactic_json, desc_json, client)
        except Exception as e:
            logger.error("  [%d/%d] M4 评估失败: %s", idx + 1, len(concepts), e)
            continue
        quality_level = classify_quality(eval_result)
        eval_result.quality_level = quality_level

        # 6. 分离双版本
        text_version = tactic_json.get("text_version", {})
        struct_version = tactic_json.get("struct_version", {})

        # 6a. 安全网：struct_version 的顶层描述字段应与 text_version 一致
        # Description/objective 是战术的整体描述，不因版本（文字/结构化）而异。
        # LLM 偶尔会为 struct_version 产出英文摘要而非详细中文描述，
        # 此处强制从 text_version 覆写。
        for field in ("Description", "objective"):
            if text_version.get(field):
                struct_version[field] = text_version[field]

        # 7. 写入结果
        level_dir_text = TACTICS_TEXT_DIR / quality_level
        level_dir_struct = TACTICS_STRUCT_DIR / quality_level
        level_dir_text.mkdir(parents=True, exist_ok=True)
        level_dir_struct.mkdir(parents=True, exist_ok=True)

        # 元数据
        text_version["_metadata"] = {
            "sub_scene_id": sub_scene_id,
            "generation_mode": mode,
            "quality_level": quality_level,
            "quality_score": eval_result.overall_score,
            "m3_review_score": review_feedback.score if review_feedback else None,
            "m4_scores": {k: v.score for k, v in eval_result.scores.items()},
            "tactic_index": idx + 1,
            "total_tactics": len(concepts),
        }
        struct_version["_metadata"] = dict(text_version["_metadata"])

        # 统一覆写 Tactic_ID：Python 侧生成，保证全局唯一
        # 格式: {sub_scene_id}_T{idx:03d}
        # LLM 自由生成的 ID 不可靠——多个概念可能产出相同 ID，
        # 跨子场景时更会静默覆盖文件或导致 ChromaDB 入库崩溃。
        tactic_id = f"{sub_scene_id}_T{idx + 1:03d}"
        text_version["Tactic_ID"] = tactic_id
        struct_version["Tactic_ID"] = tactic_id

        text_path = level_dir_text / f"{tactic_id}.json"
        struct_path = level_dir_struct / f"{tactic_id}.json"

        with open(text_path, "w", encoding="utf-8") as f:
            json.dump(text_version, f, ensure_ascii=False, indent=2)
        with open(struct_path, "w", encoding="utf-8") as f:
            json.dump(struct_version, f, ensure_ascii=False, indent=2)

        logger.info("  [%d/%d] 完成: %s (level=%s, score=%.1f)",
                     idx + 1, len(concepts), tactic_id,
                     quality_level, eval_result.overall_score)

        results.append({
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
            "seed_concept_name": concept_name,
        })

    logger.info("Phase 4 完成: %s (%d 个战术)", sub_scene_id, len(results))
    return results


def run_phase4_batch(
    desc_json_paths: List[str],
    scene_cubes_paths: List[str],
    client: Optional[MiniMaxClient] = None,
    embedding_client=None,
    vector_store=None,
    mission_phase: Optional[str] = None,
    multi_tactic: bool = True,
) -> List[Dict[str, Any]]:
    """批量运行 Phase 4（多个子场景），每场景生成多条战术"""
    all_results = []
    for desc_path, cubes_path in zip(desc_json_paths, scene_cubes_paths):
        try:
            results = run_phase4(desc_path, cubes_path, client,
                                 embedding_client=embedding_client,
                                 vector_store=vector_store,
                                 mission_phase=mission_phase,
                                 multi_tactic=multi_tactic)
            all_results.extend(results)
        except Exception as e:
            logger.error(f"Phase 4 子场景失败: {desc_path}, error={e}")
            all_results.append({"error": str(e), "desc_path": desc_path})
    return all_results


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
    parser.add_argument(
        "--single", action="store_true",
        help="禁用多战术穷举，只生成单条战术"
    )
    args = parser.parse_args()

    for p in [args.desc_json_path, args.scene_cubes_path]:
        if not Path(p).exists():
            print(f"错误: 文件不存在: {p}", file=sys.stderr)
            sys.exit(1)

    # 初始化嵌入客户端（M2 需要）
    # 支持 python -m src.phase4.pipeline 和直接运行两种方式
    embedding_client = None
    vector_store = None
    try:
        try:
            from ..kb.embedding_client import EmbeddingClient
        except ImportError:
            from src.kb.embedding_client import EmbeddingClient
        embedding_client = EmbeddingClient()
    except Exception as e:
        logger.warning("无法初始化嵌入客户端: %s", e)

    if embedding_client is not None:
        try:
            try:
                from ..kb.vector_store import VectorStore
            except ImportError:
                from src.kb.vector_store import VectorStore
            vector_store = VectorStore()
            vector_store.get_or_create_collection(
                COLLECTION_PDF_CHAPTERS
            )
        except Exception as e:
            logger.warning("ChromaDB 向量存储初始化失败 (M2 将使用 GEN 模式): %s", e)

    # Phase 4 内部使用默认目录，如需自定义可通过环境变量或修改 config
    results = run_phase4(args.desc_json_path, args.scene_cubes_path,
                         embedding_client=embedding_client,
                         vector_store=vector_store,
                         mission_phase=args.mission_phase,
                         multi_tactic=not args.single)
    print(f"Phase 4 完成，共生成 {len(results)} 条战术")
    for i, r in enumerate(results):
        print(f"--- 战术 {i + 1}/{len(results)} ---")
        print(f"  名称: {r.get('seed_concept_name', r.get('text_version', {}).get('Tactic_Name', 'N/A'))}")
        print(f"  质量等级: {r['quality_level']}")
        print(f"  综合评分: {r['m4_result'].get('overall_score', 'N/A')}")
        print(f"  文字版: {r['output_paths']['text']}")
        print(f"  结构化版: {r['output_paths']['struct']}")
