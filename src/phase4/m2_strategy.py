"""
M2: 双层自适应策略

根据 desc.json 与 PDF 章节的嵌入相似度，选择生成模式:
- RAG: 相似度 >= θ_high → 检索增强生成
- Hybrid: θ_low <= 相似度 < θ_high → 混合生成
- GEN: 相似度 < θ_low → 自主生成
"""

import logging
from typing import Dict, Optional, List, Tuple, Any

from ..config import M2_THETA_LOW, M2_THETA_HIGH, COLLECTION_PDF_CHAPTERS

logger = logging.getLogger(__name__)


def determine_mode(
    similarities: List[float],
    theta_low: Optional[float] = None,
    theta_high: Optional[float] = None,
) -> Tuple[str, float]:
    """根据最高相似度判定生成模式

    Args:
        similarities: 与各 PDF 章节的余弦相似度列表
        theta_low: 下阈值
        theta_high: 上阈值

    Returns:
        (mode, max_similarity)
    """
    theta_low = theta_low if theta_low is not None else M2_THETA_LOW
    theta_high = theta_high if theta_high is not None else M2_THETA_HIGH

    max_sim = max(similarities) if similarities else 0.0

    if max_sim >= theta_high:
        mode = "RAG"
    elif max_sim >= theta_low:
        mode = "HYBRID"
    else:
        mode = "GEN"

    logger.info(f"M2 策略判定: mode={mode}, max_similarity={max_sim:.3f} "
                f"(θ_low={theta_low}, θ_high={theta_high})")
    return mode, max_sim


def run_m2(
    desc_json: Dict[str, Any],
    embedding_client=None,
    vector_store=None,
) -> Dict[str, Any]:
    """M2 主入口

    Returns:
        {
            "mode": "RAG|HYBRID|GEN",
            "max_similarity": float,
            "reference_chapters": [...],
            "reference_content": str or None,
        }
    """
    # 若嵌入客户端或向量存储不可用，直接返回 GEN 模式
    if embedding_client is None or vector_store is None:
        logger.info("M2: 嵌入客户端不可用，默认使用 GEN 模式")
        return _gen_fallback()

    # 1. 生成 desc.json 的查询嵌入
    desc_text = _serialize_for_embedding(desc_json)

    try:
        query_emb = embedding_client.embed(desc_text)
        # 检测 stub 零向量（嵌入客户端降级到 stub 时返回全零）
        if not query_emb or all(v == 0.0 for v in query_emb):
            logger.info("M2: 嵌入向量为零向量（嵌入模型不可用），默认使用 GEN 模式")
            return _gen_fallback()

        # 2. 检索相似 PDF 章节
        results = vector_store.search(COLLECTION_PDF_CHAPTERS, query_emb, top_k=5)
        if not isinstance(results, list):
            raise TypeError(f"vector_store.search 返回非列表类型: {type(results)}")
        similarities = [
            r.get("score", 0) if isinstance(r, dict) else 0.0
            for r in results
        ]
    except Exception as e:
        logger.warning("M2: 嵌入/检索失败 (%s)，回退到 GEN 模式", e)
        return _gen_fallback()

    # 3. 判定模式
    mode, max_sim = determine_mode(similarities)

    # 4. 聚合参考资料（仅 RAG/Hybrid）
    reference_content = None
    reference_chapters = []
    if mode in ("RAG", "HYBRID"):
        for r in results:
            if r.get("score", 0) >= M2_THETA_LOW:
                reference_chapters.append(r)
        if reference_chapters:
            reference_content = "\n\n".join(
                r.get("content", "") for r in reference_chapters
            )

    return {
        "mode": mode,
        "max_similarity": max_sim,
        "reference_chapters": reference_chapters,
        "reference_content": reference_content,
    }


def _gen_fallback() -> Dict[str, Any]:
    """M2 回退到 GEN 模式的统一返回值"""
    return {
        "mode": "GEN",
        "max_similarity": 0.0,
        "reference_chapters": [],
        "reference_content": None,
    }


def _serialize_for_embedding(desc_json: Dict[str, Any]) -> str:
    """将 desc.json 序列化为适合嵌入的文本"""
    parts = [
        f"战术角色: {desc_json.get('tactical_role', '')}",
        f"任务: {desc_json.get('task_hint', '')}",
        f"空间描述: {desc_json.get('spatial_description', '')}",
        f"标签: {', '.join(desc_json.get('inferred_tags', []))}",
    ]
    # 添加区域信息
    for z in desc_json.get("zones", []):
        parts.append(f"区域 {z.get('zone_id')}: {z.get('type')} - {z.get('description', '')}")
    # 添加威胁信息
    for t in desc_json.get("inferred_threats", []):
        parts.append(f"威胁: {t.get('type')} ({t.get('severity')}) - {t.get('description', '')}")
    return "\n".join(parts)
