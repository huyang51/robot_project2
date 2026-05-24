"""
M2: 双层自适应策略

根据 desc.json 与 PDF 章节的嵌入相似度，选择生成模式:
- RAG: 相似度 >= θ_high → 检索增强生成
- Hybrid: θ_low <= 相似度 < θ_high → 混合生成
- GEN: 相似度 < θ_low → 自主生成
"""

import logging
from typing import Dict, Optional, List, Tuple, Any

from ..config import M2_THETA_LOW, M2_THETA_HIGH

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


def compute_similarity(
    query_embedding: List[float],
    doc_embeddings: Dict[str, List[float]],
) -> List[float]:
    """计算查询嵌入与文档嵌入的余弦相似度

    Args:
        query_embedding: 子场景 desc.json 的嵌入向量
        doc_embeddings: {chapter_id: embedding} PDF 章节嵌入

    Returns:
        与各章节的相似度列表
    """
    if not query_embedding or not doc_embeddings:
        return []

    def _cosine(a: List[float], b: List[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = (sum(x ** 2 for x in a)) ** 0.5
        norm_b = (sum(x ** 2 for x in b)) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    return [_cosine(query_embedding, emb) for emb in doc_embeddings.values()]


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
        return {
            "mode": "GEN",
            "max_similarity": 0.0,
            "reference_chapters": [],
            "reference_content": None,
        }

    # 1. 生成 desc.json 的查询嵌入
    desc_text = _serialize_for_embedding(desc_json)
    query_emb = embedding_client.embed(desc_text)

    # 2. 检索相似 PDF 章节
    from ..config import COLLECTION_PDF_CHAPTERS
    results = vector_store.search(COLLECTION_PDF_CHAPTERS, query_emb, top_k=5)
    similarities = [r.get("score", 0) for r in results]

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
