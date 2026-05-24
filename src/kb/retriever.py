"""
知识库检索器

提供战术知识检索和 PDF 章节检索接口。
"""

import logging
from typing import List, Dict, Optional, Any

from ..config import COLLECTION_TACTICS_TEXT, COLLECTION_TACTICS_STRUCT, COLLECTION_PDF_CHAPTERS
from .embedding_client import EmbeddingClient
from .vector_store import VectorStore

logger = logging.getLogger(__name__)


class TacticalRetriever:
    """战术知识检索器"""

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_client: EmbeddingClient,
    ):
        self.vs = vector_store
        self.ec = embedding_client

    def search_tactics(
        self,
        query: str,
        version: str = "text",
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """检索相关战术

        Args:
            query: 查询文本（场景描述、任务描述等）
            version: "text" (文字描述版) | "struct" (结构化描述版)
            top_k: 返回数量
        """
        collection = COLLECTION_TACTICS_TEXT if version == "text" else COLLECTION_TACTICS_STRUCT
        query_emb = self.ec.embed(query)
        return self.vs.search(collection, query_emb, top_k=top_k)

    def search_pdf_chapters(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """检索相关 PDF 章节"""
        query_emb = self.ec.embed(query)
        return self.vs.search(COLLECTION_PDF_CHAPTERS, query_emb, top_k=top_k)

    def search_by_sub_scene(
        self,
        desc_json: Dict[str, Any],
        version: str = "text",
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """根据子场景语义标注检索相关战术

        将 desc_json 序列化为查询文本。
        """
        query_text = _serialize_desc_for_query(desc_json)
        return self.search_tactics(query_text, version=version, top_k=top_k)


def _serialize_desc_for_query(desc: Dict[str, Any]) -> str:
    """将 desc.json 序列化为适合检索查询的文本

    格式与 m2_strategy._serialize_for_embedding 保持一致，
    确保 M2 阶段和手动检索使用相同的嵌入表示。
    """
    parts = [
        f"战术角色: {desc.get('tactical_role', '')}",
        f"任务: {desc.get('task_hint', '')}",
        f"空间描述: {desc.get('spatial_description', '')}",
        f"标签: {', '.join(desc.get('inferred_tags', []))}",
    ]
    for z in desc.get("zones", []):
        parts.append(f"区域 {z.get('zone_id')}: {z.get('type')} - {z.get('description', '')}")
    for t in desc.get("inferred_threats", []):
        parts.append(f"威胁: {t.get('type')} ({t.get('severity')}) - {t.get('description', '')}")
    return "\n".join(parts)
