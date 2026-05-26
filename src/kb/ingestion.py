"""
战术知识入库

将 Phase 4 生成的战术 JSON 和嵌入向量写入 ChromaDB。
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any

from ..config import COLLECTION_TACTICS_TEXT, COLLECTION_TACTICS_STRUCT
from .embedding_client import EmbeddingClient
from .vector_store import VectorStore

logger = logging.getLogger(__name__)


def ingest_tactic(
    tactic_json: Dict[str, Any],
    version: str,  # "text" | "struct"
    vector_store: VectorStore,
    embedding_client: EmbeddingClient,
) -> bool:
    """将单条战术入库

    Args:
        tactic_json: 战术 JSON（含 _metadata）
        version: "text" 或 "struct"
        vector_store: 向量存储
        embedding_client: 嵌入客户端

    Returns:
        成功则 True
    """
    collection = COLLECTION_TACTICS_TEXT if version == "text" else COLLECTION_TACTICS_STRUCT
    tactic_id = tactic_json.get("Tactic_ID", "unknown")
    metadata = tactic_json.pop("_metadata", {})

    try:
        # 序列化战术为文本
        tactic_text = json.dumps(tactic_json, ensure_ascii=False)

        # 生成嵌入
        embedding = embedding_client.embed(tactic_text)

        # 入 ChromaDB
        try:
            vector_store.add(
                collection_name=collection,
                ids=[tactic_id],
                embeddings=[embedding],
                metadatas=[{
                    "tactic_id": tactic_id,
                    "tactic_name": tactic_json.get("Tactic_Name", ""),
                    "mission_phase": tactic_json.get("Mission_Phase", ""),
                    "tactic_type": tactic_json.get("Tactic_Type", ""),
                    "quality_level": metadata.get("quality_level", "L"),
                    "sub_scene_id": metadata.get("sub_scene_id", ""),
                    "generation_mode": metadata.get("generation_mode", ""),
                }],
                documents=[tactic_text],
            )
        except Exception as add_err:
            error_msg = str(add_err).lower()
            if "duplicate" in error_msg or "already exists" in error_msg:
                logger.warning("战术 ID 重复，跳过入库: %s (%s)", tactic_id, version)
                return False
            raise
    finally:
        # 始终恢复 _metadata，避免调用方字典被不可逆修改
        tactic_json["_metadata"] = metadata

    logger.info(f"战术入库完成: {tactic_id} ({version}, {metadata.get('quality_level', 'L')})")
    return True


def ingest_directory(
    dir_path: str,
    version: str,
    vector_store: VectorStore,
    embedding_client: EmbeddingClient,
    quality_filter: Optional[List[str]] = None,
) -> int:
    """批量入库目录中的所有战术 JSON

    Args:
        dir_path: 战术 JSON 目录
        version: "text" | "struct"
        vector_store: 向量存储
        embedding_client: 嵌入客户端
        quality_filter: 仅入库指定质量等级的战术（如 ["H", "M"]）

    Returns:
        成功入库的数量
    """
    path = Path(dir_path)
    if not path.exists():
        logger.warning(f"目录不存在: {dir_path}")
        return 0

    count = 0
    for json_file in path.rglob("*.json"):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                tactic = json.load(f)

            metadata = tactic.get("_metadata", {})
            level = metadata.get("quality_level", "L")

            if quality_filter and level not in quality_filter:
                continue

            if ingest_tactic(tactic, version, vector_store, embedding_client):
                count += 1
        except Exception as e:
            logger.error(f"入库失败 {json_file}: {e}")

    logger.info(f"批量入库完成: {count} 条 ({dir_path})")
    return count


def get_ingestion_stats(vector_store: VectorStore) -> Dict[str, int]:
    """获取入库统计"""
    return {
        "tactics_text": vector_store.collection_count(COLLECTION_TACTICS_TEXT),
        "tactics_struct": vector_store.collection_count(COLLECTION_TACTICS_STRUCT),
    }
