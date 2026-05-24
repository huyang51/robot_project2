"""
ChromaDB 向量存储封装

提供战术知识库的存储和检索接口。
"""

import json
import logging
from typing import List, Dict, Optional, Any

from ..config import CHROMA_PERSISTENCE_DIR
from ..core.exceptions import CollectionError

logger = logging.getLogger(__name__)


class VectorStore:
    """ChromaDB 向量存储封装"""

    def __init__(self, persist_dir: Optional[str] = None):
        self.persist_dir = persist_dir or CHROMA_PERSISTENCE_DIR
        self._client = None
        self._collections = {}

    @property
    def client(self):
        """懒加载 ChromaDB 客户端"""
        if self._client is None:
            try:
                import chromadb
                self._client = chromadb.PersistentClient(path=self.persist_dir)
                logger.info(f"ChromaDB 已连接: {self.persist_dir}")
            except ImportError:
                logger.error("chromadb 包未安装")
                raise CollectionError("chromadb 包未安装")
            except Exception as e:
                logger.error(f"ChromaDB 连接失败: {e}")
                raise CollectionError(f"ChromaDB 连接失败: {e}")
        return self._client

    def get_or_create_collection(self, name: str) -> Any:
        """获取或创建 Collection"""
        if name not in self._collections:
            try:
                self._collections[name] = self.client.get_or_create_collection(
                    name=name,
                    metadata={"hnsw:space": "cosine"},
                )
            except Exception as e:
                raise CollectionError(f"Collection '{name}' 操作失败: {e}")
        return self._collections[name]

    def add(
        self,
        collection_name: str,
        ids: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict]] = None,
        documents: Optional[List[str]] = None,
    ) -> None:
        """向 Collection 添加向量"""
        try:
            collection = self.get_or_create_collection(collection_name)
            collection.add(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents,
            )
            logger.info(f"已添加 {len(ids)} 条记录到 '{collection_name}'")
        except Exception as e:
            raise CollectionError(f"添加向量失败: {e}")

    def search(
        self,
        collection_name: str,
        query_embedding: List[float],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """语义检索"""
        try:
            collection = self.get_or_create_collection(collection_name)
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
            )
            # 格式化结果
            formatted = []
            if results and results.get("ids") and results["ids"][0]:
                for i, doc_id in enumerate(results["ids"][0]):
                    item = {"id": doc_id}
                    if results.get("distances") and results["distances"][0]:
                        item["score"] = 1.0 - results["distances"][0][i]
                    if results.get("metadatas") and results["metadatas"][0]:
                        item["metadata"] = results["metadatas"][0][i]
                    if results.get("documents") and results["documents"][0]:
                        item["content"] = results["documents"][0][i]
                    formatted.append(item)
            return formatted
        except Exception as e:
            logger.error(f"检索失败: {e}")
            return []

    def delete_collection(self, name: str) -> None:
        """删除 Collection"""
        try:
            self.client.delete_collection(name=name)
            self._collections.pop(name, None)
            logger.info(f"已删除 Collection '{name}'")
        except Exception as e:
            raise CollectionError(f"删除 Collection 失败: {e}")

    def collection_count(self, name: str) -> int:
        """获取 Collection 中的记录数"""
        try:
            collection = self.get_or_create_collection(name)
            return collection.count()
        except Exception:
            return 0
