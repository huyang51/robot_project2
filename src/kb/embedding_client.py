"""
嵌入向量生成客户端

默认使用本地 SentenceTransformer 模型（无需 API Key），
与 robot_project 的 EmbeddingGenerator 保持一致。

后端优先级: Local(ST, 默认) → MiniMax → OpenAI → stub

用法:
    client = EmbeddingClient()                        # 默认本地模型
    client = EmbeddingClient(use_minimax=True)        # 强制 MiniMax
    client = EmbeddingClient(use_openai=True)         # 强制 OpenAI

GPU 支持:
    本地模型自动检测 CUDA 并使用 GPU (SentenceTransformer 默认行为)。
    RTX PRO 6000 (96GB VRAM) 可轻松运行 bge-large-zh-v1.5 (1.3GB)。
"""

import logging
import os
from typing import List, Optional

from ..config import (
    EMBEDDING_MODEL, EMBEDDING_DIMENSION,
    MINIMAX_EMBEDDING_MODEL, MINIMAX_EMBEDDING_DIMENSION,
    MINIMAX_API_KEY, MINIMAX_BASE_URL,
    EMBEDDING_MODEL_OPENAI, EMBEDDING_DIMENSION_OPENAI,
)

logger = logging.getLogger(__name__)


class EmbeddingClient:
    """嵌入向量生成客户端

    降级链 (单向, 无循环):
      _embed_local() → 失败则 _fallback_embed()
        └─ _try_minimax() → 失败则 _try_openai()
             └─ _try_openai() → 失败则 _embed_stub()

    各后端方法为"叶子"函数，只返回有效嵌入或空列表，
    绝不调用其他后端或 _fallback_embed。
    """

    def __init__(
        self,
        model: Optional[str] = None,
        use_local: bool = True,
        use_minimax: bool = False,
        use_openai: bool = False,
        api_key: Optional[str] = None,
    ):
        self.minimax_key = api_key or MINIMAX_API_KEY
        self.openai_key = self._validate_openai_key(
            os.environ.get("OPENAI_API_KEY", "")
        )
        self._local_model = None
        self._openai_client = None

        if use_minimax and self.minimax_key:
            self.model = model or MINIMAX_EMBEDDING_MODEL
            self.dimension = MINIMAX_EMBEDDING_DIMENSION
            self._backend = "minimax"
        elif use_openai and self.openai_key:
            self.model = model or EMBEDDING_MODEL_OPENAI
            self.dimension = EMBEDDING_DIMENSION_OPENAI
            self._backend = "openai"
        else:
            # 默认本地模型（无需 API Key）
            self.model = model or EMBEDDING_MODEL
            self.dimension = EMBEDDING_DIMENSION
            self._backend = "local"

    # ── Public API ─────────────────────────────────────────

    def embed(self, text: str) -> List[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: List[str], batch_size: int = 50) -> List[List[float]]:
        """批量生成嵌入向量

        降级: local → MiniMax → OpenAI → stub (单向, 不循环)
        """
        if self._backend == "local":
            result = self._try_local(texts, batch_size)
            if result:
                return result
            return self._fallback_embed(texts, batch_size)

        elif self._backend == "minimax":
            result = self._try_minimax(texts)
            if result:
                return result
            return self._fallback_embed(texts, batch_size)

        elif self._backend == "openai":
            result = self._try_openai(texts, batch_size)
            if result:
                return result
            return self._fallback_embed(texts, batch_size)

        else:
            return self._embed_stub(texts)

    # ── Fallback orchestrator ──────────────────────────────

    def _fallback_embed(self, texts: List[str], batch_size: int) -> List[List[float]]:
        """降级链编排器: MiniMax → OpenAI → stub (单向, 无递归)"""
        # 尝试 MiniMax
        if self.minimax_key:
            result = self._try_minimax(texts)
            if result:
                logger.info("降级到 MiniMax 嵌入成功")
                return result

        # 尝试 OpenAI
        if self.openai_key:
            result = self._try_openai(texts, batch_size)
            if result:
                logger.info("降级到 OpenAI 嵌入成功")
                return result

        # 最终回退
        logger.warning("所有嵌入后端均不可用，使用 stub 零向量 (M2 将退化为 GEN 模式)")
        return self._embed_stub(texts)

    # ── Leaf backends (return valid embeddings or []) ──────

    def _try_local(self, texts: List[str], batch_size: int) -> List[List[float]]:
        """本地 SentenceTransformer (与 robot_project 一致)"""
        try:
            from sentence_transformers import SentenceTransformer
            if self._local_model is None:
                self._local_model = SentenceTransformer(
                    self.model,
                    device="cuda" if self._cuda_available() else "cpu",
                )
                logger.info("加载本地嵌入模型: %s (dim=%d, device=%s)",
                            self.model, self.dimension,
                            "cuda" if self._cuda_available() else "cpu")
            embeddings = self._local_model.encode(
                texts,
                batch_size=batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            return embeddings.tolist()
        except Exception as e:
            logger.warning("本地嵌入模型不可用 (%s: %s)", type(e).__name__, e)
            return []

    def _try_minimax(self, texts: List[str]) -> List[List[float]]:
        """MiniMax embo-01 API (叶子方法, 失败返回 [])"""
        import requests
        try:
            resp = requests.post(
                f"{MINIMAX_BASE_URL}/embeddings",
                headers={
                    "Authorization": f"Bearer {self.minimax_key}",
                    "Content-Type": "application/json",
                },
                json={"texts": texts, "model": self.model, "type": "db"},
                timeout=120,
            )
            if not resp.ok:
                logger.warning("MiniMax embedding HTTP %d: %s",
                               resp.status_code, resp.text[:150])
                return []
            data = resp.json()
            vectors = data.get("vectors")
            if vectors is None:
                base_resp = data.get("base_resp", {})
                logger.warning("MiniMax embedding 不可用: %s",
                               base_resp.get("status_msg", "vectors is null"))
                return []
            if len(vectors) > 0 and len(vectors[0]) > 0:
                logger.info("MiniMax embedding: %d texts, dim=%d",
                            len(texts), len(vectors[0]))
                return vectors
            logger.warning("MiniMax embedding 返回空向量")
            return []
        except Exception as e:
            logger.warning("MiniMax embedding 失败 (%s: %s)", type(e).__name__, e)
            return []

    def _try_openai(self, texts: List[str], batch_size: int) -> List[List[float]]:
        """OpenAI text-embedding-3-large (叶子方法, 失败返回 [])"""
        try:
            from openai import OpenAI
        except ImportError:
            logger.warning("openai 未安装, 跳过 OpenAI 嵌入")
            return []
        try:
            if self._openai_client is None:
                self._openai_client = OpenAI(
                    api_key=self.openai_key,
                    timeout=10.0,      # 降级场景下快速超时
                    max_retries=1,      # 降级场景下减少重试
                )
            all_embeddings = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                response = self._openai_client.embeddings.create(
                    model=self.model, input=batch
                )
                all_embeddings.extend([d.embedding for d in response.data])
            return all_embeddings
        except Exception as e:
            logger.warning("OpenAI embedding 失败 (%s: %s)", type(e).__name__, e)
            return []

    def _embed_stub(self, texts: List[str]) -> List[List[float]]:
        """零向量 stub (M2 将判定为 GEN 模式)"""
        return [[0.0] * self.dimension for _ in texts]

    # ── Helpers ────────────────────────────────────────────

    @staticmethod
    def _validate_openai_key(key: str) -> str:
        """验证 OpenAI API Key 格式，过滤占位符和无效 Key"""
        if not key:
            return ""
        # 过滤明显的占位符
        placeholders = (
            "your_openai_api_key_here", "your_api_key_here",
            "sk-your-", "sk-xxx", "sk-test",
        )
        key_lower = key.lower()
        for ph in placeholders:
            if ph in key_lower:
                logger.info("检测到 OpenAI API Key 占位符，跳过 OpenAI 后端")
                return ""
        # 标准 OpenAI Key 格式: sk-proj-... 或 sk-...
        if not key.startswith("sk-"):
            logger.info("OpenAI API Key 格式无效（不以 sk- 开头），跳过 OpenAI 后端")
            return ""
        return key

    @staticmethod
    def _cuda_available() -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False


# ── 向后兼容子类 ─────────────────────────────────────────

class OpenAIEmbeddingClient(EmbeddingClient):
    """显式 OpenAI 嵌入客户端"""
    def __init__(self, api_key=None, model="text-embedding-3-large"):
        super().__init__(model=model, use_local=False, use_openai=True, api_key=api_key)
