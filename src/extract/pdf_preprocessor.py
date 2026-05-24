"""
PDF 预处理编排器（从 robot_project 适配）

编排 PDF 预处理全管线：
1. PDFExtractor: 从 data/raw/ 扫描 PDF，逐页提取文本
2. TextChunker: 按章节+固定大小分块
3. EmbeddingClient: 批量生成嵌入向量
4. VectorStore: 写入 ChromaDB pdf_chapters Collection

用法:
    python -m src.extract.pdf_preprocessor           # 增量处理
    python -m src.extract.pdf_preprocessor --force    # 强制清空重建
"""
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

from ..config import (
    RAW_DATA_DIR,
    COLLECTION_PDF_CHAPTERS,
    PDF_EMBEDDING_BATCH_SIZE,
)
from ..kb.embedding_client import EmbeddingClient
from ..kb.vector_store import VectorStore
from .pdf_extractor import PDFExtractor, PDFExtractError
from .text_chunker import TextChunker

logger = logging.getLogger(__name__)


class PDFPreprocessor:
    """PDF 预处理编排器

    将 data/raw/ 下的 PDF 参考资料处理为 ChromaDB 中的嵌入向量，
    供 Phase 4 M2 检索增强生成 (RAG/Hybrid) 使用。
    """

    def __init__(
        self,
        pdf_dir: Optional[Path] = None,
        vector_store: Optional[VectorStore] = None,
        embedding_client: Optional[EmbeddingClient] = None,
    ):
        self.pdf_dir = Path(pdf_dir) if pdf_dir else RAW_DATA_DIR
        self.pdf_extractor = PDFExtractor(self.pdf_dir)
        self.text_chunker = TextChunker()

        # 使用 robot_project2 已有的基础设施
        self.vs = vector_store or VectorStore()
        self.ec = embedding_client or EmbeddingClient()

    def preprocess(self, force: bool = False) -> Dict:
        """执行完整 PDF 预处理管线

        Args:
            force: True 则清空已有 pdf_chapters Collection 后重建

        Returns:
            处理统计 {"pdf_count": int, "chunk_count": int, "errors": [...]}
        """
        stats = {"pdf_count": 0, "chunk_count": 0, "errors": []}

        # 1. 处理已有 Collection
        if force:
            try:
                self.vs.delete_collection(COLLECTION_PDF_CHAPTERS)
                logger.info("已清空 Collection '%s'", COLLECTION_PDF_CHAPTERS)
            except Exception:
                pass  # Collection 可能本来就不存在

        # 确保 Collection 存在
        self.vs.get_or_create_collection(COLLECTION_PDF_CHAPTERS)
        existing_count = self.vs.collection_count(COLLECTION_PDF_CHAPTERS)
        if existing_count > 0 and not force:
            logger.info(
                "Collection '%s' 已有 %d 条记录，跳过预处理（使用 --force 强制重建）",
                COLLECTION_PDF_CHAPTERS, existing_count,
            )
            stats["chunk_count"] = existing_count
            stats["pdf_count"] = "unknown (已有数据)"
            return stats

        # 2. 提取 PDF
        logger.info("扫描 PDF 目录: %s", self.pdf_dir)
        try:
            pdf_results = self.pdf_extractor.extract_from_directory(self.pdf_dir)
        except Exception as e:
            logger.error("PDF 扫描失败: %s", e)
            stats["errors"].append(str(e))
            return stats

        if not pdf_results:
            logger.warning("未找到 PDF 文件，请在 %s 下放置 .pdf 参考资料", self.pdf_dir)
            return stats

        stats["pdf_count"] = len(pdf_results)
        logger.info("成功提取 %d 个 PDF", len(pdf_results))

        # 3. 分块
        all_chunks = []
        for result in pdf_results:
            try:
                chunks = self.text_chunker.chunk_document(result)
                all_chunks.extend(chunks)
            except Exception as e:
                logger.error("分块失败 %s: %s", result.get("metadata", {}).get("source_file", "?"), e)
                stats["errors"].append(str(e))

        if not all_chunks:
            logger.warning("未生成任何文本块")
            return stats

        logger.info("共生成 %d 个文本块", len(all_chunks))

        # 4. 生成嵌入向量并写入 ChromaDB
        texts = [c["text"] for c in all_chunks]
        total_chunks = len(all_chunks)

        for batch_start in range(0, total_chunks, PDF_EMBEDDING_BATCH_SIZE):
            batch_end = min(batch_start + PDF_EMBEDDING_BATCH_SIZE, total_chunks)
            batch_chunks = all_chunks[batch_start:batch_end]
            batch_texts = texts[batch_start:batch_end]

            try:
                # 生成嵌入
                embeddings = self.ec.embed_batch(batch_texts)

                # 准备 ChromaDB 写入参数
                ids = [
                    f"pdf_{c['source']}_{c['chunk_index']}"
                    for c in batch_chunks
                ]
                metadatas = [
                    {
                        "source": c["source"],
                        "section": c.get("section", ""),
                        "chunk_index": c["chunk_index"],
                        "source_file": c.get("metadata", {}).get("source_file", ""),
                        "page_num": c.get("metadata", {}).get("total_pages", 0),
                    }
                    for c in batch_chunks
                ]

                self.vs.add(
                    collection_name=COLLECTION_PDF_CHAPTERS,
                    ids=ids,
                    embeddings=embeddings,
                    metadatas=metadatas,
                    documents=batch_texts,
                )
                stats["chunk_count"] += len(batch_chunks)
                logger.info(
                    "批次 [%d-%d]: 已写入 %d 条 (总计 %d)",
                    batch_start, batch_end - 1,
                    len(batch_chunks), stats["chunk_count"],
                )

            except Exception as e:
                logger.error("嵌入/写入失败 批次 [%d-%d]: %s", batch_start, batch_end - 1, e)
                stats["errors"].append(str(e))

        logger.info(
            "PDF 预处理完成: %d 个 PDF → %d 个文本块 (errors=%d)",
            stats["pdf_count"], stats["chunk_count"], len(stats["errors"]),
        )
        return stats


# ── CLI 入口 ────────────────────────────────────────────────

def main():
    """命令行入口"""
    import argparse

    parser = argparse.ArgumentParser(
        description="PDF 预处理: 提取 → 分块 → 嵌入 → ChromaDB"
    )
    parser.add_argument(
        "--pdf-dir", "-d",
        default=None,
        help=f"PDF 文件目录 (默认: {RAW_DATA_DIR})",
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="强制清空已有 Collection 并重建",
    )
    parser.add_argument(
        "--log-level", "-l",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    pdf_dir = Path(args.pdf_dir) if args.pdf_dir else RAW_DATA_DIR

    print("=" * 60)
    print("PDF 预处理开始")
    print(f"  PDF 目录: {pdf_dir}")
    print(f"  强制重建: {'是' if args.force else '否'}")
    print("=" * 60)

    preprocessor = PDFPreprocessor(pdf_dir=pdf_dir)
    stats = preprocessor.preprocess(force=args.force)

    print(f"\n处理完成:")
    print(f"  PDF 文件数: {stats.get('pdf_count', 0)}")
    print(f"  文本块数:   {stats.get('chunk_count', 0)}")
    errors = stats.get("errors", [])
    if errors:
        print(f"  错误数:     {len(errors)}")
        for e in errors:
            print(f"    - {e}")


if __name__ == "__main__":
    main()
