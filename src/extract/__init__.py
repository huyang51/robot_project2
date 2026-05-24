"""
src.extract — PDF 文本提取与预处理

提供 PDF → 文本分块 → 嵌入向量 → ChromaDB 的完整预处理管线，
为 Phase 4 M2 的 PDF 检索增强生成 (RAG/Hybrid) 提供数据基础。

用法:
    python -m src.extract.pdf_preprocessor           # 处理 data/raw/*.pdf
    python -m src.extract.pdf_preprocessor --force    # 强制重建

直接导入:
    from src.extract.pdf_extractor import PDFExtractor
    from src.extract.text_chunker import TextChunker
    from src.extract.pdf_preprocessor import PDFPreprocessor
"""

__all__ = ["PDFExtractor", "TextChunker", "PDFPreprocessor"]
