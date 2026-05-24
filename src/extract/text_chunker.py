"""
文本分块模块（从 robot_project 适配）

按章节标题 + 固定大小双重策略对 PDF 文本分块。
"""
import re
from typing import Dict, List

from ..config import CHUNK_SIZE, CHUNK_OVERLAP, PDF_CHAPTER_PATTERNS


class TextChunker:
    """文本分块器

    两层策略：
    1. 按章节标题（中/英文）将全文拆分为章节级片段
    2. 超过 chunk_size 的章节按固定大小再拆，在句末断句
    """

    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
        chapter_patterns: List[str] = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chapter_patterns = chapter_patterns or PDF_CHAPTER_PATTERNS

    def chunk_document(self, document: Dict) -> List[Dict]:
        """将 PDF 提取结果分块

        Args:
            document: PDFExtractor 返回的提取结果
                {"metadata": {...}, "pages": [...], "full_text": str}

        Returns:
            [{"text": str, "source": str, "section": str, "chunk_index": int, "metadata": dict}, ...]
        """
        full_text = document.get("full_text", "")
        metadata = document.get("metadata", {})
        source = metadata.get("source_file", "")

        # 第一层：按章节拆分
        sections = self._split_by_chapters(full_text)

        # 第二层：超长章节按大小拆分
        chunks = []
        for section in sections:
            if len(section["text"]) > self.chunk_size:
                sub_texts = self._chunk_by_size(section["text"])
                for sub_text in sub_texts:
                    chunks.append({
                        "text": sub_text,
                        "source": source,
                        "section": section.get("title", ""),
                        "chunk_index": len(chunks),
                        "metadata": metadata,
                    })
            else:
                if section["text"].strip():
                    chunks.append({
                        "text": section["text"],
                        "source": source,
                        "section": section.get("title", ""),
                        "chunk_index": len(chunks),
                        "metadata": metadata,
                    })

        return chunks

    def _split_by_chapters(self, text: str) -> List[Dict]:
        """按章节标题拆分为 [{title, text}]"""
        sections = []
        current_section = {"title": "", "text": ""}

        chapter_re = "|".join(self.chapter_patterns)

        for line in text.split("\n"):
            stripped = line.strip()
            if stripped and re.match(chapter_re, stripped):
                # 遇到新章节标题，保存前一节
                if current_section["text"].strip():
                    sections.append(current_section)
                current_section = {"title": stripped, "text": ""}
            else:
                if current_section["text"]:
                    current_section["text"] += "\n"
                current_section["text"] += line

        # 保存最后一节
        if current_section["text"].strip():
            sections.append(current_section)

        # 如果章节拆分没有发现任何章节标题（纯文本 PDF），整篇作为一节
        if not sections:
            sections = [{"title": "", "text": text}]

        return sections

    def _chunk_by_size(self, text: str) -> List[str]:
        """按固定大小分块（带重叠），在句末断句"""
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []

        chunks = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]

            # 在边界处找最佳断句点
            if end < len(text):
                break_at = self._find_break_point(chunk)
                if break_at > self.chunk_size // 4:
                    chunk = chunk[:break_at]
                    end = start + break_at

            chunks.append(chunk)
            start = end - self.chunk_overlap

        return chunks

    def _find_break_point(self, text: str) -> int:
        """在文本末尾附近找到最佳断句点"""
        # 优先在句末断句
        for mark in ["。", "！", "？", "；", "\n"]:
            # 从后往前搜索（最后 ~200 字符范围）
            search_start = max(0, len(text) - 200)
            for i in range(len(text) - 1, search_start - 1, -1):
                if text[i] == mark:
                    return i + 1

        # 退而求其次，在逗号处断句
        for i in range(len(text) - 1, max(0, len(text) - 200) - 1, -1):
            if text[i] in ("，", ",", "、"):
                return i + 1

        return len(text)
