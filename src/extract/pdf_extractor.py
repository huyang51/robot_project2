"""
PDF 文本提取模块（从 robot_project 适配）

使用 PyMuPDF (fitz) 逐页提取 PDF 文本。
"""
import logging
from pathlib import Path
from typing import Dict, List, Optional

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

from ..config import PDF_EXTENSIONS, RAW_DATA_DIR
from ..core.exceptions import GTKGError

logger = logging.getLogger(__name__)


class PDFExtractError(GTKGError):
    """PDF 提取错误"""


class PDFExtractor:
    """PDF 文本提取器

    使用 PyMuPDF 逐页提取文本，返回包含元数据和页面内容的结构化结果。
    """

    def __init__(self, raw_data_dir: Optional[Path] = None):
        self.raw_data_dir = Path(raw_data_dir) if raw_data_dir else RAW_DATA_DIR

    def extract_from_file(self, pdf_path: Path) -> Dict:
        """从单个 PDF 文件提取文本

        Args:
            pdf_path: PDF 文件路径

        Returns:
            {"metadata": {...}, "pages": [...], "full_text": str}
        """
        if fitz is None:
            raise PDFExtractError(
                "PyMuPDF (fitz) 未安装。请运行: pip install PyMuPDF"
            )

        if not pdf_path.exists():
            raise PDFExtractError(f"PDF 文件不存在: {pdf_path}")

        if pdf_path.suffix.lower() not in PDF_EXTENSIONS:
            raise PDFExtractError(f"不支持的文件类型: {pdf_path.suffix}")

        try:
            doc = fitz.open(str(pdf_path))
            pages = []
            full_text_parts = []

            for page_num, page in enumerate(doc):
                text = page.get_text("text")
                pages.append({
                    "page_num": page_num + 1,
                    "text": text,
                    "source": f"{pdf_path.name}#page={page_num + 1}",
                })
                full_text_parts.append(text)

            metadata = {
                "source_file": pdf_path.name,
                "total_pages": len(pages),
                "title": doc.metadata.get("title", "") if doc.metadata else "",
                "author": doc.metadata.get("author", "") if doc.metadata else "",
            }
            doc.close()

            result = {
                "metadata": metadata,
                "pages": pages,
                "full_text": "\n\n".join(full_text_parts),
            }
            logger.info(f"PDF 提取完成: {pdf_path.name} ({metadata['total_pages']} 页)")
            return result

        except PDFExtractError:
            raise
        except Exception as e:
            raise PDFExtractError(f"PDF 提取失败 {pdf_path.name}: {e}")

    def extract_from_directory(self, directory: Optional[Path] = None) -> List[Dict]:
        """批量提取目录中的所有 PDF

        Args:
            directory: PDF 所在目录，默认 RAW_DATA_DIR

        Returns:
            每个 PDF 的提取结果列表
        """
        dir_path = Path(directory) if directory else self.raw_data_dir
        if not dir_path.exists():
            logger.warning(f"目录不存在: {dir_path}")
            return []

        pdf_files = list(dir_path.rglob("*.pdf"))
        if not pdf_files:
            logger.warning(f"目录中未找到 PDF 文件: {dir_path}")
            return []

        results = []
        for pdf_file in sorted(pdf_files):
            try:
                result = self.extract_from_file(pdf_file)
                results.append(result)
            except PDFExtractError as e:
                logger.error(f"提取失败: {e}")

        logger.info(f"批量提取完成: {len(results)}/{len(pdf_files)} 个 PDF")
        return results
