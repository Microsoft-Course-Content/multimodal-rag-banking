"""
Document Cracker — Extracts text, tables, and images from PDF documents.
Uses PyMuPDF (fitz) for high-fidelity extraction with spatial awareness.
"""

import fitz  # PyMuPDF
import io
import logging
from dataclasses import dataclass, field
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class ExtractedPage:
    """Content extracted from a single PDF page."""
    page_number: int
    text: str
    tables: list[dict] = field(default_factory=list)
    images: list[dict] = field(default_factory=list)
    # image dict: {"bytes": bytes, "bbox": tuple, "caption": str, "format": str}


@dataclass
class CrackedDocument:
    """Fully extracted document with all content types."""
    filename: str
    total_pages: int
    pages: list[ExtractedPage]
    metadata: dict = field(default_factory=dict)


class DocumentCracker:
    """
    Extracts text, images, and tables from PDF documents
    for multimodal RAG ingestion.
    """

    def __init__(self, min_image_size: int = 100):
        """
        Args:
            min_image_size: Minimum image dimension (px) to extract.
                           Filters out tiny icons/logos.
        """
        self.min_image_size = min_image_size

    async def crack(self, file_bytes: bytes, filename: str) -> CrackedDocument:
        """
        Extract all content from a PDF document.

        Args:
            file_bytes: Raw PDF bytes
            filename: Original filename

        Returns:
            CrackedDocument with all pages, text, images, and tables
        """
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        pages = []

        logger.info(f"Cracking document: {filename} ({doc.page_count} pages)")

        for page_num in range(doc.page_count):
            page = doc[page_num]
            extracted = await self._extract_page(page, page_num + 1)
            pages.append(extracted)

        metadata = {
            "title": doc.metadata.get("title", filename),
            "author": doc.metadata.get("author", ""),
            "creation_date": doc.metadata.get("creationDate", ""),
            "total_pages": doc.page_count,
        }

        doc.close()

        result = CrackedDocument(
            filename=filename,
            total_pages=len(pages),
            pages=pages,
            metadata=metadata,
        )

        total_images = sum(len(p.images) for p in pages)
        total_text_chars = sum(len(p.text) for p in pages)
        logger.info(
            f"Cracked {filename}: {len(pages)} pages, "
            f"{total_text_chars} chars, {total_images} images"
        )
        return result

    async def _extract_page(self, page: fitz.Page, page_number: int) -> ExtractedPage:
        """Extract all content from a single page."""
        # Extract text with layout preservation
        text = page.get_text("text")

        # Extract images
        images = self._extract_images(page, page_number)

        # Extract tables (using text blocks spatial analysis)
        tables = self._extract_tables(page)

        return ExtractedPage(
            page_number=page_number,
            text=text.strip(),
            tables=tables,
            images=images,
        )

    def _extract_images(self, page: fitz.Page, page_number: int) -> list[dict]:
        """Extract images from a page, filtering small icons."""
        images = []
        image_list = page.get_images(full=True)

        for img_idx, img_info in enumerate(image_list):
            xref = img_info[0]
            try:
                base_image = page.parent.extract_image(xref)
                if not base_image:
                    continue

                image_bytes = base_image["image"]
                img = Image.open(io.BytesIO(image_bytes))
                width, height = img.size

                # Filter small images (icons, logos, bullets)
                if width < self.min_image_size or height < self.min_image_size:
                    continue

                # Try to find caption near the image
                caption = self._find_image_caption(page, img_info)

                images.append({
                    "bytes": image_bytes,
                    "width": width,
                    "height": height,
                    "format": base_image.get("ext", "png"),
                    "page_number": page_number,
                    "image_index": img_idx,
                    "caption": caption,
                })

            except Exception as e:
                logger.warning(f"Failed to extract image {img_idx} on page {page_number}: {e}")

        return images

    def _find_image_caption(self, page: fitz.Page, img_info) -> str:
        """
        Attempt to find a caption associated with an image.
        Looks for text blocks below or near the image with
        patterns like 'Figure X:', 'Chart:', 'Graph:', etc.
        """
        text_blocks = page.get_text("dict")["blocks"]
        caption_keywords = ["figure", "chart", "graph", "table", "exhibit", "fig."]

        for block in text_blocks:
            if block["type"] == 0:  # Text block
                text = ""
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        text += span.get("text", "")

                text_lower = text.lower().strip()
                if any(kw in text_lower for kw in caption_keywords):
                    return text.strip()[:200]  # Limit caption length

        return ""

    def _extract_tables(self, page: fitz.Page) -> list[dict]:
        """
        Extract tables using spatial analysis of text blocks.
        Groups aligned text blocks into table structures.
        """
        tables = []
        try:
            # Use PyMuPDF's built-in table detection
            tabs = page.find_tables()
            if tabs and tabs.tables:
                for tab_idx, table in enumerate(tabs.tables):
                    table_data = table.extract()
                    if table_data:
                        tables.append({
                            "table_index": tab_idx,
                            "headers": table_data[0] if table_data else [],
                            "rows": table_data[1:] if len(table_data) > 1 else [],
                            "row_count": len(table_data),
                            "col_count": len(table_data[0]) if table_data else 0,
                        })
        except Exception as e:
            logger.debug(f"Table extraction fallback: {e}")

        return tables
