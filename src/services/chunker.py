"""
Semantic Text Chunker.
Smart chunking that preserves context around tables, figures,
and section boundaries in banking documents.
"""

import re
import logging
from dataclasses import dataclass
from ..config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """A single text chunk with metadata."""
    chunk_id: str
    content: str
    page_number: int
    chunk_index: int
    token_count: int
    has_table: bool = False
    has_image_reference: bool = False
    section_title: str = ""
    source_document: str = ""


class SemanticChunker:
    """
    Chunks documents using semantic boundaries rather than fixed sizes.
    Preserves tables intact, keeps figure references with their context,
    and respects section boundaries in financial documents.
    """

    # Banking document section patterns
    SECTION_PATTERNS = [
        r"^#{1,3}\s+.+",                           # Markdown headers
        r"^(?:Chapter|Section|Part)\s+\d+",         # Formal sections
        r"^(?:Executive Summary|Financial Highlights|Risk Factors)",
        r"^(?:Balance Sheet|Income Statement|Cash Flow)",
        r"^(?:Notes to |Management Discussion|Auditor)",
        r"^\d+\.\s+[A-Z]",                         # Numbered sections
    ]

    TABLE_MARKERS = ["table", "|", "───", "---", "==="]
    FIGURE_MARKERS = ["figure", "chart", "graph", "exhibit", "fig."]

    def __init__(self):
        settings = get_settings()
        self.chunk_size = settings.chunk_size
        self.chunk_overlap = settings.chunk_overlap

    def chunk_document(
        self, pages: list, filename: str
    ) -> list[Chunk]:
        """
        Chunk a cracked document into semantic chunks.

        Args:
            pages: List of ExtractedPage objects
            filename: Source document filename

        Returns:
            List of Chunk objects
        """
        all_chunks = []
        chunk_index = 0

        for page in pages:
            # Split page text into semantic sections
            sections = self._split_into_sections(page.text)

            for section_title, section_text in sections:
                if not section_text.strip():
                    continue

                # Check if section contains a table — keep intact
                if self._contains_table(section_text):
                    chunk = Chunk(
                        chunk_id=f"{filename}_p{page.page_number}_c{chunk_index}",
                        content=section_text,
                        page_number=page.page_number,
                        chunk_index=chunk_index,
                        token_count=self._estimate_tokens(section_text),
                        has_table=True,
                        section_title=section_title,
                        source_document=filename,
                    )
                    all_chunks.append(chunk)
                    chunk_index += 1
                    continue

                # Check for figure references
                has_figure_ref = self._references_figure(section_text)

                # Chunk long sections with overlap
                if self._estimate_tokens(section_text) > self.chunk_size:
                    sub_chunks = self._chunk_with_overlap(section_text)
                    for sub_text in sub_chunks:
                        chunk = Chunk(
                            chunk_id=f"{filename}_p{page.page_number}_c{chunk_index}",
                            content=sub_text,
                            page_number=page.page_number,
                            chunk_index=chunk_index,
                            token_count=self._estimate_tokens(sub_text),
                            has_image_reference=has_figure_ref,
                            section_title=section_title,
                            source_document=filename,
                        )
                        all_chunks.append(chunk)
                        chunk_index += 1
                else:
                    chunk = Chunk(
                        chunk_id=f"{filename}_p{page.page_number}_c{chunk_index}",
                        content=section_text,
                        page_number=page.page_number,
                        chunk_index=chunk_index,
                        token_count=self._estimate_tokens(section_text),
                        has_image_reference=has_figure_ref,
                        section_title=section_title,
                        source_document=filename,
                    )
                    all_chunks.append(chunk)
                    chunk_index += 1

            # Add table chunks (from structured table extraction)
            for table in page.tables:
                table_text = self._table_to_text(table)
                chunk = Chunk(
                    chunk_id=f"{filename}_p{page.page_number}_t{table['table_index']}",
                    content=table_text,
                    page_number=page.page_number,
                    chunk_index=chunk_index,
                    token_count=self._estimate_tokens(table_text),
                    has_table=True,
                    section_title=f"Table {table['table_index'] + 1}",
                    source_document=filename,
                )
                all_chunks.append(chunk)
                chunk_index += 1

        logger.info(f"Created {len(all_chunks)} chunks from {filename}")
        return all_chunks

    def _split_into_sections(self, text: str) -> list[tuple[str, str]]:
        """Split text into sections based on semantic boundaries."""
        lines = text.split("\n")
        sections = []
        current_title = ""
        current_lines = []

        for line in lines:
            is_section_header = any(
                re.match(pattern, line.strip(), re.IGNORECASE)
                for pattern in self.SECTION_PATTERNS
            )

            if is_section_header and current_lines:
                sections.append((current_title, "\n".join(current_lines)))
                current_title = line.strip()
                current_lines = []
            else:
                current_lines.append(line)

        if current_lines:
            sections.append((current_title, "\n".join(current_lines)))

        return sections if sections else [("", text)]

    def _chunk_with_overlap(self, text: str) -> list[str]:
        """Split long text into overlapping chunks by sentences."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = []
        current_tokens = 0

        for sentence in sentences:
            sent_tokens = self._estimate_tokens(sentence)

            if current_tokens + sent_tokens > self.chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))

                # Calculate overlap
                overlap_tokens = 0
                overlap_start = len(current_chunk)
                for i in range(len(current_chunk) - 1, -1, -1):
                    overlap_tokens += self._estimate_tokens(current_chunk[i])
                    if overlap_tokens >= self.chunk_overlap:
                        overlap_start = i
                        break

                current_chunk = current_chunk[overlap_start:]
                current_tokens = sum(self._estimate_tokens(s) for s in current_chunk)

            current_chunk.append(sentence)
            current_tokens += sent_tokens

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def _contains_table(self, text: str) -> bool:
        """Check if text contains a table structure."""
        text_lower = text.lower()
        pipe_count = text.count("|")
        return pipe_count > 4 or any(m in text_lower for m in self.TABLE_MARKERS[:1])

    def _references_figure(self, text: str) -> bool:
        """Check if text references a figure or chart."""
        text_lower = text.lower()
        return any(m in text_lower for m in self.FIGURE_MARKERS)

    def _table_to_text(self, table: dict) -> str:
        """Convert structured table to readable text for embedding."""
        lines = []
        if table.get("headers"):
            lines.append(" | ".join(str(h) for h in table["headers"]))
            lines.append("-" * 40)
        for row in table.get("rows", []):
            lines.append(" | ".join(str(cell) for cell in row))
        return "\n".join(lines)

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Estimate token count (rough: ~4 chars per token)."""
        return len(text) // 4
