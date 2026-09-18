"""
TRUSTRAG — Pluggable Chunking Strategies.

Provides multiple chunking strategies that can be selected at runtime
via models.yaml configuration. All strategies produce consistent output
format compatible with the ingestion pipeline.
"""

from __future__ import annotations

from typing import Any

from app.core.logging import get_logger
from app.ingestion.chunker import chunk_text
from app.ingestion.preprocessor import detect_chunk_zone, normalize_text

logger = get_logger(__name__)


class ChunkingStrategy:
    """Abstract base class for chunking strategies."""

    def chunk(
        self,
        pages: list[dict[str, Any]],
        chunk_size: int,
        chunk_overlap: int,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Execute chunking according to this strategy."""
        raise NotImplementedError()


class SlidingWindowStrategy(ChunkingStrategy):
    """Standard sliding window chunking (default behavior)."""

    def chunk(
        self,
        pages: list[dict[str, Any]],
        chunk_size: int,
        chunk_overlap: int,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        return chunk_text(pages, chunk_size=chunk_size, chunk_overlap=chunk_overlap)


class SemanticChunkingStrategy(ChunkingStrategy):
    """Semantic-aware chunking that respects document structure.

    Attempts to keep related content together based on detected headings,
    sections, or semantic boundaries.
    """

    def chunk(
        self,
        pages: list[dict[str, Any]],
        chunk_size: int,
        chunk_overlap: int,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        chunks: list[dict[str, Any]] = []
        chunk_index = 0

        for page_obj in pages:
            page_num = page_obj["page"]
            raw_text = page_obj.get("text", "")
            text = normalize_text(raw_text)

            if not text.strip():
                continue

            ocr_used = bool(page_obj.get("ocr_used", False))
            ocr_confidence = page_obj.get("ocr_confidence")

            # Detect potential section boundaries (headings, etc.)
            lines = text.split("\n")
            sections: list[str] = []
            current_section: list[str] = []

            for line in lines:
                # Heuristic: lines that look like headings (all caps, short, start with #)
                # NOTE: normalize_text lowercases, so the isupper() branch only fires
                # for digit/symbol lines; '#' markdown headings are the live signal.
                is_heading = (
                    line.strip().startswith("#")
                    or (len(line.strip()) < 100 and line.strip().isupper())
                    or line.strip().startswith(("##", "###", "####"))
                )
                if is_heading and current_section:
                    sections.append("\n".join(current_section))
                    current_section = [line]
                elif is_heading:
                    current_section = [line]
                else:
                    current_section.append(line)

            if current_section:
                sections.append("\n".join(current_section))

            # Chunk each section independently, keeping TRUE page offsets so
            # evidence citations (character_offset) stay traceable.
            cursor = 0
            for section in sections:
                if not section.strip():
                    continue
                offset = text.find(section, cursor)
                if offset < 0:
                    offset = cursor
                # Use the standard chunker on each section
                section_result = chunk_text(
                    [
                        {
                            "page": page_num,
                            "text": section,
                            "ocr_used": ocr_used,
                            "ocr_confidence": ocr_confidence,
                        }
                    ],
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )
                for c in section_result:
                    c["chunk_index"] = chunk_index
                    chunk_index += 1
                    c["page"] = page_num
                    c["character_offset"] = offset + c["character_offset"]
                    c["zone"] = detect_chunk_zone(c["text"], page=page_num)
                chunks.extend(section_result)
                cursor = offset + len(section)

        return chunks


class ProgressiveChunkingStrategy(ChunkingStrategy):
    """Progressive chunking that starts small and grows.

    Useful for documents where initial context is sufficient, but deeper
    content may need larger chunks for coherence.
    """

    def chunk(
        self,
        pages: list[dict[str, Any]],
        chunk_size: int,
        chunk_overlap: int,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        # Use standard chunking but with progressive size adjustment
        all_chunks: list[dict[str, Any]] = []
        chunk_index = 0

        for page_obj in pages:
            page_num = page_obj["page"]
            raw_text = page_obj.get("text", "")
            text = normalize_text(raw_text)

            if not text.strip():
                continue

            ocr_used = bool(page_obj.get("ocr_used", False))
            ocr_confidence = page_obj.get("ocr_confidence")

            length = len(text)
            start = 0

            while start < length:
                # Use progressively larger chunks near the beginning
                progress = start / max(length, 1)
                effective_chunk_size = int(chunk_size * (0.5 + 0.5 * progress))
                end = min(start + effective_chunk_size, length)
                chunk_content = text[start:end].strip()

                if chunk_content:
                    zone = detect_chunk_zone(chunk_content, page=page_num)
                    all_chunks.append(
                        {
                            "text": chunk_content,
                            "page": page_num,
                            "chunk_index": chunk_index,
                            "character_offset": start,
                            "zone": zone,
                            "ocr_used": ocr_used,
                            "ocr_confidence": ocr_confidence,
                        }
                    )
                    chunk_index += 1

                if end >= length:
                    break
                # Step scales with the effective window: a fixed full-size step
                # would skip text while windows are still small (silent gaps).
                start += max(1, effective_chunk_size - chunk_overlap)

        return all_chunks


def lines_of(text: str) -> list[str]:
    """Split normalized page text into lines (newlines are preserved by normalize_text)."""
    return text.split("\n")


def is_table_line(line: str) -> bool:
    """Heuristic: pipe-separated or wide multi-space lines look like table rows."""
    stripped = line.strip()
    return (
        "|" in line or stripped.startswith("|") or (len(stripped) > 20 and stripped.count(" ") > 3)
    )


class LayoutAwareChunkingStrategy(ChunkingStrategy):
    """Layout-aware chunking that respects document structure like tables,
    figures, and formatted sections.

    Preserves table boundaries and keeps related visual/text content together.
    """

    def chunk(
        self,
        pages: list[dict[str, Any]],
        chunk_size: int,
        chunk_overlap: int,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        chunks: list[dict[str, Any]] = []
        chunk_index = 0

        for page_obj in pages:
            page_num = page_obj["page"]
            raw_text = page_obj.get("text", "")
            text = normalize_text(raw_text)

            if not text.strip():
                continue

            ocr_used = bool(page_obj.get("ocr_used", False))
            ocr_confidence = page_obj.get("ocr_confidence")

            # Group consecutive lines into table vs prose blocks, preserving
            # page order. Tables are chunked as whole blocks (never split
            # row-by-row); prose blocks go through the standard chunker.
            blocks: list[tuple[str, list[str]]] = []
            for line in lines_of(text):
                if not line.strip():
                    continue
                kind = "table" if is_table_line(line) else "text"
                if blocks and blocks[-1][0] == kind:
                    blocks[-1][1].append(line)
                else:
                    blocks.append((kind, [line]))

            for kind, block_lines in blocks:
                if kind == "table":
                    table_chunks = self._chunk_table_content(
                        block_lines,
                        page_num,
                        chunk_index,
                        chunk_size,
                        chunk_overlap,
                        ocr_used,
                        ocr_confidence,
                    )
                    chunks.extend(table_chunks)
                    chunk_index += len(table_chunks)
                else:
                    section_result = chunk_text(
                        [
                            {
                                "page": page_num,
                                "text": "\n".join(block_lines),
                                "ocr_used": ocr_used,
                                "ocr_confidence": ocr_confidence,
                            }
                        ],
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                    )
                    for c in section_result:
                        c["chunk_index"] = chunk_index
                        chunk_index += 1
                        c["page"] = page_num
                    chunks.extend(section_result)

        return chunks

    def _chunk_table_content(
        self,
        table_rows: list[str],
        page_num: int,
        chunk_index: int,
        chunk_size: int,
        chunk_overlap: int,
        ocr_used: bool = False,
        ocr_confidence: float | None = None,
    ) -> list[dict[str, Any]]:
        """Chunk table-related content while preserving row structure."""
        if not table_rows:
            return []

        combined = " ".join(table_rows)
        result = chunk_text(
            [
                {
                    "page": page_num,
                    "text": combined,
                    "ocr_used": ocr_used,
                    "ocr_confidence": ocr_confidence,
                }
            ],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        for i, c in enumerate(result):
            c["chunk_index"] = chunk_index + i
            c["page"] = page_num
            c["zone"] = "table"

        return result


# Global strategy instance
_chunking_strategy: ChunkingStrategy | None = None


def get_chunking_strategy() -> ChunkingStrategy:
    """Get the global chunking strategy instance from models.yaml config."""
    global _chunking_strategy
    if _chunking_strategy is None:
        strategy_name = _get_strategy_from_config()
        _chunking_strategy = _create_strategy(strategy_name)
    return _chunking_strategy


def _get_strategy_from_config() -> str:
    """Retrieve the chunking strategy name from models.yaml config."""
    from app.core.config import get_model_config

    cfg = get_model_config()
    strategy = getattr(cfg, "_chunking_strategy", None)
    if strategy is None:
        raw = cfg._data.get("ingestion", {})
        strategy = raw.get("chunking_strategy", "sliding_window")
    return strategy


def _create_strategy(strategy_name: str) -> ChunkingStrategy:
    """Create a chunking strategy instance based on the config name."""
    strategies: dict[str, type[ChunkingStrategy]] = {
        "sliding_window": SlidingWindowStrategy,
        "semantic": SemanticChunkingStrategy,
        "progressive": ProgressiveChunkingStrategy,
        "layout_aware": LayoutAwareChunkingStrategy,
    }
    strategy_class = strategies.get(strategy_name.lower(), SlidingWindowStrategy)
    logger.info("Using chunking strategy", strategy=strategy_name)
    return strategy_class()


def clear_chunking_strategy() -> None:
    """Clear the cached chunking strategy (useful for config reloads)."""
    global _chunking_strategy
    _chunking_strategy = None
