"""Versioned, backend-independent Markdown chunk pipeline."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable

from app.rag.schemas import KnowledgeChunk, KnowledgeDocument


PIPELINE_VERSIONS = {"section-v1", "sentence-v2"}
_SECTION_RE = re.compile(r"^##\s+", re.MULTILINE)
_SENTENCE_RE = re.compile(r"(?<=[。！？!?])\s+|\n{2,}")


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def split_markdown_sections(text: str) -> list[tuple[str | None, str]]:
    """Split Markdown at level-two headings while preserving source text."""

    sections: list[tuple[str | None, list[str]]] = []
    current_title: str | None = None
    current_lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("## "):
            if current_lines:
                sections.append((current_title, current_lines))
            current_title = stripped.lstrip("#").strip()
            current_lines = [line]
        else:
            current_lines.append(line)
    if current_lines:
        sections.append((current_title, current_lines))
    return [
        (title, "\n".join(lines).strip())
        for title, lines in sections
        if "\n".join(lines).strip()
    ]


def split_markdown_sections_with_offsets(
    text: str,
) -> list[tuple[str | None, str, int, int]]:
    """Return section text together with an absolute source span."""

    sections: list[tuple[str | None, str, int, int]] = []
    current_title: str | None = None
    current_start = 0
    position = 0
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        if stripped.startswith("## "):
            if position > current_start:
                raw = text[current_start:position]
                value = raw.strip()
                if value:
                    leading = len(raw) - len(raw.lstrip())
                    trailing = len(raw.rstrip())
                    sections.append(
                        (current_title, value, current_start + leading, current_start + trailing)
                    )
            current_title = stripped.lstrip("#").strip()
            current_start = position
        position += len(line)
    if position >= current_start:
        raw = text[current_start:position]
        value = raw.strip()
        if value:
            leading = len(raw) - len(raw.lstrip())
            trailing = len(raw.rstrip())
            sections.append(
                (current_title, value, current_start + leading, current_start + trailing)
            )
    return sections


def _token_count(text: str, tokenizer: Callable[[str], list[int]] | None) -> int:
    return len(tokenizer(text)) if tokenizer is not None else len(text)


def _split_sentence_text(
    text: str,
    *,
    tokenizer: Callable[[str], list[int]] | None,
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    """Pack sentence units by the supplied model tokenizer budget."""

    return [
        value
        for value, _start, _end in _split_sentence_text_with_offsets(
            text,
            tokenizer=tokenizer,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    ]


def _split_sentence_text_with_offsets(
    text: str,
    *,
    tokenizer: Callable[[str], list[int]] | None,
    chunk_size: int,
    chunk_overlap: int,
) -> list[tuple[str, int, int]]:
    """Pack sentence units and retain exact offsets inside ``text``."""

    units: list[tuple[str, int, int]] = []
    cursor = 0
    for match in _SENTENCE_RE.finditer(text):
        raw = text[cursor:match.start()]
        value = raw.strip()
        if value:
            leading = len(raw) - len(raw.lstrip())
            trailing = len(raw.rstrip())
            units.append((value, cursor + leading, cursor + trailing))
        cursor = match.end()
    raw = text[cursor:]
    value = raw.strip()
    if value:
        leading = len(raw) - len(raw.lstrip())
        trailing = len(raw.rstrip())
        units.append((value, cursor + leading, cursor + trailing))
    if not units:
        return []
    result: list[str] = []
    spans: list[tuple[str, int, int]] = []
    current: list[tuple[str, int, int]] = []
    current_size = 0
    for unit, unit_start, unit_end in units:
        unit_size = _token_count(unit, tokenizer)
        if current and current_size + unit_size > chunk_size:
            result.append("\n".join(item[0] for item in current).strip())
            spans.append((result[-1], current[0][1], current[-1][2]))
            overlap_units: list[tuple[str, int, int]] = []
            overlap_size = 0
            for previous in reversed(current):
                previous_size = _token_count(previous[0], tokenizer)
                if overlap_size + previous_size > chunk_overlap:
                    break
                overlap_units.insert(0, previous)
                overlap_size += previous_size
            current = overlap_units
            current_size = overlap_size
        # A single oversized unit is split at character boundaries.  This is
        # rare for Markdown prose but prevents an unbounded chunk when a code
        # block or URL has no sentence boundary.
        if unit_size > chunk_size:
            parts: list[str] = []
            if tokenizer is None:
                step = max(1, chunk_size)
                parts = [unit[start : start + step] for start in range(0, len(unit), step)]
            else:
                part_start = 0
                for char_index in range(1, len(unit) + 1):
                    if _token_count(unit[part_start:char_index], tokenizer) > chunk_size:
                        if char_index - 1 > part_start:
                            parts.append(unit[part_start : char_index - 1])
                            part_start = char_index - 1
                if part_start < len(unit):
                    parts.append(unit[part_start:])
            part_cursor = 0
            for part in parts:
                part = part.strip()
                if part:
                    if current:
                        result.append("\n".join(item[0] for item in current).strip())
                        spans.append((result[-1], current[0][1], current[-1][2]))
                        current = []
                        current_size = 0
                    part_offset = unit.find(part, part_cursor)
                    if part_offset < 0:
                        part_offset = part_cursor
                    spans.append((part, unit_start + part_offset, unit_start + part_offset + len(part)))
                    result.append(part)
                    part_cursor = part_offset + len(part)
            continue
        current.append((unit, unit_start, unit_end))
        current_size += unit_size
    if current:
        result.append("\n".join(item[0] for item in current).strip())
        spans.append((result[-1], current[0][1], current[-1][2]))
    return [(value, start, end) for value, start, end in spans]


def build_document_chunks(
    document: KnowledgeDocument,
    *,
    pipeline_version: str,
    chunk_size: int = 512,
    chunk_overlap: int = 80,
    tokenizer: Callable[[str], list[int]] | None = None,
) -> list[KnowledgeChunk]:
    """Build final chunks consumed identically by all retrieval backends."""

    if pipeline_version not in PIPELINE_VERSIONS:
        raise ValueError(f"Unsupported chunk pipeline version: {pipeline_version}")
    chunks: list[KnowledgeChunk] = []
    ordinal = 0
    for section_index, (section, section_text, source_start, source_end) in enumerate(
        split_markdown_sections_with_offsets(document.text), start=1,
    ):
        chunks_with_offsets = (
            [(section_text, 0, len(section_text))]
            if pipeline_version == "section-v1"
            else _split_sentence_text_with_offsets(
                section_text,
                tokenizer=tokenizer,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        )
        for local_index, (text, relative_start, relative_end) in enumerate(chunks_with_offsets, start=1):
            ordinal += 1
            chunk_id = (
                f"{document.doc_id}#chunk-{ordinal}"
                if pipeline_version == "section-v1"
                else f"{document.doc_id}#{pipeline_version}#section-{section_index}-{local_index}"
            )
            chunks.append(
                KnowledgeChunk(
                    chunk_id=chunk_id,
                    doc_id=document.doc_id,
                    title=document.title,
                    text=text,
                    source_path=document.source_path,
                    doc_type=document.doc_type,
                    metric_name=document.metric_name,
                    tags=document.tags,
                    section=section,
                    document_hash=document.document_hash,
                    index_version=document.index_version,
                    pipeline_version=pipeline_version,
                    content_hash=_content_hash(text),
                    source_start=source_start + relative_start,
                    source_end=min(source_start + relative_end, source_end),
                )
            )
    return chunks
