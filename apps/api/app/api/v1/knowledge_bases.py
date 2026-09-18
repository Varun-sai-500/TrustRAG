"""
TRUSTRAG API — Knowledge Base routes.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
from collections.abc import Mapping
from typing import Any

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    HTTPException,
    Request,
    UploadFile,
    status,
)
from pydantic import BaseModel, Field, HttpUrl

from app.api.deps import get_current_user
from app.api.v1.schemas.kb import DocResponse, KBCreate, KBResponse
from app.core.config import get_model_config, get_settings
from app.core.exceptions import FileTooLargeError, UnsupportedFormatError
from app.core.rate_limiter import limiter
from app.ingestion.chunking_strategies import get_chunking_strategy
from app.ingestion.parser import parse_document
from app.ingestion.pipeline import index_parsed_chunks
from app.services import kb_service
from app.services.search_service import fetch_document_from_url, validate_ingestion_url

router = APIRouter(prefix="/knowledge-bases", tags=["knowledge-bases"])


# ─── URL Ingestion Schema ─────────────────────────────────────────────────────


class URLDocumentRequest(BaseModel):
    """Request model for URL-based document ingestion."""

    url: HttpUrl = Field(..., description="URL of the document to ingest")
    filename: str | None = Field(None, description="Optional custom filename")
    allowlist: list[str] | None = Field(None, description="Optional custom URL allowlist prefixes")


@router.post(
    "",
    response_model=KBResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new knowledge base",
)
async def create_kb_endpoint(
    schema: KBCreate, current_user: Mapping[str, Any] = Depends(get_current_user)
) -> KBResponse:
    """Create a new collection of documents under user ownership."""
    return await kb_service.create_kb(schema, str(current_user["_id"]))


@router.get("", response_model=list[KBResponse], summary="List all user knowledge bases")
async def list_kbs_endpoint(
    current_user: Mapping[str, Any] = Depends(get_current_user),
) -> list[KBResponse]:
    """List all knowledge bases owned by the authenticated user."""
    return await kb_service.list_kbs(str(current_user["_id"]))


@router.get("/{kb_id}", response_model=KBResponse, summary="Retrieve knowledge base metadata")
async def get_kb_endpoint(
    kb_id: str, current_user: Mapping[str, Any] = Depends(get_current_user)
) -> KBResponse:
    """Retrieve detailed knowledge base configuration and verify ownership."""
    return await kb_service.get_kb(kb_id, str(current_user["_id"]))


@router.delete(
    "/{kb_id}", status_code=status.HTTP_204_NO_CONTENT, summary="Delete a knowledge base"
)
async def delete_kb_endpoint(
    kb_id: str, current_user: Mapping[str, Any] = Depends(get_current_user)
) -> None:
    """Delete knowledge base and all registered documents from DB and vector storage."""
    await kb_service.delete_kb(kb_id, str(current_user["_id"]))


@router.post(
    "/{kb_id}/snapshots",
    response_model=KBResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Snapshot a knowledge base for rollback",
)
async def create_snapshot_endpoint(
    kb_id: str,
    version: str = "1.0",
    current_user: Mapping[str, Any] = Depends(get_current_user),
) -> KBResponse:
    """Copy documents, chunks, and vectors into a versioned snapshot KB."""
    return await kb_service.create_kb_snapshot(kb_id, str(current_user["_id"]), version=version)


@router.post(
    "/{kb_id}/rollback/{snapshot_id}",
    response_model=KBResponse,
    summary="Roll back a knowledge base to a snapshot",
)
async def rollback_kb_endpoint(
    kb_id: str,
    snapshot_id: str,
    current_user: Mapping[str, Any] = Depends(get_current_user),
) -> KBResponse:
    """Restore the snapshot's state as the live KB.

    The restored KB keeps the *snapshot's* id, not the original live id —
    clients must swap to the returned id. Refuses snapshots with no searchable
    vectors (409) instead of restoring an empty KB.
    """
    return await kb_service.rollback_kb_to_snapshot(kb_id, snapshot_id, str(current_user["_id"]))


@router.get(
    "/{kb_id}/documents",
    response_model=list[DocResponse],
    summary="List documents in knowledge base",
)
async def list_documents_endpoint(
    kb_id: str, current_user: Mapping[str, Any] = Depends(get_current_user)
) -> list[DocResponse]:
    """List all documents registered in this knowledge base."""
    return await kb_service.list_kb_documents(kb_id, str(current_user["_id"]))


@router.post(
    "/{kb_id}/documents",
    response_model=DocResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and register document",
)
@limiter.limit(lambda: f"{get_settings().rate_limit_upload_per_minute}/minute")
async def upload_document_endpoint(
    request: Request,
    kb_id: str,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    current_user: Mapping[str, Any] = Depends(get_current_user),
) -> DocResponse:
    """
    Upload and parse a document.

    Accepts PDF, TXT, or MD files up to 20MB.
    Enforces format validation, parses content immediately,
    and runs dense/sparse embedding indexing in background.
    """
    # Verify file extension
    cfg = get_model_config()
    allowed_extensions = {
        ext if ext.startswith(".") else f".{ext}" for ext in cfg.supported_formats
    }
    from pathlib import Path

    raw_filename = file.filename or "document.txt"
    filename = Path(raw_filename).name.replace("\x00", "").strip() or "document.txt"
    ext = "." + filename.split(".")[-1].lower() if "." in filename else ""
    if ext not in allowed_extensions:
        raise UnsupportedFormatError(
            f"Unsupported file format '{ext}'",
            detail=f"Only the following formats are accepted: {sorted(allowed_extensions)}",
        )

    # Read content in 1MB chunks with streaming size guard to prevent OOM
    max_file_size = cfg.max_file_size_mb * 1024 * 1024
    chunk_size = 1024 * 1024
    content_chunks = []
    total_bytes = 0

    while chunk := await file.read(chunk_size):
        total_bytes += len(chunk)
        if total_bytes > max_file_size:
            raise FileTooLargeError(
                "Document upload failed",
                detail=(
                    f"File exceeds maximum limit of {cfg.max_file_size_mb}MB "
                    f"(stream halted after reading {total_bytes / (1024 * 1024):.2f}MB)"
                ),
            )
        content_chunks.append(chunk)

    content = b"".join(content_chunks)
    file_size = len(content)

    # Compute content hash
    content_hash = hashlib.sha256(content).hexdigest()

    # Parse document immediately to extract pages and dates without blocking
    # the event loop on CPU-heavy PDF/DOCX work.
    stream = io.BytesIO(content)
    pages, eff_from, eff_until = await asyncio.to_thread(parse_document, filename, stream)

    chunks = await asyncio.to_thread(
        get_chunking_strategy().chunk,
        pages,
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
    )

    # Save metadata record in MongoDB
    doc = await kb_service.add_document(
        kb_id_str=kb_id,
        filename=filename,
        file_size=file_size,
        content_hash=content_hash,
        user_id_str=str(current_user["_id"]),
        effective_from=eff_from,
        effective_until=eff_until,
    )

    # Trigger background indexing
    background_tasks.add_task(
        index_parsed_chunks, doc_id_str=doc.id, kb_id_str=kb_id, chunks=chunks
    )

    return doc


@router.post(
    "/{kb_id}/documents/from-url",
    response_model=DocResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Ingest document from URL",
)
@limiter.limit(lambda: f"{get_settings().rate_limit_url_ingest_per_minute}/minute")
async def ingest_document_from_url_endpoint(
    request: Request,
    kb_id: str,
    background_tasks: BackgroundTasks,
    url_request: URLDocumentRequest,
    current_user: Mapping[str, Any] = Depends(get_current_user),
) -> DocResponse:
    """
    Ingest a document from a URL with SSRF protection.

    Implements defense-in-depth SSRF protection:
    - URL allowlist validation (configurable per-request or global defaults)
    - Internal IP/hostname blocking (loopback, private ranges, cloud metadata)
    - DNS resolution verification
    - Response size limits (10MB default)
    - Request timeout (15s default)
    - Content type validation (text, PDF, JSON, XML, CSV, MD)
    - No automatic redirects to internal addresses

    Allowed content types: text/*, application/pdf, application/json,
    application/xml, text/csv, text/markdown

    Default allowlist includes: wikipedia.org, arxiv.org, github.com,
    python.org, mozilla.org, w3.org, ietf.org, rfc-editor.org
    """
    cfg = get_model_config()

    # Validate URL with SSRF protection
    url_str = str(url_request.url)
    allowlist = set(url_request.allowlist) if url_request.allowlist else None

    is_valid, error = validate_ingestion_url(url_str, allowlist)
    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"URL validation failed: {error}",
        )

    # Fetch document content with SSRF protection
    content, fetch_error = await fetch_document_from_url(url_str, allowlist)
    if fetch_error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to fetch document: {fetch_error}",
        )

    assert content is not None
    file_size = len(content)

    # Verify file size limit
    max_file_size = cfg.max_file_size_mb * 1024 * 1024
    if file_size > max_file_size:
        raise FileTooLargeError(
            "Document from URL exceeds size limit",
            detail=(
                f"Document size {file_size / (1024 * 1024):.2f}MB "
                f"exceeds limit of {cfg.max_file_size_mb}MB"
            ),
        )

    # Determine filename
    if url_request.filename:
        filename = url_request.filename
    else:
        # Extract filename from URL path
        from urllib.parse import urlparse

        parsed = urlparse(url_str)
        filename = parsed.path.split("/")[-1] or "document"
        # Ensure it has an extension
        if "." not in filename:
            # Guess from content type or default to .txt
            filename += ".txt"

    # Clean filename
    from pathlib import Path

    filename = Path(filename).name.replace("\x00", "").strip() or "document.txt"
    ext = "." + filename.split(".")[-1].lower() if "." in filename else ""
    allowed_extensions = {
        ext if ext.startswith(".") else f".{ext}" for ext in cfg.supported_formats
    }
    if ext not in allowed_extensions:
        raise UnsupportedFormatError(
            f"Unsupported file format '{ext}'",
            detail=f"Only the following formats are accepted: {sorted(allowed_extensions)}",
        )

    # Compute content hash
    content_hash = hashlib.sha256(content).hexdigest()

    # Parse document immediately to extract pages and dates without blocking
    # the event loop on CPU-heavy remote-file parsing.
    stream = io.BytesIO(content)
    pages, eff_from, eff_until = await asyncio.to_thread(parse_document, filename, stream)

    chunks = await asyncio.to_thread(
        get_chunking_strategy().chunk,
        pages,
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
    )

    # Save metadata record in MongoDB
    doc = await kb_service.add_document(
        kb_id_str=kb_id,
        filename=filename,
        file_size=file_size,
        content_hash=content_hash,
        user_id_str=str(current_user["_id"]),
        effective_from=eff_from,
        effective_until=eff_until,
    )

    # Trigger background indexing
    background_tasks.add_task(
        index_parsed_chunks, doc_id_str=doc.id, kb_id_str=kb_id, chunks=chunks
    )

    return doc
