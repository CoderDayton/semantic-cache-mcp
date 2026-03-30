"""Explicit tool response models for FastMCP output schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class ToolResponseModel(BaseModel):
    """Base model that tolerates compact/truncated fallbacks."""

    model_config = ConfigDict(extra="ignore")

    truncated: bool | None = None


class ReadLineRange(ToolResponseModel):
    start: int | None = None
    end: int | None = None
    total: int | None = None


class ReadParams(ToolResponseModel):
    max_size: int | None = None
    diff_mode: bool | None = None
    offset: int | None = None
    limit: int | None = None


class ReadResponse(ToolResponseModel):
    path: str | None = None
    content: str | None = None
    lines: ReadLineRange | None = None
    unchanged: bool | None = None
    is_diff: bool | None = None
    semantic_match: str | None = None
    total_tokens: int | None = None
    hint: str | None = None
    from_cache: bool | None = None
    tokens_saved: int | None = None
    tokens_original: int | None = None
    tokens_returned: int | None = None
    params: ReadParams | None = None


class ClearResponse(ToolResponseModel):
    status: str | None = None
    count: int | None = None
    output_mode: str | None = None


class DeleteResponse(ToolResponseModel):
    status: str | None = None
    path: str | None = None
    deleted: bool | None = None
    dry_run: bool | None = None
    cache_removed: bool | None = None
    symlink: bool | None = None


class StatsStorage(ToolResponseModel):
    files_cached: int | None = None
    total_tokens_cached: int | None = None
    total_documents: int | None = None
    db_size_mb: float | None = None


class StatsSession(ToolResponseModel):
    uptime_s: float | None = None
    tokens_saved: int | None = None
    tokens_original: int | None = None
    tokens_returned: int | None = None
    cache_hits: int | None = None
    cache_misses: int | None = None
    hit_rate_pct: float | int | None = None
    files_read: int | None = None
    files_written: int | None = None
    files_edited: int | None = None
    diffs_served: int | None = None
    tool_calls: dict[str, int] | None = None


class StatsLifetime(ToolResponseModel):
    total_sessions: int | None = None
    tokens_saved: int | None = None
    tokens_original: int | None = None
    tokens_returned: int | None = None
    cache_hits: int | None = None
    cache_misses: int | None = None
    hit_rate_pct: float | int | None = None
    files_read: int | None = None
    files_written: int | None = None
    files_edited: int | None = None


class StatsEmbedding(ToolResponseModel):
    model: str | None = None
    provider: str | None = None
    ready: bool | None = None
    process_rss_mb: float | None = None


class StatsResponse(ToolResponseModel):
    mode: str | None = None
    storage: StatsStorage | None = None
    session: StatsSession | None = None
    lifetime: StatsLifetime | None = None
    embedding: StatsEmbedding | None = None


class WriteResponse(ToolResponseModel):
    status: str | None = None
    path: str | None = None
    diff: str | None = None
    diff_omitted: bool | None = None
    created: bool | None = None
    dry_run: bool | None = None
    tokens_saved: int | None = None
    bytes_written: int | None = None
    tokens_written: int | None = None
    diff_stats: dict[str, Any] | None = None
    content_hash: str | None = None
    from_cache: bool | None = None


class EditParams(ToolResponseModel):
    replace_all: bool | None = None
    dry_run: bool | None = None
    auto_format: bool | None = None


class EditResponse(ToolResponseModel):
    status: str | None = None
    path: str | None = None
    replaced: int | None = None
    line_numbers: list[int] | None = None
    diff: str | None = None
    diff_omitted: bool | None = None
    tokens_saved: int | None = None
    diff_stats: dict[str, Any] | None = None
    content_hash: str | None = None
    from_cache: bool | None = None
    params: EditParams | None = None


class BatchEditFailure(ToolResponseModel):
    old: str | None = None
    error: str | None = None


class BatchEditOutcome(ToolResponseModel):
    old: str | None = None
    new: str | None = None
    success: bool | None = None
    line_number: int | None = None
    error: str | None = None


class BatchEditParams(ToolResponseModel):
    dry_run: bool | None = None
    auto_format: bool | None = None


class BatchEditResponse(ToolResponseModel):
    status: str | None = None
    path: str | None = None
    succeeded: int | None = None
    failed: int | None = None
    failures: list[BatchEditFailure] | None = None
    diff: str | None = None
    diff_omitted: bool | None = None
    tokens_saved: int | None = None
    outcomes: list[BatchEditOutcome] | None = None
    diff_stats: dict[str, Any] | None = None
    content_hash: str | None = None
    from_cache: bool | None = None
    params: BatchEditParams | None = None


class SearchMatch(ToolResponseModel):
    path: str | None = None
    similarity: float | None = None
    tokens: int | None = None
    preview: str | None = None


class SearchResponse(ToolResponseModel):
    query: str | None = None
    matches: list[SearchMatch] | None = None
    count: int | None = None
    cached_files: int | None = None
    files_searched: int | None = None
    k: int | None = None
    directory: str | None = None


class DiffResponse(ToolResponseModel):
    path1: str | None = None
    path2: str | None = None
    diff: str | None = None
    similarity: float | None = None
    diff_stats: dict[str, Any] | None = None
    tokens_saved: int | None = None
    from_cache: bool | None = None
    context_lines: int | None = None


class BatchReadSummary(ToolResponseModel):
    files_read: int | None = None
    files_skipped: int | None = None
    total_tokens: int | None = None
    tokens_saved: int | None = None
    unchanged: list[str] | None = None
    unchanged_count: int | None = None


class BatchReadSkipped(ToolResponseModel):
    path: str | None = None
    est_tokens: int | None = None
    hint: str | None = None


class BatchReadFile(ToolResponseModel):
    path: str | None = None
    status: str | None = None
    content: str | None = None
    hint: str | None = None
    tokens: int | None = None
    from_cache: bool | None = None


class BatchReadResponse(ToolResponseModel):
    summary: BatchReadSummary | None = None
    skipped: list[BatchReadSkipped] | None = None
    files: list[BatchReadFile] | None = None


class SimilarFile(ToolResponseModel):
    path: str | None = None
    similarity: float | None = None
    tokens: int | None = None


class SimilarResponse(ToolResponseModel):
    source_path: str | None = None
    similar_files: list[SimilarFile] | None = None
    source_tokens: int | None = None
    files_searched: int | None = None
    k: int | None = None


class GlobMatch(ToolResponseModel):
    path: str | None = None
    cached: bool | None = None
    tokens: int | None = None
    mtime: float | None = None


class GlobResponse(ToolResponseModel):
    pattern: str | None = None
    directory: str | None = None
    matches: list[GlobMatch] | None = None
    total_matches: int | None = None
    cached_count: int | None = None
    total_cached_tokens: int | None = None


class GrepMatch(ToolResponseModel):
    line_number: int | None = None
    line: str | None = None
    before: list[str] | None = None
    after: list[str] | None = None


class GrepFile(ToolResponseModel):
    path: str | None = None
    count: int | None = None
    matches: list[GrepMatch] | None = None


class GrepResponse(ToolResponseModel):
    pattern: str | None = None
    path: str | None = None
    total_matches: int | None = None
    files_matched: int | None = None
    files: list[GrepFile] | None = None
    fixed_string: bool | None = None
    case_sensitive: bool | None = None
    context_lines: int | None = None


def output_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Build an explicit FastMCP output schema with a stable title."""
    return model.model_json_schema(mode="serialization")
