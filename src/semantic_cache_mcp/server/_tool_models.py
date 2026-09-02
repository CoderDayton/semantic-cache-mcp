"""Explicit tool response models for FastMCP output schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

from ..config import PUBLISH_OUTPUT_SCHEMA


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
    # What the body actually holds. Emitted in every response mode: a caller
    # must never have to sniff the content for `@@` headers to tell a diff or
    # a summary apart from the file itself. (`truncated` marks the summary and
    # is inherited from ToolResponseModel.)
    is_diff: bool | None = None
    total_tokens: int | None = None
    hint: str | None = None
    from_cache: bool | None = None
    tokens_saved: int | None = None
    tokens_original: int | None = None
    tokens_returned: int | None = None
    params: ReadParams | None = None
    is_binary: bool | None = None
    size: int | None = None
    mime: str | None = None
    content_hash: str | None = None
    # Hash of a file only partly delivered (line range or summary). Namespaced
    # with a `partial:` prefix so it cannot be redeemed as a `known_hash`.
    file_hash: str | None = None
    # Signed record of the line ranges a ranged read has delivered. Echoed back
    # as `known_hash` it buys `unchanged` for a window already held; unlike
    # `file_hash` it is redeemable, because it names what the caller was sent.
    coverage_token: str | None = None
    total_lines: int | None = None
    # Set by `read(outline=true)`: the body is a map of the file's definitions,
    # not its text. `symbols` counts them, and `reason` says why a body is
    # missing when none were recognized.
    outline: bool | None = None
    symbols: int | None = None
    reason: str | None = None


class ReadImageResponse(ToolResponseModel):
    path: str | None = None
    size: int | None = None
    mime: str | None = None


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


class StatsResponse(ToolResponseModel):
    mode: str | None = None
    storage: StatsStorage | None = None
    session: StatsSession | None = None
    lifetime: StatsLifetime | None = None
    process_rss_mb: float | None = None


class WriteResponse(ToolResponseModel):
    status: str | None = None
    path: str | None = None
    diff: str | None = None
    diff_state: str | None = None
    diff_omitted: bool | None = None
    created: bool | None = None
    dry_run: bool | None = None
    tokens_saved: int | None = None
    bytes_written: int | None = None
    tokens_written: int | None = None
    diff_stats: dict[str, Any] | None = None
    content_hash: str | None = None
    file_hash: str | None = None
    from_cache: bool | None = None


class EditParams(ToolResponseModel):
    replace_all: bool | None = None
    dry_run: bool | None = None
    auto_format: bool | None = None
    show_diff: bool | None = None


class EditResponse(ToolResponseModel):
    status: str | None = None
    path: str | None = None
    dry_run: bool | None = None
    replaced: int | None = None
    line_numbers: list[int] | None = None
    diff: str | None = None
    diff_state: str | None = None
    diff_omitted: bool | None = None
    tokens_saved: int | None = None
    diff_stats: dict[str, Any] | None = None
    content_hash: str | None = None
    file_hash: str | None = None
    from_cache: bool | None = None
    params: EditParams | None = None


class EditPreviewMatch(ToolResponseModel):
    line: int | None = None
    snippet: str | None = None


class EditPreviewResponse(ToolResponseModel):
    path: str | None = None
    found: bool | None = None
    match_count: int | None = None
    line_numbers: list[int] | None = None
    context: list[EditPreviewMatch] | None = None
    truncated: bool | None = None


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
    show_diff: bool | None = None


class BatchEditResponse(ToolResponseModel):
    status: str | None = None
    path: str | None = None
    dry_run: bool | None = None
    succeeded: int | None = None
    failed: int | None = None
    failures: list[BatchEditFailure] | None = None
    diff: str | None = None
    diff_state: str | None = None
    diff_omitted: bool | None = None
    tokens_saved: int | None = None
    outcomes: list[BatchEditOutcome] | None = None
    diff_stats: dict[str, Any] | None = None
    content_hash: str | None = None
    file_hash: str | None = None
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
    # Directory every `matches[].path` is relative to, named once instead of
    # repeated on each entry. Absent when no prefix was worth reporting, in
    # which case the paths are absolute.
    root: str | None = None
    cached_files: int | None = None
    k: int | None = None
    directory: str | None = None
    show_preview: bool | None = None


class BatchReadSummary(ToolResponseModel):
    files_read: int | None = None
    files_skipped: int | None = None
    total_tokens: int | None = None
    tokens_saved: int | None = None
    unchanged: list[str] | None = None
    unchanged_count: int | None = None
    hint: str | None = None


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
    content_hash: str | None = None


class BatchReadResponse(ToolResponseModel):
    summary: BatchReadSummary | None = None
    # Directory every `files[].path` / `skipped[].path` is relative to, named once.
    root: str | None = None
    skipped: list[BatchReadSkipped] | None = None
    files: list[BatchReadFile] | None = None


class GlobMatch(ToolResponseModel):
    path: str | None = None
    cached: bool | None = None
    tokens: int | None = None
    mtime: float | None = None


class GlobResponse(ToolResponseModel):
    pattern: str | None = None
    directory: str | None = None
    root: str | None = None
    matches: list[GlobMatch] | None = None
    total_matches: int | None = None
    cached_count: int | None = None
    total_cached_tokens: int | None = None


class WarmFailure(ToolResponseModel):
    path: str | None = None
    reason: str | None = None


class WarmResponse(ToolResponseModel):
    warmed: int | None = None
    already_current: int | None = None
    skipped: int | None = None
    tokens_indexed: int | None = None
    failures: list[WarmFailure] | None = None
    # True when a cap stopped the walk before every named file was reached, so
    # `warmed` is a floor and some files are still invisible to grep/search.
    incomplete: bool | None = None
    hint: str | None = None


class GrepFile(ToolResponseModel):
    path: str | None = None
    # One entry per line delivered, rendered as `"<line>:<text>"` for a line
    # that matched and `"<line>-<text>"` for a context line. A string instead
    # of an object per match: the keys were ~16 tokens of pure envelope each.
    # Absent in `output="paths"` mode.
    lines: list[str] | None = None


class GrepResponse(ToolResponseModel):
    pattern: str | None = None
    path: str | None = None
    # Directory every `files[].path` is relative to, named once.
    root: str | None = None
    total_matches: int | None = None
    files_matched: int | None = None
    files: list[GrepFile] | None = None
    truncated_matches: int | None = None
    truncated_files: int | None = None
    # False when a cap stopped the scan early, so `total_matches` counts only
    # what was examined. Absent means the scan ran to completion.
    complete: bool | None = None
    limit_reached: str | None = None
    files_not_searched: int | None = None
    files_in_response: int | None = None
    # Why an empty result was empty, and what to do about it.
    reason: str | None = None
    hint: str | None = None
    fixed_string: bool | None = None
    case_sensitive: bool | None = None
    context_lines: int | None = None


def output_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Build an explicit FastMCP output schema with a stable title."""
    return model.model_json_schema(mode="serialization")


# Tool name -> the model that declares its response shape. Populated at import
# time by `register_response_model` as each tool is defined.
#
# The registry exists because the published output schema does not. Schemas are
# the majority of this server's advertised bytes and the Messages API has no
# field to receive them, so they stay off the wire — but the *contract* they
# encoded is still worth enforcing, and the response-contract tests check
# payload keys against these models rather than against `tools/list`.
TOOL_RESPONSE_MODELS: dict[str, type[BaseModel]] = {}


def register_response_model(tool: str, model: type[BaseModel]) -> dict[str, Any] | None:
    """Declare *tool*'s response shape; return the schema only if it is published.

    Raises:
        ValueError: two different models were registered for one tool name,
            which would leave the contract tests checking the wrong shape.
    """
    existing = TOOL_RESPONSE_MODELS.get(tool)
    if existing is not None and existing is not model:
        raise ValueError(
            f"tool {tool!r} already declared response model {existing.__name__}; "
            f"refusing to replace it with {model.__name__}"
        )
    TOOL_RESPONSE_MODELS[tool] = model
    return output_schema(model) if PUBLISH_OUTPUT_SCHEMA else None
