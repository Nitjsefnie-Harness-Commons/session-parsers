#!/usr/bin/env python3
"""
Session JSONL Parser — investigation utility for violation analysis.

Parses Claude Code session JSONL transcripts into structured, readable output.
Full untruncated output. Parallel batch detection. Structured filtering.

Event type filters (combine freely — union of all selected types):
    --tools             tool calls + agent spawns
    --results           tool results only
    --user              user messages only
    --assistant         assistant text only
    --thinking          thinking blocks only
    --agents            agent spawns with full prompts
    --conversation      user + assistant text (no tools)
    --pairs             tool calls paired with their results
    --errors            tool results containing error indicators

Tool-specific filters:
    --tool Bash         filter by tool name (exact match, comma-separated for multiple)
    --tool "Bash,Edit"  multiple tool names

Analysis modes:
    --agent-tasks       async Agent launches (agentId, output file, completion)
    --api-errors        API errors separate from tool errors (system/api_error)
    --bg-tasks          background Bash task lifecycle (launch/notify/read).
                        Sections: normal lifecycle, LAUNCHED BUT NEVER NOTIFIED
                        (still running or lost), NOTIFIED BUT NEVER READ (result
                        ignored — may indicate dropped work). TaskStop calls
                        count as terminal notifications (status=stopped) — an
                        explicitly-killed task is NOT an orphan.
    --cache             prompt cache hit rate + token totals (per-model if mixed)
    --cache-anomalies   single-turn outliers: cache_create >=25% of session
                        total, cache_read >=1M (approaching 1M limit), or
                        output >=5k tokens
    --commands          extract Bash command strings only
    --compact           compact events (system/compact_boundary)
    --context-growth    context size over time (turn number, timestamp, input
                        tokens, output tokens, delta from previous turn)
    --edits             summary of all Edit calls (file, old/new snippets, structuredPatch)
    --files             summary of all files touched (read, edited, written, grepped, globbed)
    --latency           rank tool calls by wall-clock latency (result_ts - use_ts)
    --reply-latency     gap between each user message and the assistant's first response event
    --reply-latency-words  words in user prompts ranked by their effect (slower/faster) on reply latency
    --reply-latency-word WORD  with-vs-without reply latency for ONE specific word, split by model
    --rulecheck         find Bash tool_uses missing or with mismatched HR0
                        Rulecheck pairs. Presence (pair exists), shape (regex
                        matches), and slot correctness (|=N counts pipes,
                        timeout matches tool_input.timeout/bg, redirect matches
                        actual command form). One line per violation:
                        `L<n>\\ttool_use=<id>\\tmiss=<reason>\\tcmd=<cmd>`.
    --script-churn      detect /tmp/<base>_*.{sh,py,js} write churn — N+
                        distinct paths in the same family (default N=3).
                        Mirrors the goonbot tmp-script-churn.py hook.
    --shell-errors      Bash calls with stderr, interrupted, or non-zero exit
    --slash             slash commands invoked (system/local_command)
    --last-assistant-message  last turn's assistant text, verbatim (raw)
    --stats             session statistics overview (incl. compute time, gaps, queue ops)
    --timeline          chronological interleave of ALL events with compact one-line formatting.
                        Works with --after, --before, --grep filters
    --tool-orphans      tool_use events without a matching tool_result (crashed/dropped calls)
    --turns             per-turn duration + token spend from system/turn_duration
    --writes            summary of all Write calls (file, content size, preview)

Path filter:
    --path PATTERN      filter events by file path (substring match on tool inputs)

Range/time/text filters (combine with any of the above):
    --range N-M         JSONL line range
    --after HH:MM:SS    events after time
    --before HH:MM:SS   events before time
    --grep PATTERN      regex filter on formatted output

Output:
    --full              disable truncation (show full result bodies and edit diffs)
    -v, --verbose       show full agent spawn prompts (auto-enabled with --agents)
    -o FILE             write to file

Chained mode (--and):
    Run multiple modes against the same input, sharing the parse phase.
    The jsonl is opened once, extract_events / annotate_parallel /
    link_pairs run once, then each segment dispatches its mode against
    the shared master events list.

    Segments are split on the literal token `--and`. Per-segment flags:
    mode flag, -o FILE, --range, --after, --before, --path, --tool,
    --grep, --full, --verbose, --version, --cheat-sheet. Each segment
    must specify exactly ONE mode flag (multi-mode-per-segment errors).

    Restricted to segment 0 (before first --and):
    - jsonl positional argument
    - --line-grep (pre-parse filter; cannot vary per-segment)

    Output:
    - Segments with -o FILE write to that file.
    - Segments without -o stream to stdout with `===== <mode> =====`
      separator headers.

    Errors in one segment do not abort the chain — load is the
    expensive phase, individual mode failures are reported on stderr
    and the next segment runs.

Examples:
    parse_session.py <file/folder> --tools --tool Bash          # all Bash calls
    parse_session.py <file/folder> --pairs --tool Edit           # Edit calls with results
    parse_session.py <file/folder> --pairs --full                # pairs with full result bodies
    parse_session.py <file/folder> --files                       # all files touched
    parse_session.py <file/folder> --writes                      # all files created/overwritten
    parse_session.py <file/folder> --errors                      # find failures
    parse_session.py <file/folder> --tools --grep parallel       # find parallel batches
    parse_session.py <file/folder> --thinking --range 50-80      # thinking in range
    parse_session.py <file/folder> --commands --after 10:25:00   # commands after time
    parse_session.py <file/folder> --edits --full                # edits with full diffs
    parse_session.py <file/folder> --path web/browse.py          # all events touching file
    parse_session.py <file/folder> --cache -o c.txt --and --latency -o l.txt --and --turns
                                                          # 3 modes, single load
"""

__version__ = "1.34.0"
# Cross-platform: must work on Linux AND Windows. No POSIX-only calls.
# Bump __version__ (SemVer) on every substantive change. See docs/update_protocol.md.

import io
import json
import lzma
import os
import sys
import argparse
import re
import time
from pathlib import Path
from datetime import datetime, timedelta, timezone
from collections import Counter, defaultdict
from typing import TextIO

# Session JSONLs may be stored plain or transparently xz-compressed (the R2
# bucket archives them per-object as `<name>.jsonl.xz`). xz decompression is
# stdlib (`lzma`), so reading either form needs no third-party dependency.
JSONL_SUFFIXES = (".jsonl", ".jsonl.xz")

# The two shell tools. They carry the same `command` / `run_in_background` /
# result shape, so every mode that reasons about "a shell call" must accept
# both. Keying on Bash alone made PowerShell activity invisible to bg-task
# tracking, the command log, shell-error triage and the redirect-then-read
# check — on a Windows box that is a large fraction of all shell calls.
_SHELL_TOOLS = ("Bash", "PowerShell")


def _open_text(path) -> TextIO:
    """Open a .jsonl or .jsonl.xz path as a UTF-8 text stream.

    Transparent xz handling via stdlib `lzma` — a drop-in for
    `open(path, encoding="utf-8")` at every JSONL read site, with zero
    third-party dependency. Plain files take the unchanged `open()` path.
    """
    if str(path).endswith(".xz"):
        return io.TextIOWrapper(lzma.LZMAFile(path, "rb"), encoding="utf-8")
    return open(path, encoding="utf-8")


def parse_timestamp(ts: str) -> datetime | None:
    """Parse ISO timestamp from JSONL."""
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


def short_time(ts: str) -> str:
    """Format ISO timestamp as 'YYYY-MM-DD HH:MM:SS' (UTC).

    Width: 19 chars. Sites that print this in tabular columns reserve
    width 19 ('time' header, value cells). Sites that wrap in '[...]'
    accept the wider value without alignment churn.
    """
    dt = parse_timestamp(ts)
    if not dt:
        return "????-??-?? ??:??:??"
    return dt.strftime("%Y-%m-%d %H:%M:%S")


# Message types to skip outright — noise, no event or meta record emitted.
_SKIP_MSG_TYPES = {"progress", "file-history-snapshot"}

_LINE_GREP_CHEAT_SHEET = """\
--line-grep cheat sheet (verified patterns per mode)
====================================================

Pointless on a single jsonl — the filter overhead > savings unless you're
walking a folder with many files. Recommend using only in folder mode on
corpora of >50 files.

Pattern matching is literal substring when the value has no regex metachars
(.^$*+?{}[]\\|()), regex otherwise. Literal is ~5x faster per line; prefer
literals when possible.

Verified bit-identical (output matches no-filter run on a frozen file):

  Mode                    Filter                    Notes
  --------------------    ----------------------    ----------------------
  --cache                 '"usage":'                see "minor diff" below
  --cache-anomalies       '"usage":'                see "minor diff" below
  --writes                '"name":"Write"'          clean
  --tools --tool TOOL     TOOL                      e.g. Bash, Edit, Read
  --thinking              '"type":"think'           clean (R2 ~5x speedup)

Almost-clean (filter loses some annotations but core data matches):

  Mode                    Filter                    What's lost
  --------------------    ----------------------    ----------------------
  --commands              '"name":"Bash"'           [parallel N/M] tags when
                                                    batch had non-Bash
                                                    siblings — AND every
                                                    PowerShell call (the mode
                                                    covers both shells; use
                                                    no filter on a mixed
                                                    session)
  --edits                 '"name":"Edit"'           +N/-N line stats from
                                                    paired tool_result

Not recommended (mode consumes events broadly; filter breaks output):

  --turns, --context-growth, --hooks, --bg-tasks, --agent-tasks,
  --stats, --fast-bg, --timeline, --rulecheck, --redirect-read,
  --script-churn, --tool-orphans, --shell-errors, --reply-latency,
  --reply-latency-words, --reply-latency-word, --latency, --files

Minor diff caveat:
  --line-grep can drop records that contribute only to turn counts (e.g.
  <synthetic> assistant turns whose JSON shape differs slightly) or to
  paired-result-derived stats. Dollar/token sums are unaffected and remain
  correct. If you need exact turn counts or auxiliary stats, omit the filter.

Speedup (full corpus, ~800 files):
  --cache                 13.9s -> 6.5s   (2.1x)
  --tools --tool TOOL     12.2s -> 2.1s   (5.8x, narrow literal)
"""


def _meta_queue_operation(obj: dict, line_num: int, ts: str) -> dict:
    return {
        "line": line_num, "type": "queue-operation", "ts": ts,
        "operation": obj.get("operation", ""),
        "content": obj.get("content", ""),
        "raw": obj,
    }


def _meta_system(obj: dict, line_num: int, ts: str) -> dict:
    return {
        "line": line_num, "type": "system", "ts": ts,
        "subtype": obj.get("subtype", ""),
        "content": obj.get("content", ""),
        "raw": obj,
    }


def _meta_attachment(obj: dict, line_num: int, ts: str) -> dict:
    attachment = obj.get("attachment", {}) or {}
    return {
        "line": line_num, "type": "attachment", "ts": ts,
        "attachment_type": attachment.get("type", ""),
        "raw": obj,
    }


def _meta_permission_mode(obj: dict, line_num: int, ts: str) -> dict:
    return {
        "line": line_num, "type": "permission-mode", "ts": ts,
        "permissionMode": obj.get("permissionMode", ""),
        "raw": obj,
    }


def _meta_last_prompt(obj: dict, line_num: int, ts: str) -> dict:
    return {
        "line": line_num, "type": "last-prompt", "ts": ts,
        "content": obj.get("lastPrompt", ""),
        "raw": obj,
    }


_META_HANDLERS = {
    "queue-operation": _meta_queue_operation,
    "system": _meta_system,
    "attachment": _meta_attachment,
    "permission-mode": _meta_permission_mode,
    "last-prompt": _meta_last_prompt,
}


def _build_tool_call_event(c: dict, line_num: int, ts: str) -> dict:
    """Build a tool_call or agent_spawn event from a content block."""
    tool_name = c.get("name", "")
    tool_input = c.get("input", {})
    tool_use_id = c.get("id", "")
    event = {
        "line": line_num, "type": "tool_call", "ts": ts,
        "tool_name": tool_name,
        "tool_input": tool_input,
        "tool_use_id": tool_use_id,
        "detail": "",
    }
    if tool_name == "Agent":
        event["type"] = "agent_spawn"
        event["agent_name"] = tool_input.get("name", "?")
        event["agent_model"] = tool_input.get("model", "(default)")
        event["agent_bg"] = tool_input.get("run_in_background", False)
        event["agent_team"] = tool_input.get("team_name", "(none)")
        event["agent_prompt"] = tool_input.get("prompt", "")
        event["agent_subagent_type"] = tool_input.get("subagent_type", "(default)")
        event["detail"] = (
            f"Agent({event['agent_name']}, "
            f"model={event['agent_model']}, "
            f"bg={event['agent_bg']}, "
            f"team={event['agent_team']}, "
            f"subagent_type={event['agent_subagent_type']})"
        )
    # else: detail left empty; format_event/etc. lazily render from tool_input
    # via _tool_call_detail when needed. Saves ~1s of json.dumps on large
    # corpora when meta-only modes (--cache, --turns, ...) never format these.
    return event


def _tool_call_detail(event: dict) -> str:
    """Lazy renderer for the display string of a tool_call event."""
    cached = event.get("detail")
    if cached:
        return cached
    rendered = f"{event.get('tool_name', '')}({json.dumps(event.get('tool_input', {}))})"
    event["detail"] = rendered
    return rendered


def _parse_user_content(content, tool_use_result, line_num: int, ts: str,
                        rec_uuid: str = "",
                        task_delivery: dict | None = None) -> list[dict]:
    """Events emitted by one user-role record's content field.

    ``task_delivery`` is the record's top-level ``taskDelivery`` object when
    it carries one (harness-delivered background-task completion,
    ``origin.kind: task-notification``). It rides on the emitted
    ``user_message`` event so the bg-task read detector can credit the
    delivery as a read (issue #383).
    """
    if isinstance(content, str):
        ev = {
            "line": line_num, "type": "user_message", "ts": ts, "detail": content,
            "uuid": rec_uuid,
        }
        if task_delivery is not None:
            ev["taskDelivery"] = task_delivery
        return [ev]
    if not isinstance(content, list):
        return []
    out: list[dict] = []
    for c in content:
        ctype = c.get("type")
        if ctype == "tool_result":
            raw_content = c.get("content", "")
            if isinstance(raw_content, list):
                result_text = "\n".join(
                    (x.get("text", "") if isinstance(x, dict) else str(x))
                    for x in raw_content
                )
            else:
                result_text = str(raw_content)
            out.append({
                "line": line_num, "type": "tool_result", "ts": ts,
                "tool_use_id": c.get("tool_use_id", ""),
                "is_error": c.get("is_error", False),
                "detail": result_text,
                "toolUseResult": tool_use_result,
            })
        elif ctype == "text":
            ev = {
                "line": line_num, "type": "user_message", "ts": ts, "detail": c["text"],
                "uuid": rec_uuid,
            }
            if task_delivery is not None:
                ev["taskDelivery"] = task_delivery
            out.append(ev)
    return out


def _extract_server_result_text(raw) -> tuple[str, bool]:
    """Pull display text + an error flag from a server-tool result block's `content`.

    Server tools carry their result payload in different shapes:
      - advisor:    a dict {"type": "advisor_result", "text": "..."}  (success)
                    or {"type": "advisor_tool_result_error", "error_code": "..."} (error)
      - web_search: a list of result dicts (each may carry "text")
      - others/MCP: a plain string
    Returns (text, is_error). is_error is set when the payload type ends in
    "_error" or carries an error_code / is_error marker.
    """
    if isinstance(raw, dict):
        rtype = str(raw.get("type", ""))
        is_error = rtype.endswith("_error") or bool(raw.get("error_code")) or bool(raw.get("is_error"))
        if "text" in raw:
            return str(raw.get("text", "")), is_error
        return json.dumps(raw, ensure_ascii=False), is_error
    if isinstance(raw, list):
        parts = [
            (x.get("text") or json.dumps(x, ensure_ascii=False)) if isinstance(x, dict) else str(x)
            for x in raw
        ]
        return "\n".join(parts), False
    return str(raw), False


def _parse_assistant_content(content, line_num: int, ts: str, model: str = "(unknown)") -> list[dict]:
    """Events emitted by one assistant-role record's content field (non-usage).

    Every emitted event is tagged with the responding `model` so downstream
    analyses (e.g. --reply-latency-words) can attribute a reply to its model
    without re-walking usage records."""
    if not isinstance(content, list):
        return []
    out: list[dict] = []
    for c in content:
        ctype = c.get("type")
        if ctype == "text":
            out.append({
                "line": line_num, "type": "assistant_text", "ts": ts, "detail": c["text"],
            })
        elif ctype in ("tool_use", "server_tool_use"):
            # server_tool_use (advisor, web_search, web_fetch, ...) is a
            # provider-executed tool call. Route it through the same path as a
            # normal tool_use so every server tool surfaces in
            # --tools/--stats/--tool/--timeline/--latency, instead of the
            # near-dead "server_tool_call" type that no mode consumed.
            out.append(_build_tool_call_event(c, line_num, ts))
        elif ctype == "thinking":
            think = c.get("thinking", "")
            if think:
                out.append({
                    "line": line_num, "type": "thinking", "ts": ts, "detail": think,
                })
        elif isinstance(ctype, str) and ctype.endswith("_tool_result"):
            # Server-tool results (advisor_tool_result, web_search_tool_result,
            # ...) arrive as assistant-content blocks carrying the originating
            # call's tool_use_id. Emit them as tool_result events so they pair
            # with the call (no false --tool-orphans) and surface in
            # --results / --pairs / --errors / --latency.
            result_text, is_error = _extract_server_result_text(c.get("content", ""))
            out.append({
                "line": line_num, "type": "tool_result", "ts": ts,
                "tool_use_id": c.get("tool_use_id", ""),
                "is_error": is_error,
                "detail": result_text,
                "toolUseResult": None,
            })
    for ev in out:
        ev.setdefault("model", model)
    return out


_USAGE_INT_KEYS = (
    "input_tokens",
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
    "output_tokens",
)
_USAGE_NESTED_KEYS = ("cache_creation", "server_tool_use")


def _merge_usage_max(a: dict, b: dict) -> dict:
    """Merge two usage dicts by per-key max for numeric values.

    Specialized for the Anthropic usage schema:
      - flat int counters: input_tokens, cache_*_input_tokens, output_tokens
      - nested int dicts: cache_creation (ephemeral_*), server_tool_use

    Used to consolidate multiple records for the same requestId where
    output_tokens may differ due to streaming. Unknown keys (forward-compat
    for new API fields) are kept from `a` if present, else copied from `b`.
    """
    out = dict(a)
    # Flat int counters — type() is faster than isinstance() in the hot loop
    for k in _USAGE_INT_KEYS:
        vb = b.get(k)
        if vb is None:
            continue
        va = out.get(k)
        if va is None:
            out[k] = vb
        # `type() is int` on purpose: isinstance(True, int) is True, and a
        # bool arriving in a usage counter must not win a `>` against an int.
        # pylint: disable-next=unidiomatic-typecheck
        elif type(vb) is int and type(va) is int and vb > va:
            out[k] = vb
    # Nested int dicts — one level deep, no recursion
    for k in _USAGE_NESTED_KEYS:
        nb = b.get(k)
        if not isinstance(nb, dict):
            continue
        na = out.get(k)
        if not isinstance(na, dict):
            out[k] = dict(nb)
            continue
        merged = dict(na)
        for nk, nvb in nb.items():
            nva = merged.get(nk)
            # pylint: disable-next=unidiomatic-typecheck
            if nva is None or (type(nvb) is int and type(nva) is int and nvb > nva):
                merged[nk] = nvb
        out[k] = merged
    # Unknown keys — forward-compat
    for k, vb in b.items():
        if k in _USAGE_INT_KEYS or k in _USAGE_NESTED_KEYS or k in out:
            continue
        out[k] = vb
    return out


def _flatten_usage(usage: dict) -> dict:
    """Flatten a usage dict by summing iterations[] into top-level fields.

    Anthropic's API returns multi-turn responses (e.g. advisor_message
    iterations) as an ``iterations`` array. The top-level ``input_tokens``,
    ``output_tokens``, etc. in these records are partial snapshots — only
    the sum across iterations reflects the true token spend.
    """
    iters = usage.get("iterations")
    if not iters:
        return usage
    out = dict(usage)
    # Flat int counters — sum across iterations
    for k in _USAGE_INT_KEYS:
        total = sum((i.get(k) or 0) for i in iters)
        out[k] = total
    # Nested int dicts — sum across iterations (cache_creation, server_tool_use)
    for k in _USAGE_NESTED_KEYS:
        merged: dict[str, int] = {}
        for i in iters:
            nested = i.get(k)
            if not isinstance(nested, dict):
                continue
            for nk, nv in nested.items():
                if isinstance(nv, int):
                    merged[nk] = merged.get(nk, 0) + nv
        if merged:
            out[k] = merged
    return out


_REGEX_META = re.compile(r"[.^$*+?{}\[\]\\|()]")


def _close_torn_prefix(prefix: str):
    """Best-effort close of a JSON object truncated mid-record.

    Walks `prefix` tracking open quotes, escapes, and brace/bracket
    nesting. Appends the minimal closing chars needed and retries
    json.loads. Returns the parsed object on success, None if the
    prefix is too damaged (cut mid-key, mid-number, mid-escape, etc.)
    to round-trip. Marks the result with `_torn_truncated: True` so
    consumers know the record is partial.
    """
    in_string = False
    escape = False
    stack: list[str] = []
    for ch in prefix:
        if escape:
            escape = False
            continue
        if in_string:
            if ch == '\\':
                escape = True
            elif ch == '"':
                in_string = False
        else:
            if ch == '"':
                in_string = True
            elif ch in '{[':
                stack.append(ch)
            elif ch == '}' and stack and stack[-1] == '{':
                stack.pop()
            elif ch == ']' and stack and stack[-1] == '[':
                stack.pop()
    closing: list[str] = []
    if in_string:
        closing.append('"')
    while stack:
        closing.append('}' if stack.pop() == '{' else ']')
    candidate = prefix + ''.join(closing)
    try:
        obj = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    if isinstance(obj, dict):
        obj["_torn_truncated"] = True
    return obj


def _iter_json_objects(line: str):
    """Yield each decodable top-level JSON object in `line`.

    Tolerates torn writes that concatenate a truncated record with a
    complete next record (no separator) — observed when Claude Code is
    killed mid-flush and the next session's first write tacks onto the
    partial line. On a parse failure, resyncs to the next `{"` boundary
    (the JSONL record-start convention) and tries `_close_torn_prefix`
    on the skipped chunk to salvage the truncated header (type / ts /
    sessionId / partial content). Yields nothing if no complete object
    can be decoded.
    """
    decoder = json.JSONDecoder()
    pos = 0
    n = len(line)
    while pos < n:
        while pos < n and line[pos].isspace():
            pos += 1
        if pos >= n:
            return
        try:
            obj, end = decoder.raw_decode(line, pos)
        except json.JSONDecodeError:
            # Find next JSONL record boundary (`{"`) and try to close
            # the truncated prefix between pos and there.
            next_open = line.find('{"', pos + 1)
            if next_open == -1:
                return
            closed = _close_torn_prefix(line[pos:next_open])
            if closed is not None:
                yield closed
            pos = next_open
            continue
        yield obj
        pos = end


def _process_record_objs(objs, line_num, events, meta_events,
                         seen_request_events, seen_uuids):
    """Turn one raw line's parsed JSON object(s) into events/meta_events,
    accumulating into the caller's lists + dedup/merge state. Shared by
    extract_events (read+parse path) and _build_from_cache (cached-parse path used
    by --find-linegrep) so both apply IDENTICAL record logic. Empty `objs` (a line
    that failed every JSON decode) becomes a parse_error event."""
    if not objs:
        events.append({
            "line": line_num, "type": "parse_error", "ts": "",
            "detail": f"Invalid JSON at line {line_num}",
        })
        return
    for obj in objs:
        if seen_uuids is not None:
            rec_uuid = obj.get("uuid")
            if rec_uuid:
                if rec_uuid in seen_uuids:
                    continue
                seen_uuids.add(rec_uuid)

        ts = obj.get("timestamp", "")
        msg_type = obj.get("type", "")

        if msg_type in _SKIP_MSG_TYPES:
            continue

        handler = _META_HANDLERS.get(msg_type)
        if handler:
            meta_events.append(handler(obj, line_num, ts))
            continue

        msg = obj.get("message", {})
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role == "user":
            events.extend(_parse_user_content(
                content, obj.get("toolUseResult"), line_num, ts,
                obj.get("uuid", ""), obj.get("taskDelivery")
            ))
        elif role == "assistant":
            usage = msg.get("usage")
            if usage:
                usage = _flatten_usage(usage)
                req_id = obj.get("requestId", "")
                if req_id and req_id in seen_request_events:
                    existing = seen_request_events[req_id]["usage"]
                    seen_request_events[req_id]["usage"] = _merge_usage_max(existing, usage)
                else:
                    ev = {
                        "line": line_num, "type": "assistant_usage", "ts": ts,
                        "model": msg.get("model") or "(unknown)",
                        "requestId": req_id,
                        "uuid": obj.get("uuid", ""),
                        "usage": usage,
                    }
                    if req_id:
                        seen_request_events[req_id] = ev
                    meta_events.append(ev)
            events.extend(_parse_assistant_content(
                content, line_num, ts, msg.get("model") or "(unknown)"))


def extract_events(
    jsonl_path: str,
    line_filter: "re.Pattern[str] | str | None" = None,
    seen_uuids: "set[str] | None" = None,
) -> tuple[list[dict], list[dict]]:
    """Parse JSONL into structured events and sibling meta events.

    Returns (events, meta_events).
    - events: conversational events (user/assistant/tool_call/tool_result/thinking)
    - meta_events: queue-operation, system/*, attachment, permission-mode,
      last-prompt — plus assistant usage records (as ``assistant_usage`` meta
      events) that are needed for per-turn token accounting.

    line_filter: optional pre-parse regex applied to the raw JSONL text. Lines
    not matching the pattern are skipped without json.loads — useful as a
    fast path on large corpora when the caller only needs records mentioning
    a specific token (tool name, requestId, etc.). Differs from ``--grep``
    which applies to the formatted output after parse + mode dispatch.

    seen_uuids: optional cross-file dedup set, keyed by the JSONL record's
    top-level `uuid` field. When the directory walker passes the same set to
    every per-file call, the SAME API call recorded into two files is
    processed exactly once. The canonical case is a session's main
    `<uuid>.jsonl` and its `data/subagents/agent-*.jsonl` companion: the
    subagent file is a strict subset of the parent's records but with a
    different wrapper (`isSidechain:true`, an `agentId`, a `slug`), so raw-
    line hashing does not catch them — but the inner `uuid` is identical.
    None disables the cross-file pass; single-file invocations rely solely
    on per-file requestId merging (`_merge_usage_max`) and are unaffected.
    Records without a `uuid` (e.g. legacy parse_error rows) are not deduped.
    """
    events: list[dict] = []
    meta_events: list[dict] = []
    # Claude Code splits one assistant API response into N JSONL records when
    # the response has multiple content blocks (thinking + text + tool_use).
    # Each record usually carries the SAME top-level `usage` block — but for
    # streaming responses the per-record `output_tokens` can lag (intermediate
    # records report a partial count, the final record carries the aggregate).
    # input/cache_create/cache_read are reliably identical across records.
    # Dedup by requestId; on duplicate, MERGE numeric usage fields by max so
    # we capture the final aggregate regardless of record order. Records with
    # empty requestId (legacy data) are not deduped.
    seen_request_events: dict[str, dict] = {}
    json_loads = json.loads
    # Fast path: when caller passes a pattern with no regex metacharacters
    # (or we get a plain string), use substring `in` — ~5x faster than
    # re.Pattern.search on non-matching lines.
    literal_substring: str | None = None
    line_search = None
    if isinstance(line_filter, str):
        if _REGEX_META.search(line_filter) is None:
            literal_substring = line_filter
        else:
            line_search = re.compile(line_filter, re.IGNORECASE).search
    elif line_filter is not None:
        # Already-compiled pattern
        line_search = line_filter.search
    with _open_text(jsonl_path) as f:
        for line_num, line in enumerate(f, 1):
            # rstrip("\n") only — jsonl lines don't carry stray ws and full
            # strip() showed up in the profile at ~0.3s on a 791-file corpus.
            if line and line[-1] == "\n":
                line = line[:-1]
            if not line:
                continue
            # Pre-parse line filter (fast path): skip json.loads when the
            # raw line doesn't match. Substantially faster than parsing every
            # line and discarding post-format.
            if literal_substring is not None:
                if literal_substring not in line:
                    continue
            elif line_search is not None and line_search(line) is None:
                continue
            # Fast path: most lines are exactly one JSON object. Try the
            # cheap single-decode first; only on failure invoke the
            # raw_decode loop that tolerates torn-write concatenations.
            try:
                objs: list = [json_loads(line)]
            except json.JSONDecodeError:
                objs = list(_iter_json_objects(line))
            _process_record_objs(objs, line_num, events, meta_events,
                                 seen_request_events, seen_uuids)

    # Attach source path so folder-mode formatters can emit provenance
    # via --with-source. One-pass tag at the boundary; per-event parsers
    # stay unchanged.
    for e in events:
        e["src"] = jsonl_path
    for m in meta_events:
        m["src"] = jsonl_path
    return events, meta_events


def annotate_parallel(events: list[dict]) -> None:
    """Mark tool calls dispatched in the same parallel batch.

    Parallel calls detected by timestamp proximity: consecutive tool calls
    with <2s between them form a batch. Each gets batch_size and batch_index.
    """
    tool_events = [e for e in events if e["type"] in ("tool_call", "agent_spawn")]
    if not tool_events:
        return

    batches: list[list[dict]] = [[tool_events[0]]]
    for e in tool_events[1:]:
        prev = batches[-1][-1]
        t_prev = parse_timestamp(prev["ts"])
        t_curr = parse_timestamp(e["ts"])
        if t_prev and t_curr and abs((t_curr - t_prev).total_seconds()) < 2.0:
            batches[-1].append(e)
        else:
            batches.append([e])

    for batch in batches:
        size = len(batch)
        for idx, e in enumerate(batch, 1):
            e["batch_size"] = size
            e["batch_index"] = idx


def link_pairs(events: list[dict]) -> None:
    """Link tool calls to their results via tool_use_id."""
    call_map: dict[str, dict] = {}
    for e in events:
        if e["type"] in ("tool_call", "agent_spawn") and e.get("tool_use_id"):
            call_map[e["tool_use_id"]] = e
    for e in events:
        if e["type"] == "tool_result" and e.get("tool_use_id"):
            call = call_map.get(e["tool_use_id"])
            if call:
                e["paired_call"] = call
                call["paired_result"] = e


def format_event(event: dict, verbose: bool = False, with_source: bool = False) -> str:
    """Format a single event for display."""
    line = event["line"]
    time = short_time(event["ts"])
    etype = event["type"]
    detail = event["detail"] if etype != "tool_call" else _tool_call_detail(event)

    # Parallel batch tag for tool calls
    batch_tag = ""
    batch_size = event.get("batch_size", 1)
    if batch_size > 1:
        batch_tag = f" [parallel {event['batch_index']}/{batch_size}]"

    # Source-provenance tag (folder mode). Inserted between L<n> and
    # the timestamp so block-detection (^L) keeps working.
    src_tag = ""
    if with_source:
        src = event.get("src", "")
        if src:
            base = os.path.basename(src)
            if base.endswith(".jsonl.xz"):
                base = base[:-9]
            elif base.endswith(".jsonl"):
                base = base[:-6]
            src_tag = f"<{base}> "

    if etype == "user_message":
        return f"L{line:<4} {src_tag}[{time}] USER: {detail}"
    elif etype == "assistant_text":
        return f"L{line:<4} {src_tag}[{time}] ASSISTANT: {detail}"
    elif etype == "tool_call":
        return f"L{line:<4} {src_tag}[{time}] TOOL{batch_tag}: {detail}"
    elif etype == "tool_result":
        err = " [ERROR]" if event.get("is_error") else ""
        return f"L{line:<4} {src_tag}[{time}] RESULT{err}: {detail}"
    elif etype == "agent_spawn":
        header = f"L{line:<4} {src_tag}[{time}] AGENT_SPAWN{batch_tag}: {detail}"
        if verbose:
            prompt = event.get("agent_prompt", "")
            if prompt:
                header += f"\n{'='*60}\nPROMPT ({len(prompt)} chars):\n{prompt}\n{'='*60}"
        return header
    elif etype == "thinking":
        return f"L{line:<4} {src_tag}[{time}] THINKING: {detail}"
    elif etype == "parse_error":
        return f"L{line:<4} {src_tag}[ERROR] {detail}"
    else:
        return f"L{line:<4} {src_tag}[{time}] {etype}: {detail}"


def format_pair(call: dict, result: dict | None, verbose: bool = False, with_source: bool = False) -> str:
    """Format a tool call + result pair."""
    lines = [format_event(call, verbose=verbose, with_source=with_source)]
    if result:
        detail = result["detail"]
        err = " [ERROR]" if result.get("is_error") else ""
        if verbose:
            lines.append(f"  └─ RESULT{err} (L{result['line']}, {len(detail)} chars):\n{detail}")
        else:
            preview = detail[:200] + ("..." if len(detail) > 200 else "")
            lines.append(f"  └─ RESULT{err} (L{result['line']}, {len(detail)} chars): {preview}")
    else:
        lines.append("  └─ RESULT: (not found)")
    return "\n".join(lines)


# --- Analysis modes ---

def _event_side_label(event: dict) -> str:
    """Short label describing an event for gap annotation."""
    etype = event["type"]
    if etype == "tool_call":
        return f"TOOL {event.get('tool_name', '?')}"
    if etype == "agent_spawn":
        return f"AGENT {event.get('agent_name', '?')}"
    if etype == "tool_result":
        call = event.get("paired_call") or {}
        name = call.get("tool_name") or call.get("agent_name") or "?"
        err = " [ERR]" if event.get("is_error") else ""
        return f"RESULT {name}{err}"
    if etype == "user_message":
        return "USER"
    if etype == "assistant_text":
        return "ASSISTANT"
    if etype == "thinking":
        return "THINKING"
    return etype


def compute_stats(events: list[dict], meta_events: list[dict] | None = None) -> str:
    """Compute session statistics."""
    meta_events = meta_events or []
    lines = []
    tool_counts = Counter()
    agent_spawns = []
    user_messages = 0
    assistant_messages = 0
    tool_results = 0
    error_results = 0
    thinking_blocks = 0
    parallel_batches = 0
    parallel_calls = 0
    first_ts = None
    last_ts = None

    for e in events:
        ts = parse_timestamp(e["ts"])
        if ts:
            if first_ts is None:
                first_ts = ts
            last_ts = ts

        if e["type"] == "tool_call":
            tool_counts[e["tool_name"]] += 1
        elif e["type"] == "agent_spawn":
            agent_spawns.append(e)
            tool_counts["Agent"] += 1
        elif e["type"] == "user_message":
            user_messages += 1
        elif e["type"] == "assistant_text":
            assistant_messages += 1
        elif e["type"] == "tool_result":
            tool_results += 1
            if e.get("is_error"):
                error_results += 1
        elif e["type"] == "thinking":
            thinking_blocks += 1

        if e.get("batch_size", 1) > 1 and e.get("batch_index") == 1:
            parallel_batches += 1
            parallel_calls += e["batch_size"]

    # Total turn compute time from system/turn_duration records
    turn_duration_ms_total = 0
    turn_duration_count = 0
    for m in meta_events:
        if m["type"] == "system" and m.get("subtype") == "turn_duration":
            dur = m["raw"].get("durationMs", 0) or 0
            turn_duration_ms_total += dur
            turn_duration_count += 1

    # queue-operation breakdown
    queue_ops: Counter = Counter()
    queue_statuses: Counter = Counter()
    for m in meta_events:
        if m["type"] != "queue-operation":
            continue
        queue_ops[m.get("operation", "")] += 1
        content = m.get("content", "") or ""
        sm = re.search(r"<status>([^<]+)</status>", content)
        if sm:
            queue_statuses[sm.group(1)] += 1

    wall_duration = None
    if first_ts and last_ts:
        delta = last_ts - first_ts
        wall_duration = delta.total_seconds()
        minutes = int(wall_duration // 60)
        seconds = int(wall_duration % 60)
        duration = f"{minutes}m {seconds}s"
    else:
        duration = ""

    lines.append("SESSION STATISTICS")
    lines.append("=" * 40)
    if first_ts:
        lines.append(f"Start:      {first_ts.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    if last_ts:
        lines.append(f"End:        {last_ts.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    if duration:
        lines.append(f"Duration:   {duration}  (wall)")
    if turn_duration_count:
        td_secs = turn_duration_ms_total / 1000.0
        td_m = int(td_secs // 60)
        td_s = int(td_secs % 60)
        td_str = f"{td_m}m {td_s}s"
        if wall_duration and wall_duration > 0:
            ratio = (td_secs / wall_duration) * 100.0
            lines.append(
                f"Compute:    {td_str}  ({turn_duration_count} turns, "
                f"{ratio:.1f}% of wall)"
            )
        else:
            lines.append(f"Compute:    {td_str}  ({turn_duration_count} turns)")
    lines.append(f"Events:     {len(events)}")
    lines.append(f"User msgs:  {user_messages}")
    lines.append(f"Asst msgs:  {assistant_messages}")
    lines.append(f"Thinking:   {thinking_blocks}")
    lines.append(f"Tool calls: {sum(tool_counts.values())}")
    lines.append(f"Results:    {tool_results} ({error_results} errors)")
    if parallel_batches:
        lines.append(f"Parallel:   {parallel_batches} batches ({parallel_calls} calls)")
    lines.append("")

    if tool_counts:
        lines.append("TOOL USAGE:")
        for tool, count in tool_counts.most_common():
            lines.append(f"  {tool:<20} {count:>4}")
        lines.append("")

    if queue_ops or queue_statuses:
        lines.append("QUEUE OPERATIONS:")
        for op, count in queue_ops.most_common():
            lines.append(f"  {op:<20} {count:>4}")
        if queue_statuses:
            lines.append("  -- task-notification statuses --")
            for status, count in queue_statuses.most_common():
                lines.append(f"  {status:<20} {count:>4}")
        lines.append("")

    if agent_spawns:
        lines.append("AGENT SPAWNS:")
        for a in agent_spawns:
            bg = "bg" if a.get("agent_bg") else "FG"
            team = a.get("agent_team", "none")
            lines.append(
                f"  L{a['line']:<4} [{short_time(a['ts'])}] "
                f"{a['agent_name']} (model={a['agent_model']}, "
                f"{bg}, team={team}, prompt={len(a.get('agent_prompt', ''))} chars)"
            )
        lines.append("")

    # Time gaps > 10s — annotated with prev/next event labels
    gaps = []
    for i in range(1, len(events)):
        t1 = parse_timestamp(events[i - 1]["ts"])
        t2 = parse_timestamp(events[i]["ts"])
        if t1 and t2:
            gap = (t2 - t1).total_seconds()
            if gap > 10:
                gaps.append((events[i - 1], events[i], gap))

    if gaps:
        lines.append("NOTABLE GAPS (>10s):")
        # Sort biggest first
        for prev, curr, gap in sorted(gaps, key=lambda x: -x[2]):
            prev_label = _event_side_label(prev)
            next_label = _event_side_label(curr)
            # AFK heuristic: gap ends with a user_message (human typed something)
            kind = "AFK" if curr["type"] == "user_message" and gap > 120 else "stall"
            lines.append(
                f"  L{prev['line']}->L{curr['line']}  {gap:>7.1f}s  [{kind:<5}]  "
                f"{prev_label}  ->  {next_label}"
            )
        lines.append("")

    return "\n".join(lines)


def compute_files(events: list[dict]) -> str:
    """Summarize all files touched in the session."""
    files: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))  # path -> op -> [lines]

    for e in events:
        if e["type"] != "tool_call":
            continue
        name = e.get("tool_name", "")
        inp = e.get("tool_input", {})
        line = e["line"]

        if name == "Read":
            files[inp.get("file_path", "?")]["read"].append(line)
        elif name == "Write":
            files[inp.get("file_path", "?")]["write"].append(line)
        elif name == "Edit":
            files[inp.get("file_path", "?")]["edit"].append(line)
        elif name == "Grep":
            path = inp.get("path", "(cwd)")
            files[path]["grep"].append(line)
        elif name == "Glob":
            path = inp.get("path", "(cwd)")
            pattern = inp.get("pattern", "")
            files[f"{path} [{pattern}]"]["glob"].append(line)
        elif name in _SHELL_TOOLS:
            # Don't parse shell commands for file paths — too unreliable
            pass

    out = ["FILES TOUCHED", "=" * 40]
    for path in sorted(files.keys()):
        ops_summary = []
        all_lines = []
        for op in sorted(files[path].keys()):
            op_lines = files[path][op]
            all_lines.extend(op_lines)
            ops_summary.append(f"{op}({len(op_lines)})")
        ops_str = ", ".join(ops_summary)
        line_refs = ",".join(f"L{ln}" for ln in sorted(all_lines)[:5])
        if len(all_lines) > 5:
            line_refs += f"…+{len(all_lines)-5}"
        out.append(f"  [{ops_str:>20}]  {path}  ({line_refs})")
    out.append(f"\nTotal: {len(files)} files/paths")
    return "\n".join(out)


def compute_commands(events: list[dict]) -> str:
    """Extract all shell commands (Bash and PowerShell).

    The prompt marker distinguishes them — `$` for Bash, `PS>` for PowerShell —
    so a mixed-shell session reads correctly instead of silently presenting
    PowerShell as Bash (or, as before, omitting it entirely).
    """
    lines = ["SHELL COMMANDS", "=" * 40]
    for e in events:
        if e["type"] == "tool_call" and e.get("tool_name") in _SHELL_TOOLS:
            cmd = e["tool_input"].get("command", "")
            desc = e["tool_input"].get("description", "")
            bg = " [bg]" if e["tool_input"].get("run_in_background") else ""
            batch_tag = ""
            if e.get("batch_size", 1) > 1:
                batch_tag = f" [parallel {e['batch_index']}/{e['batch_size']}]"
            header = f"L{e['line']:<4} [{short_time(e['ts'])}]{batch_tag}{bg}"
            if desc:
                header += f" — {desc}"
            prompt = "PS>" if e.get("tool_name") == "PowerShell" else "  $"
            lines.append(header)
            lines.append(f"{prompt} {cmd}")
            lines.append("")
    return "\n".join(lines)


def compute_edits(events: list[dict], verbose: bool = False) -> str:
    """Summarize all Edit calls (accepted vs user-modified, per-file totals)."""
    lines = ["EDITS", "=" * 40]
    # path -> {count, added, removed, modified_count}
    per_file: dict[str, dict[str, int]] = defaultdict(
        lambda: {"count": 0, "added": 0, "removed": 0, "modified_count": 0}
    )
    total_accepted = 0
    total_modified = 0
    total_added = 0
    total_removed = 0

    for e in events:
        if not (e["type"] == "tool_call" and e.get("tool_name") == "Edit"):
            continue
        inp = e["tool_input"]
        path = inp.get("file_path", "?")
        old = inp.get("old_string", "")
        new = inp.get("new_string", "")
        replace_all = inp.get("replace_all", False)

        # Pull structuredPatch / userModified from paired result
        result = e.get("paired_result") or {}
        tur = result.get("toolUseResult") or {}
        if not isinstance(tur, dict):
            tur = {}
        user_modified = bool(tur.get("userModified"))
        structured_patch = tur.get("structuredPatch") or []
        added = 0
        removed = 0
        for hunk in structured_patch:
            for patch_line in hunk.get("lines") or []:
                if not patch_line:
                    continue
                if patch_line[0] == "+":
                    added += 1
                elif patch_line[0] == "-":
                    removed += 1

        stat = per_file[path]
        stat["count"] += 1
        stat["added"] += added
        stat["removed"] += removed
        if user_modified:
            stat["modified_count"] += 1
            total_modified += 1
        else:
            total_accepted += 1
        total_added += added
        total_removed += removed

        if verbose:
            old_preview = old
            new_preview = new
        else:
            old_preview = old[:100] + ("..." if len(old) > 100 else "")
            new_preview = new[:100] + ("..." if len(new) > 100 else "")
        ra = " [replace_all]" if replace_all else ""
        status_tag = " [USER-MODIFIED]" if user_modified else ""
        err_tag = " [ERROR]" if result.get("is_error") else ""

        lines.append(
            f"L{e['line']:<4} [{short_time(e['ts'])}] {path}{ra}{status_tag}{err_tag}"
        )
        lines.append(f"  - {old_preview!r}")
        lines.append(f"  + {new_preview!r}")
        lines.append(
            f"  ({len(old)} chars -> {len(new)} chars, +{added}/-{removed} lines)"
        )
        lines.append("")

    if total_accepted + total_modified == 0:
        lines.append("(no edits found)")
        return "\n".join(lines)

    # Per-file summary
    if per_file:
        lines.append("PER-FILE SUMMARY")
        lines.append("-" * 40)
        lines.append(
            f"  {'edits':>5}  {'mod':>4}  {'+lines':>7}  {'-lines':>7}  path"
        )
        rows = sorted(
            per_file.items(),
            key=lambda kv: (-kv[1]["count"], -(kv[1]["added"] + kv[1]["removed"])),
        )
        for path, stat in rows:
            lines.append(
                f"  {stat['count']:>5}  {stat['modified_count']:>4}  "
                f"{stat['added']:>7}  {stat['removed']:>7}  {path}"
            )
        lines.append("")
        lines.append(
            f"TOTAL: {total_accepted + total_modified} edits "
            f"({total_accepted} accepted, {total_modified} user-modified), "
            f"+{total_added}/-{total_removed} lines across {len(per_file)} files"
        )

    return "\n".join(lines)


# Column definitions for the TOP-N turn tables: header -> (data_key, width).
_TOKEN_COLS: dict[str, tuple[str, int]] = {
    "output":      ("output", 8),
    "c_read":      ("read",   10),
    "c_create":    ("create", 10),
    "c_create_1h": ("eph1h",  11),
    "c_create_5m": ("eph5",   11),
    "fresh":       ("fresh",  8),
}


def _top_turns_table(
    title: str,
    rows: list[dict],
    sort_data_key: str,
    col_order: list[str],
    *,
    require_positive_top: bool = False,
) -> list[str]:
    """Render one 'TOP 10 TURNS BY X' table. Returns [] if no rows or guard fails.

    ``sort_data_key`` names the per-request field (e.g. 'output', 'read', 'create').
    ``col_order`` names display headers from _TOKEN_COLS in the order they appear.
    ``require_positive_top`` gates rendering on top row's sort key > 0.
    """
    top = sorted(rows, key=lambda r: -r[sort_data_key])[:10]
    if not top:
        return []
    if require_positive_top and top[0][sort_data_key] <= 0:
        return []

    hdr = "  ".join(f"{col:>{_TOKEN_COLS[col][1]}}" for col in col_order)
    out = [
        "",
        f"{title}:",
        f"  {'time':<19}  {'L#':>5}  {'requestId':<30}  {hdr}  {'$':>7}",
    ]
    for r in top:
        data_cells = "  ".join(
            f"{r[_TOKEN_COLS[col][0]]:>{_TOKEN_COLS[col][1]},}"
            for col in col_order
        )
        out.append(
            f"  {short_time(r['first_ts']):<19}  "
            f"L{r['first_line']:<4}  "
            f"{r['requestId']:<30}  "
            f"{data_cells}  "
            f"${r['cost']:>6.3f}"
        )
    return out


def compute_cache(meta_events: list[dict]) -> str:
    """Compute prompt cache hit rate from assistant usage meta events.

    hit_rate = cache_read / (cache_read + cache_creation + fresh_input)
    Includes per-turn breakdown and ephemeral_5m/ephemeral_1h split.
    """
    # model -> {fresh, create, read, output, turns, eph5, eph1h}
    per_model: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "fresh": 0,
            "create": 0,
            "read": 0,
            "output": 0,
            "turns": 0,
            "eph5": 0,
            "eph1h": 0,
        }
    )

    # Per-request accumulation (a turn can span multiple assistant records
    # when tool calls are chained — group by requestId).
    per_request: dict[str, dict] = {}

    for m in meta_events:
        if m["type"] != "assistant_usage":
            continue
        # Skip <synthetic> records — harness-injected stubs with no real
        # API usage (e.g. post-/exit ack, interrupted partial responses).
        # Including them inflates turn counts and pollutes per-model rows
        # with zero-everything entries.
        if m.get("model") == "<synthetic>":
            continue
        usage = m["usage"]
        model = m.get("model") or "(unknown)"
        bucket = per_model[model]
        fresh = usage.get("input_tokens", 0) or 0
        create = usage.get("cache_creation_input_tokens", 0) or 0
        read = usage.get("cache_read_input_tokens", 0) or 0
        output = usage.get("output_tokens", 0) or 0
        bucket["fresh"] += fresh
        bucket["create"] += create
        bucket["read"] += read
        bucket["output"] += output
        bucket["turns"] += 1
        eph = usage.get("cache_creation") or {}
        bucket["eph5"] += eph.get("ephemeral_5m_input_tokens", 0) or 0
        bucket["eph1h"] += eph.get("ephemeral_1h_input_tokens", 0) or 0

        req_id = m.get("requestId") or f"(no-req-L{m['line']})"
        pr = per_request.setdefault(
            req_id,
            {
                "requestId": req_id,
                "first_line": m["line"],
                "first_ts": m["ts"],
                "model": model,
                "fresh": 0,
                "create": 0,
                "read": 0,
                "output": 0,
                "eph5": 0,
                "eph1h": 0,
            },
        )
        pr["fresh"] += fresh
        pr["create"] += create
        pr["read"] += read
        pr["output"] += output
        pr["eph5"] += eph.get("ephemeral_5m_input_tokens", 0) or 0
        pr["eph1h"] += eph.get("ephemeral_1h_input_tokens", 0) or 0

    lines = ["CACHE / TOKEN USAGE", "=" * 40]

    if not per_model:
        lines.append("(no assistant usage records found)")
        return "\n".join(lines)

    def fmt(name: str, b: dict[str, int]) -> list[str]:
        total_in = b["fresh"] + b["create"] + b["read"]
        hit = (b["read"] / total_in * 100.0) if total_in else 0.0
        out = [
            f"{name}  ({b['turns']} turns)",
            f"  input total: {total_in:>12,}   "
            f"fresh: {b['fresh']:>11,}   "
            f"cache_create: {b['create']:>11,}   "
            f"cache_read: {b['read']:>12,}",
            f"  output:      {b['output']:>12,}   "
            f"hit rate: {hit:5.1f}%",
        ]
        if b.get("eph5") or b.get("eph1h"):
            out.append(
                f"  cache_create split: "
                f"ephemeral_5m: {b.get('eph5', 0):>11,}   "
                f"ephemeral_1h: {b.get('eph1h', 0):>11,}"
            )
        return out

    if len(per_model) > 1:
        totals = {
            "fresh": 0,
            "create": 0,
            "read": 0,
            "output": 0,
            "turns": 0,
            "eph5": 0,
            "eph1h": 0,
        }
        for model in sorted(per_model.keys()):
            lines.extend(fmt(model, per_model[model]))
            for k in totals:
                totals[k] += per_model[model][k]
        lines.append("")
        lines.extend(fmt("SESSION TOTAL", totals))
    else:
        model = next(iter(per_model))
        lines.extend(fmt(model, per_model[model]))

    # Cost estimation — per-model API pricing per
    # https://platform.claude.com/docs/en/about-claude/pricing.md (USD/MTok,
    # ex-VAT). Cache writes are split by TTL: 5m write = 1.25× base input, 1h
    # write = 2× base input. Tokens recorded as `cache_creation_input_tokens`
    # without an `ephemeral_*` split (older SDK versions) are charged at the
    # 5m rate (the cheaper of the two — conservative undercount, not overcount).
    # Order: most-specific keys first so substring match doesn't misroute
    # `claude-opus-4-7` to `claude-opus-4` etc.
    # Naming conventions in real model IDs:
    #   Claude 4.x family: `claude-<family>-<major>-<minor>[-<date>]`
    #     e.g. claude-opus-4-7, claude-haiku-4-5-20251001
    #   Claude 3.x family: `claude-3[-<minor>]-<family>-<date>`
    #     e.g. claude-3-haiku-20240307, claude-3-5-haiku-20241022,
    #          claude-3-opus-20240229, claude-3-7-sonnet-20250219
    # Substrings below MUST match the real ID format. Order matters
    # (most-specific first) so the substring loop doesn't misroute
    # claude-opus-4-7 to claude-opus-4 etc.
    MODEL_RATES = {
        # Claude 5 family
        "claude-fable-5":      {"fresh": 10.00, "create_5m": 12.50, "create_1h": 20.00, "read": 1.00, "output": 50.00},
        "claude-opus-5":       {"fresh": 5.00,  "create_5m": 6.25,  "create_1h": 10.00, "read": 0.50, "output": 25.00},
        "claude-sonnet-5":     {"fresh": 2.00,  "create_5m": 2.50,  "create_1h": 4.00,  "read": 0.20, "output": 10.00},
        # Claude 4.x family
        "claude-opus-4-8":     {"fresh": 5.00,  "create_5m": 6.25,  "create_1h": 10.00, "read": 0.50, "output": 25.00},
        "claude-opus-4-7":     {"fresh": 5.00,  "create_5m": 6.25,  "create_1h": 10.00, "read": 0.50, "output": 25.00},
        "claude-opus-4-6":     {"fresh": 5.00,  "create_5m": 6.25,  "create_1h": 10.00, "read": 0.50, "output": 25.00},
        "claude-opus-4-5":     {"fresh": 5.00,  "create_5m": 6.25,  "create_1h": 10.00, "read": 0.50, "output": 25.00},
        "claude-opus-4-1":     {"fresh": 15.00, "create_5m": 18.75, "create_1h": 30.00, "read": 1.50, "output": 75.00},
        "claude-opus-4":       {"fresh": 15.00, "create_5m": 18.75, "create_1h": 30.00, "read": 1.50, "output": 75.00},
        "claude-sonnet-4-6":   {"fresh": 3.00,  "create_5m": 3.75,  "create_1h": 6.00,  "read": 0.30, "output": 15.00},
        "claude-sonnet-4-5":   {"fresh": 3.00,  "create_5m": 3.75,  "create_1h": 6.00,  "read": 0.30, "output": 15.00},
        "claude-sonnet-4":     {"fresh": 3.00,  "create_5m": 3.75,  "create_1h": 6.00,  "read": 0.30, "output": 15.00},
        "claude-haiku-4-5":    {"fresh": 1.00,  "create_5m": 1.25,  "create_1h": 2.00,  "read": 0.10, "output": 5.00},
        # Claude 3.x family (real ID layout: claude-3[-<minor>]-<family>-<date>)
        "claude-3-7-sonnet-":  {"fresh": 3.00,  "create_5m": 3.75,  "create_1h": 6.00,  "read": 0.30, "output": 15.00},
        "claude-3-5-sonnet-":  {"fresh": 3.00,  "create_5m": 3.75,  "create_1h": 6.00,  "read": 0.30, "output": 15.00},
        "claude-3-5-haiku-":   {"fresh": 0.80,  "create_5m": 1.00,  "create_1h": 1.60,  "read": 0.08, "output": 4.00},
        "claude-3-opus-":      {"fresh": 15.00, "create_5m": 18.75, "create_1h": 30.00, "read": 1.50, "output": 75.00},
        "claude-3-haiku-":     {"fresh": 0.25,  "create_5m": 0.30,  "create_1h": 0.50,  "read": 0.03, "output": 1.25},
    }
    DEFAULT_RATES = MODEL_RATES["claude-opus-4-7"]  # fallback
    # Attach per-turn cost to each request so it can be shown in top-10 tables
    # and summed for the totals block below.
    cost_totals = {"fresh": 0.0, "create_5m": 0.0, "create_1h": 0.0, "read": 0.0, "output": 0.0}
    fallback_models = set()
    for r in per_request.values():
        model = r.get("model", "")
        rates = None
        for key, r_map in MODEL_RATES.items():
            if key in model:
                rates = r_map
                break
        if rates is None:
            rates = DEFAULT_RATES
            fallback_models.add(model or "(unknown)")
        # Split cache_create tokens into 5m / 1h via per-request eph5/eph1h.
        # Tokens unaccounted for in either bucket (legacy SDK) → charge at 5m.
        eph5 = r.get("eph5", 0)
        eph1h = r.get("eph1h", 0)
        unsplit_create = max(0, r["create"] - eph5 - eph1h)
        f_cost = r["fresh"] * rates["fresh"] / 1_000_000
        c5_cost = (eph5 + unsplit_create) * rates["create_5m"] / 1_000_000
        c1h_cost = eph1h * rates["create_1h"] / 1_000_000
        rd_cost = r["read"] * rates["read"] / 1_000_000
        o_cost = r["output"] * rates["output"] / 1_000_000
        r["cost"] = f_cost + c5_cost + c1h_cost + rd_cost + o_cost
        cost_totals["fresh"] += f_cost
        cost_totals["create_5m"] += c5_cost
        cost_totals["create_1h"] += c1h_cost
        cost_totals["read"] += rd_cost
        cost_totals["output"] += o_cost

    all_requests = list(per_request.values())
    lines.extend(_top_turns_table(
        "TOP 10 TURNS BY OUTPUT TOKENS",
        all_requests,
        "output",
        ["output", "c_read", "c_create_1h", "c_create_5m", "fresh"],
    ))
    lines.extend(_top_turns_table(
        "TOP 10 TURNS BY CACHE CREATE",
        all_requests,
        "create",
        ["c_create_1h", "c_create_5m", "c_read", "output", "fresh"],
        require_positive_top=True,
    ))
    lines.extend(_top_turns_table(
        "TOP 10 TURNS BY CACHE READ",
        all_requests,
        "read",
        ["c_read", "c_create_1h", "c_create_5m", "output", "fresh"],
        require_positive_top=True,
    ))

    total_cost = sum(cost_totals.values())
    total_turns = sum(b["turns"] for b in per_model.values())
    if total_cost > 0:
        lines.append("")
        lines.append("ESTIMATED API COST")
        lines.append("  Rates (USD ex-VAT, fresh / 5m_write / 1h_write / read / output per MTok): "
                     "Fable 5 $10/$12.50/$20/$1/$50, "
                     "Opus 5 / Opus 4.5+ $5/$6.25/$10/$0.50/$25, "
                     "Sonnet 5 $2/$2.50/$4/$0.20/$10, "
                     "Sonnet 4.x $3/$3.75/$6/$0.30/$15, "
                     "Haiku 4.5 $1/$1.25/$2/$0.10/$5. "
                     "5m write = 1.25× base input, 1h write = 2× base input. "
                     "Tokens with no ephemeral_* split (legacy SDK) charged at 5m rate.")
        if fallback_models:
            lines.append(f"  NOTE: no rate key matched {', '.join(sorted(fallback_models))} — "
                         f"priced at the Opus 4.x fallback rate; add it to MODEL_RATES.")
        lines.append("")
        lines.append(f"  {'Category':<22} {'Cost':>10}  {'%':>6}")
        lines.append(f"  {'-'*22} {'-'*10}  {'-'*6}")
        for cat, label in [("read", "Cache read"),
                           ("create_1h", "Cache create 1h"),
                           ("create_5m", "Cache create 5m"),
                           ("output", "Output"),
                           ("fresh", "Fresh input")]:
            c = cost_totals[cat]
            pct = c / total_cost * 100 if total_cost else 0
            lines.append(f"  {label:<22} ${c:>9.2f}  {pct:>5.1f}%")
        lines.append(f"  {'-'*22} {'-'*10}  {'-'*6}")
        lines.append(f"  {'TOTAL':<22} ${total_cost:>9.2f}")
        if total_turns:
            per_turn = total_cost / total_turns
            lines.append(f"  {'PER TURN':<22} ${per_turn:>9.4f}  ({total_turns} turns)")

    return "\n".join(lines)


def compute_writes(events: list[dict], verbose: bool = False) -> str:
    """Summarize all Write calls."""
    lines = ["WRITES", "=" * 40]
    count = 0
    for e in events:
        if e["type"] == "tool_call" and e.get("tool_name") == "Write":
            inp = e["tool_input"]
            path = inp.get("file_path", "?")
            content = inp.get("content", "")
            count += 1

            lines.append(f"L{e['line']:<4} [{short_time(e['ts'])}] {path} ({len(content)} chars)")
            if verbose:
                lines.append(content)
            else:
                preview = content[:200] + ("..." if len(content) > 200 else "")
                lines.append(f"  {preview}")
            lines.append("")
    lines.append(f"Total: {count} files written")
    return "\n".join(lines)


def _tool_call_excerpt(call: dict, limit: int = 80) -> str:
    """One-line excerpt for a tool call (command/path/pattern depending on tool)."""
    if call["type"] == "agent_spawn":
        return f"agent={call.get('agent_name', '?')}"
    inp = call.get("tool_input", {}) or {}
    name = call.get("tool_name", "")
    if name in _SHELL_TOOLS:
        text = inp.get("command", "")
    elif name in ("Read", "Write", "Edit", "NotebookEdit"):
        text = inp.get("file_path", "")
    elif name == "Grep":
        text = f"{inp.get('pattern', '')}  in {inp.get('path', '(cwd)')}"
    elif name == "Glob":
        text = f"{inp.get('pattern', '')}  in {inp.get('path', '(cwd)')}"
    elif name == "WebFetch":
        text = inp.get("url", "")
    else:
        text = json.dumps(inp)[:200]
    text = text.replace("\n", " ")
    if len(text) > limit:
        text = text[:limit] + "..."
    return text


def _bucket_label(value: float, spec: list[tuple[str, float, float]]) -> str | None:
    """Return the label of the [lo, hi) bucket containing value, or None."""
    for label, lo, hi in spec:
        if lo <= value < hi:
            return label
    return None


def compute_latency(events: list[dict]) -> str:
    """Rank tool calls by wall-clock latency (result_ts - use_ts)."""
    lines = ["TOOL LATENCY", "=" * 40]

    # per-tool histogram buckets — math-interval notation [a, b) = a inclusive, b exclusive
    buckets_spec = [
        ("[0, 100ms)",     0.0,    0.1),
        ("[100ms, 1s)",    0.1,    1.0),
        ("[1s, 5s)",       1.0,    5.0),
        ("[5s, 30s)",      5.0,   30.0),
        ("[30s, 2m)",     30.0,  120.0),
        ("[2m, inf)",    120.0, float("inf")),
    ]

    per_tool_hist: dict[str, Counter] = defaultdict(Counter)
    per_tool_totals: dict[str, dict[str, float]] = defaultdict(
        lambda: {"count": 0, "total": 0.0, "max": 0.0}
    )
    all_pairs: list[tuple[float, dict, dict]] = []

    for e in events:
        if e["type"] not in ("tool_call", "agent_spawn"):
            continue
        result = e.get("paired_result")
        if not result:
            continue
        t_call = parse_timestamp(e["ts"])
        t_res = parse_timestamp(result["ts"])
        if not t_call or not t_res:
            continue
        dur = (t_res - t_call).total_seconds()
        if dur < 0:
            dur = 0.0
        name = e.get("tool_name") or ("Agent" if e["type"] == "agent_spawn" else "?")
        per_tool_totals[name]["count"] += 1
        per_tool_totals[name]["total"] += dur
        if dur > per_tool_totals[name]["max"]:
            per_tool_totals[name]["max"] = dur
        lbl = _bucket_label(dur, buckets_spec)
        if lbl:
            per_tool_hist[name][lbl] += 1
        all_pairs.append((dur, e, result))

    if not all_pairs:
        lines.append("(no paired tool call/result events)")
        return "\n".join(lines)

    lines.append("PER-TOOL HISTOGRAM:")
    header = (
        f"  {'tool':<18}  {'n':>5}  {'avg':>8}  {'max':>8}   "
        + "  ".join(f"{lbl:>13}" for lbl, _, _ in buckets_spec)
    )
    lines.append(header)
    for name in sorted(per_tool_totals.keys(), key=lambda n: -per_tool_totals[n]["total"]):
        t = per_tool_totals[name]
        avg = t["total"] / t["count"] if t["count"] else 0.0
        hist = per_tool_hist[name]
        buckets = "  ".join(f"{hist.get(lbl, 0):>13}" for lbl, _, _ in buckets_spec)
        lines.append(
            f"  {name:<18}  {int(t['count']):>5}  "
            f"{avg:>7.2f}s  {t['max']:>7.2f}s   {buckets}"
        )
    lines.append("")

    lines.append("SLOWEST 20 CALLS:")
    lines.append(f"  {'time':<19}  {'L#':>5}  {'dur':>9}  {'tool':<14}  detail")
    for dur, call, result in sorted(all_pairs, key=lambda x: -x[0])[:20]:
        name = call.get("tool_name") or ("Agent" if call["type"] == "agent_spawn" else "?")
        lines.append(
            f"  {short_time(call['ts']):<19}  "
            f"L{call['line']:<4}  "
            f"{dur:>8.2f}s  {name:<14}  {_tool_call_excerpt(call)}"
        )

    return "\n".join(lines)


# Event types that count as "the assistant has started responding."
_ASSISTANT_SIDE_TYPES = {"assistant_text", "thinking", "tool_call", "agent_spawn"}


_INSTRUMENTATION_USER_PREFIXES = (
    "<bash-input>",
    "<bash-stdout>",
    "<bash-stderr>",
    "<local-command-caveat>",
    "<local-command-stdout>",
    "<local-command-stderr>",
    "<command-name>",
    "<command-message>",
    "<command-args>",
)
_INTERRUPT_MARKER = "[Request interrupted by user"  # covers both "...user]" and "...user for tool use]"


def _is_instrumentation_user_msg(event: dict) -> bool:
    """True if the user_message is harness instrumentation (bash IO + caveat).

    These are explicitly flagged "DO NOT respond" and never receive an
    assistant reply, so they should not anchor a reply-latency measurement.
    """
    if event.get("type") != "user_message":
        return False
    text = (event.get("detail") or "").lstrip()
    return any(text.startswith(p) for p in _INSTRUMENTATION_USER_PREFIXES)


def _is_interrupt_user_msg(event: dict) -> bool:
    """True if the user_message is a `[Request interrupted by user]` marker.

    Interrupts terminate a latency window the same way an assistant response
    does — they mark 'user gave up waiting'. The gap up to the interrupt
    counts as reply latency.
    """
    if event.get("type") != "user_message":
        return False
    text = (event.get("detail") or "").lstrip()
    return text.startswith(_INTERRUPT_MARKER)


# A token starts and ends alphanumeric, with internal dashes/underscores/
# apostrophes allowed (so "reply-latency", "min_count", "don't" stay whole). The
# aggregate ranking additionally drops tokens shorter than 3 chars and any token
# containing a digit (`_is_word` — kills hex/uuid/dimension junk like 71e3af8,
# 800px, 11t15); the explicit single-word mode matches against this raw set so a
# caller can still search anything.
_WORD_RE = re.compile(r"[a-z0-9](?:[a-z0-9_'-]*[a-z0-9])?")
# Common English / harness-structural tokens that carry no signal about reply
# speed; excluded so the word-effect ranking surfaces meaningful terms.
_REPLY_LATENCY_STOPWORDS = frozenset("""
the and for are but not you all any can had her was one our out his has how man
its let put say she too use that with have this will your from they know want been
good much some time very when come here just like long make many more only over
such take than them well were what where which while who whom why would there their
then else into onto upon about after again could should might must does did done
your you're it's i'm that's don't can't won't isn't aren't doesn't didn't couldn't
shouldn't wouldn't yes please now also still even ever never always thing things
""".split())


def _is_word(tok: str) -> bool:
    """A token qualifies for the aggregate word ranking: >=3 chars, no digit (so
    hex/uuid/dimension junk like 71e3af8 / 800px / 11t15 is dropped), not a
    stopword. Dashes/underscores inside the token are fine — see _WORD_RE."""
    return (len(tok) >= 3
            and not any(c.isdigit() for c in tok)
            and tok not in _REPLY_LATENCY_STOPWORDS)


# A user-role record whose STRING content starts with one of these markers is
# harness/synthetic (a notification, observer payload, slash-command expansion,
# bash IO) — NOT a human-typed prompt. The parser stores both as `user_message`
# string content with no structural flag, so word attribution distinguishes them
# by leading marker (see _is_human_prompt). Real tool_results are a separate
# event type and never anchor a reply-latency pair to begin with.
_NON_HUMAN_PROMPT_PREFIXES = _INSTRUMENTATION_USER_PREFIXES + (
    "<task-notification>",
    "<observed_from_primary_session>",
    "<system-reminder>",
    "<post-tool-use>",
)
# Harness-injected blocks that ride INSIDE an otherwise-human prompt (a trailing
# system-reminder, an embedded observer block); stripped before tokenizing so
# their words aren't attributed to the human's request.
_INJECTED_BLOCK_RE = re.compile(
    r"<(system-reminder|observed_from_primary_session)\b.*?</\1>", re.DOTALL)


def _is_human_prompt(event: dict) -> bool:
    """True if a reply-latency anchor is a genuine human-typed prompt, not a
    harness/synthetic user-role event (notification, observer payload, command
    expansion, bash IO). Distinguishes only by leading marker — the parser stores
    both kinds as `user_message` string content with no structural flag."""
    text = (event.get("detail") or "").lstrip()
    return bool(text) and not text.startswith(_NON_HUMAN_PROMPT_PREFIXES)


def _replayed_user_message_lines(events: list[dict]) -> set[tuple[str | None, int]]:
    """Return (src, line) pairs of user_message events that replay a previous
    prompt within the same source file.

    A subagent re-dispatch appends a second copy of the original user record
    with the SAME uuid but a DIFFERENT line number. Those replayed records must
    not anchor, supersede, or terminate reply-latency measurements. A record
    that emits several user_message events for multiple text blocks all share
    the same line, so same-uuid at the SAME line is NOT a replay."""
    seen: dict[tuple[str | None, str], int] = {}
    replays: set[tuple[str | None, int]] = set()
    for e in events:
        if e["type"] != "user_message":
            continue
        u = e.get("uuid", "")
        if not u:
            continue
        src = e.get("src")
        line = e.get("line", 0)
        key = (src, u)
        first_line = seen.get(key)
        if first_line is None:
            seen[key] = line
        elif first_line != line:
            replays.add((src, line))
    return replays


def _reply_latency_pairs(events: list[dict]) -> list[tuple[float, dict, str]]:
    """(latency_seconds, anchoring user_event, responding_model) per measurable reply.

    Shared by compute_reply_latency and compute_reply_latency_words so both use
    the IDENTICAL anchor filter: skip harness-instrumentation + interrupt user
    messages, skip replayed re-dispatch prompts (duplicate-uuid user records at a
    later line in the same file), drop anchors superseded by a later real user
    message, drop measurements with no following terminator or a negative
    (compaction-artifact) duration. See compute_reply_latency's docstring for
    the full rule set. The model is the terminator's tagged model, or
    '(interrupted)' when the reply was cut off by a `[Request interrupted by
    user]` marker."""
    pairs: list[tuple[float, dict, str]] = []
    replayed = _replayed_user_message_lines(events)
    n = len(events)
    for i, e in enumerate(events):
        if e["type"] != "user_message":
            continue
        if _is_instrumentation_user_msg(e):
            continue
        if _is_interrupt_user_msg(e):
            continue  # interrupt is a terminator, never an anchor
        # Folder mode merges every session file into one ts-sorted list, so an
        # anchor near the end of session A can otherwise pair with the first
        # event of a DIFFERENT session B that started hours later (a bogus
        # multi-hour "reply latency"). Confine the terminator walk to the
        # anchor's own source file; a cross-file event is skipped, never a
        # terminator or a superseder. Single-file mode: every `src` is equal, so
        # this is a no-op. `src` is always set by extract_events / _build_from_cache.
        src = e.get("src")
        if (src, e.get("line", 0)) in replayed:
            continue  # re-dispatch replay: invisible for latency pairing
        t_user = parse_timestamp(e["ts"])
        if not t_user:
            continue
        found = None
        superseded = False
        for j in range(i + 1, n):
            nxt = events[j]
            if nxt.get("src") != src:
                continue  # event from another session file — not this anchor's
            if nxt["type"] in _ASSISTANT_SIDE_TYPES:
                found = nxt
                break
            if nxt["type"] == "user_message":
                if (src, nxt.get("line", 0)) in replayed:
                    continue  # replayed prompt is invisible, keep walking
                if _is_instrumentation_user_msg(nxt):
                    continue
                if _is_interrupt_user_msg(nxt):
                    found = nxt
                    break
                superseded = True  # real user message supersedes this anchor
                break
        if superseded or not found:
            continue
        t_reply = parse_timestamp(found["ts"])
        if not t_reply:
            continue
        dur = (t_reply - t_user).total_seconds()
        if dur < 0:
            continue  # compaction/session-restore artifact — drop, don't clamp
        if found["type"] in _ASSISTANT_SIDE_TYPES:
            model = found.get("model") or "(unknown)"
        else:
            model = "(interrupted)"  # terminator was an interrupt marker
        pairs.append((dur, e, model))
    return pairs


def _reply_latency_pairs_for_paths(
    src_paths: list[str],
    after: "str | None" = None,
    before: "str | None" = None,
) -> list[tuple[float, dict, str]]:
    """Reply-latency pairs computed PER SESSION FILE, then concatenated.

    Reply latency is a within-session sequential measurement, so folder mode must
    NOT run the anchor walk over the globally-merged, ts-sorted, cross-file-
    deduped event stream. Two artifacts come from doing so:
      1. An anchor near the end of session A pairs with the first event of a
         later session B — a bogus multi-hour "latency".
      2. Cross-file uuid dedup strips a session's OWN reply records (they survive
         only under a duplicate log's src), so the anchor bridges to a far later
         same-file event — an even larger phantom gap.
    Both vanish when each file is parsed on its own COMPLETE records (no
    ``seen_uuids`` dedup) and walked in native order. `src_paths` is the loaded
    source set (folder mode = every contributing file); a file that was fully
    deduped away is a strict subset of an included file, whose complete re-read
    here already contains those records — so every turn is counted, and records
    shared by genuinely duplicated session logs are counted once per file, a
    negligible over-count next to the artifacts removed. `after`/`before` scope
    each file's events by timestamp (parity with the merged path); --range /
    --path / --tool do not apply (per-file line ranges are meaningless across a
    corpus, same caveat as --branches)."""
    pairs: list[tuple[float, dict, str]] = []
    for path in src_paths:
        try:
            evs, _ = extract_events(path)  # complete records; NO cross-file dedup
        except OSError:
            continue
        if after or before:
            evs = filter_time(evs, after, before)
        pairs.extend(_reply_latency_pairs(evs))
    return pairs


def _collect_reply_latency_pairs(
    events: list[dict],
    src_paths: "list[str] | None",
    after: "str | None",
    before: "str | None",
) -> list[tuple[float, dict, str]]:
    """Pair source for the whole --reply-latency* family. With 2+ source files
    (folder mode) compute per-file to avoid cross-session bridging (see
    _reply_latency_pairs_for_paths); otherwise (single file, or a pre-built event
    list such as a test's) walk `events` directly so range/path/tool filters
    already applied upstream are honored."""
    if src_paths and len(src_paths) > 1:
        return _reply_latency_pairs_for_paths(src_paths, after, before)
    return _reply_latency_pairs(events)


def compute_reply_latency(
    events: list[dict],
    top_n: int = 20,
    src_paths: "list[str] | None" = None,
    after: "str | None" = None,
    before: "str | None" = None,
) -> str:
    """Wall-clock gap between each user message and the assistant's first response event.

    For each user_message event, walks forward through the transcript until
    it finds a terminator — either an assistant-side event (assistant_text,
    thinking, tool_call, agent_spawn) or a
    `[Request interrupted by user]` marker (which counts the gap as 'took
    N seconds and didn't respond'). The latency is (terminator_ts - user_event_ts).

    Filtered user messages (do NOT anchor a latency measurement):
    - `<bash-input>`, `<bash-stdout>`, `<bash-stderr>`, `<local-command-caveat>`
      — harness instrumentation emitted around every local `!bash` command,
      explicitly flagged "DO NOT respond". These arrive in bursts of 3 with
      identical millisecond timestamps and would otherwise triple-count.
    - Replayed re-dispatch prompts (a duplicate-uuid user record at a later line
      in the same file, e.g. a subagent re-dispatch) — they replay a stale
      timestamp and must not anchor, supersede, or terminate a latency pair.

    Kept user messages (do anchor a latency measurement):
    - Plain text prompts.
    - `<task-notification>` and other harness events that I actually respond to.

    `[Request interrupted by user]` is treated as an assistant-side terminator
    only — it stops the clock on the preceding message but does NOT anchor a
    new latency measurement. Semantically it's the user aborting the reply,
    not sending a new prompt.

    Skipped:
    - User messages with no following terminator (session ended).
    - Intervening tool_result events or other non-assistant filler — the
      walk continues past them.

    Folder mode: latency is a within-session metric, so it is computed PER FILE
    on each file's complete records rather than over the merged, ts-sorted,
    cross-file-deduped stream — see _reply_latency_pairs_for_paths. This prevents
    an end-of-session anchor from pairing with the next session's first event.
    """
    lines = ["REPLY LATENCY", "=" * 40]
    lines.append("(gap from each user message to the assistant's first response event)")
    lines.append("")

    # Collect (latency_seconds, user_event) pairs. Folder mode → per-file;
    # single file / pre-built list → walk `events` directly.
    pairs = _collect_reply_latency_pairs(events, src_paths, after, before)

    if not pairs:
        lines.append("(no user messages with following assistant events)")
        return "\n".join(lines)

    # Summary statistics.
    durations = sorted(d for d, _, _ in pairs)
    count = len(durations)
    total = sum(durations)
    mean = total / count
    mn = durations[0]
    mx = durations[-1]
    median = durations[count // 2] if count % 2 == 1 else (durations[count // 2 - 1] + durations[count // 2]) / 2.0
    p95_idx = max(0, min(count - 1, int(round(0.95 * (count - 1)))))
    p95 = durations[p95_idx]

    lines.append(
        f"count: {count}   min: {mn:.2f}s   max: {mx:.2f}s   "
        f"median: {median:.2f}s   p95: {p95:.2f}s   mean: {mean:.2f}s"
    )
    lines.append("")

    # Bucket histogram. Math-interval notation: [a, b) means a inclusive,
    # b exclusive. Removes the "<5s" ambiguity (does it mean [0,5) or [2,5)?).
    buckets_spec = [
        ("[0, 0.5s)",   0.0,    0.5),
        ("[0.5s, 1s)",  0.5,    1.0),
        ("[1s, 2s)",    1.0,    2.0),
        ("[2s, 5s)",    2.0,    5.0),
        ("[5s, 10s)",   5.0,   10.0),
        ("[10s, 30s)", 10.0,   30.0),
        ("[30s, 2m)",  30.0,  120.0),
        ("[2m, inf)", 120.0, float("inf")),
    ]
    hist: Counter = Counter()
    for d in durations:
        lbl = _bucket_label(d, buckets_spec)
        if lbl:
            hist[lbl] += 1
    lines.append("HISTOGRAM:")
    for label, _, _ in buckets_spec:
        c = hist.get(label, 0)
        bar = "#" * min(c, 60)
        lines.append(f"  {label:<13}  {c:>5}  {bar}")
    lines.append("")

    # Top-N slowest.
    lines.append(f"SLOWEST {top_n}:")
    lines.append(f"  {'time':<19}  {'L#':>5}  {'latency':>9}  preview")
    slowest = sorted(pairs, key=lambda x: -x[0])[:top_n]
    for dur, ev, _model in slowest:
        preview = (ev.get("detail", "") or "").replace("\n", " ").strip()
        if len(preview) > 60:
            preview = preview[:57] + "..."
        lines.append(
            f"  {short_time(ev['ts']):<19}  "
            f"L{ev['line']:<4}  "
            f"{dur:>8.2f}s  {preview}"
        )

    return "\n".join(lines)


def _prompt_preview(text: str, limit: int = 100) -> str:
    """First `limit` chars of `text`, or up to the first newline — whichever is
    SHORTER. Leading whitespace is trimmed so an injected-block remnant or a blank
    lead doesn't produce an empty preview."""
    s = (text or "").lstrip()
    nl = s.find("\n")
    if nl != -1:
        s = s[:nl]
    return s[:limit].rstrip()


def _word_effect_rows(pairs: list[tuple[float, dict]], min_count: int):
    """(rows, count, mean) for one set of (latency, human-prompt-event) pairs.

    rows = (effect, word, n, with_mean, without_mean, first_prompt_preview), one
    per word occurring in >= min_count prompts (and not in every prompt, which
    would have no contrast). effect = mean latency of prompts WITH the word minus
    mean WITHOUT it; positive = slower."""
    total_count = len(pairs)
    total_sum = sum(d for d, _ in pairs)
    word_durs: dict[str, list[float]] = defaultdict(list)
    word_first: dict[str, str] = {}
    for dur, ev in pairs:
        cleaned = _INJECTED_BLOCK_RE.sub(" ", ev.get("detail") or "")
        for w in set(_WORD_RE.findall(cleaned.lower())):
            if not _is_word(w):
                continue
            word_durs[w].append(dur)
            if w not in word_first:
                word_first[w] = _prompt_preview(cleaned)
    rows = []
    for w, durs in word_durs.items():
        c = len(durs)
        if c < min_count or c == total_count:
            continue
        with_sum = sum(durs)
        with_mean = with_sum / c
        without_mean = (total_sum - with_sum) / (total_count - c)
        rows.append((with_mean - without_mean, w, c, with_mean, without_mean, word_first[w]))
    mean = total_sum / total_count if total_count else 0.0
    return rows, total_count, mean


def compute_reply_latency_words(
    events: list[dict],
    top_n: int = 10,
    min_count: int = 5,
    src_paths: "list[str] | None" = None,
    after: "str | None" = None,
    before: "str | None" = None,
) -> str:
    """Words in human prompts ranked by their effect on reply latency, SPLIT BY MODEL.

    For every measured reply (same anchor logic as --reply-latency) whose anchor is
    a GENUINE HUMAN PROMPT (synthetic user-role events — task-notifications, observer
    payloads, slash-command expansions, bash IO — excluded via _is_human_prompt;
    harness blocks injected inside a prompt are stripped), the prompt is tokenized
    into distinct lowercased word tokens (>=3 chars, stopwords removed). Pairs are
    grouped by the RESPONDING MODEL and, within each model,
        effect = mean(latency | prompt CONTAINS the word)
               - mean(latency | prompt does NOT contain the word)
    Positive = prompts with that word were answered SLOWER (within that model);
    negative = FASTER. Each word counts a prompt once. Words in fewer than
    `min_count` of a model's prompts are dropped. Each row also shows the first 100
    chars (or first line, whichever is shorter) of the first prompt carrying the
    word. Per model, the top `top_n` words each direction are printed, by effect.

    Splitting by model is the point: latency varies far more BETWEEN models than
    between word choices, so a global ranking mostly sorts words by which model
    happened to answer them. Within-model contrast removes that confound. What
    remains is still correlational (response length co-varies with word choice),
    not causal.

    Folder mode computes pairs PER FILE (see _reply_latency_pairs_for_paths) so
    anchors never bridge across session boundaries."""
    triples = [t for t in _collect_reply_latency_pairs(events, src_paths, after, before)
               if _is_human_prompt(t[1])]
    lines = ["REPLY LATENCY BY WORD (split by model)", "=" * 40]
    if len(triples) < 2 * min_count:
        lines.append(f"(not enough samples: {len(triples)} human-prompt replies, "
                     f"need >= {2 * min_count} for min_count={min_count})")
        return "\n".join(lines)

    by_model: dict[str, list[tuple[float, dict]]] = defaultdict(list)
    for dur, ev, model in triples:
        by_model[model].append((dur, ev))

    lines.append(f"({len(triples)} human-prompt replies across {len(by_model)} model(s); "
                 f"effect computed WITHIN each model, min_count={min_count})")
    lines.append("(human prompts only; correlational — response length co-varies with words — not causal)")
    lines.append("(preview = first 100 chars / first line of the first prompt with the word)")

    header = f"    {'effect':>9}  {'word':<20}  {'count':<7}  with/without    first prompt"

    def _fmt(r):
        eff, w, c, wm, wom, prev = r
        sign = "+" if eff >= 0 else "-"
        return (f"    {sign}{abs(eff):>7.2f}s  {w:<20}  n={c:<5}  "
                f"{wm:6.2f}/{wom:6.2f}s  {prev}")

    skipped = []
    for model in sorted(by_model, key=lambda k: -len(by_model[k])):
        mpairs = by_model[model]
        if len(mpairs) < 2 * min_count:
            skipped.append((model, len(mpairs)))
            continue
        rows, mcount, mmean = _word_effect_rows(mpairs, min_count)
        lines.append("")
        lines.append(f"### {model}  —  {mcount} replies, mean {mmean:.2f}s, {len(rows)} words qualified")
        if not rows:
            lines.append(f"    (no words reached min_count={min_count})")
            continue
        slower = sorted(rows, key=lambda r: -r[0])[:top_n]
        faster = sorted(rows, key=lambda r: r[0])[:top_n]
        lines.append(f"  -> SLOWER (top {len(slower)}):")
        lines.append(header)
        lines.extend(_fmt(r) for r in slower)
        lines.append(f"  -> FASTER (top {len(faster)}):")
        lines.append(header)
        lines.extend(_fmt(r) for r in faster)

    if skipped:
        lines.append("")
        lines.append(f"(skipped {len(skipped)} model(s) with < {2 * min_count} human-prompt replies: "
                     + ", ".join(f"{m} [{n}]" for m, n in skipped) + ")")
    return "\n".join(lines)


def compute_reply_latency_word(
    events: list[dict],
    word: str,
    src_paths: "list[str] | None" = None,
    after: "str | None" = None,
    before: "str | None" = None,
) -> str:
    """With-vs-without reply latency for ONE specific word, split by model.

    Like --reply-latency-words but for a single caller-chosen token: for each
    model, splits that model's human-prompt replies into the ones whose prompt
    CONTAINS `word` and the ones that don't, and reports n + mean latency for
    each side plus the effect (mean WITH minus mean WITHOUT). Because the word is
    chosen explicitly, NO stopword filter applies and NO min_count gate — every
    model where the word appears at least once is shown, with up to 3 example
    prompt previews. Same anchor/human-prompt filtering and injected-block
    stripping as the aggregate mode. Matching is token-level (the same tokenizer
    as --reply-latency-words), so `word` must be a single word, not a phrase.

    Still correlational: a word co-varies with the kind of request it rides on."""
    target = (word or "").strip().lower()
    lines = [f'REPLY LATENCY FOR WORD "{target}" (split by model)', "=" * 40]
    if not target:
        lines.append("(no word given)")
        return "\n".join(lines)

    triples = [t for t in _collect_reply_latency_pairs(events, src_paths, after, before)
               if _is_human_prompt(t[1])]
    if not triples:
        lines.append("(no human-prompt replies)")
        return "\n".join(lines)

    by_model: dict[str, list[tuple[float, dict]]] = defaultdict(list)
    for dur, ev, model in triples:
        by_model[model].append((dur, ev))

    lines.append(f'({len(triples)} human-prompt replies across {len(by_model)} model(s); '
                 f'effect = mean WITH "{target}" minus mean WITHOUT, within each model)')
    lines.append("(human prompts only; correlational, not causal)")

    any_hit = False
    grand_with = 0
    for model in sorted(by_model, key=lambda k: -len(by_model[k])):
        mpairs = by_model[model]
        with_d: list[float] = []
        without_d: list[float] = []
        examples: list[tuple[float, str]] = []
        for dur, ev in mpairs:
            cleaned = _INJECTED_BLOCK_RE.sub(" ", ev.get("detail") or "")
            if target in set(_WORD_RE.findall(cleaned.lower())):
                with_d.append(dur)
                if len(examples) < 3:
                    examples.append((dur, _prompt_preview(cleaned)))
            else:
                without_d.append(dur)
        if not with_d:
            continue
        any_hit = True
        grand_with += len(with_d)
        wmean = sum(with_d) / len(with_d)
        womean = sum(without_d) / len(without_d) if without_d else 0.0
        eff = wmean - womean
        sign = "+" if eff >= 0 else "-"
        verdict = "slower" if eff >= 0 else "faster"
        lines.append("")
        lines.append(f"### {model}  —  {len(mpairs)} replies")
        lines.append(f'    WITH "{target}":  n={len(with_d):<5}  mean={wmean:7.2f}s')
        lines.append(f"    WITHOUT       :  n={len(without_d):<5}  mean={womean:7.2f}s")
        lines.append(f"    effect        :  {sign}{abs(eff):.2f}s ({verdict} with the word)")
        for d, prev in examples:
            lines.append(f"      e.g. [{d:6.2f}s] {prev}")

    if not any_hit:
        lines.append("")
        lines.append(f'(word "{target}" did not appear in any human prompt)')
    else:
        lines.append("")
        lines.append(f'(total: "{target}" appeared in {grand_with} human prompt(s))')
    return "\n".join(lines)


_TASK_ID_RE = re.compile(r"Command running in background with ID:\s*(\w+)")
_TASK_OUTPUT_RE = re.compile(
    # Anchor on `.output` (canonical bg-task suffix) so the capture stops at
    # the path, not at trailing sentence punctuation. Claude Code 2.1.143+
    # tool_results phrase the launch confirmation as
    # `Output is being written to: <path>. You will be notified...` — the
    # `\S+` form gobbled the sentence-terminating period and broke Pass 3's
    # exact-string lookup against Read/Bash file_path arguments.
    r"Output is being written to:\s*(\S+?\.output)\b"
)
_NOTIF_TASK_ID_RE = re.compile(r"<task-id>([^<]+)</task-id>")
_NOTIF_TOOL_USE_RE = re.compile(r"<tool-use-id>([^<]+)</tool-use-id>")
_NOTIF_STATUS_RE = re.compile(r"<status>([^<]+)</status>")
_NOTIF_OUTFILE_RE = re.compile(r"<output-file>([^<]+)</output-file>")
_NOTIF_SUMMARY_RE = re.compile(r"<summary>([\s\S]*?)</summary>")
_AGENT_ID_RE = re.compile(r"agentId:\s*(\w+)")
_AGENT_OUTPUT_RE = re.compile(r"output_file:\s*(\S+)")


def _new_task_record(
    task_id: str,
    *,
    tool_use_id: str | None = None,
    launch_ts: str | None = None,
    launch_line: int | None = None,
    command: str = "",
    output_file: str | None = None,
) -> dict:
    """Fresh bg-task lifecycle record with all downstream fields defaulted."""
    return {
        "task_id": task_id,
        "tool_use_id": tool_use_id,
        "launch_ts": launch_ts,
        "launch_line": launch_line,
        "command": command,
        "output_file": output_file,
        "notified_ts": None,
        "notified_line": None,
        "status": None,
        "summary": "",
        "read_ts": None,
        "read_line": None,
    }


def _delivered_bg_text(delivery: dict | None) -> str:
    """Delivered output text of a harness background-task completion —
    '' when the delivery carries nothing read-worthy.

    Accepts either delivery shape observed in transcripts (issue #383):
    the ``taskDelivery`` object of a delivered completion (top-level on a
    task-notification ``user`` record, or nested under ``attachment`` on a
    ``queued_command`` attachment record), or the ``toolUseResult`` of an
    explicit ``GetTask`` call. Both carry
    ``{taskId, status, result{isError, content[]}}``.

    A delivery counts only with ``status: completed`` and non-empty
    ``result.content``. ``status: working`` results carry no ``result`` at
    all, and a completed delivery with empty content carries nothing
    read-worthy; both return ''. ``result.isError`` is deliberately ignored:
    for a background COMMAND it means the command exited nonzero, not that
    the delivery failed — and a failed task's output is the one that most
    needs reading.
    """
    if not isinstance(delivery, dict):
        return ""
    if str(delivery.get("status") or "") != "completed":
        return ""
    result = delivery.get("result")
    if not isinstance(result, dict):
        return ""
    content = result.get("content")
    if isinstance(content, str):
        return content if content.strip() else ""
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                parts.append(str(block.get("text", "")))
            elif isinstance(block, str):
                parts.append(block)
            else:
                parts.append(json.dumps(block, ensure_ascii=False))
        text = "\n".join(parts)
        return text if text.strip() else ""
    return ""


def compute_bg_tasks(events: list[dict], meta_events: list[dict]) -> str:
    """Background shell task lifecycle: launch -> notification -> output read."""
    lines = ["BACKGROUND TASKS", "=" * 40]

    # task_id -> record
    tasks: dict[str, dict] = {}

    # Pass 1: collect bg shell launches (Bash/PowerShell) via tool_use_id
    for e in events:
        if e["type"] != "tool_call":
            continue
        if e.get("tool_name") not in _SHELL_TOOLS:
            continue
        if not (e.get("tool_input") or {}).get("run_in_background"):
            continue
        result = e.get("paired_result")
        if not result:
            continue
        detail = result.get("detail", "") or ""
        m_id = _TASK_ID_RE.search(detail)
        m_out = _TASK_OUTPUT_RE.search(detail)
        if not m_id:
            continue
        task_id = m_id.group(1)
        tasks[task_id] = _new_task_record(
            task_id,
            tool_use_id=e.get("tool_use_id"),
            launch_ts=e["ts"],
            launch_line=e["line"],
            command=(e.get("tool_input") or {}).get("command", ""),
            output_file=m_out.group(1) if m_out else None,
        )

    # Pass 2: scan meta_events for task-notifications (queue-operation enqueue)
    for m in meta_events:
        if m["type"] != "queue-operation":
            continue
        content = m.get("content", "") or ""
        if "task-notification" not in content:
            continue
        # Skip Monitor event notifications — persistent Monitor tasks emit a
        # notification per stdout event, which breaks the "one read per notify"
        # lifecycle model this detector assumes.
        if "<summary>Monitor event:" in content:
            continue
        mid = _NOTIF_TASK_ID_RE.search(content)
        if not mid:
            continue
        task_id = mid.group(1)
        task = tasks.get(task_id)
        if not task:
            # Notification without matching launch — still record it
            task = _new_task_record(task_id)
            tasks[task_id] = task
        task["notified_ts"] = m["ts"]
        task["notified_line"] = m["line"]
        ms = _NOTIF_STATUS_RE.search(content)
        if ms:
            task["status"] = ms.group(1)
        m_out = _NOTIF_OUTFILE_RE.search(content)
        if m_out and not task["output_file"]:
            task["output_file"] = m_out.group(1)
        m_summary = _NOTIF_SUMMARY_RE.search(content)
        if m_summary:
            task["summary"] = m_summary.group(1).strip()

    # Pass 2.5: TaskStop calls terminate a task — treat as a notification with
    # status="stopped". An agent that explicitly killed a task isn't waiting for
    # its output, so it should NOT show up in launched-never-notified.
    for e in events:
        if e["type"] != "tool_call":
            continue
        if e.get("tool_name") != "TaskStop":
            continue
        ti = e.get("tool_input") or {}
        task_id = ti.get("task_id") or ti.get("shell_id") or ""
        if not task_id:
            continue
        task = tasks.get(task_id)
        if task is None:
            continue
        if task["notified_ts"] is None:
            task["notified_ts"] = e["ts"]
            task["notified_line"] = e["line"]
            task["status"] = "stopped"
            task["summary"] = "Stopped by agent (TaskStop)"

    # Pass 3: find successful Read / Bash / TaskOutput access of each task.
    # Path comparisons are separator- and case-normalized: on Windows the
    # launch detail records backslash paths while agents routinely Read/tail
    # the same file with forward slashes — an exact-string match silently
    # drops those legitimate reads (observed 2026-06-10: 10 tasks flagged
    # never-read despite tail/TaskOutput consumption of every one).
    def _norm_path(p: str) -> str:
        # Canonicalize Windows drive spellings too: C:\x, C:/x, /c/x and
        # /cygdrive/c/x all denote the same file, but the launch record
        # stores the first while git-bash commands use the third. The
        # lookbehind admits start/whitespace/quotes/`(`/`=` because this
        # also normalizes whole shell commands, where the path follows the
        # command name (e.g. `tail -20 /c/Users/...`).
        return re.sub(
            r"(?:(?<=^)|(?<=[\s'\"(=]))/(?:cygdrive/)?([a-z])/",
            lambda m: m.group(1) + ":/",
            p.replace("\\", "/").casefold(),
        )

    output_file_to_task: dict[str, dict] = {
        _norm_path(t["output_file"]): t for t in tasks.values() if t.get("output_file")
    }
    for e in events:
        if e["type"] != "tool_call":
            continue
        # Skip if the tool call errored (e.g. blocked by hook)
        result = e.get("paired_result") or {}
        if result.get("is_error"):
            continue
        tool = e.get("tool_name")
        if tool == "Read":
            path = _norm_path((e.get("tool_input") or {}).get("file_path", ""))
            task = output_file_to_task.get(path)
            if task and task["read_ts"] is None:
                task["read_ts"] = e["ts"]
                task["read_line"] = e["line"]
        elif tool == "Grep":
            # The Grep TOOL pointed at the output file is a real content read —
            # the same act as `grep <pattern> <output-file>` through Bash, which
            # has always counted. Omitting it produced a FALSE "never read"
            # violation (observed 2026-07-28: a task Grepped twice, then flagged
            # unread by the Stop hook). Only `output_mode: "content"` counts:
            # the default `files_with_matches` returns just the path, which is
            # metadata, not the output — the same reason ls/stat don't count.
            ti = e.get("tool_input") or {}
            if str(ti.get("output_mode") or "") == "content":
                task = output_file_to_task.get(_norm_path(str(ti.get("path") or "")))
                if task and task["read_ts"] is None:
                    task["read_ts"] = e["ts"]
                    task["read_line"] = e["line"]
        elif tool == "TaskOutput":
            # TaskOutput is the harness-native consumption path: with
            # block=true it returns the task's stdout/stderr inline. It
            # identifies the task by id, not by output path.
            ti = e.get("tool_input") or {}
            task = tasks.get(ti.get("task_id") or ti.get("shell_id") or "")
            if task and task["read_ts"] is None:
                task["read_ts"] = e["ts"]
                task["read_line"] = e["line"]
        elif tool == "GetTask":
            # Explicit model-issued GetTask (issue #383): the harness-native
            # poll for a background task, keyed by taskId like TaskOutput. A
            # result with status completed + non-empty result.content IS the
            # task output — the model read it there. status=working results
            # carry no result.content yet and do NOT count (nothing read).
            # result.isError is ignored: for a background command it means
            # the command exited nonzero, not that the read failed.
            ti = e.get("tool_input") or {}
            task = tasks.get(ti.get("taskId") or "")
            if task:
                result = e.get("paired_result") or {}
                if _delivered_bg_text(result.get("toolUseResult")):
                    if task["read_ts"] is None:
                        task["read_ts"] = e["ts"]
                        task["read_line"] = e["line"]
        elif tool in _SHELL_TOOLS:
            cmd = (e.get("tool_input") or {}).get("command", "")
            # Only count commands that read file content, not ls/stat/lsof/wc.
            # Content-consuming tools: pager/printer (cat, head, tail, less, more),
            # search (grep, rg), stream processors (awk, sed), structured-data
            # readers (jq). Metadata-only tools (ls, stat, file, lsof, wc) are
            # excluded — they don't read content in a way that counts as "I saw
            # the output." The PowerShell cmdlet forms (Get-Content/gc, Select-
            # String/sls, findstr) count for the same reason their Unix
            # equivalents do.
            if not re.search(
                r"\b(cat|head|tail|less|more|grep|rg|awk|sed|jq"
                r"|Get-Content|gc|Select-String|sls|findstr)\b", cmd, re.IGNORECASE):
                continue
            cmd_norm = _norm_path(cmd)
            for opath, task in output_file_to_task.items():
                if opath in cmd_norm and task["read_ts"] is None:
                    task["read_ts"] = e["ts"]
                    task["read_line"] = e["line"]

    # Pass 3.5: harness-delivered completions (issue #383). The completion
    # notification can carry the task's output INLINE as taskDelivery — the
    # session read it there, whether or not any tool ever opened the .output
    # file. Two record shapes deliver it: top-level on a task-notification
    # `user` record (origin.kind: task-notification), and nested under
    # `attachment` on a queued_command attachment record (mid-turn delivery).
    # result.isError is ignored: for a background command it means the command
    # exited nonzero, not that the delivery failed — and a failed task's
    # output is the one that most needs reading.
    #
    # A delivery usually PRECEDES any .output-file read (it arrives with the
    # notification; a file read after a false "never read" block is a
    # redundant reread of output the session already had). Pass 3 has already
    # run by the time we get here, so where both exist the EARLIER
    # consumption wins the read citation: a delivery rewrites an
    # already-recorded read only when its timestamp is strictly older. An
    # empty-timestamp event never wins (it would corrupt the read fields).
    for e in events:
        if e.get("type") != "user_message":
            continue
        delivery = e.get("taskDelivery")
        if not isinstance(delivery, dict) or not _delivered_bg_text(delivery):
            continue
        task = tasks.get(delivery.get("taskId") or "")
        if task and e["ts"] and (task["read_ts"] is None or e["ts"] < task["read_ts"]):
            task["read_ts"] = e["ts"]
            task["read_line"] = e["line"]
    for m in meta_events:
        if m.get("type") != "attachment":
            continue
        attach = (m.get("raw") or {}).get("attachment") or {}
        if str(attach.get("type") or "") != "queued_command":
            continue
        delivery = attach.get("taskDelivery")
        if not isinstance(delivery, dict) or not _delivered_bg_text(delivery):
            continue
        task = tasks.get(delivery.get("taskId") or "")
        if task and m["ts"] and (task["read_ts"] is None or m["ts"] < task["read_ts"]):
            task["read_ts"] = m["ts"]
            task["read_line"] = m["line"]

    # Notification delivery: a task-notification is "delivered to the agent"
    # once the server has moved it out of the pending queue via any of three
    # observed paths:
    #
    #   1. `dequeue`: normal path — the agent's next turn pops the
    #      notification off the queue as fresh input.
    #   2. `enqueue(task-notif) → remove`: server cleanup after inline
    #      delivery via a Stop-hook warning.  The warning text contains the
    #      task summary, so the agent saw it; the queue entry is then
    #      removed as redundant.
    #   3. Raw `<task-notification>` content injected directly into a
    #      user-message event (bypasses the queue entirely — the
    #      notification appears as a literal user content block at the head
    #      of the next turn's input).
    #
    # Before a notification hits any of these, it's genuinely pending and
    # the agent has no way to know the task finished — any Stop-hook check
    # in that window would falsely classify it as "notified but never read"
    # (the original race bug).
    #
    # FIFO pairing runs against ALL enqueues (not just task-notifications)
    # because user-typed text sits in the same queue and a dequeue could be
    # consuming either.  `popAll`'s effect on the queue is not simulated
    # (semantics not fully characterized from transcript evidence alone);
    # the conservative consequence is that popAll-cleared task-notifs land
    # in in-flight rather than never-read.
    pending_queue: list[tuple[int, str | None]] = []
    delivered: dict[str, int] = {}  # task_id -> delivery-signal line
    prev_qop: dict | None = None
    for m in meta_events:
        if m["type"] != "queue-operation":
            continue
        op = m.get("operation", "")
        content = m.get("content", "") or ""
        if op == "enqueue":
            tid = None
            if "task-notification" in content:
                mid = _NOTIF_TASK_ID_RE.search(content)
                if mid:
                    tid = mid.group(1)
            pending_queue.append((m["line"], tid))
        elif op == "dequeue" and pending_queue:
            _, tid = pending_queue.pop(0)
            if tid:
                delivered[tid] = m["line"]
        elif op == "remove" and prev_qop is not None and prev_qop.get("operation") == "enqueue":
            # Path 2: enqueue → remove pair = server cleanup after inline
            # hook delivery.  Only treat as delivered if (a) the enqueued
            # item was a task-notification (user-text enqueue→remove is
            # the "user edited the typing buffer" pattern, not delivery),
            # AND (b) the remove happens within ~1s of the enqueue.  A
            # remove seconds/minutes after the enqueue is targeting some
            # other item, not the just-enqueued one — the task-notif will
            # be delivered later via a normal dequeue.
            prev_content = prev_qop.get("content", "") or ""
            if "task-notification" in prev_content:
                t_enq = parse_timestamp(prev_qop.get("ts", ""))
                t_rem = parse_timestamp(m.get("ts", ""))
                gap = (t_rem - t_enq).total_seconds() if (t_enq and t_rem) else 99.0
                if 0.0 <= gap <= 1.0:
                    mid = _NOTIF_TASK_ID_RE.search(prev_content)
                    if mid:
                        tid = mid.group(1)
                        delivered[tid] = m["line"]
                        for idx in range(len(pending_queue) - 1, -1, -1):
                            if pending_queue[idx][1] == tid:
                                pending_queue.pop(idx)
                                break
        prev_qop = m

    # Path 3: raw `<task-notification>` content injected as a user_message
    # (not a tool_result, not a stop-hook warning — direct content inject
    # at the head of a turn).  Only the first appearance per tid counts,
    # and only if that tid isn't already recorded as delivered via 1 or 2.
    for e in events:
        if e.get("type") != "user_message":
            continue
        detail = e.get("detail", "") or ""
        if not detail.startswith("<task-notification>"):
            continue
        mid = _NOTIF_TASK_ID_RE.search(detail)
        if mid:
            tid = mid.group(1)
            delivered.setdefault(tid, e["line"])

    # `notified_line`/`notified_ts` on each task stays at the enqueue (or
    # the TaskStop call for Pass 2.5 cases) — that's when the notification
    # was created.  `delivered` is used only for categorization (in-flight
    # vs never-read); it doesn't overwrite the notify fields.

    # Categorize (skip agent tasks — their summary is in the notification)
    launched_never_notified = []
    notified_never_read = []
    notified_in_flight = []
    full = []
    expired = []  # output file no longer exists on disk
    for t in tasks.values():
        if not t["launch_ts"]:
            continue  # Agent task — no Bash launch, summary delivered inline
        if t["launch_ts"] and not t["notified_ts"]:
            launched_never_notified.append(t)
        elif t["notified_ts"] and not t["read_ts"]:
            # Skip "never read" classification when the output file is gone.
            # Cross-session restarts and /tmp cleanup can delete /tmp/claude-*
            # task outputs, leaving notified-but-never-read tasks unreadable.
            # Reporting them as violations spams the bg-task-read-check Stop
            # hook every turn and the agent has no way to satisfy the rule.
            # (dispatcher reported 9 such tasks on 2026-05-05.)
            outf = t.get("output_file")
            if outf and not os.path.exists(outf):
                expired.append(t)
                continue
            if t.get("status") == "stopped":
                # TaskStop'd by the agent: partial output may still be worth
                # reading to confirm why the kill was warranted.  Skip the
                # delivery check — TaskStop is its own terminal signal from
                # Pass 2.5, independent of the notification queue.
                notified_never_read.append(t)
            else:
                tid = t.get("task_id") or ""
                delivered_line = delivered.get(tid)
                if delivered_line is None:
                    # Enqueued but never dequeued → not yet delivered to agent
                    notified_in_flight.append(t)
                else:
                    t["delivered_line"] = delivered_line
                    notified_never_read.append(t)
        elif t["launch_ts"] and t["notified_ts"] and t["read_ts"]:
            full.append(t)

    def fmt_task(t: dict) -> str:
        parts = [f"  task={t['task_id']:<12}"]
        if t["launch_ts"]:
            parts.append(
                f"launch=L{t['launch_line']} [{short_time(t['launch_ts'])}]"
            )
        if t["notified_ts"]:
            notif_part = (
                f"notify=L{t['notified_line']} [{short_time(t['notified_ts'])}]"
            )
            if t["launch_ts"]:
                t1 = parse_timestamp(t["launch_ts"])
                t2 = parse_timestamp(t["notified_ts"])
                if t1 and t2:
                    notif_part += f" (+{(t2 - t1).total_seconds():.1f}s)"
            parts.append(notif_part)
        if t["read_ts"]:
            read_part = f"read=L{t['read_line']} [{short_time(t['read_ts'])}]"
            if t["notified_ts"]:
                t1 = parse_timestamp(t["notified_ts"])
                t2 = parse_timestamp(t["read_ts"])
                if t1 and t2:
                    read_part += f" (+{(t2 - t1).total_seconds():.1f}s)"
            parts.append(read_part)
        status = t.get("status") or ""
        if status:
            parts.append(f"status={status}")
        return "  ".join(parts)

    counted = (
        len(full)
        + len(launched_never_notified)
        + len(notified_never_read)
        + len(notified_in_flight)
        + len(expired)
    )
    lines.append(
        f"Totals: {counted} tasks  "
        f"({len(full)} full, "
        f"{len(launched_never_notified)} launched-never-notified, "
        f"{len(notified_never_read)} notified-never-read, "
        f"{len(notified_in_flight)} notified-in-flight, "
        f"{len(expired)} expired)"
    )
    lines.append("")

    if full:
        lines.append("FULL LIFECYCLES:")
        for t in sorted(full, key=lambda x: x["launch_line"] or 0):
            lines.append(fmt_task(t))
            if t.get("command"):
                cmd_excerpt = t["command"].replace("\n", " ")[:100]
                lines.append(f"    cmd: {cmd_excerpt}")
            if t.get("summary"):
                lines.append(f"    summary: {t['summary'][:120]}")
        lines.append("")

    if launched_never_notified:
        lines.append("LAUNCHED BUT NEVER NOTIFIED:")
        for t in sorted(launched_never_notified, key=lambda x: x["launch_line"] or 0):
            lines.append(fmt_task(t))
            if t.get("command"):
                cmd_excerpt = t["command"].replace("\n", " ")[:100]
                lines.append(f"    cmd: {cmd_excerpt}")
        lines.append("")

    if notified_never_read:
        lines.append("NOTIFIED BUT NEVER READ:")
        for t in sorted(notified_never_read, key=lambda x: x["notified_line"] or 0):
            lines.append(fmt_task(t))
            if t.get("summary"):
                lines.append(f"    summary: {t['summary'][:120]}")
        lines.append("")

    if notified_in_flight:
        lines.append(
            "NOTIFIED IN FLIGHT (queued in current turn — first readable next turn):"
        )
        for t in sorted(notified_in_flight, key=lambda x: x["notified_line"] or 0):
            lines.append(fmt_task(t))
            if t.get("summary"):
                lines.append(f"    summary: {t['summary'][:120]}")
        lines.append("")

    if expired:
        lines.append(
            "EXPIRED (output file removed since notification — read no longer possible):"
        )
        for t in sorted(expired, key=lambda x: x["notified_line"] or 0):
            lines.append(fmt_task(t))
            if t.get("summary"):
                lines.append(f"    summary: {t['summary'][:120]}")
        lines.append("")

    return "\n".join(lines)


def compute_turns(events: list[dict], meta_events: list[dict]) -> str:
    """Per-turn duration + token spend from system/turn_duration records."""
    lines = ["TURNS", "=" * 40]

    # Build a uuid -> assistant_usage index
    usage_by_uuid: dict[str, dict] = {}
    for m in meta_events:
        if m["type"] == "assistant_usage" and m.get("uuid"):
            usage_by_uuid[m["uuid"]] = m

    # Build a lookup from uuid -> raw assistant record (for requestId / usage).
    # We can use assistant_usage's uuid (same as record uuid).
    turn_records = []
    for m in meta_events:
        if m["type"] != "system" or m.get("subtype") != "turn_duration":
            continue
        raw = m["raw"]
        parent_uuid = raw.get("parentUuid")
        parent_usage = usage_by_uuid.get(parent_uuid)
        entry = {
            "ts": m["ts"],
            "line": m["line"],
            "durationMs": raw.get("durationMs", 0) or 0,
            "messageCount": raw.get("messageCount"),
            "requestId": (parent_usage or {}).get("requestId", ""),
            "fresh": 0,
            "create": 0,
            "read": 0,
            "output": 0,
        }
        if parent_usage:
            u = parent_usage["usage"]
            entry["fresh"] = u.get("input_tokens", 0) or 0
            entry["create"] = u.get("cache_creation_input_tokens", 0) or 0
            entry["read"] = u.get("cache_read_input_tokens", 0) or 0
            entry["output"] = u.get("output_tokens", 0) or 0
        turn_records.append(entry)

    if not turn_records:
        lines.append("(no system/turn_duration records found — modern Claude Code JSONLs don't emit these; use --context-growth instead)")
        return "\n".join(lines)

    total_ms = sum(t["durationMs"] for t in turn_records)
    lines.append(
        f"Total turns: {len(turn_records)}  "
        f"total compute: {total_ms/1000:.1f}s"
    )
    lines.append("")
    lines.append(
        f"  {'time':<19}  {'L#':>5}  {'durMs':>8}  {'msgs':>5}  "
        f"{'requestId':<30}  {'input':>10}  {'output':>8}"
    )
    for t in sorted(turn_records, key=lambda x: -x["durationMs"]):
        total_in = t["fresh"] + t["create"] + t["read"]
        lines.append(
            f"  {short_time(t['ts']):<19}  "
            f"L{t['line']:<4}  "
            f"{t['durationMs']:>8}  "
            f"{(t['messageCount'] or 0):>5}  "
            f"{t['requestId']:<30}  "
            f"{total_in:>10,}  "
            f"{t['output']:>8,}"
        )

    return "\n".join(lines)


def compute_agent_tasks(events: list[dict], meta_events: list[dict]) -> str:
    """Async Agent launches: agentId, output file, completion, duration."""
    lines = ["AGENT TASKS", "=" * 40]

    # tool_use_id -> agent entry
    agents: dict[str, dict] = {}
    # agentId -> tool_use_id mapping (for matching)
    agent_id_to_tool_use: dict[str, str] = {}

    for e in events:
        if e["type"] != "agent_spawn":
            continue
        result = e.get("paired_result") or {}
        detail = result.get("detail", "") or ""
        m_id = _AGENT_ID_RE.search(detail)
        m_out = _AGENT_OUTPUT_RE.search(detail)
        agent_id = m_id.group(1) if m_id else None
        output_file = m_out.group(1) if m_out else None
        is_bg = bool(e.get("agent_bg"))
        entry = {
            "line": e["line"],
            "ts": e["ts"],
            "tool_use_id": e.get("tool_use_id"),
            "agent_id": agent_id,
            "output_file": output_file,
            "prompt_size": len(e.get("agent_prompt", "")),
            "bg": is_bg,
            "subagent_type": e.get("agent_subagent_type", "(default)"),
            "notify_ts": None,
            "notify_line": None,
            "status": None,
            "summary": "",
            "read_ts": None,
            "read_line": None,
            "read_result_len": None,
            "result_len": len(detail),
        }
        # Foreground agents complete via the synchronous tool_result itself —
        # use that timestamp as the completion signal.
        if not is_bg and result:
            entry["notify_ts"] = result.get("ts")
            entry["notify_line"] = result.get("line")
            entry["status"] = "completed"
            entry["summary"] = f"(FG result, {len(detail)} chars)"
        agents[e.get("tool_use_id") or f"anon-L{e['line']}"] = entry
        if agent_id:
            agent_id_to_tool_use[agent_id] = e.get("tool_use_id") or ""

    # Look up completion via queue-operation task-notification that carries
    # the matching <tool-use-id>. Fall back to <task-id> matching agent_id.
    for m in meta_events:
        if m["type"] != "queue-operation":
            continue
        content = m.get("content", "") or ""
        if "task-notification" not in content:
            continue
        mtool = _NOTIF_TOOL_USE_RE.search(content)
        mtask = _NOTIF_TASK_ID_RE.search(content)
        tool_use_id = mtool.group(1) if mtool else None
        task_id = mtask.group(1) if mtask else None
        entry = None
        if tool_use_id and tool_use_id in agents:
            entry = agents[tool_use_id]
        elif task_id:
            # Task id may match agent_id
            tu = agent_id_to_tool_use.get(task_id)
            if tu:
                entry = agents.get(tu)
        if not entry:
            continue
        entry["notify_ts"] = m["ts"]
        entry["notify_line"] = m["line"]
        ms = _NOTIF_STATUS_RE.search(content)
        if ms:
            entry["status"] = ms.group(1)
        m_summary = _NOTIF_SUMMARY_RE.search(content)
        if m_summary:
            entry["summary"] = m_summary.group(1).strip()

    # Output file reads
    output_to_entry = {
        a["output_file"]: a for a in agents.values() if a.get("output_file")
    }
    for e in events:
        if e["type"] != "tool_call" or e.get("tool_name") != "Read":
            continue
        path = (e.get("tool_input") or {}).get("file_path", "")
        entry = output_to_entry.get(path)
        if entry and entry["read_ts"] is None:
            entry["read_ts"] = e["ts"]
            entry["read_line"] = e["line"]
            result = e.get("paired_result") or {}
            entry["read_result_len"] = len(result.get("detail", "") or "")

    if not agents:
        lines.append("(no Agent spawns found)")
        return "\n".join(lines)

    lines.append(f"Total agent launches: {len(agents)}")
    lines.append("")
    for entry in sorted(agents.values(), key=lambda x: x["line"]):
        bg_tag = "bg" if entry["bg"] else "FG"
        dur_str = ""
        if entry["notify_ts"]:
            t1 = parse_timestamp(entry["ts"])
            t2 = parse_timestamp(entry["notify_ts"])
            if t1 and t2:
                dur_str = f"  duration={(t2 - t1).total_seconds():.1f}s"
        lines.append(
            f"L{entry['line']:<4} [{short_time(entry['ts'])}] {bg_tag}  "
            f"type={entry['subagent_type']}  "
            f"agent_id={entry['agent_id'] or '?'}  "
            f"prompt={entry['prompt_size']} chars{dur_str}"
        )
        if entry["output_file"]:
            lines.append(f"    output_file: {entry['output_file']}")
        if entry["notify_ts"]:
            lines.append(
                f"    notified L{entry['notify_line']} [{short_time(entry['notify_ts'])}] "
                f"status={entry['status']}"
            )
            if entry["summary"]:
                lines.append(f"    summary: {entry['summary'][:160]}")
        else:
            lines.append("    (no completion notification found)")
        if entry["read_ts"]:
            lines.append(
                f"    read L{entry['read_line']} [{short_time(entry['read_ts'])}]  "
                f"result={entry['read_result_len']} chars"
            )
        else:
            lines.append("    (output file never read)")

    return "\n".join(lines)


def _hook_preview(s: str, n: int = 200) -> str:
    """Single-line-ify + truncate for hook body previews."""
    s = (s or "").replace("\n", "\\n")
    return s if len(s) <= n else s[:n] + f"... [{len(s)-n} more chars]"


def _render_hook_success(att: dict, verbose: bool) -> list[str]:
    cmd = att.get("command", "") or ""
    exit_code = att.get("exitCode")
    duration = att.get("durationMs")
    stdout = att.get("stdout", "") or ""
    stderr = att.get("stderr", "") or ""
    content_inline = att.get("content", "") or ""
    meta = (
        f"  exit={exit_code}  duration={duration}ms"
        f"  stdout={len(stdout):,}B  stderr={len(stderr):,}B"
    )
    if content_inline:
        meta += f"  content={len(content_inline):,}B"
    out = [meta]
    if cmd:
        out.append(f"  cmd: {_hook_preview(cmd, 400)}")
    for label, body in (("stdout", stdout), ("stderr", stderr)):
        if not body:
            continue
        if verbose:
            out.append(f"  {label}:")
            out.extend(f"    {ln}" for ln in body.splitlines())
        else:
            out.append(f"  {label}: {_hook_preview(body, 300)}")
    if content_inline and not stdout:
        body = str(content_inline)
        if verbose:
            out.append("  content:")
            out.extend(f"    {ln}" for ln in body.splitlines())
        else:
            out.append(f"  content: {_hook_preview(body, 300)}")
    return out


def _render_hook_blocking_error(att: dict, verbose: bool) -> list[str]:
    be = att.get("blockingError") or {}
    if isinstance(be, dict):
        msg = be.get("blockingError", "") or ""
        cmd = be.get("command", "") or ""
    else:
        msg = str(be)
        cmd = ""
    out: list[str] = []
    if cmd:
        out.append(f"  cmd: {_hook_preview(cmd, 400)}")
    if msg:
        if verbose:
            out.append("  error:")
            out.extend(f"    {ln}" for ln in msg.splitlines())
        else:
            out.append(f"  error: {_hook_preview(msg, 400)}")
    return out


def _render_hook_permission_decision(att: dict, verbose: bool) -> list[str]:
    decision = att.get("decision", "?")
    reason = att.get("reason") or att.get("message") or ""
    out = [f"  decision={decision}"]
    if reason:
        out.append(f"  reason: {_hook_preview(str(reason), 300)}")
    return out


def _render_hook_additional_context(att: dict, verbose: bool) -> list[str]:
    content = att.get("content")
    out: list[str] = []
    if isinstance(content, list):
        total = sum(len(s) for s in content if isinstance(s, str))
        out.append(f"  segments={len(content)}  total={total:,}B")
        for i, seg in enumerate(content):
            seg_s = seg if isinstance(seg, str) else str(seg)
            if verbose:
                out.append(f"  [{i}] ({len(seg_s):,}B):")
                out.extend(f"    {ln}" for ln in seg_s.splitlines())
            else:
                out.append(f"  [{i}] ({len(seg_s):,}B) {_hook_preview(seg_s, 200)}")
    elif isinstance(content, str):
        out.append(f"  content={len(content):,}B")
        if verbose:
            out.extend(f"    {ln}" for ln in content.splitlines())
        else:
            out.append(f"  {_hook_preview(content, 300)}")
    return out


_HOOK_RENDERERS = {
    "hook_success": _render_hook_success,
    "hook_blocking_error": _render_hook_blocking_error,
    "hook_permission_decision": _render_hook_permission_decision,
    "hook_additional_context": _render_hook_additional_context,
}


def compute_hooks(meta_events: list[dict], event_filter: str | None = None,
                  verbose: bool = False) -> str:
    """All hook invocations (SessionStart, PreToolUse, PostToolUse, Stop,
    PermissionRequest, ...). Covers every attachment whose payload carries a
    ``hookEvent`` field — that is, attachment types
    ``hook_success``, ``hook_blocking_error``, ``hook_permission_decision``,
    and ``hook_additional_context``.

    ``event_filter``: if set, only rows whose hookEvent matches (case-insensitive).
    ``verbose``: include full stdout/stderr bodies and full additionalContext
    segments instead of size + preview.
    """
    header = "HOOK INVOCATIONS"
    if event_filter:
        header += f"  (filter: hookEvent={event_filter})"
    lines = [header, "=" * 60]

    rows: list[dict] = []
    for m in meta_events:
        if m["type"] != "attachment":
            continue
        raw = m["raw"]
        att = raw.get("attachment") or {}
        hook_event = att.get("hookEvent") or ""
        if not hook_event:
            continue
        if event_filter and hook_event.lower() != event_filter.lower():
            continue
        rows.append({
            "line": m["line"],
            "ts": m["ts"],
            "att_type": att.get("type", ""),
            "hook_event": hook_event,
            "hook_name": att.get("hookName", ""),
            "tool_use_id": att.get("toolUseID", ""),
            "att": att,
        })

    if not rows:
        lines.append("(no matching hook events)")
        return "\n".join(lines)

    # Summary counts
    by_event: Counter = Counter()
    by_type: Counter = Counter()
    by_hookname: Counter = Counter()
    total_duration_ms = 0
    duration_samples = 0
    blocking_count = 0
    denied_count = 0
    for r in rows:
        by_event[r["hook_event"]] += 1
        by_type[r["att_type"]] += 1
        if r["hook_name"]:
            by_hookname[r["hook_name"]] += 1
        if r["att_type"] == "hook_success":
            d = r["att"].get("durationMs")
            if isinstance(d, (int, float)):
                total_duration_ms += d
                duration_samples += 1
        elif r["att_type"] == "hook_blocking_error":
            blocking_count += 1
        elif r["att_type"] == "hook_permission_decision":
            if (r["att"].get("decision") or "").lower() == "deny":
                denied_count += 1

    lines.append(f"Total hook attachments: {len(rows)}")
    lines.append("  By hookEvent:")
    for ev, n in sorted(by_event.items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f"    {ev:<20} {n}")
    lines.append("  By attachment type:")
    for t, n in sorted(by_type.items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f"    {t:<30} {n}")
    if by_hookname:
        lines.append("  By hookName (top 10):")
        for h, n in by_hookname.most_common(10):
            lines.append(f"    {h:<40} {n}")
    if duration_samples:
        avg = total_duration_ms / duration_samples
        lines.append(
            f"  hook_success durations: n={duration_samples}  "
            f"total={total_duration_ms:,}ms  avg={avg:,.1f}ms"
        )
    if blocking_count:
        lines.append(f"  hook_blocking_error total: {blocking_count}")
    if denied_count:
        lines.append(f"  hook_permission_decision denies: {denied_count}")
    lines.append("")
    lines.append("TIMELINE")
    lines.append("-" * 60)

    for r in rows:
        att = r["att"]
        att_type = r["att_type"]
        head = (
            f"L{r['line']:<5} [{short_time(r['ts'])}] "
            f"{r['hook_event']:<17} {att_type}"
        )
        if r["hook_name"]:
            head += f"  hookName={r['hook_name']}"
        if r["tool_use_id"]:
            head += f"  toolUseID={r['tool_use_id']}"
        lines.append(head)

        renderer = _HOOK_RENDERERS.get(att_type)
        if renderer:
            lines.extend(renderer(att, verbose))

        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def compute_compact(meta_events: list[dict]) -> str:
    """Compact events (system/compact_boundary)."""
    lines = ["COMPACT EVENTS", "=" * 40]
    found = 0
    for m in meta_events:
        if m["type"] != "system" or m.get("subtype") != "compact_boundary":
            continue
        found += 1
        raw = m["raw"]
        meta = raw.get("compactMetadata") or {}
        trigger = meta.get("trigger", "?")
        pre_tokens = meta.get("preTokens", 0) or 0
        lines.append(
            f"L{m['line']:<4} [{short_time(m['ts'])}] "
            f"trigger={trigger}  preTokens={pre_tokens:,}"
        )
        content_preview = (raw.get("content", "") or "").replace("\n", " ")[:120]
        if content_preview:
            lines.append(f"  content: {content_preview}")
    if not found:
        lines.append("(no compact_boundary events)")
    return "\n".join(lines)


def compute_api_errors(meta_events: list[dict]) -> str:
    """API errors separate from tool errors (system/api_error)."""
    lines = ["API ERRORS", "=" * 40]
    found = 0
    for m in meta_events:
        if m["type"] != "system" or m.get("subtype") != "api_error":
            continue
        found += 1
        raw = m["raw"]
        err = raw.get("error") or {}
        status = err.get("status", "?")
        inner = err.get("error") or {}
        inner_inner = inner.get("error") or {}
        err_type = (
            inner_inner.get("type")
            or inner.get("type")
            or err.get("type")
            or "?"
        )
        err_msg = inner_inner.get("message") or ""
        retry_ms = raw.get("retryInMs")
        retry_attempt = raw.get("retryAttempt")
        max_retries = raw.get("maxRetries")
        head = (
            f"L{m['line']:<4} [{short_time(m['ts'])}] "
            f"status={status}  type={err_type}  "
            f"attempt={retry_attempt}/{max_retries}"
        )
        if isinstance(retry_ms, (int, float)):
            head += f"  retryIn={retry_ms:.0f}ms"
        lines.append(head)
        if err_msg:
            lines.append(f"  msg: {err_msg[:200]}")
    if not found:
        lines.append("(no api_error events)")
    return "\n".join(lines)


_SLASH_CMD_RE = re.compile(r"<command-name>([^<]+)</command-name>")
_SLASH_ARGS_RE = re.compile(r"<command-args>([^<]*)</command-args>")


def compute_last_assistant_message(events: list[dict]) -> str:
    """Assistant-authored text from the LAST turn, verbatim.

    "Last turn" is everything after the final user message, so a closing reply
    split across several text blocks (text, tool call, more text) comes back
    whole rather than just its final fragment.

    Emitted RAW -- no banner, no `L<n>` refs, no truncation -- unlike every
    other mode here. The consumer is a Stop-side hook matching on what the
    agent actually said, and a banner is something every such caller would
    have to strip back off. Returns an empty string when the last turn
    produced no text (pure tool-call turn).
    """
    tail: list[str] = []
    for e in reversed(events):
        if e["type"] == "user_message":
            break
        if e["type"] == "assistant_text":
            tail.append(e.get("detail") or "")
    return "\n".join(reversed(tail)).strip()


def compute_slash(meta_events: list[dict]) -> str:
    """Slash commands invoked (system/local_command with <command-name>)."""
    lines = ["SLASH COMMANDS", "=" * 40]
    found = 0
    for m in meta_events:
        if m["type"] != "system" or m.get("subtype") != "local_command":
            continue
        content = m.get("content", "") or ""
        mc = _SLASH_CMD_RE.search(content)
        if not mc:
            continue
        found += 1
        cmd = mc.group(1)
        ma = _SLASH_ARGS_RE.search(content)
        args = ma.group(1).strip() if ma else ""
        args_tag = f"  args={args}" if args else ""
        lines.append(f"L{m['line']:<4} [{short_time(m['ts'])}] {cmd}{args_tag}")
    if not found:
        lines.append("(no slash commands found)")
    return "\n".join(lines)


def compute_shell_errors(events: list[dict]) -> str:
    """Shell calls (Bash/PowerShell) with non-empty stderr, interrupted, or
    non-zero exit."""
    lines = ["SHELL ERRORS", "=" * 40]
    found = 0
    for e in events:
        if e["type"] != "tool_call" or e.get("tool_name") not in _SHELL_TOOLS:
            continue
        result = e.get("paired_result") or {}
        tur = result.get("toolUseResult") or {}
        stderr = (tur.get("stderr") or "").strip() if isinstance(tur, dict) else ""
        interrupted = bool(tur.get("interrupted")) if isinstance(tur, dict) else False
        detail = result.get("detail", "") or ""
        is_error = bool(result.get("is_error"))
        exit_match = re.match(r"Exit code (\d+)", detail)
        exit_code = int(exit_match.group(1)) if exit_match else None
        nonzero_exit = exit_code is not None and exit_code != 0

        if not (stderr or interrupted or nonzero_exit or is_error):
            continue
        found += 1
        inp = e.get("tool_input") or {}
        cmd_excerpt = (inp.get("command") or "").replace("\n", " ")[:120]
        status_parts = []
        if exit_code is not None:
            status_parts.append(f"exit={exit_code}")
        if interrupted:
            status_parts.append("interrupted")
        if is_error and exit_code is None:
            status_parts.append("is_error")
        status = " ".join(status_parts) or "error"
        lines.append(
            f"L{e['line']:<4} [{short_time(e['ts'])}]  {status}"
        )
        lines.append(f"    {'PS>' if e.get('tool_name') == 'PowerShell' else '$'} {cmd_excerpt}")
        if stderr:
            stderr_preview = stderr.replace("\n", " ")[:200]
            lines.append(f"    stderr: {stderr_preview}")
        elif detail:
            detail_preview = detail.replace("\n", " ")[:200]
            lines.append(f"    out: {detail_preview}")
    if not found:
        lines.append("(no shell errors found)")
    else:
        lines.append("")
        lines.append(f"Total: {found} shell errors")
    return "\n".join(lines)


def compute_tool_orphans(events: list[dict]) -> str:
    """Find tool_use events without a matching tool_result.

    Requires link_pairs() to have been called first.
    """
    lines = ["TOOL ORPHANS", "=" * 40]
    orphans = []
    for e in events:
        if e["type"] not in ("tool_call", "agent_spawn"):
            continue
        if e.get("paired_result") is None:
            orphans.append(e)

    if not orphans:
        lines.append("(no orphaned tool calls — all have matching results)")
        lines.append("")
        lines.append("0 orphaned calls")
        return "\n".join(lines)

    for e in orphans:
        name = e.get("tool_name") or ("Agent" if e["type"] == "agent_spawn" else "?")
        excerpt = _tool_call_excerpt(e, limit=80)
        lines.append(
            f"L{e['line']:<4} [{short_time(e['ts'])}] {name:<14}  {excerpt}"
        )
    lines.append("")
    lines.append(f"{len(orphans)} orphaned call{'s' if len(orphans) != 1 else ''}")
    return "\n".join(lines)


def compute_timeline(events: list[dict], verbose: bool = False) -> str:
    """Chronological interleave of ALL event types with compact one-line formatting."""
    lines = ["TIMELINE", "=" * 40]

    if not events:
        lines.append("(no events)")
        lines.append("0 events")
        return "\n".join(lines)

    type_labels = {
        "user_message": "USER",
        "assistant_text": "ASST",
        "thinking": "THINK",
        "tool_call": "TOOL",
        "tool_result": "RESULT",
        "agent_spawn": "TOOL",
        "parse_error": "ERROR",
    }

    preview_limit = 200 if verbose else 80
    for e in events:
        label = type_labels.get(e["type"], e["type"].upper())
        # Add tool name for tool calls
        if e["type"] in ("tool_call", "agent_spawn"):
            name = e.get("tool_name") or ("Agent" if e["type"] == "agent_spawn" else "?")
            label = f"TOOL:{name}"
        elif e["type"] == "tool_result":
            call = e.get("paired_call") or {}
            name = call.get("tool_name") or call.get("agent_name") or "?"
            err = "[ERR]" if e.get("is_error") else ""
            label = f"RESULT:{name}{err}"

        if e["type"] == "tool_call":
            detail = _tool_call_detail(e).replace("\n", " ")
        else:
            detail = (e.get("detail") or "").replace("\n", " ")
        if len(detail) > preview_limit:
            detail = detail[:preview_limit] + "..."

        lines.append(
            f"L{e['line']:<4} [{short_time(e['ts'])}] {label:<20} {detail}"
        )

    lines.append("")
    lines.append(f"{len(events)} events")
    return "\n".join(lines)


def compute_context_growth(events: list[dict], meta_events: list[dict]) -> str:
    """Cumulative token count over time from assistant_usage records.

    Derives turns from user text messages (turn boundaries) and
    assistant_usage records (token snapshots).  For each turn the last
    assistant_usage before the next user message is the representative
    data point.  This replaces the old turn_duration join which silently
    dropped turns that lacked a turn_duration record.

    Shows turn number, timestamp, input tokens, output tokens, delta from
    previous input total.
    """
    lines = ["CONTEXT GROWTH", "=" * 40]

    # Collect assistant_usage records chronologically. Skip <synthetic>
    # records — Claude Code injects these for harness-driven stubs (e.g.
    # post-/exit "No response requested.", interrupted partial responses)
    # with all-zero usage and no requestId. They aren't real model turns
    # and would otherwise clobber the legitimate "last usage before next
    # boundary" pick further down.
    usage_records: list[dict] = []
    for m in meta_events:
        if m["type"] != "assistant_usage":
            continue
        if m.get("model") == "<synthetic>":
            continue
        u = m.get("usage", {})
        fresh = u.get("input_tokens", 0) or 0
        create = u.get("cache_creation_input_tokens", 0) or 0
        read = u.get("cache_read_input_tokens", 0) or 0
        output = u.get("output_tokens", 0) or 0
        # CONTEXT GROWTH must reflect the size of context fed to a SINGLE API
        # call, not the per-turn billing rollup. When `usage.iterations` is
        # present with >1 entries (advisor() sub-calls, sub-agent invocations,
        # retry iterations), the top-level usage sums per-iteration cache_read
        # across separate API calls — that sum is billing, not context. Take
        # the per-iteration peak so the column actually measures "context".
        iters = u.get("iterations") or []
        if len(iters) > 1:
            total_input = max(
                ((it.get("input_tokens", 0) or 0)
                 + (it.get("cache_creation_input_tokens", 0) or 0)
                 + (it.get("cache_read_input_tokens", 0) or 0))
                for it in iters
            )
        else:
            total_input = fresh + create + read
        usage_records.append({
            "ts": m["ts"],
            "line": m["line"],
            "input": total_input,
            "output": output,
        })

    if not usage_records:
        lines.append("(no assistant_usage records found)")
        return "\n".join(lines)

    # Collect user text message lines as turn boundaries.
    # A user message that is purely tool_result content is mid-turn
    # (the assistant called a tool, got a result, and continues).
    # Only messages with actual user text start a new turn.
    # Same anchor filter as compute_reply_latency: harness instrumentation
    # bursts (<bash-input>, <command-name>, …) and `[Request interrupted
    # by user]` markers are NOT turn boundaries — they fire mid-turn or
    # in groups of 3 with identical timestamps.
    user_lines: set[int] = set()
    for e in events:
        if e["type"] != "user_message":
            continue
        if _is_instrumentation_user_msg(e):
            continue
        if _is_interrupt_user_msg(e):
            continue
        user_lines.add(e["line"])

    # Build turn records: walk usage records chronologically.
    # When we encounter a user_message line between two usage records,
    # that's a turn boundary — flush the previous turn's last usage.
    usage_records.sort(key=lambda r: (r["ts"], r["line"]))

    # Collect all line numbers that are turn boundaries, sorted
    boundary_lines = sorted(user_lines)

    # For each turn, find the last assistant_usage before the next
    # turn boundary.  Strategy: walk usage records; when the next
    # boundary is at or before the current usage line, flush.
    turn_records: list[dict] = []
    last_usage: dict | None = None
    bi = 0  # boundary index

    for rec in usage_records:
        # Flush if we crossed a boundary
        while bi < len(boundary_lines) and boundary_lines[bi] <= rec["line"]:
            if last_usage is not None:
                turn_records.append(last_usage)
                last_usage = None
            bi += 1
        last_usage = rec

    # Flush final turn
    if last_usage is not None:
        turn_records.append(last_usage)

    # Drop turns with 0 input tokens — these are API refusals or interrupts
    # that produced no tokens, and including them corrupts delta calculations
    # (prev_input becomes 0, making the next real turn's delta equal to its
    # entire cumulative context).
    turn_records = [t for t in turn_records if t["input"] > 0]

    if not turn_records:
        lines.append("(no turns with usage data)")
        return "\n".join(lines)

    lines.append(
        f"  {'#':>4}  {'time':<19}  {'L#':>5}  {'input':>10}  {'output':>8}  "
        f"{'delta':>10}"
    )

    prev_input = 0
    for idx, t in enumerate(turn_records, 1):
        delta = t["input"] - prev_input
        lines.append(
            f"  {idx:>4}  {short_time(t['ts']):<19}  "
            f"L{t['line']:<4}  "
            f"{t['input']:>10,}  "
            f"{t['output']:>8,}  "
            f"{delta:>+10,}"
        )
        prev_input = t["input"]

    lines.append("")
    lines.append(
        f"Total: {len(turn_records)} turns, final context: {turn_records[-1]['input']:,} input tokens"
    )
    return "\n".join(lines)


def filter_by_path(events: list[dict], pattern: str) -> list[dict]:
    """Filter events to only those touching a file path matching pattern."""
    pattern_lower = pattern.lower()
    result = []
    for e in events:
        if e["type"] in ("tool_call", "agent_spawn"):
            inp = e.get("tool_input", {})
            # Check common file path fields
            file_path = inp.get("file_path", "") or inp.get("path", "")
            command = inp.get("command", "")
            if pattern_lower in file_path.lower() or pattern_lower in command.lower():
                result.append(e)
                continue
        elif e["type"] == "tool_result":
            # Include results paired to matching calls
            paired = e.get("paired_call")
            if paired:
                inp = paired.get("tool_input", {})
                file_path = inp.get("file_path", "") or inp.get("path", "")
                command = inp.get("command", "")
                if pattern_lower in file_path.lower() or pattern_lower in command.lower():
                    result.append(e)
                    continue
        # Non-tool events pass through (user/assistant messages give context)
        if e["type"] in ("user_message", "assistant_text", "thinking"):
            result.append(e)
    return result


def filter_time(events: list[dict], after: str | None, before: str | None) -> list[dict]:
    """Filter events by time range.

    Accepts:
      - HH:MM:SS / HH:MM / H — intra-day seconds-of-day comparison (back-compat;
        useful for slicing a single session, useless across days)
      - YYYY-MM-DD — absolute date floor (treated as 00:00:00 UTC)
      - YYYY-MM-DDTHH:MM:SS — absolute datetime (UTC if no tz)
      - Nd / Nh / Nw — relative lookback from "now" (e.g. ``--after 30d``)

    Returns events whose timestamp passes the comparator. Events without a
    parseable timestamp are dropped when a filter is active.
    """
    def parse_time_arg(t: str) -> tuple[str, int | datetime]:
        m = re.match(r"^(\d+)([dhw])$", t)
        if m:
            n, unit = int(m.group(1)), m.group(2)
            mul = {"d": 86400, "h": 3600, "w": 86400 * 7}[unit]
            return ("absolute", datetime.now(timezone.utc) - timedelta(seconds=n * mul))
        if "-" in t:
            try:
                dt = datetime.fromisoformat(t)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return ("absolute", dt)
            except ValueError:
                pass
        parts = t.split(":")
        s = int(parts[0]) * 3600
        if len(parts) > 1:
            s += int(parts[1]) * 60
        if len(parts) > 2:
            s += int(parts[2])
        return ("intraday", s)

    def passes(e: dict, mode: str, val, op) -> bool:
        dt = parse_timestamp(e["ts"])
        if not dt:
            return False
        if mode == "intraday":
            es = dt.hour * 3600 + dt.minute * 60 + dt.second
            return op(es, val)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return op(dt, val)

    if after:
        mode, val = parse_time_arg(after)
        events = [e for e in events if passes(e, mode, val, lambda a, b: a >= b)]
    if before:
        mode, val = parse_time_arg(before)
        events = [e for e in events if passes(e, mode, val, lambda a, b: a <= b)]
    return events


def filter_line_range(events: list[dict], range_str: str) -> list[dict]:
    """Filter events by JSONL line range."""
    start, end = range_str.split("-")
    start, end = int(start), int(end)
    return [e for e in events if start <= e["line"] <= end]


def compute_cache_anomalies(meta_events: list[dict]) -> str:
    """Flag single-turn cache outliers vs session totals.

    Surfaces:
      - Turns whose cache_create > 25% of session-total cache_create
      - Turns whose cache_read >= 1,000,000 (approaching 1M context limit)
      - Turns whose output_tokens > 5,000 (large generation)
    """
    per_request: dict[str, dict] = {}
    totals = {"create": 0, "read": 0, "output": 0, "fresh": 0}

    for m in meta_events:
        if m["type"] != "assistant_usage":
            continue
        # Skip <synthetic> records — harness-injected stubs with no real API
        # usage. Same filter as compute_cache / compute_context_growth; they
        # add (no-req-L<n>) noise entries to the per-request map otherwise.
        if m.get("model") == "<synthetic>":
            continue
        usage = m["usage"]
        fresh = usage.get("input_tokens", 0) or 0
        create = usage.get("cache_creation_input_tokens", 0) or 0
        read = usage.get("cache_read_input_tokens", 0) or 0
        output = usage.get("output_tokens", 0) or 0
        totals["create"] += create
        totals["read"] += read
        totals["output"] += output
        totals["fresh"] += fresh

        req_id = m.get("requestId") or f"(no-req-L{m['line']})"
        pr = per_request.setdefault(
            req_id,
            {
                "requestId": req_id,
                "line": m["line"],
                "ts": m["ts"],
                "create": 0,
                "read": 0,
                "output": 0,
                "fresh": 0,
            },
        )
        pr["create"] += create
        pr["read"] += read
        pr["output"] += output
        pr["fresh"] += fresh

    lines = ["CACHE ANOMALIES", "=" * 40]
    if not per_request:
        lines.append("(no assistant usage records found)")
        return "\n".join(lines)

    lines.append(
        f"Session totals — cache_create: {totals['create']:,}   "
        f"cache_read: {totals['read']:,}   "
        f"output: {totals['output']:,}   "
        f"fresh: {totals['fresh']:,}"
    )
    lines.append("")

    create_threshold = max(1, int(totals["create"] * 0.25))
    create_anomalies = sorted(
        [r for r in per_request.values() if r["create"] >= create_threshold and r["create"] > 0],
        key=lambda r: -r["create"],
    )
    lines.append(f"CACHE_CREATE OUTLIERS (single turn >= 25% of session total = {create_threshold:,} tokens):")
    if not create_anomalies:
        lines.append("  (none)")
    else:
        for r in create_anomalies:
            pct = (r["create"] / totals["create"] * 100.0) if totals["create"] else 0.0
            lines.append(
                f"  L{r['line']:<5} [{short_time(r['ts'])}]  "
                f"cache_create={r['create']:>10,}  "
                f"({pct:5.1f}% of session)  "
                f"req={r['requestId']}"
            )
    lines.append("")

    read_threshold = 1_000_000
    read_anomalies = sorted(
        [r for r in per_request.values() if r["read"] >= read_threshold],
        key=lambda r: -r["read"],
    )
    lines.append(f"CACHE_READ OUTLIERS (single turn >= {read_threshold:,} = approaching 1M context limit):")
    if not read_anomalies:
        lines.append("  (none)")
    else:
        for r in read_anomalies:
            lines.append(
                f"  L{r['line']:<5} [{short_time(r['ts'])}]  "
                f"cache_read={r['read']:>10,}  "
                f"req={r['requestId']}"
            )
    lines.append("")

    output_threshold = 5000
    output_anomalies = sorted(
        [r for r in per_request.values() if r["output"] >= output_threshold],
        key=lambda r: -r["output"],
    )
    lines.append(f"OUTPUT TOKEN OUTLIERS (single turn >= {output_threshold:,} tokens):")
    if not output_anomalies:
        lines.append("  (none)")
    else:
        for r in output_anomalies:
            lines.append(
                f"  L{r['line']:<5} [{short_time(r['ts'])}]  "
                f"output={r['output']:>6,}  "
                f"req={r['requestId']}"
            )

    return "\n".join(lines)


_TMP_SCRIPT_RE = re.compile(r"^/tmp/([A-Za-z0-9]+)(?:_[^/]*)?\.(sh|py|js)$")


def compute_script_churn(events: list[dict], threshold: int = 3) -> str:
    """Detect /tmp/<base>_*.{sh,py,js} write churn — multiple distinct paths
    in same family suggest the script should be promoted to scripts/.

    Family key: (first _-separated token of stem, ext). Counts DISTINCT paths
    per family — overwriting the same file doesn't count, new siblings do.
    Mirrors the goonbot Stop hook's detection logic so analyses and the hook
    surface the same families.
    """
    families: dict[tuple[str, str], list[tuple[int, str, str]]] = defaultdict(list)
    seen_paths: set[str] = set()

    for e in events:
        if e["type"] != "tool_call" or e.get("tool_name") != "Write":
            continue
        path = (e.get("tool_input") or {}).get("file_path", "")
        if not path or path in seen_paths:
            continue
        m = _TMP_SCRIPT_RE.match(path)
        if not m:
            continue
        seen_paths.add(path)
        base = m.group(1)
        ext = m.group(2)
        families[(base, ext)].append((e["line"], e["ts"], path))

    lines = ["SCRIPT CHURN", "=" * 40]
    lines.append(f"(threshold: {threshold}+ distinct /tmp/<base>_*.{{sh,py,js}} writes per family)")
    lines.append("")

    flagged = sorted(
        [(k, v) for k, v in families.items() if len(v) >= threshold],
        key=lambda kv: -len(kv[1]),
    )

    if not flagged:
        lines.append("(no churn families detected)")
    else:
        lines.append(f"FLAGGED FAMILIES ({len(flagged)}):")
        for (base, ext), paths in flagged:
            lines.append(f"  {base}_*.{ext}  ({len(paths)} distinct files)")
            for line_num, ts, path in sorted(paths):
                lines.append(f"    L{line_num:<5} [{short_time(ts)}]  {path}")
            lines.append("")

    near = sorted(
        [(k, v) for k, v in families.items() if len(v) == threshold - 1],
        key=lambda kv: -len(kv[1]),
    )
    if near:
        lines.append(f"NEAR THRESHOLD ({threshold-1} distinct, will flag on next sibling):")
        for (base, ext), paths in near:
            lines.append(f"  {base}_*.{ext}  ({len(paths)} distinct)")

    return "\n".join(lines)


# Redirect forms, both shells: `> F 2>&1` and `&> F` (POSIX, and `> F 2>&1`
# works verbatim in PowerShell), plus PowerShell's `*> F` and the pipeline
# forms `| Out-File F` / `| Tee-Object -FilePath F` — the way a PowerShell call
# actually parks output in a file.
_REDIRECT_RE = re.compile(
    r"(?:>\s*(\S+?)\s+2>&1"
    r"|&>\s*(\S+)"
    r"|\*>\s*(\S+)"
    r"|(?:Out-File|Tee-Object)(?:\s+-(?:FilePath|Path))?\s+[\"']?([^\"'\s|]+))",
    re.IGNORECASE,
)
_INSPECT_CMDS_RE = re.compile(
    r"\b(cat|head|tail|less|more|grep|rg|awk|sed|jq"
    r"|Get-Content|gc|Select-String|sls|findstr)\b", re.IGNORECASE)


def compute_redirect_read(events: list[dict], size_threshold: int = 2000) -> str:
    """Flag cross-tool-call instances of the redirect-then-read antipattern.

    Pattern: a foreground shell call (Bash or PowerShell) with ``> FILE 2>&1``
    / ``&> FILE`` / ``*> FILE`` / ``| Out-File FILE`` / ``| Tee-Object FILE``,
    followed by a separate tool call that reads the full contents of FILE
    (Read with no offset/limit, or a shell call with cat/head/tail/grep/.../jq
    or Get-Content/Select-String/findstr), where the downstream result body is
    small enough (< ``size_threshold`` bytes) that a bare command would have
    fit inline.

    Ignored:
    - Backgrounded shell calls (bg output files are part of a different flow).
    - Read with offset or limit (deliberate slice, not full read).
    - Downstream results larger than the threshold (redirect was prudent).
    """
    redirects: list[dict] = []
    for e in events:
        if e["type"] != "tool_call" or e.get("tool_name") not in _SHELL_TOOLS:
            continue
        inp = e.get("tool_input") or {}
        if inp.get("run_in_background"):
            continue
        cmd = inp.get("command", "") or ""
        m = _REDIRECT_RE.search(cmd)
        if not m:
            continue
        fname = m.group(1) or m.group(2) or m.group(3) or m.group(4)
        if not fname:
            continue
        redirects.append({
            "line": e["line"],
            "ts": e["ts"],
            "fname": fname,
            "cmd": cmd,
        })

    pairs: list[dict] = []
    for r in redirects:
        for e in events:
            if e["line"] <= r["line"] or e["type"] != "tool_call":
                continue
            name = e.get("tool_name", "")
            inp = e.get("tool_input") or {}
            downstream_kind = ""
            if name == "Read":
                if inp.get("file_path") != r["fname"]:
                    continue
                if inp.get("offset") is not None or inp.get("limit") is not None:
                    continue
                downstream_kind = "Read"
            elif name in _SHELL_TOOLS:
                c = inp.get("command", "") or ""
                m = _INSPECT_CMDS_RE.search(c) if r["fname"] in c else None
                if m is None:
                    continue
                downstream_kind = f"{name}:{m.group(1)}"
            else:
                continue
            result = e.get("paired_result") or {}
            detail = result.get("detail", "") or ""
            if len(detail) >= size_threshold:
                break  # large enough that redirect was justified
            pairs.append({
                "src_line": r["line"],
                "src_ts": r["ts"],
                "src_cmd": r["cmd"],
                "fname": r["fname"],
                "dst_line": e["line"],
                "dst_ts": e["ts"],
                "dst_kind": downstream_kind,
                "dst_bytes": len(detail),
            })
            break

    lines = [
        f"REDIRECT-THEN-READ ANTIPATTERN (across tool calls, result < {size_threshold}B)",
        "=" * 60,
    ]
    if not pairs:
        lines.append("(none)")
        return "\n".join(lines)

    for p in sorted(pairs, key=lambda x: x["src_line"]):
        cmd_excerpt = p["src_cmd"].replace("\n", " ")[:80]
        lines.append(
            f"  L{p['src_line']:<5} [{short_time(p['src_ts'])}] Bash: {cmd_excerpt}"
        )
        lines.append(f"    redirect target: {p['fname']}")
        lines.append(
            f"    L{p['dst_line']:<5} [{short_time(p['dst_ts'])}] "
            f"{p['dst_kind']} ({p['dst_bytes']}B returned)"
        )
    lines.append("")
    lines.append(
        f"Total: {len(pairs)} pair(s). Bare execution would have returned the "
        f"output inline without the intermediate file."
    )
    return "\n".join(lines)


def compute_fast_bg(events: list[dict], meta_events: list[dict], threshold_s: float = 5.0) -> str:
    """Flag bg shell tasks (Bash/PowerShell) whose launch->notify gap is under
    `threshold_s`.

    Short-lived backgrounded commands didn't benefit from bg mechanics —
    the ceremony (task-id, notification, read-output file) costs more than
    running foreground with a small timeout would have.  Emits one row per
    qualifying task; the Stop hook paired with this mode dedupes against
    prior warnings (see `bg-task-fast-check.py`).
    """
    tasks: dict[str, dict] = {}
    for e in events:
        if e["type"] != "tool_call" or e.get("tool_name") not in _SHELL_TOOLS:
            continue
        if not (e.get("tool_input") or {}).get("run_in_background"):
            continue
        result = e.get("paired_result")
        if not result:
            continue
        detail = result.get("detail", "") or ""
        m_id = _TASK_ID_RE.search(detail)
        if not m_id:
            continue
        task_id = m_id.group(1)
        tasks[task_id] = {
            "task_id": task_id,
            "launch_ts": e["ts"],
            "launch_line": e["line"],
            "command": (e.get("tool_input") or {}).get("command", ""),
            "status": None,
        }

    # Notify time from queue-op enqueues (task completion signal). Keep the
    # terminal status: a fast failed/killed/stopped task is not a successful
    # backgrounding mistake and must not feed the Stop-side success nudge.
    for m in meta_events:
        if m["type"] != "queue-operation" or m.get("operation") != "enqueue":
            continue
        content = m.get("content", "") or ""
        if "task-notification" not in content:
            continue
        mid = _NOTIF_TASK_ID_RE.search(content)
        if not mid:
            continue
        t = tasks.get(mid.group(1))
        if t and "notified_ts" not in t:
            t["notified_ts"] = m["ts"]
            t["notified_line"] = m["line"]
            status = _NOTIF_STATUS_RE.search(content)
            if status:
                t["status"] = status.group(1).strip().lower()

    lines = [
        f"FAST BG TASKS (launch->notify < {threshold_s}s — may not have needed backgrounding)",
        "=" * 60,
    ]
    fast: list[dict] = []
    for t in tasks.values():
        if t.get("status") != "completed":
            continue
        if "notified_ts" not in t:
            continue
        t_launch = parse_timestamp(t["launch_ts"])
        t_notify = parse_timestamp(t["notified_ts"])
        if not t_launch or not t_notify:
            continue
        gap = (t_notify - t_launch).total_seconds()
        if 0 <= gap < threshold_s:
            t["gap"] = gap
            fast.append(t)

    if not fast:
        lines.append("(none)")
        return "\n".join(lines)

    for t in sorted(fast, key=lambda x: x["launch_line"]):
        cmd = (t["command"] or "").replace("\n", " ")[:80]
        lines.append(
            f"  task={t['task_id']:<12} launch=L{t['launch_line']:<5} "
            f"[{short_time(t['launch_ts'])}]  (+{t['gap']:.1f}s)  cmd: {cmd}"
        )
    lines.append("")
    lines.append(
        f"Total: {len(fast)} fast bg task(s). Foreground Bash with "
        f"timeout<=15s would have returned inline."
    )
    return "\n".join(lines)


# The marker names the tool it covers (`Rulecheck Bash:` / `Rulecheck
# PowerShell:`) — captured as group 1 so a pair written for one shell cannot
# satisfy a call made through the other.
_RULECHECK_LINE_RE = re.compile(
    r"^Rulecheck (Bash|PowerShell):\s*\|=(\d+),\s*timeout=(\d+|bg),"
    r"\s*redirect=(.+?),\s*read-first=(.+?)\s*$"
)
_RULECHECK_BACKTICK_RE = re.compile(r"^\s*`(.+)`\s*$")
_RULECHECK_BACKTICK_OPEN_RE = re.compile(r"^\s*`(.*)$")
_RULECHECK_BACKTICK_CLOSE_RE = re.compile(r"^(.*)`\s*$")


def _hr0_extract_backticked(pre_lines: list[str], k: int):
    """Extract the backticked command literal whose closing line is pre_lines[k-1].

    Single-line ``cmd`` returns "cmd". For multi-line spans (heredoc commit
    bodies), walks back to a line opening with ` and rejoins the literal
    across newlines. Middle lines must contain no backticks. Returns None
    if no well-formed backtick block ends on line k-1.
    """
    if k <= 0:
        return None
    above = pre_lines[k - 1]
    mb = _RULECHECK_BACKTICK_RE.match(above)
    if mb:
        return mb.group(1).strip()
    mc = _RULECHECK_BACKTICK_CLOSE_RE.match(above)
    if not mc:
        return None
    closing_head = mc.group(1)
    if "`" in closing_head:
        return None
    for j in range(k - 2, -1, -1):
        line = pre_lines[j]
        n = line.count("`")
        if n == 0:
            continue
        if n != 1:
            return None
        mo = _RULECHECK_BACKTICK_OPEN_RE.match(line)
        if not mo:
            return None
        opening_tail = mo.group(1)
        middle = list(pre_lines[j + 1 : k - 1])
        return "\n".join([opening_tail] + middle + [closing_head]).strip()
    return None


def _hr0_strip_quotes(cmd: str) -> str:
    """Remove single- and double-quoted spans (matches hr2-pipe-redirect.py)."""
    s = re.sub(r"'[^']*'", '""', cmd)
    s = re.sub(r'"(?:[^"\\]|\\.)*"', '""', s)
    return s


def _hr0_count_pipes(cmd: str) -> int:
    """Count `|` outside quoted strings, treating `||` as not pipes."""
    s = _hr0_strip_quotes(cmd)
    return len(re.findall(r"(?<!\|)\|(?!\|)", s))


def _hr0_redirect_form(cmd: str) -> str:
    """Classify cmd's stdout/stderr redirect: 'none', '> file 2>&1', '&> file', 'other'."""
    s = _hr0_strip_quotes(cmd)
    # Drop stdin redirects so they don't trigger the `>` fallback.
    s = re.sub(r"<<<\s*\S+", "", s)
    s = re.sub(r"<<-?\s*\S+", "", s)
    s = re.sub(r"(?<![<>])<\s*\S+", "", s)
    if re.search(r"&>\s*\S+", s):
        return "&> file"
    if re.search(r">\s*\S+\s+2>&1", s) or re.search(r"2>&1\s*>\s*\S+", s):
        return "> file 2>&1"
    if ">" in s:
        return "other"
    # PowerShell parks output in a file through the pipeline rather than a `>`
    # operator, so classify Out-File / Tee-Object as a redirect ("other")
    # instead of reporting "none" — the latter would make a correct claim look
    # like a mismatch.
    if re.search(r"\b(?:Out-File|Tee-Object)\b", s, re.IGNORECASE):
        return "other"
    return "none"


def _hr0_check_pair(cmd: str, is_bg: bool, timeout, preceding: str,
                    tool: str = "Bash"):
    """Return None if a valid HR0 Rulecheck pair exists in `preceding` for this
    shell call; else a short reason describing the violation.

    Pair shape (the marker names the tool — Bash or PowerShell):
        `<command literal>`
        Rulecheck Bash: |=N, timeout=M|bg, redirect=R, read-first=P|N/A
    """
    pre_lines = preceding.split("\n")
    cmd_first = cmd.split("\n", 1)[0].strip()
    cmd_full = cmd.strip()

    # Locate every Rulecheck line in the preceding text.
    candidates: list[tuple[int, "re.Match[str]"]] = []
    for k, line in enumerate(pre_lines):
        m = _RULECHECK_LINE_RE.match(line)
        if m:
            candidates.append((k, m))

    if not candidates:
        return "no-pair"

    matched = None
    for k, m in candidates:
        if k == 0:
            continue
        literal = _hr0_extract_backticked(pre_lines, k)
        if literal is None:
            literal = pre_lines[k - 1].strip()
        if literal == cmd_first or literal == cmd_full:
            matched = m
            break

    if matched is None:
        return "no-pair"

    tool_str, n_pipe_str, timeout_str, redirect_str, _readfirst = matched.groups()
    if tool_str != tool:
        return f"tool-mismatch (claimed={tool_str} actual={tool})"
    actual_pipes = _hr0_count_pipes(cmd)
    try:
        claimed_pipes = int(n_pipe_str)
    except ValueError:
        claimed_pipes = -1
    if actual_pipes != claimed_pipes:
        return f"pipe-mismatch (claimed=|={claimed_pipes} actual=|={actual_pipes})"

    timeout_str = timeout_str.strip()
    if is_bg:
        if timeout_str != "bg":
            return f"timeout-mismatch (claimed={timeout_str} actual=bg)"
    else:
        actual_to = timeout if timeout is not None else 120000
        try:
            claimed_to = int(timeout_str)
        except ValueError:
            return f"timeout-mismatch (claimed={timeout_str} actual={actual_to})"
        if claimed_to != actual_to:
            return f"timeout-mismatch (claimed={claimed_to} actual={actual_to})"

    actual_redir = _hr0_redirect_form(cmd)
    claimed_redir = redirect_str.strip()
    if claimed_redir != actual_redir:
        return f"redirect-mismatch (claimed={claimed_redir!r} actual={actual_redir!r})"

    return None


def compute_rulecheck(events: list[dict]) -> str:
    """Find shell tool_uses with missing or mismatched HR0 Rulecheck pairs.

    The HR0 contract requires every shell call — Bash or PowerShell — to be
    preceded, in the same assistant message body, by a two-line pair: the
    literal command on its own line, then
    `Rulecheck <tool>: |=N, timeout=M|bg, redirect=R, read-first=P|N/A`
    where each slot value matches the actual invocation and `<tool>` names the
    shell it was made through.

    Output one line per offender (tab-delimited so hooks can parse cheaply):
        L<line>\\ttool_use=<id>\\tmiss=<reason>\\tcmd=<cmd[:80]>
    Empty output means no violations found.
    """
    out_lines: list[str] = []
    for i, e in enumerate(events):
        if e["type"] != "tool_call" or e.get("tool_name") not in _SHELL_TOOLS:
            continue
        ti = e.get("tool_input") or {}
        cmd = ti.get("command", "")
        is_bg = bool(ti.get("run_in_background"))
        timeout = ti.get("timeout")
        tool = e.get("tool_name") or "Bash"
        tid = e.get("tool_use_id", "")
        line_n = e["line"]

        # Walk back collecting assistant_text up to the true turn boundary
        # (user_message). tool_result is intra-turn — Claude Code splits one
        # API response across multiple JSONL records and the harness sequences
        # parallel tool_uses through tool_results, so the Rulecheck pair for a
        # later tool_use can live in an earlier record's text.
        chunks: list[str] = []
        for j in range(i - 1, -1, -1):
            ee = events[j]
            t = ee["type"]
            if t == "user_message":
                break
            if t == "assistant_text":
                chunks.insert(0, ee.get("detail", ""))
        preceding = "\n".join(chunks)

        miss = _hr0_check_pair(cmd, is_bg, timeout, preceding, tool)
        if miss is None:
            continue
        cmd_short = cmd.replace("\n", " ").replace("\t", " ")[:80]
        out_lines.append(f"L{line_n}\ttool_use={tid}\tmiss={miss}\tcmd={cmd_short}")

    return "\n".join(out_lines)


ERROR_PATTERNS = re.compile(
    r"error|Error|ERROR|Traceback|FAILED|exception|Exception|"
    r"AttributeError|NameError|ImportError|TypeError|ValueError|"
    r"KeyError|IndexError|FileNotFoundError|ModuleNotFoundError|"
    r"column .* does not exist|no such file|permission denied|"
    r"Exit code [1-9]",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# --branches: genuine rewind / conversation-branch detection.
#
# A Claude Code session JSONL is a TREE (each record carries parentUuid), not a
# list. When the user rewinds/checkpoint-restores and continues, the abandoned
# branch's records stay on disk — every other mode counts them linearly. This
# mode reconstructs the tree and surfaces the abandoned branches.
#
# The load-bearing subtlety: a parent with 2+ children is NOT automatically a
# fork. Normal threading produces multi-child parents that are NOT rewinds:
#   * streaming-split — one assistant API response is written as N records that
#     chain parent→child, all sharing message.id;
#   * tool pairing — an assistant tool_use record's children are BOTH its
#     streaming continuation AND the tool_result (a user-role record).
# So a "genuine" fork requires 2+ children that survive collapsing (a) children
# sharing the parent's message.id and (b) tool_result children. This is why the
# mode reconstructs the RAW record graph itself rather than reusing the
# sub-block event stream (which splits one record into thinking/text/tool_use
# events and drops parentUuid) — one graph node must map to one JSONL record.
# ---------------------------------------------------------------------------


def _branch_msg_id(rec: dict):
    m = rec.get("message")
    return m.get("id") if isinstance(m, dict) else None


def _branch_is_tool_result(rec: dict) -> bool:
    m = rec.get("message")
    c = m.get("content") if isinstance(m, dict) else None
    return isinstance(c, list) and any(
        isinstance(b, dict) and b.get("type") == "tool_result" for b in c
    )


def _branch_summary(rec: dict, limit: int) -> str:
    """One-line content summary of a raw record for branch display."""
    m = rec.get("message", {})
    c = m.get("content") if isinstance(m, dict) else None
    if isinstance(c, str):
        s = " ".join(c.split())
        return s if len(s) <= limit else s[:limit] + "…"
    parts: list[str] = []
    if isinstance(c, list):
        for b in c:
            if not isinstance(b, dict):
                continue
            bt = b.get("type")
            if bt == "text":
                parts.append(" ".join(str(b.get("text", "")).split()))
            elif bt == "thinking":
                parts.append("thinking…")
            elif bt == "tool_use":
                inp = json.dumps(b.get("input", {}), ensure_ascii=False)
                parts.append(f"TOOL {b.get('name')}: {inp}")
            elif bt == "tool_result":
                rc = b.get("content")
                s = rc if isinstance(rc, str) else json.dumps(rc, ensure_ascii=False)
                parts.append(f"result: {' '.join(str(s).split())}")
    joined = " | ".join(p for p in parts if p)
    return joined if len(joined) <= limit else joined[:limit] + "…"


def _branch_read_records(paths: list[str]) -> list[tuple[str, list[dict]]]:
    """Read raw record-level JSONL (one node per line) from each path, deduped
    by uuid across files. Returns [(path, records)] preserving per-file grouping
    so each session's tree is reconstructed independently — uuids are unique per
    session, so a folder join of DISTINCT sessions must not collapse to one HEAD.
    The cross-file dedup only ever fires for a session's subagent companion file
    (which repeats the parent's uuids)."""
    per_file: list[tuple[str, list[dict]]] = []
    seen: set[str] = set()
    for p in paths:
        recs: list[dict] = []
        try:
            with _open_text(p) as f:
                for ln, line in enumerate(f, 1):
                    line = line.rstrip("\n")
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except (json.JSONDecodeError, ValueError):
                        continue
                    if not isinstance(obj, dict):
                        continue
                    u = obj.get("uuid")
                    if u:
                        if u in seen:
                            continue
                        seen.add(u)
                    obj["_line"] = ln
                    recs.append(obj)
        except OSError:
            continue
        per_file.append((p, recs))
    return per_file


def _branch_stats(sub: list[dict]) -> tuple[int, int, int, int]:
    """(message records, tool_use calls, output tokens, subagent spawns) over a
    subtree's records. Counts user+assistant records (incl. sidechain — abandoned
    subagent work is abandoned work)."""
    msgs = tools = out_tok = subagents = 0
    for r in sub:
        if r.get("type") not in ("user", "assistant"):
            continue
        msgs += 1
        m = r.get("message", {})
        c = m.get("content") if isinstance(m, dict) else None
        if isinstance(c, list):
            for b in c:
                if isinstance(b, dict) and b.get("type") == "tool_use":
                    tools += 1
                    if b.get("name") in ("Agent", "Task"):
                        subagents += 1
        u = m.get("usage") if isinstance(m, dict) else None
        if isinstance(u, dict):
            out_tok += u.get("output_tokens") or 0
    return msgs, tools, out_tok, subagents


def _branch_file_section(path: str, recs: list[dict], after, before, verbose: bool,
                         with_source: bool) -> tuple[list[str], int, int]:
    """Build the report lines for ONE session file. Returns (lines, n_forks,
    n_joins_ignored)."""
    prev = 60 if not verbose else 10 ** 9
    body_prev = 100 if not verbose else 10 ** 9
    by_uuid = {r["uuid"]: r for r in recs if r.get("uuid")}

    # children adjacency over ALL records (for full subtree descent)
    all_kids: dict[str, list[dict]] = {}
    # children adjacency over MESSAGE records only (for fork detection)
    msg_kids: dict[str, list[dict]] = {}
    for r in recs:
        p = r.get("parentUuid")
        if not p or not r.get("uuid"):
            continue
        all_kids.setdefault(p, []).append(r)
        if r.get("type") in ("user", "assistant") and not r.get("isSidechain"):
            msg_kids.setdefault(p, []).append(r)

    # HEAD = last non-sidechain user/assistant record in file order (anchor the
    # live-branch walk here, NOT on trailing meta records that carry no parent).
    head = None
    for r in recs:
        if r.get("type") in ("user", "assistant") and r.get("uuid") and not r.get("isSidechain"):
            head = r
    live: set[str] = set()
    cur = head
    while isinstance(cur, dict):
        u = cur.get("uuid")
        if u in live:
            break
        if u:
            live.add(u)
        pp = cur.get("parentUuid")
        cur = by_uuid.get(pp) if pp else None

    def descend(start_uuid: str) -> list[dict]:
        out: list[dict] = []
        stack = [start_uuid]
        seen: set[str] = set()
        while stack:
            u = stack.pop()
            if u in seen:
                continue
            seen.add(u)
            node = by_uuid.get(u)
            if node is not None:
                out.append(node)
            for c in all_kids.get(u, []):
                cu = c.get("uuid")
                if cu:
                    stack.append(cu)
        return out

    # Detect forks.
    forks: list[dict] = []
    joins_ignored = 0
    for parent_uuid, kids in msg_kids.items():
        if len(kids) < 2:
            continue
        parent = by_uuid.get(parent_uuid)
        pmid = _branch_msg_id(parent) if parent else None
        genuine = [
            k for k in kids
            if not (pmid and _branch_msg_id(k) == pmid) and not _branch_is_tool_result(k)
        ]
        if len(genuine) < 2:
            joins_ignored += 1
            continue
        forks.append({
            "parent": parent,
            "genuine": genuine,
            "ts": (parent or {}).get("timestamp", ""),
            "line": (parent or {}).get("_line", 0),
        })

    # Honor --after/--before on the fork's branch-point timestamp.
    if after or before:
        forks = filter_time(forks, after, before)
    forks.sort(key=lambda fk: fk["line"])

    # Session-level abandoned/live totals.
    abandoned_recs: dict[str, dict] = {}
    for fk in forks:
        for child in fk["genuine"]:
            if child.get("uuid") in live:
                continue
            for r in descend(child["uuid"]):
                if r.get("uuid"):
                    abandoned_recs[r["uuid"]] = r
    a_msgs, a_tools, a_tok, a_sub = _branch_stats(list(abandoned_recs.values()))
    live_recs = [r for r in recs if r.get("uuid") in live]
    l_msgs, l_tools, l_tok, _ = _branch_stats(live_recs)

    lines: list[str] = []
    src_tag = f"[{os.path.basename(path)}] " if with_source else ""
    lines.append(f"{src_tag}{len(forks)} genuine rewind{'s' if len(forks) != 1 else ''} "
                 f"· {joins_ignored} tool-thread join{'s' if joins_ignored != 1 else ''} ignored")
    lines.append(f"    abandoned: {a_msgs} msgs / {a_tools} tools / {a_tok:,} output tok"
                 + (f" / {a_sub} subagent spawn(s)" if a_sub else "")
                 + f"      live: {l_msgs} msgs / {l_tools} tools / {l_tok:,} output tok")

    for fk in forks:
        parent = fk["parent"]
        pt = short_time((parent or {}).get("timestamp", ""))
        pl = (parent or {}).get("_line", 0)
        psum = _branch_summary(parent, prev) if parent else "(session root)"
        lines.append("")
        lines.append(f"L{pl:<5} rewind @ [{pt}]  branch point: {psum}")
        for child in fk["genuine"]:
            sub = descend(child["uuid"])
            c_msgs, c_tools, c_tok, c_sub = _branch_stats(sub)
            cl = child.get("_line", 0)
            csum = _branch_summary(child, prev)
            if child.get("uuid") in live:
                extra = f" / {c_sub} subagent(s)" if c_sub else ""
                lines.append(f"    ● LIVE      L{cl:<5} {csum}")
                lines.append(f"                → continues to HEAD  ({c_msgs} msgs / {c_tools} tools / {c_tok:,} tok{extra})")
            else:
                extra = f" / {c_sub} subagent spawn(s)" if c_sub else ""
                lines.append(f"    ○ ABANDONED L{cl:<5} {csum}")
                lines.append(f"                {c_msgs} msgs / {c_tools} tools / {c_tok:,} tok{extra}  — full trace:")
                for r in sorted(sub, key=lambda x: x.get("_line", 0)):
                    if r.get("type") not in ("user", "assistant"):
                        continue
                    rt = "asst" if r.get("type") == "assistant" else "user"
                    sc = " SC" if r.get("isSidechain") else ""
                    rl = r.get("_line", 0)
                    lines.append(f"        L{rl:<5} [{short_time(r.get('timestamp', ''))}] {rt}{sc}: {_branch_summary(r, body_prev)}")
    return lines, len(forks), joins_ignored


def compute_branches(paths: list[str], after: str | None = None,
                     before: str | None = None, verbose: bool = False,
                     with_source: bool = False) -> str:
    """--branches: reconstruct the conversation tree(s) and surface abandoned
    rewind branches in full. Does NOT alter any other mode — it re-reads the raw
    session file(s) record-by-record (see the block comment above for why).

    paths: the resolved source jsonl file(s). Each is treated as an independent
    session tree. Live branches are summarized (visible in every other mode);
    abandoned branches are dumped in full (visible nowhere else)."""
    if not paths:
        return "BRANCHES / REWINDS\n" + "=" * 40 + "\n(no source files)"
    per_file = _branch_read_records(paths)
    multi = len([pf for pf in per_file if pf[1]]) > 1
    out: list[str] = ["BRANCHES / REWINDS", "=" * 40]
    total_forks = 0
    for path, recs in per_file:
        if not recs:
            continue
        section, nf, _ = _branch_file_section(
            path, recs, after, before, verbose, with_source or multi)
        total_forks += nf
        out.extend(section)
    if total_forks == 0:
        out.append("")
        out.append("No genuine rewinds — every branch point is normal tool/streaming threading.")
    return "\n".join(out)


# Mode-flag attribute names on the parsed args namespace. Used to
# count modes per segment in --and chained mode (exactly one mode flag
# required per segment) and to label stdout section headers.
_MODE_FLAG_NAMES = (
    "stats", "files", "commands", "edits", "writes",
    "cache", "cache_anomalies", "script_churn", "fast_bg",
    "redirect_read", "rulecheck", "latency", "reply_latency",
    "reply_latency_words", "reply_latency_word",
    "bg_tasks", "turns", "agent_tasks", "compact", "hooks",
    "api_errors", "slash", "shell_errors", "tool_orphans",
    "last_assistant_message",
    "timeline", "context_growth", "pairs", "branches",
)
# Type filter attribute names (--tools, --results, ...) — these route
# to the type-filter fallback path. They count as a "mode" only when
# no _MODE_FLAG_NAMES flag is set in the segment.
_TYPE_FILTER_NAMES = (
    "tools", "results", "user", "assistant", "thinking",
    "agents", "conversation", "errors",
)


def _split_argv_on_and(argv: list[str]) -> list[list[str]]:
    """Split argv on the literal token `--and`. Always returns >=1 chunk.

    A chain `--cache --and --latency --and --turns` splits to three
    chunks. The `--and` token itself is consumed (not present in any
    chunk). Empty chunks (e.g. `--and --and`) are preserved so the
    caller can detect the misuse and error.
    """
    chunks: list[list[str]] = [[]]
    for tok in argv:
        if tok == "--and":
            chunks.append([])
        else:
            chunks[-1].append(tok)
    return chunks


def _active_mode_flags(args: argparse.Namespace) -> list[str]:
    """Return the list of mode-flag names active on `args`.

    `--hooks` and `--hook-event` both route to compute_hooks; either
    counts as activating the hooks mode. --hook-event without --hooks
    counts as one mode.
    """
    active: list[str] = []
    for name in _MODE_FLAG_NAMES:
        if getattr(args, name, False):
            active.append(name)
    if getattr(args, "hook_event", None):
        if "hooks" not in active:
            active.append("hooks")
    return active


def _segment_label(args: argparse.Namespace) -> str:
    """Pick a short label for a segment (used in stdout section headers
    and error messages). Short-circuit flags (--cheat-sheet, --version)
    take precedence over mode/type-filter flags so a `--version` segment
    is labelled clearly. Falls back to `(default)` for a bare segment."""
    if args.cheat_sheet:
        return "--cheat-sheet"
    if args.version:
        return "--version"
    modes = _active_mode_flags(args)
    if modes:
        # Render --hook_event back as --hook-event to match the CLI form.
        return "--" + modes[0].replace("_", "-")
    type_filters = [n for n in _TYPE_FILTER_NAMES if getattr(args, n, False)]
    if type_filters:
        return "+".join("--" + n for n in type_filters)
    return "(default)"


def _run_segment(
    args: argparse.Namespace,
    events: list[dict],
    meta_events: list[dict],
) -> str:
    """Apply this segment's filters, dispatch its mode, return formatted output.

    Pure with respect to the input lists: filters return new lists,
    so calling this multiple times against the same shared `events` /
    `meta_events` is safe.
    """
    if args.full:
        args.verbose = True

    # Per-segment cheat-sheet / version short-circuit.
    if args.cheat_sheet:
        return _LINE_GREP_CHEAT_SHEET
    if args.version:
        return f"parse_session.py {__version__}"

    # Capture the source file set BEFORE filters reassign `events` — the
    # --branches mode re-reads the raw session file(s) record-by-record (it
    # needs the full parentUuid graph, which the sub-block event stream drops).
    _src_paths = sorted(
        {str(x.get("src")) for x in events if x.get("src")}
        | {str(x.get("src")) for x in meta_events if x.get("src")}
    )

    # Apply range / time / path / tool filters (each returns a new list).
    if args.range:
        events = filter_line_range(events, args.range)
        meta_events = filter_line_range(meta_events, args.range)
    events = filter_time(events, args.after, args.before)
    meta_events = filter_time(meta_events, args.after, args.before)
    if args.path:
        events = filter_by_path(events, args.path)
    if args.tool:
        tool_names = {t.strip() for t in args.tool.split(",")}
        events = [
            e for e in events
            if (e["type"] in ("tool_call", "agent_spawn") and e.get("tool_name", "") in tool_names)
            or (e["type"] == "tool_result" and e.get("paired_call", {}).get("tool_name", "") in tool_names)
            or e["type"] not in ("tool_call", "agent_spawn", "tool_result")
        ]

    # Mode dispatch — first matching predicate wins.
    mode_dispatch = [
        (lambda a: a.stats,           lambda a, e, m: compute_stats(e, m)),
        (lambda a: a.files,           lambda a, e, m: compute_files(e)),
        (lambda a: a.commands,        lambda a, e, m: compute_commands(e)),
        (lambda a: a.edits,           lambda a, e, m: compute_edits(e, verbose=a.verbose)),
        (lambda a: a.writes,          lambda a, e, m: compute_writes(e, verbose=a.verbose)),
        (lambda a: a.cache,           lambda a, e, m: compute_cache(m)),
        (lambda a: a.cache_anomalies, lambda a, e, m: compute_cache_anomalies(m)),
        (lambda a: a.script_churn,    lambda a, e, m: compute_script_churn(e)),
        (lambda a: a.fast_bg,          lambda a, e, m: compute_fast_bg(e, m)),
        (lambda a: a.redirect_read,    lambda a, e, m: compute_redirect_read(e)),
        (lambda a: a.rulecheck,        lambda a, e, m: compute_rulecheck(e)),
        (lambda a: a.latency,         lambda a, e, m: compute_latency(e)),
        (lambda a: a.reply_latency,   lambda a, e, m: compute_reply_latency(
            e, src_paths=_src_paths, after=a.after, before=a.before)),
        (lambda a: a.reply_latency_words, lambda a, e, m: compute_reply_latency_words(
            e, src_paths=_src_paths, after=a.after, before=a.before)),
        (lambda a: a.reply_latency_word, lambda a, e, m: compute_reply_latency_word(
            e, a.reply_latency_word, src_paths=_src_paths, after=a.after, before=a.before)),
        (lambda a: a.bg_tasks,        lambda a, e, m: compute_bg_tasks(e, m)),
        (lambda a: a.turns,           lambda a, e, m: compute_turns(e, m)),
        (lambda a: a.agent_tasks,     lambda a, e, m: compute_agent_tasks(e, m)),
        (lambda a: a.compact,         lambda a, e, m: compute_compact(m)),
        (lambda a: a.hooks or a.hook_event,
                                      lambda a, e, m: compute_hooks(m, event_filter=a.hook_event, verbose=a.verbose)),
        (lambda a: a.api_errors,      lambda a, e, m: compute_api_errors(m)),
        (lambda a: a.slash,           lambda a, e, m: compute_slash(m)),
        (lambda a: a.last_assistant_message,
                                      lambda a, e, m: compute_last_assistant_message(e)),
        (lambda a: a.shell_errors,    lambda a, e, m: compute_shell_errors(e)),
        (lambda a: a.tool_orphans,    lambda a, e, m: compute_tool_orphans(e)),
        (lambda a: a.timeline,        lambda a, e, m: compute_timeline(e, verbose=a.verbose)),
        (lambda a: a.context_growth,  lambda a, e, m: compute_context_growth(e, m)),
        (lambda a: a.pairs,           lambda a, e, m: "\n".join(
            format_pair(ev, ev.get("paired_result"), verbose=a.verbose, with_source=a.with_source)
            for ev in e if ev["type"] in ("tool_call", "agent_spawn")
        )),
        (lambda a: a.branches,        lambda a, e, m: compute_branches(
            _src_paths, after=a.after, before=a.before,
            verbose=a.verbose, with_source=a.with_source)),
    ]
    output = None
    for predicate, handler in mode_dispatch:
        if predicate(args):
            output = handler(args, events, meta_events)
            break

    if output is None:
        # Type filter fallback (no analysis mode selected).
        type_flags = {
            "tools": args.tools,
            "results": args.results,
            "user": args.user,
            "assistant": args.assistant,
            "thinking": args.thinking,
            "agents": args.agents,
            "conversation": args.conversation,
            "errors": args.errors,
        }
        active_filters = {k for k, v in type_flags.items() if v}

        if active_filters:
            allowed_types: set[str] = set()
            if "tools" in active_filters:
                allowed_types |= {"tool_call", "agent_spawn"}
            if "results" in active_filters:
                allowed_types.add("tool_result")
            if "user" in active_filters:
                allowed_types.add("user_message")
            if "assistant" in active_filters:
                allowed_types.add("assistant_text")
            if "thinking" in active_filters:
                allowed_types.add("thinking")
            if "agents" in active_filters:
                allowed_types.add("agent_spawn")
                args.verbose = True
            if "conversation" in active_filters:
                allowed_types |= {"user_message", "assistant_text"}
            if "errors" in active_filters:
                allowed_types.add("tool_result")

            events = [e for e in events if e["type"] in allowed_types]

            if "errors" in active_filters and active_filters == {"errors"}:
                events = [
                    e for e in events
                    if e["type"] != "tool_result" or e.get("is_error") or ERROR_PATTERNS.search(e.get("detail", ""))
                ]

        formatted_lines = [format_event(e, verbose=args.verbose, with_source=args.with_source) for e in events]
        output = "\n".join(formatted_lines)

    # --grep block-level filter.
    if args.grep:
        pattern = re.compile(args.grep, re.IGNORECASE)
        blocks: list[str] = []
        current: list[str] = []
        for line in output.split("\n"):
            if line.startswith("L") and current:
                blocks.append("\n".join(current))
                current = [line]
            elif line.startswith("  └─") and current:
                current.append(line)
            elif line == "" and current:
                current.append(line)
            else:
                current.append(line)
        if current:
            blocks.append("\n".join(current))
        output = "\n".join(block for block in blocks if pattern.search(block))

    return output


_TYPE_PROBE_RE = re.compile(r'"type":"([A-Za-z0-9_-]+)"')
_NAME_PROBE_RE = re.compile(r'"name":"([A-Za-z0-9_.-]+)"')
_KEY_PROBE_RE = re.compile(r'"([a-zA-Z_][a-zA-Z0-9_]*)":')


def _shared_value_substrings(values: "set[str]", min_len: int = 3,
                             min_share: int = 2, cap: int = 200) -> list[str]:
    """Substrings (>=min_len chars) shared by >=min_share of the given short
    values. Surfaces cross-record discriminators a per-value `"type":"X"`
    candidate can't — e.g. 'tool_' from {tool_use, tool_result}, letting one
    filter cover several record kinds at once (the --pairs case). Longest first,
    capped."""
    counter: Counter = Counter()
    for v in values:
        seen: set[str] = set()
        n = len(v)
        for i in range(n):
            for j in range(i + min_len, n + 1):
                s = v[i:j]
                if s not in seen:
                    seen.add(s)
                    counter[s] += 1
    shared = [s for s, c in counter.items() if c >= min_share]
    shared.sort(key=lambda s: -len(s))
    return shared[:cap]


def _gather_jsonl(jsonl_path: str) -> list[str]:
    """The .jsonl files for a target: a single file, or every *.jsonl under a dir
    (sorted), mirroring main's resolution so find_linegrep sees the same corpus."""
    p = Path(jsonl_path)
    if p.is_dir():
        files: list[str] = []
        for dp, _, fns in os.walk(jsonl_path, followlinks=True):
            files += [os.path.join(dp, fn) for fn in fns if fn.endswith(JSONL_SUFFIXES)]
        files.sort()
        return files
    return [jsonl_path]


def _cache_parse(path: str):
    """[(line_num, raw_line, objs), ...] — read + json-parse a file ONCE, no
    filter. Lets find_linegrep test many candidate line-filters against the same
    parsed records (via _build_from_cache) without re-reading/re-parsing per
    candidate — the json.loads is the dominant cost, so this is the ~100x win."""
    out = []
    with _open_text(path) as f:
        for line_num, line in enumerate(f, 1):
            if line and line[-1] == "\n":
                line = line[:-1]
            if not line:
                continue
            try:
                objs: list = [json.loads(line)]
            except json.JSONDecodeError:
                objs = list(_iter_json_objects(line))
            out.append((line_num, line, objs))
    return out


def _build_from_cache(records, path, line_filter, seen_uuids):
    """(events, meta_events) from cached records, applying `line_filter` to the
    raw line — byte-identical to extract_events(path, line_filter) but reusing a
    parse. Same pre-parse filter semantics (literal substring fast path / regex)
    and the SAME _process_record_objs record logic."""
    literal_substring = None
    line_search = None
    if isinstance(line_filter, str):
        if _REGEX_META.search(line_filter) is None:
            literal_substring = line_filter
        else:
            line_search = re.compile(line_filter, re.IGNORECASE).search
    elif line_filter is not None:
        line_search = line_filter.search
    events: list[dict] = []
    meta_events: list[dict] = []
    seen_request_events: dict[str, dict] = {}
    for line_num, line, objs in records:
        if literal_substring is not None:
            if literal_substring not in line:
                continue
        elif line_search is not None and line_search(line) is None:
            continue
        _process_record_objs(objs, line_num, events, meta_events,
                             seen_request_events, seen_uuids)
    for e in events:
        e["src"] = path
    for m in meta_events:
        m["src"] = path
    return events, meta_events


def _load_from_caches(caches, line_filter):
    """extract → sort → annotate → link over pre-parsed caches [(path, records),
    ...] — same result as main's load path but with no file re-read and no
    re-parse. Used by find_linegrep per candidate."""
    events: list[dict] = []
    meta_events: list[dict] = []
    seen: "set[str] | None" = set() if len(caches) > 1 else None
    for path, records in caches:
        ev, me = _build_from_cache(records, path, line_filter, seen)
        events.extend(ev)
        meta_events.extend(me)
    events.sort(key=lambda e: e.get("ts") or "￿")
    meta_events.sort(key=lambda m: m.get("ts") or "￿")
    annotate_parallel(events)
    link_pairs(events)
    return events, meta_events


def _scan_linegrep_candidates(files: list[str], mode_tools: "str | None"):
    """Cheap raw pass(es). Returns (total_lines, {pattern: admitted_lines}).

    Candidate pool — DATA-MINED single substrings, not a hardcoded key list:
      - every distinct `"type":"X"` and `"name":"X"` substring,
      - every distinct `"key":` substring (any JSON field, so discriminators
        beyond type/name surface on their own),
      - substrings SHARED across >=2 type/name values (e.g. `tool_` from
        tool_use/tool_result) so one filter can span several record kinds,
      - `"role":"user"`, `"role":"assistant"`, `"usage":`, and `--tool` names.
    `admitted` = how many raw lines each substring lets through."""
    extras = ['"role":"user"', '"role":"assistant"', '"usage":']
    for t in (mode_tools or "").split(","):
        t = t.strip()
        if t:
            extras.append(f'"name":"{t}"')
    counts: dict[str, int] = defaultdict(int)
    values: set[str] = set()
    total = 0
    for f in files:
        try:
            fh = _open_text(f)
        except OSError:
            continue
        with fh:
            for line in fh:
                if not line.strip():
                    continue
                total += 1
                hits: set[str] = set()
                for m in _TYPE_PROBE_RE.finditer(line):
                    hits.add(f'"type":"{m.group(1)}"')
                    values.add(m.group(1))
                for m in _NAME_PROBE_RE.finditer(line):
                    hits.add(f'"name":"{m.group(1)}"')
                    values.add(m.group(1))
                for m in _KEY_PROBE_RE.finditer(line):
                    hits.add(f'"{m.group(1)}":')
                for e in extras:
                    if e in line:
                        hits.add(e)
                for h in hits:
                    counts[h] += 1
    # Second pass: count lines admitted by cross-value shared substrings (e.g.
    # `tool_`), which aren't captured by the per-line `"type":"X"` candidates.
    shared = _shared_value_substrings(values)
    if shared:
        for f in files:
            try:
                fh = _open_text(f)
            except OSError:
                continue
            with fh:
                for line in fh:
                    if not line.strip():
                        continue
                    for s in shared:
                        if s in line:
                            counts[s] += 1
    return total, counts


def _count_admitted(caches, pattern: str) -> int:
    """Raw lines a (possibly-regex) line-grep pattern would admit, over the cached
    records (no file re-read)."""
    rx = re.compile(pattern, re.IGNORECASE)
    n = 0
    for _path, records in caches:
        for _ln, line, _objs in records:
            if rx.search(line):
                n += 1
    return n


def _try_or_union(caches, base_seg, baseline, cands, total, pool: int = 12):
    """None-safe fallback: try the cheapest result-preserving 2-way union `A|B`
    among the `pool` cheapest singles — for a mode whose needed lines split into
    two disjoint record kinds no single substring spans. Bounded (<= pool*(pool-1)/2
    verifies). Returns (pattern, admitted, (A, B)) or None. (Most none-safe modes
    genuinely need ~all lines, so this usually finds nothing — it's best-effort.)"""
    pool_pats = [p for p, _ in cands[:pool]]
    best = None
    for i in range(len(pool_pats)):
        for j in range(i + 1, len(pool_pats)):
            a, b = pool_pats[i], pool_pats[j]
            pat = f"{re.escape(a)}|{re.escape(b)}"
            if _run_segment(base_seg, *_load_from_caches(caches, pat)) == baseline:
                cnt = _count_admitted(caches, pat)
                # `best is None` short-circuits before the subscript.
                # pylint: disable-next=unsubscriptable-object
                if best is None or cnt < best[1]:
                    best = (pat, cnt, (a, b))
    return best


def find_linegrep(jsonl_path: str, args0: argparse.Namespace, max_tests: int = 100_000) -> str:
    """Find the cheapest result-preserving --line-grep for args0's (single) mode.

    Loads the baseline (no filter) once, derives single-substring candidates from
    the raw lines (see _scan_linegrep_candidates), then — CHEAPEST-ADMITTING FIRST
    — re-runs the mode under each candidate and diffs the output against baseline,
    stopping at the first byte-identical (SAFE) one (which is therefore the
    cheapest, given ascending order). Reports the tested candidates and the
    recommendation, or '(none)' if no single substring is both cheaper and
    result-preserving (the correct answer for genuinely multi-record modes).

    Cost: tests ALL reducing candidates (early-stop at the cheapest SAFE), so it
    scales with candidate-count x corpus-size — for a broad-safe mode (e.g.
    --pairs, whose safe cover admits many lines and thus ranks late) that can be
    hundreds of re-load+re-run passes. RUN IT ON A SMALL SUBSET (a few files);
    the discovered pattern applies to the full corpus. Beyond single substrings it
    also tries AND (intersect all safe singles → tightest cover, when the best
    single is broad) and, when no single covers the needed lines, a bounded 2-way
    OR (union of two patterns)."""
    # Works for whatever the segment produces — a named mode, one or more type
    # filters (--thinking, --tools --tool X, ...), or the default event dump.
    mode_label = _segment_label(args0)
    lines = ["FIND --line-grep (cheapest result-preserving filter)", "=" * 40]

    base_seg = argparse.Namespace(**vars(args0))
    base_seg.find_linegrep = False
    base_seg.line_grep = None
    base_seg.output = None

    files = _gather_jsonl(jsonl_path)
    if not files:
        lines.append(f"(no *.jsonl under {jsonl_path})")
        return "\n".join(lines)

    # Parse every file ONCE; all candidate filters below reuse these caches
    # (no re-read, no re-parse) — the json.loads is the dominant per-candidate cost.
    caches = [(f, _cache_parse(f)) for f in files]

    base_ev, base_me = _load_from_caches(caches, None)
    baseline = _run_segment(base_seg, base_ev, base_me)
    baseline_n = baseline.count("\n") + 1

    total, counts = _scan_linegrep_candidates(files, getattr(args0, "tool", None))
    cands = sorted(((p, c) for p, c in counts.items() if 0 < c < total),
                   key=lambda x: x[1])

    lines.append(f"mode: {mode_label}   files: {len(files)}   total lines: {total:,}   "
                 f"baseline output: {baseline_n:,} lines")

    # Degeneracy guard: if filtering to ZERO admitted lines reproduces the
    # baseline, the output doesn't depend on any record — the mode matched
    # nothing on this corpus, so every filter is trivially "safe" and a
    # recommendation would be meaningless (the --thinking-with-no-thinking trap).
    empty_out = _run_segment(base_seg, *_load_from_caches(caches, "\x00\x00no_match\x00\x00"))
    if empty_out == baseline:
        lines.append("")
        lines.append("(baseline output is data-independent: this mode matched nothing on this "
                     "corpus, so every filter is trivially 'safe'. Nothing to recommend — run "
                     "on a corpus where this mode actually produces output.)")
        return "\n".join(lines)

    lines.append(f"(testing {min(len(cands), max_tests)} of {len(cands)} reducing candidates, "
                 f"cheapest first; single → AND-tighten if broad → OR if none safe)")
    lines.append("")
    lines.append(f"  {'admitted':>18}  lit  verdict  pattern")

    # ---- single-substring pass (ascending by admitted count) ----
    # Collect EVERY safe single. Stop early only if the cheapest safe single is
    # already selective — then there's nothing for AND to tighten.
    TIGHTEN_PCT = 5.0
    safe_singles: list[tuple[str, int]] = []
    first_safe = None
    for pat, cnt in cands[:max_tests]:
        safe = _run_segment(base_seg, *_load_from_caches(caches, pat)) == baseline
        literal = _REGEX_META.search(pat) is None
        lines.append(f"  {cnt:>9,} ({100.0 * cnt / total:5.1f}%)  {'Y' if literal else 'n'}    "
                     f"{'SAFE' if safe else 'diff'}     {pat}")
        if safe:
            safe_singles.append((pat, cnt))
            if first_safe is None:
                first_safe = (pat, cnt, literal)
                if cnt <= TIGHTEN_PCT / 100.0 * total:
                    break

    # ---- AND: greedily build the MINIMAL intersection of safe singles ----
    # Each safe single admits needed + its own extra, so any intersection still
    # ⊇ needed (safe). But ANDing every safe single is full of redundancy — a term
    # that's a substring of another (e.g. `tool_` under `tool_use`) is implied, and
    # a term that doesn't shrink the set earns nothing. So add a term only if it is
    # (a) not a substring of an already-chosen term, and (b) strictly reduces the
    # admitted count. Most-selective first. The result is a clean minimal cover.
    and_rec = None
    if first_safe and first_safe[1] > TIGHTEN_PCT / 100.0 * total and len(safe_singles) >= 2:
        chosen: list[str] = []
        chosen_cnt = total
        for p, _c in sorted(safe_singles, key=lambda x: x[1]):
            if any(p in q for q in chosen):
                continue  # substring of a chosen term ⇒ already implied by the AND
            trial = chosen + [p]
            tc = _count_admitted(caches, "".join(f"(?=.*{re.escape(t)})" for t in trial))
            if tc < chosen_cnt:
                chosen, chosen_cnt = trial, tc
        # Final substring-dedup (order usually prevents it, but be safe).
        chosen = [t for t in chosen if not any(t != u and t in u for u in chosen)]
        if len(chosen) >= 2 and chosen_cnt < first_safe[1]:
            and_pat = "".join(f"(?=.*{re.escape(t)})" for t in chosen)
            if _run_segment(base_seg, *_load_from_caches(caches, and_pat)) == baseline:
                and_rec = (and_pat, chosen_cnt, chosen, first_safe[0], first_safe[1])

    # ---- OR: union two patterns when NO single covers the needed lines ----
    or_rec = _try_or_union(caches, base_seg, baseline, cands, total) if first_safe is None else None

    lines.append("")
    if and_rec:
        ap, ac, parts, bsp, bsc = and_rec
        lines.append(f"RECOMMEND (AND — tightened from the best single): --line-grep '{ap}'")
        lines.append(f"  admits {ac:,}/{total:,} = {100.0 * ac / total:.1f}% of lines "
                     f"(best single '{bsp}' admitted {100.0 * bsc / total:.1f}%), "
                     f"regex, byte-identical.")
        lines.append(f"  = every line containing ALL of: {', '.join(parts)}")
    elif first_safe:
        pat, cnt, literal = first_safe
        lines.append(f"RECOMMEND: --line-grep '{pat}'")
        lines.append(f"  admits {cnt:,}/{total:,} = {100.0 * cnt / total:.1f}% of lines, "
                     f"{'literal (fast path)' if literal else 'regex'}, byte-identical to no-filter.")
    elif or_rec:
        op, oc, (a, b) = or_rec
        lines.append(f"RECOMMEND (OR — no single substring covered the needed lines): --line-grep '{op}'")
        lines.append(f"  admits {oc:,}/{total:,} = {100.0 * oc / total:.1f}% of lines, regex, "
                     f"byte-identical. Union of '{a}' and '{b}'.")
    else:
        lines.append(f"RECOMMEND: (none) — no single substring, AND-of-safe-singles, or 2-way OR "
                     f"of the {len(cands)} candidates was result-preserving. Run {mode_label} "
                     f"without --line-grep.")
    return "\n".join(lines)


def main():
    # Force UTF-8 on stdout/stderr so print() / file=sys.stderr work on Windows
    # locales like cp1250 (Czech) / cp932 (Japanese) where the session content
    # contains non-ANSI chars (e.g. U+2192 →). No-op on Linux where they already
    # default to UTF-8.
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8")
            except (OSError, ValueError):
                pass  # non-text stream or already-detached; nothing to do
    parser = argparse.ArgumentParser(
        description="Parse Claude Code session JSONL for investigation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    # Custom --version (store_true, not argparse's auto-exiting `version`
    # action) so chained --and segments can short-circuit individually
    # without killing the whole process.
    parser.add_argument("--version", action="store_true", help=f"Print 'parse_session.py {__version__}' and exit (or short-circuit one --and segment)")
    parser.add_argument("jsonl", nargs="?", help="Path to session .jsonl file, session ID, or directory (recursively joins all *.jsonl, sorted by timestamp). Required unless --cheat-sheet is given.")

    # Event type filters
    type_group = parser.add_argument_group("event type filters (union if multiple)")
    type_group.add_argument("--tools", action="store_true", help="Tool calls + agent spawns")
    type_group.add_argument("--results", action="store_true", help="Tool results only")
    type_group.add_argument("--user", action="store_true", help="User messages only")
    type_group.add_argument("--assistant", action="store_true", help="Assistant text only")
    type_group.add_argument("--thinking", action="store_true", help="Thinking blocks only")
    type_group.add_argument("--agents", action="store_true", help="Agent spawns (with full prompts)")
    type_group.add_argument("--conversation", action="store_true", help="User + assistant text only")
    type_group.add_argument("--pairs", action="store_true", help="Tool calls paired with results")
    type_group.add_argument("--errors", action="store_true", help="Tool results containing errors")

    # Tool-specific filter
    parser.add_argument("--tool", metavar="NAME", help="Filter by tool name (comma-separated for multiple)")

    # Analysis modes (alphabetical)
    analysis_group = parser.add_argument_group("analysis modes")
    analysis_group.add_argument("--agent-tasks", action="store_true", help="Async Agent launches (agentId, output file, completion)")
    analysis_group.add_argument("--api-errors", action="store_true", help="API errors (system/api_error)")
    analysis_group.add_argument("--bg-tasks", action="store_true", help="Background Bash task lifecycle (launch/notify/read)")
    analysis_group.add_argument("--branches", action="store_true", help="Conversation-tree rewind branches: genuine forks (collapsing streaming-split + tool_result joins), abandoned branches dumped in full, live branches summarized. Additive — does not change any other mode. --range/--path/--tool don't apply (reconstructs from full session files); --after/--before scope by branch-point time")
    analysis_group.add_argument("--cache", action="store_true", help="Prompt cache hit rate + token totals (per-model)")
    analysis_group.add_argument("--cache-anomalies", action="store_true", help="Flag single-turn cache outliers (>=25%% of session cache_create, >=1M cache_read, >=5k output)")
    analysis_group.add_argument("--commands", action="store_true", help="All Bash commands")
    analysis_group.add_argument("--compact", action="store_true", help="Compact events (system/compact_boundary)")
    analysis_group.add_argument("--context-growth", action="store_true", help="Context size over time (per-turn input, output, delta)")
    analysis_group.add_argument("--edits", action="store_true", help="All Edit summaries")
    analysis_group.add_argument("--files", action="store_true", help="All files touched")
    analysis_group.add_argument("--latency", action="store_true", help="Rank tool calls by wall-clock latency")
    analysis_group.add_argument("--reply-latency", action="store_true", help="Gap from each user message to the assistant's first response event")
    analysis_group.add_argument("--reply-latency-words", action="store_true", help="Words in user prompts ranked by their effect (slower/faster) on reply latency")
    analysis_group.add_argument("--reply-latency-word", metavar="WORD", help="With-vs-without reply latency for ONE specific word, split by model")
    analysis_group.add_argument("--script-churn", action="store_true", help="Detect /tmp/<base>_*.{sh,py,js} write churn — N+ distinct paths in same family")
    analysis_group.add_argument("--fast-bg", action="store_true", help="Flag bg Bash tasks with launch->notify gap < 5s (should have run foreground)")
    analysis_group.add_argument("--redirect-read", action="store_true", help="Flag cross-call redirect-then-read antipattern (Bash `> file 2>&1` followed by Read/cat of that file, result <2000B)")
    analysis_group.add_argument("--rulecheck", action="store_true", help="Find Bash tool_uses missing or with mismatched HR0 Rulecheck pairs (presence/shape/slot correctness)")
    analysis_group.add_argument("--shell-errors", action="store_true", help="Bash calls with stderr/interrupted/non-zero exit")
    analysis_group.add_argument("--slash", action="store_true", help="Slash commands invoked")
    analysis_group.add_argument("--last-assistant-message", action="store_true",
                                help="Last turn's assistant text, verbatim (raw: no banner/refs) — for Stop-side hooks")
    analysis_group.add_argument("--stats", action="store_true", help="Session statistics")
    analysis_group.add_argument("--timeline", action="store_true", help="Chronological interleave of ALL events (compact one-liner per event)")
    analysis_group.add_argument("--tool-orphans", action="store_true", help="tool_use events without matching tool_result (crashed/dropped)")
    analysis_group.add_argument("--turns", action="store_true", help="Per-turn duration + token spend")
    analysis_group.add_argument("--writes", action="store_true", help="All Write calls (file creation/overwrite)")
    analysis_group.add_argument("--hooks", action="store_true", help="All hook invocations (success/blocked/permission/additional_context)")
    analysis_group.add_argument("--hook-event", metavar="EVENT", help="Filter --hooks by hookEvent (e.g. SessionStart, PreToolUse, Stop)")

    # Path filter
    parser.add_argument("--path", metavar="PATTERN", help="Filter events by file path (substring match)")

    # Range/time/text filters
    parser.add_argument("--range", metavar="N-M", help="JSONL line range")
    parser.add_argument("--after", metavar="WHEN", help="Events at/after WHEN. Accepts HH:MM:SS (intra-day), YYYY-MM-DD, YYYY-MM-DDTHH:MM:SS, or Nd/Nh/Nw relative to now (e.g. 30d).")
    parser.add_argument("--before", metavar="WHEN", help="Events at/before WHEN. Same formats as --after.")
    parser.add_argument("--grep", metavar="PATTERN", help="Regex filter on the formatted output (post-parse, post-mode). Block-level: keeps each L<n> block whose text matches.")
    parser.add_argument("--line-grep", metavar="PATTERN", help=(
        "Regex pre-filter on raw JSONL lines, applied INSIDE extract_events. "
        "Literal patterns (no regex metachars) take a substring fast path; "
        "regex patterns work but are ~5x slower per line. Pointless on a "
        "single jsonl (overhead > savings); only useful in folder mode on "
        "large corpora. Output may show minor differences in turn counts / "
        "parallel-annotation / paired-result-derived stats (e.g. <synthetic> "
        "turns in --cache, [parallel N/M] in --commands, +N/-N lines in "
        "--edits) when the filter excludes records that contribute only "
        "secondary annotations. Dollar/token sums are unaffected. "
        "Use --cheat-sheet to see verified per-mode patterns."
    ))
    parser.add_argument("--cheat-sheet", action="store_true", help="Print a table of recommended --line-grep patterns per mode, then exit. Pointless on a single jsonl.")
    parser.add_argument("--find-linegrep", action="store_true", help="For the chosen mode, empirically find the cheapest --line-grep that yields byte-identical output (tests candidates against an unfiltered baseline), then exit. Needs a folder + one mode.")

    # Output
    parser.add_argument("--full", action="store_true", help="Disable truncation (full result bodies and edit diffs)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show full agent spawn prompts")
    parser.add_argument("-o", "--output", metavar="FILE", help="Write to file")
    parser.add_argument("--with-source", action="store_true", help="Insert source jsonl basename (without .jsonl) between L<n> and timestamp on every formatted event. Useful in folder mode where events from many files are interleaved by timestamp. Per-segment flag.")
    parser.add_argument("--timing", action="store_true", help="Print per-phase wall time to stderr (load/annotate/sort/filter/dispatch)")

    raw_argv = sys.argv[1:]
    chunks = _split_argv_on_and(raw_argv)
    is_chained = len(chunks) > 1

    # Reject empty chunks (`--and --and`, trailing `--and`, leading `--and`).
    for i, chunk in enumerate(chunks):
        if not chunk and is_chained:
            parser.error(
                f"--and segment {i} is empty. Each segment must specify "
                f"at least one mode flag."
            )

    # Parse segment 0 with the full parser (gets jsonl + --line-grep + globals).
    args0 = parser.parse_args(chunks[0])

    # Parse remaining segments. Reuse the same parser; jsonl is nargs='?'
    # so its absence is fine, and we then validate that segments past the
    # first don't carry the constraints reserved for segment 0.
    seg_args: list[argparse.Namespace] = [args0]
    for i, chunk in enumerate(chunks[1:], 1):
        a = parser.parse_args(chunk)
        if a.jsonl is not None:
            parser.error(
                f"--and segment {i}: the jsonl path may appear only in "
                f"the first segment (got {a.jsonl!r}). Move it before the "
                f"first --and."
            )
        if a.line_grep is not None:
            parser.error(
                f"--and segment {i}: --line-grep is a pre-parse filter and "
                f"must appear in the first segment (before --and). All "
                f"--and segments share the same loaded events."
            )
        seg_args.append(a)

    # In chained mode each segment must specify EXACTLY ONE mode flag.
    # Segments with zero modes are still allowed (they fall through to the
    # type-filter / default formatter path inside _run_segment); zero is a
    # legitimate way to dump filtered events. Two or more is ambiguous.
    if is_chained:
        for i, a in enumerate(seg_args):
            modes = _active_mode_flags(a)
            if len(modes) > 1:
                shown = ", ".join("--" + m.replace("_", "-") for m in modes)
                parser.error(
                    f"--and segment {i}: each segment must specify exactly "
                    f"ONE mode flag. Got: {shown}.\n"
                    f"Use one mode per segment, e.g. "
                    f"`parse_session.py JSONL --cache --and --latency`."
                )

    # Single-segment short-circuits: --cheat-sheet / --version do not
    # require a jsonl. Preserves the historical zero-arg behavior.
    if not is_chained:
        if args0.cheat_sheet:
            print(_LINE_GREP_CHEAT_SHEET)
            sys.exit(0)
        if args0.version:
            print(f"parse_session.py {__version__}")
            sys.exit(0)

    # Need a jsonl unless every segment is a pure short-circuit
    # (--cheat-sheet / --version only — no mode flag, no type filter).
    needs_load = any(
        not (a.cheat_sheet or a.version) for a in seg_args
    )
    if needs_load and not args0.jsonl:
        parser.error("the following arguments are required: jsonl (or pass --cheat-sheet)")

    # --timing is global: if any segment requested it, enable.
    timing_enabled = any(a.timing for a in seg_args)
    timings: list[tuple[str, float]] = []

    def _tick(label: str, t0: float) -> None:
        if timing_enabled:
            timings.append((label, time.perf_counter() - t0))

    events: list[dict] = []
    meta_events: list[dict] = []

    if needs_load:
        # Resolve JSONL path. Argument may be a file, a session ID (probe
        # ~/.claude/projects/*/<id>.jsonl), or a directory (walk recursively).
        jsonl_path = args0.jsonl
        jsonl_path_obj = Path(jsonl_path)
        if not jsonl_path_obj.exists():
            projects_root = Path.home() / ".claude" / "projects"
            matches = sorted(
                [*projects_root.glob(f"*/{jsonl_path}.jsonl"),
                 *projects_root.glob(f"*/{jsonl_path}.jsonl.xz")]
            ) if projects_root.is_dir() else []
            if len(matches) == 1:
                jsonl_path = str(matches[0])
                jsonl_path_obj = Path(jsonl_path)
            elif len(matches) > 1:
                print(
                    f"Ambiguous session ID '{jsonl_path}' — matches {len(matches)} files:",
                    file=sys.stderr,
                )
                for p in matches:
                    print(f"  {p}", file=sys.stderr)
                sys.exit(1)
            else:
                print(f"File not found: {jsonl_path}", file=sys.stderr)
                sys.exit(1)

        line_filter: "str | None" = args0.line_grep

        # --find-linegrep: instead of running the mode, discover the cheapest
        # --line-grep that yields identical output for it, then exit.
        if getattr(args0, "find_linegrep", False):
            print(find_linegrep(jsonl_path, args0))
            return

        if jsonl_path_obj.is_dir():
            # os.walk with followlinks=True so callers can stage a corpus by
            # symlinking source dirs into one folder. Path.rglob() does not
            # traverse directory symlinks before Python 3.13.
            t0 = time.perf_counter()
            files: list[str] = []
            for dp, _, fns in os.walk(jsonl_path, followlinks=True):
                for fn in fns:
                    if fn.endswith(JSONL_SUFFIXES):
                        files.append(os.path.join(dp, fn))
            files.sort()
            _tick(f"walk ({len(files)} files)", t0)
            if not files:
                print(f"No *.jsonl files under {jsonl_path}", file=sys.stderr)
                sys.exit(1)
            t0 = time.perf_counter()
            # Cross-file dedup set keyed on the JSONL record uuid so the
            # same API call appearing in two files (main jsonl + subagent
            # data/subagents/agent-*.jsonl) is processed exactly once.
            seen_uuids: set[str] = set()
            for f in files:
                ev, me = extract_events(f, line_filter=line_filter, seen_uuids=seen_uuids)
                events.extend(ev)
                meta_events.extend(me)
            _tick(f"extract_events ({len(events):,} ev / {len(meta_events):,} meta, "
                  f"{len(seen_uuids):,} unique record uuids)", t0)
            t0 = time.perf_counter()
            events.sort(key=lambda e: e.get("ts") or "￿")
            meta_events.sort(key=lambda m: m.get("ts") or "￿")
            _tick("sort by ts", t0)
        else:
            t0 = time.perf_counter()
            events, meta_events = extract_events(jsonl_path, line_filter=line_filter)
            _tick(f"extract_events ({len(events):,} ev / {len(meta_events):,} meta)", t0)

        t0 = time.perf_counter()
        annotate_parallel(events)
        _tick("annotate_parallel", t0)
        t0 = time.perf_counter()
        link_pairs(events)
        _tick("link_pairs", t0)

    # Per-segment dispatch. Errors in one segment do NOT abort the chain —
    # the load was the expensive phase. Failures land on stderr.
    failures: list[tuple[int, str, str]] = []
    for i, seg in enumerate(seg_args):
        label = _segment_label(seg)
        t0 = time.perf_counter()
        try:
            output = _run_segment(seg, events, meta_events)
        except Exception as exc:
            failures.append((i, label, str(exc)))
            print(f"[--and segment {i} ({label}) failed: {exc}]", file=sys.stderr)
            _tick(f"segment {i} {label} (FAILED)", t0)
            continue
        _tick(f"segment {i} {label}", t0)

        if seg.output:
            Path(seg.output).write_text(output, encoding="utf-8")
            print(f"Output written to {seg.output}")
        else:
            if is_chained:
                print(f"\n===== {label} =====")
            print(output)

    if timing_enabled and timings:
        total = sum(t for _, t in timings)
        print(f"\n--- timing (parse_session.py {__version__}) ---", file=sys.stderr)
        for tlabel, t in timings:
            print(f"  {t:7.3f}s  {tlabel}", file=sys.stderr)
        print(f"  {total:7.3f}s  TOTAL (instrumented phases)", file=sys.stderr)

    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
