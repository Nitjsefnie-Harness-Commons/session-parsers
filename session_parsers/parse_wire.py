#!/usr/bin/env python3
"""
Kimi Wire JSONL Parser — investigation utility for Kimi Code CLI transcripts.

Parses both legacy kimi-cli wire.jsonl session logs and the new kimi-code
wire format into structured, readable output. Auto-detects the format per
file. Full untruncated output. Structured filtering. Chronological timeline.

Event type filters (combine freely — union of all selected types):
    --tools             tool calls
    --results           tool results only
    --user              user messages only
    --assistant         assistant text only
    --thinking          thinking blocks only
    --conversation      user + assistant text (no tools)
    --pairs             tool calls paired with their results
    --errors            failed tool results and API-provider errors

Tool-specific filters:
    --tool NAME         filter by tool name (exact match, comma-separated for multiple)

Analysis modes:
    --stats             session statistics overview
    --timeline          chronological interleave of ALL events with compact one-line formatting
    --commands          extract Shell command strings only
    --files             summary of all files touched (read, edited, written, grepped, globbed)
    --writes            summary of all Write calls (file, content size, preview)
    --edits             summary of all StrReplaceFile calls (file, old/new snippets)
    --cache             prompt cache + token totals from StatusUpdate events
    --context-growth    context size over time (turn number, timestamp, input tokens, output tokens)
    --latency           rank tool calls by wall-clock latency (result_ts - call_ts)
    --reply-latency     gap between each user message and the assistant's first response event
    --shell-errors      Shell calls with stderr, interrupted, or non-zero exit
    --tool-orphans      tool calls without a matching tool_result (crashed/dropped calls)
    --turns             per-turn overview from TurnBegin/TurnEnd + StatusUpdate
    --subagents         subagent events summary

Path filter:
    --path PATTERN      filter events by file path (substring match on tool inputs)

Range/time/text filters (combine with any of the above):
    --range N-M         JSONL line range
    --after HH:MM:SS    events after time
    --before HH:MM:SS   events before time
    --grep PATTERN      regex filter on formatted output

Output:
    --full              disable truncation (show full result bodies and edit diffs)
    -v, --verbose       show full content
    -o FILE             write to file

Examples:
    parse_wire.py <wire.jsonl> --tools --tool Shell          # all Shell calls
    parse_wire.py <wire.jsonl> --pairs --tool Read            # Read calls with results
    parse_wire.py <wire.jsonl> --pairs --full                 # pairs with full bodies
    parse_wire.py <wire.jsonl> --files                        # all files touched
    parse_wire.py <wire.jsonl> --writes                       # all files created/overwritten
    parse_wire.py <wire.jsonl> --errors                       # find failures
    parse_wire.py <wire.jsonl> --commands --after 10:25:00    # commands after time
    parse_wire.py <wire.jsonl> --edits --full                 # edits with full diffs
    parse_wire.py <wire.jsonl> --path web/browse.py           # all events touching file
"""

__version__ = "1.4.0"

import io
import json
import lzma
import os
import sys
import argparse
import re
import time
from pathlib import Path
from datetime import datetime, timezone, timedelta
from collections import Counter, defaultdict
from typing import TextIO

# wire.jsonl files may be stored plain or transparently xz-compressed (the R2
# bucket archives them per-object as `wire.jsonl.xz`). xz is stdlib (`lzma`),
# so reading either form needs no third-party dependency.


def _open_text(path) -> TextIO:
    """Open a .jsonl or .jsonl.xz path as a UTF-8 text stream (drop-in for
    open(path, encoding='utf-8')); xz handled transparently via stdlib lzma."""
    if str(path).endswith(".xz"):
        return io.TextIOWrapper(lzma.LZMAFile(path, "rb"), encoding="utf-8")
    return open(path, encoding="utf-8")


def _wire_file(d: Path):
    """The wire file inside a session/uuid dir, plain or xz; None if absent."""
    for name in ("wire.jsonl", "wire.jsonl.xz"):
        w = d / name
        if w.is_file():
            return w
    return None


# Hardcoded model transition.  Anchor = now - 12h; effective cutoff =
# anchor - 12h.  Sessions whose first event is strictly before this UTC
# epoch are treated as kimi-k2-6, everything else as kimi-k2-7-code.
MODEL_CUTOFF_EPOCH = 1781217035
MODEL_CUTOFF_DT = datetime.fromtimestamp(MODEL_CUTOFF_EPOCH, tz=timezone.utc)

# K3_CUTOFF applies ONLY to records that carry no wire model string at all.
# A model string that is present but unrecognized is not the "no model string"
# case, so it must never be promoted to k3 by date alone — k3 is ~3x the k2
# rates and a wrong undercount beats a wrong overcount.
K3_CUTOFF_EPOCH = 1784213155      # 2026-07-16 14:45:55 UTC, earliest observed
                                  # k3 usage.record
K3_CUTOFF_DT = datetime.fromtimestamp(K3_CUTOFF_EPOCH, tz=timezone.utc)

K2_6_RATES = {
    "fresh": 0.95,
    "cache_read": 0.16,
    "cache_create": 0.0,
    "output": 4.00}
K2_7_CODE_RATES = {
    "fresh": 0.95,
    "cache_read": 0.19,
    "cache_create": 0.0,
    "output": 4.00}
K3_RATES = {
    "fresh": 3.00,
    "cache_read": 0.30,
    "cache_create": 0.0,
    "output": 15.00}

# Raw provider id -> canonical rates. Only ids that identify a pricing model on
# their own belong here; "kimi-for-coding" deliberately does not (it spans both
# k2.6 and k2.7-code, so only a date can separate those two).
_WIRE_MODEL_MAP = {"k3": K3_RATES}


def _label_for_rates(rates: dict) -> str:
    """Name the model a rate dict came from, for display only."""
    for label, known in (("kimi-k3", K3_RATES),
                         ("kimi-k2-7-code", K2_7_CODE_RATES),
                         ("kimi-k2-6", K2_6_RATES)):
        if rates is known:
            return label
    return "(custom rates)"


def _rates_for_session(events: list[dict], meta_events: list[dict]) -> dict:
    """Pick default rates based on the session's first event timestamp.

    Only reached for records with NO wire model string, so the k3 rung is in
    play here (see K3_CUTOFF_DT).
    """
    first_ts = None
    for coll in (events, meta_events):
        for e in coll:
            ts = e.get("ts")
            if ts:
                first_ts = ts
                break
        if first_ts is not None:
            break
    dt = parse_timestamp(first_ts) if first_ts else None
    if dt is None:
        return K2_7_CODE_RATES
    if dt < MODEL_CUTOFF_DT:
        return K2_6_RATES
    if dt < K3_CUTOFF_DT:
        return K2_7_CODE_RATES
    return K3_RATES


def _rates_for_model(model: str | None) -> dict:
    """Pick rates from a kimi-code usage.record model name.

    The wire wins when it names a model unambiguously. An id may arrive
    provider-prefixed ("kimi-code/k3"), so match on the last path segment too.
    """
    if not model:
        return K2_6_RATES
    tail = model.rsplit("/", 1)[-1]
    if tail in _WIRE_MODEL_MAP:
        return _WIRE_MODEL_MAP[tail]
    if "kimi-k3" in model:
        return K3_RATES
    if "kimi-k2-7-code" in model:
        return K2_7_CODE_RATES
    return K2_6_RATES


def parse_timestamp(ts: float | str) -> datetime | None:
    """Parse Unix timestamp or ISO string."""
    if not ts:
        return None
    if isinstance(ts, (int, float)):
        return datetime.fromtimestamp(ts, tz=timezone.utc)
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        try:
            return datetime.fromtimestamp(float(ts), tz=timezone.utc)
        except (ValueError, TypeError):
            return None


def short_time(ts: float | str) -> str:
    """Format timestamp as 'YYYY-MM-DD HH:MM:SS' (UTC)."""
    dt = parse_timestamp(ts)
    if not dt:
        return "????-??-?? ??:??:??"
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def _close_torn_prefix(prefix: str):
    """Best-effort close of a JSON object truncated mid-record.

    Walks `prefix` tracking open quotes, escapes, and brace/bracket nesting.
    Appends the minimal closing chars needed and retries json.loads. Returns
    the parsed object on success, None if the prefix is too damaged to
    round-trip. Marks the result with `_torn_truncated: True` so consumers
    know the record is partial. Mirrors parse_session.py's salvage logic.
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
    """Yield each decodable top-level JSON object in `line`, tolerating torn writes.

    On a parse failure, resyncs to the next `{"` boundary and tries
    `_close_torn_prefix` on the skipped chunk to salvage truncated records.
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


def _detect_format(lines: list[str]) -> str:
    """Return 'kimi-code' or 'legacy' by scanning the first few records."""
    kimi_code_envelope_types = {
        "context.append_message",
        "context.append_loop_event",
        "usage.record",
        "llm.error",
        "turn.prompt",
        "turn.steer",
    }
    legacy_message_types = {
        "StatusUpdate",
        "TurnBegin",
        "ToolCall",
        "ContentPart"}

    for raw in lines:
        if not raw or not raw.strip():
            continue
        raw = raw.rstrip("\n")
        if not raw:
            continue
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            objs = list(_iter_json_objects(raw))
            if not objs:
                continue
            obj = objs[0]
        if not isinstance(obj, dict):
            continue

        # Metadata alone disambiguates: kimi-code has created_at, legacy does
        # not.
        if obj.get("type") == "metadata":
            return "kimi-code" if "created_at" in obj else "legacy"

        if obj.get("type") in kimi_code_envelope_types:
            return "kimi-code"

        if "timestamp" in obj:
            msg = obj.get("message") or {}
            if msg.get("type") in legacy_message_types:
                return "legacy"

    return "legacy"


def _join_text(parts: list[dict]) -> str:
    """Concatenate text parts from a kimi-code ContentPart array."""
    out = []
    for p in parts:
        if not isinstance(p, dict):
            continue
        pt = p.get("type")
        if pt == "text":
            out.append(str(p.get("text", "")))
        elif pt == "think":
            out.append(str(p.get("think", "")))
    return "".join(out)


def _kc_parse_tool_call(tc: dict) -> tuple[str, str | None, str]:
    """Extract name, arguments, id from a kimi-code ToolCall (v1.0 or v1.1)."""
    if tc.get("type") != "function":
        return "", None, ""
    tcid = tc.get("id", "")
    if "name" in tc:
        # v1.1+ flat shape
        return str(tc.get("name", "")), tc.get("arguments"), tcid
    # v1.0 nested function shape
    func = tc.get("function") or {}
    return str(func.get("name", "")), func.get("arguments"), tcid


def _kc_args_to_input(args) -> dict:
    if args is None:
        return {}
    if isinstance(args, dict):
        return args
    if isinstance(args, str):
        try:
            return json.loads(args) if args else {}
        except json.JSONDecodeError:
            return {"_raw": args}
    return {"_raw": args}


def _kc_result_detail(result: object) -> str:
    """Render a kimi-code ExecutableToolResult to a string."""
    if not isinstance(result, dict):
        return str(result)
    output = result.get("output")
    if isinstance(output, list):
        return "\n".join(
            (x.get("text", "") if isinstance(x, dict) else str(x))
            for x in output
        )
    return str(output) if output is not None else ""


def _content_parts_to_events(
    parts: list[dict], line_num: int, ts: float | str, events: list[dict]
) -> None:
    for p in parts:
        if not isinstance(p, dict):
            continue
        pt = p.get("type")
        if pt == "text":
            events.append({
                "line": line_num, "type": "assistant_text", "ts": ts,
                "detail": str(p.get("text", "")),
            })
        elif pt == "think":
            events.append({
                "line": line_num, "type": "thinking", "ts": ts,
                "detail": str(p.get("think", "")),
            })
        elif pt in ("image_url", "audio_url", "video_url"):
            url_obj = p.get(pt)
            url = url_obj.get("url", "") if isinstance(url_obj, dict) else ""
            events.append({
                "line": line_num, "type": "media", "ts": ts,
                "detail": f"[{pt}] {url[:80]}...",
            })


def _extract_kimi_code_events(
        wire_path: str, lines: list[str]) -> tuple[list[dict], list[dict]]:
    """Parse a kimi-code wire.jsonl into the same event/meta shapes as legacy."""
    events: list[dict] = []
    meta_events: list[dict] = []

    current_turn_id = None
    current_turn_open = False
    ts = ""

    def _close_turn(line_num: int, ts: float | str) -> None:
        nonlocal current_turn_open
        if current_turn_open:
            meta_events.append({
                "line": line_num, "type": "turn_end", "ts": ts, "raw": {},
            })
            current_turn_open = False

    def _open_turn(line_num: int, ts: float | str, raw=None) -> None:
        nonlocal current_turn_open
        if not current_turn_open:
            meta_events.append({
                "line": line_num, "type": "turn_begin", "ts": ts,
                "raw": raw if raw is not None else {},
            })
            current_turn_open = True

    def _make_status_update(
        line_num: int, ts: float | str, usage: dict, model: str | None, raw: dict
    ) -> dict:
        return {
            "line": line_num,
            "type": "status_update",
            "ts": ts,
            "token_usage": {
                "input_other": int(usage.get("inputOther") or 0),
                "output": int(usage.get("output") or 0),
                "input_cache_read": int(usage.get("inputCacheRead") or 0),
                "input_cache_creation": int(usage.get("inputCacheCreation") or 0),
            },
            "model": model,
            "raw": raw,
        }

    for line_num, raw_line in enumerate(lines, 1):
        if not raw_line:
            continue
        line = raw_line[:-1] if raw_line and raw_line[-1] == "\n" else raw_line
        if not line:
            continue

        try:
            objs = [json.loads(line)]
        except json.JSONDecodeError:
            objs = list(_iter_json_objects(line))
            if not objs:
                events.append({
                    "line": line_num, "type": "parse_error", "ts": "",
                    "detail": f"Invalid JSON at line {line_num}",
                })
                continue

        for obj in objs:
            typ = obj.get("type")
            if typ == "metadata":
                ts_ms = obj.get("created_at")
                ts = ts_ms / 1000.0 if isinstance(ts_ms, (int, float)) else ""
                meta_events.append({
                    "line": line_num, "type": "metadata", "ts": ts,
                    "protocol_version": obj.get("protocol_version"),
                    "raw": obj,
                })
                continue

            ts_ms = obj.get("time")
            ts = ts_ms / 1000.0 if isinstance(ts_ms, (int, float)) else ""

            if typ in ("turn.prompt", "turn.steer"):
                text = _join_text(obj.get("input") or [])
                event_type = "steer_input" if typ == "turn.steer" else "user_message"
                events.append({
                    "line": line_num, "type": event_type, "ts": ts,
                    "detail": text,
                })
                continue

            if typ == "context.append_message":
                msg = obj.get("message") or {}
                role = msg.get("role")
                content = msg.get("content") or []

                if role == "user":
                    events.append({
                        "line": line_num, "type": "user_message", "ts": ts,
                        "detail": _join_text(content),
                    })
                elif role == "assistant":
                    _content_parts_to_events(content, line_num, ts, events)
                    for tc in msg.get("toolCalls") or []:
                        name, args, tcid = _kc_parse_tool_call(tc)
                        events.append({
                            "line": line_num, "type": "tool_call", "ts": ts,
                            "tool_name": name,
                            "tool_input": _kc_args_to_input(args),
                            "tool_call_id": tcid,
                            "detail": "",
                        })
                elif role == "tool":
                    events.append({
                        "line": line_num, "type": "tool_result", "ts": ts,
                        "tool_call_id": msg.get("toolCallId", ""),
                        "is_error": bool(msg.get("isError")),
                        "detail": _join_text(content),
                        "message": "",
                    })
                continue

            if typ == "context.append_loop_event":
                ev = obj.get("event") or {}
                et = ev.get("type")
                turn_id = ev.get("turnId")

                if et == "step.begin":
                    if turn_id != current_turn_id:
                        _close_turn(line_num, ts)
                        current_turn_id = turn_id
                        _open_turn(line_num, ts, raw=ev)
                    meta_events.append({
                        "line": line_num, "type": "step_begin", "ts": ts,
                        "n": ev.get("step", 0), "raw": ev,
                    })
                    continue

                if et == "step.end":
                    meta_events.append({
                        "line": line_num, "type": "step_end", "ts": ts,
                        "usage": ev.get("usage"),
                        "finish_reason": ev.get("finishReason"),
                        "raw": ev,
                    })
                    continue

                if et == "content.part":
                    part = ev.get("part") or {}
                    _content_parts_to_events([part], line_num, ts, events)
                    continue

                if et == "tool.call":
                    events.append({
                        "line": line_num, "type": "tool_call", "ts": ts,
                        "tool_name": str(ev.get("name", "")),
                        "tool_input": _kc_args_to_input(ev.get("args")),
                        "tool_call_id": str(ev.get("toolCallId", "")),
                        "detail": "",
                    })
                    continue

                if et == "tool.result":
                    events.append({
                        "line": line_num, "type": "tool_result", "ts": ts,
                        "tool_call_id": str(ev.get("toolCallId", "")),
                        "is_error": bool(
                            (ev.get("result") or {}).get("isError")
                            or (ev.get("result") or {}).get("is_error")
                        ),
                        "detail": _kc_result_detail(ev.get("result")),
                        "message": "",
                    })
                    continue

                # Other loop event types fall through to catch-all meta
                meta_events.append({
                    "line": line_num, "type": f"loop_{et}", "ts": ts, "raw": ev,
                })
                continue

            if typ == "usage.record":
                model = obj.get("model")
                usage = obj.get("usage") or {}
                meta_events.append(
                    _make_status_update(line_num, ts, usage, model, obj)
                )
                continue

            if typ == "llm.error":
                events.append({
                    "line": line_num,
                    "type": "api_error",
                    "ts": ts,
                    "kind": str(obj.get("kind") or "other"),
                    "retryable": bool(obj.get("retryable")),
                    "status_code": obj.get("statusCode"),
                    "error_name": str(obj.get("errorName") or ""),
                    "model": obj.get("model"),
                    "detail": str(obj.get("message") or ""),
                    "raw": obj,
                })
                continue

            # Catch-all for other kimi-code record types (goal.*, etc.)
            meta_events.append({
                "line": line_num, "type": str(typ).lower(), "ts": ts, "raw": obj,
            })

    _close_turn(len(lines), ts if ts != "" else "")

    for e in events:
        e["src"] = wire_path
    for m in meta_events:
        m["src"] = wire_path
    return events, meta_events


def extract_events(wire_path: str) -> tuple[list[dict], list[dict]]:
    """Parse wire.jsonl into structured events and meta events.

    Returns (events, meta_events).
    - events: conversational events (user/assistant/tool_call/tool_result/thinking)
    - meta_events: StatusUpdate, TurnBegin/TurnEnd, StepBegin, etc.
    """
    with _open_text(wire_path) as f:
        lines = list(f)

    fmt = _detect_format(lines)
    if fmt == "kimi-code":
        return _extract_kimi_code_events(wire_path, lines)

    events: list[dict] = []
    meta_events: list[dict] = []

    for line_num, line in enumerate(lines, 1):
        if line and line[-1] == "\n":
            line = line[:-1]
        if not line:
            continue
        # Skip metadata line
        if line.startswith('{"type": "metadata"'):
            continue

        try:
            objs = [json.loads(line)]
        except json.JSONDecodeError:
            objs = list(_iter_json_objects(line))
            if not objs:
                events.append({
                    "line": line_num, "type": "parse_error", "ts": "",
                    "detail": f"Invalid JSON at line {line_num}",
                })
                continue

        for obj in objs:
            ts = obj.get("timestamp", "")
            msg = obj.get("message", {})
            msg_type = msg.get("type", "")
            payload = msg.get("payload", {})

            if msg_type == "TurnBegin":
                user_input = payload.get("user_input", "")
                if isinstance(user_input, list):
                    user_input = " ".join(
                        p.get("text", "") for p in user_input if isinstance(p, dict)
                    )
                events.append({
                    "line": line_num, "type": "user_message", "ts": ts,
                    "detail": str(user_input),
                })
                meta_events.append({
                    "line": line_num, "type": "turn_begin", "ts": ts,
                    "raw": obj,
                })

            elif msg_type == "TurnEnd":
                meta_events.append({
                    "line": line_num, "type": "turn_end", "ts": ts,
                    "raw": obj,
                })

            elif msg_type == "StepBegin":
                meta_events.append({
                    "line": line_num, "type": "step_begin", "ts": ts,
                    "n": payload.get("n", 0), "raw": obj,
                })

            elif msg_type == "StepInterrupted":
                meta_events.append({
                    "line": line_num, "type": "step_interrupted", "ts": ts,
                    "raw": obj,
                })

            elif msg_type == "StepRetry":
                meta_events.append({
                    "line": line_num, "type": "step_retry", "ts": ts,
                    "n": payload.get("n", 0),
                    "next_attempt": payload.get("next_attempt", 0),
                    "max_attempts": payload.get("max_attempts", 0),
                    "wait_s": payload.get("wait_s", 0),
                    "error_type": payload.get("error_type", ""),
                    "raw": obj,
                })

            elif msg_type == "CompactionBegin":
                meta_events.append({
                    "line": line_num, "type": "compaction_begin", "ts": ts,
                    "raw": obj,
                })
            elif msg_type == "CompactionEnd":
                meta_events.append({
                    "line": line_num, "type": "compaction_end", "ts": ts,
                    "raw": obj,
                })

            elif msg_type == "StatusUpdate":
                meta_events.append({
                    "line": line_num, "type": "status_update", "ts": ts,
                    "context_usage": payload.get("context_usage"),
                    "context_tokens": payload.get("context_tokens"),
                    "max_context_tokens": payload.get("max_context_tokens"),
                    "token_usage": payload.get("token_usage"),
                    "message_id": payload.get("message_id"),
                    "plan_mode": payload.get("plan_mode"),
                    "raw": obj,
                })

            elif msg_type == "ContentPart":
                part_type = payload.get("type", "")
                if part_type == "text":
                    events.append({
                        "line": line_num, "type": "assistant_text", "ts": ts,
                        "detail": payload.get("text", ""),
                    })
                elif part_type == "think":
                    events.append({
                        "line": line_num, "type": "thinking", "ts": ts,
                        "detail": payload.get("think", ""),
                    })
                elif part_type in ("image_url", "audio_url", "video_url"):
                    events.append({
                        "line": line_num, "type": "media", "ts": ts,
                        "detail": f"[{part_type}] {payload.get(part_type, {}).get('url', '')[:80]}...",
                    })

            elif msg_type == "ToolCall":
                func = payload.get("function", {})
                tool_name = func.get("name", "")
                arguments = func.get("arguments", "")
                try:
                    tool_input = json.loads(arguments) if arguments else {}
                except json.JSONDecodeError:
                    tool_input = {"_raw": arguments}
                events.append({
                    "line": line_num, "type": "tool_call", "ts": ts,
                    "tool_name": tool_name,
                    "tool_input": tool_input,
                    "tool_call_id": payload.get("id", ""),
                    "detail": "",
                })

            elif msg_type == "ToolCallPart":
                # Streaming argument fragment — skip for now or append to last ToolCall
                pass

            elif msg_type == "ToolResult":
                return_value = payload.get("return_value", {})
                detail = ""
                if isinstance(return_value, dict):
                    output = return_value.get("output", "")
                    if isinstance(output, list):
                        detail = "\n".join(
                            (x.get("text", "") if isinstance(x, dict) else str(x))
                            for x in output
                        )
                    else:
                        detail = str(output)
                events.append({
                    "line": line_num, "type": "tool_result", "ts": ts,
                    "tool_call_id": payload.get("tool_call_id", ""),
                    "is_error": return_value.get("is_error", False) if isinstance(return_value, dict) else False,
                    "detail": detail,
                    "message": return_value.get("message", "") if isinstance(return_value, dict) else "",
                })

            elif msg_type == "ApprovalRequest":
                events.append({
                    "line": line_num, "type": "approval_request", "ts": ts,
                    "tool_call_id": payload.get("tool_call_id", ""),
                    "sender": payload.get("sender", ""),
                    "action": payload.get("action", ""),
                    "description": payload.get("description", ""),
                    "detail": f"Approval: {payload.get('action', '')} — {payload.get('description', '')}",
                })

            elif msg_type == "ApprovalResponse":
                events.append({
                    "line": line_num, "type": "approval_response", "ts": ts,
                    "request_id": payload.get("request_id", ""),
                    "response": payload.get("response", ""),
                    "detail": f"ApprovalResponse: {payload.get('response', '')}",
                })

            elif msg_type == "SubagentEvent":
                events.append({
                    "line": line_num, "type": "subagent_event", "ts": ts,
                    "parent_tool_call_id": payload.get("parent_tool_call_id", ""),
                    "agent_id": payload.get("agent_id", ""),
                    "subagent_type": payload.get("subagent_type", ""),
                    "event": payload.get("event", {}),
                    "detail": f"Subagent({payload.get('subagent_type', '?')}, {payload.get('agent_id', '?')})",
                })

            elif msg_type == "BtwBegin":
                meta_events.append({
                    "line": line_num, "type": "btw_begin", "ts": ts,
                    "id": payload.get("id", ""),
                    "question": payload.get("question", ""),
                    "raw": obj,
                })
            elif msg_type == "BtwEnd":
                meta_events.append({
                    "line": line_num, "type": "btw_end", "ts": ts,
                    "id": payload.get("id", ""),
                    "response": payload.get("response"),
                    "error": payload.get("error"),
                    "raw": obj,
                })

            elif msg_type == "SteerInput":
                user_input = payload.get("user_input", "")
                if isinstance(user_input, list):
                    user_input = " ".join(
                        p.get("text", "") for p in user_input if isinstance(p, dict)
                    )
                events.append({
                    "line": line_num, "type": "steer_input", "ts": ts,
                    "detail": str(user_input),
                })

            elif msg_type == "PlanDisplay":
                events.append({
                    "line": line_num, "type": "plan_display", "ts": ts,
                    "file_path": payload.get("file_path", ""),
                    "detail": payload.get("content", ""),
                })

            elif msg_type in ("HookTriggered", "HookResolved"):
                meta_events.append({
                    "line": line_num, "type": msg_type.lower(), "ts": ts,
                    "event_type": payload.get("event", ""),
                    "target": payload.get("target", ""),
                    "raw": obj,
                })

            else:
                # Catch-all for unhandled wire message types
                meta_events.append({
                    "line": line_num, "type": msg_type.lower() or "unknown", "ts": ts,
                    "raw": obj,
                })

    for e in events:
        e["src"] = wire_path
    for m in meta_events:
        m["src"] = wire_path
    return events, meta_events


def annotate_parallel(events: list[dict]) -> None:
    """Mark tool calls dispatched in the same parallel batch.

    Parallel calls detected by timestamp proximity: consecutive tool calls
    with <2s between them form a batch.
    """
    tool_events = [e for e in events if e["type"] == "tool_call"]
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
    """Link tool calls to their results via tool_call_id."""
    call_map: dict[str, dict] = {}
    for e in events:
        if e["type"] == "tool_call" and e.get("tool_call_id"):
            call_map[e["tool_call_id"]] = e
    for e in events:
        if e["type"] == "tool_result" and e.get("tool_call_id"):
            call = call_map.get(e["tool_call_id"])
            if call:
                e["paired_call"] = call
                call["paired_result"] = e


def _tool_call_detail(event: dict) -> str:
    """Lazy renderer for the display string of a tool_call event."""
    cached = event.get("detail")
    if cached:
        return cached
    name = event.get("tool_name", "")
    inp = event.get("tool_input", {})
    if name == "Shell":
        rendered = f"Shell({inp.get('command', '')[:200]})"
    elif name in ("ReadFile", "Read"):
        rendered = f"Read({inp.get('path', '')})"
    elif name in ("WriteFile", "Write"):
        rendered = f"Write({inp.get('path', '')})"
    elif name in ("StrReplaceFile", "Edit"):
        rendered = f"Edit({inp.get('path', '')})"
    elif name == "Grep":
        rendered = f"Grep({inp.get('pattern', '')} in {inp.get('path', '(cwd)')})"
    elif name == "Glob":
        rendered = f"Glob({inp.get('pattern', '')} in {inp.get('path', '(cwd)')})"
    elif name == "Agent":
        rendered = f"Agent({inp.get('description', '?')})"
    else:
        rendered = f"{name}({json.dumps(inp)[:200]})"
    event["detail"] = rendered
    return rendered


def format_event(event: dict, verbose: bool = False,
                 with_source: bool = False) -> str:
    """Format a single event for display."""
    line = event["line"]
    time = short_time(event["ts"])
    etype = event["type"]
    detail = event["detail"] if etype != "tool_call" else _tool_call_detail(
        event)

    batch_tag = ""
    batch_size = event.get("batch_size", 1)
    if batch_size > 1:
        batch_tag = f" [parallel {event['batch_index']}/{batch_size}]"

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
    elif etype == "thinking":
        return f"L{line:<4} {src_tag}[{time}] THINKING: {detail}"
    elif etype == "tool_call":
        return f"L{line:<4} {src_tag}[{time}] TOOL{batch_tag}: {detail}"
    elif etype == "tool_result":
        err = " [ERROR]" if event.get("is_error") else ""
        return f"L{line:<4} {src_tag}[{time}] RESULT{err}: {detail[:200]}{'...' if len(detail) > 200 else ''}"
    elif etype == "api_error":
        kind = event.get("kind") or "other"
        status = event.get("status_code")
        status_tag = f" HTTP {status}" if status is not None else ""
        retry = " retryable" if event.get("retryable") else ""
        return (f"L{line:<4} {src_tag}[{time}] API ERROR "
                f"[{kind}{status_tag}{retry}]: {detail}")
    elif etype == "approval_request":
        return f"L{line:<4} {src_tag}[{time}] APPROVAL: {detail}"
    elif etype == "approval_response":
        return f"L{line:<4} {src_tag}[{time}] {detail}"
    elif etype == "subagent_event":
        return f"L{line:<4} {src_tag}[{time}] SUBAGENT: {detail}"
    elif etype == "steer_input":
        return f"L{line:<4} {src_tag}[{time}] STEER: {detail}"
    elif etype == "plan_display":
        return f"L{line:<4} {src_tag}[{time}] PLAN: {event.get('file_path', '')}"
    elif etype == "media":
        return f"L{line:<4} {src_tag}[{time}] MEDIA: {detail}"
    elif etype == "parse_error":
        return f"L{line:<4} {src_tag}[ERROR] {detail}"
    else:
        return f"L{line:<4} {src_tag}[{time}] {etype}: {detail}"


def format_pair(call: dict, result: dict | None,
                verbose: bool = False, with_source: bool = False) -> str:
    """Format a tool call + result pair."""
    lines = [format_event(call, verbose=verbose, with_source=with_source)]
    if result:
        detail = result["detail"]
        err = " [ERROR]" if result.get("is_error") else ""
        if verbose:
            lines.append(
                f"  └─ RESULT{err} (L{result['line']}, {len(detail)} chars):\n{detail}")
        else:
            preview = detail[:300] + ("..." if len(detail) > 300 else "")
            lines.append(
                f"  └─ RESULT{err} (L{result['line']}, {len(detail)} chars): {preview}")
    else:
        lines.append("  └─ RESULT: (not found)")
    return "\n".join(lines)


# --- Analysis modes ---

def compute_stats(events: list[dict],
                  meta_events: list[dict] | None = None) -> str:
    """Compute session statistics."""
    meta_events = meta_events or []
    lines = []
    tool_counts = Counter()
    user_messages = 0
    assistant_messages = 0
    thinking_blocks = 0
    tool_results = 0
    error_results = 0
    api_errors = 0
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
        elif e["type"] == "user_message":
            user_messages += 1
        elif e["type"] == "assistant_text":
            assistant_messages += 1
        elif e["type"] == "thinking":
            thinking_blocks += 1
        elif e["type"] == "tool_result":
            tool_results += 1
            if e.get("is_error"):
                error_results += 1
        elif e["type"] == "api_error":
            api_errors += 1

        if e.get("batch_size", 1) > 1 and e.get("batch_index") == 1:
            parallel_batches += 1
            parallel_calls += e["batch_size"]

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
        lines.append(
            f"Start:      {first_ts.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    if last_ts:
        lines.append(
            f"End:        {last_ts.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    if duration:
        lines.append(f"Duration:   {duration}  (wall)")
    lines.append(f"Events:     {len(events)}")
    lines.append(f"User msgs:  {user_messages}")
    lines.append(f"Asst msgs:  {assistant_messages}")
    lines.append(f"Thinking:   {thinking_blocks}")
    lines.append(f"Tool calls: {sum(tool_counts.values())}")
    lines.append(f"Results:    {tool_results} ({error_results} errors)")
    lines.append(f"API errors: {api_errors}")
    if parallel_batches:
        lines.append(
            f"Parallel:   {parallel_batches} batches ({parallel_calls} calls)")
    lines.append("")

    if tool_counts:
        lines.append("TOOL USAGE:")
        for tool, count in tool_counts.most_common():
            lines.append(f"  {tool:<20} {count:>4}")
        lines.append("")

    # Count turns from meta_events
    turn_count = sum(1 for m in meta_events if m["type"] == "turn_begin")
    step_count = sum(1 for m in meta_events if m["type"] == "step_begin")
    compaction_count = sum(
        1 for m in meta_events if m["type"] == "compaction_begin")
    lines.append(f"Turns:      {turn_count}")
    lines.append(f"Steps:      {step_count}")
    lines.append(f"Compactions:{compaction_count}")

    return "\n".join(lines)


def compute_files(events: list[dict]) -> str:
    """Summarize all files touched in the session."""
    files: dict[str, dict[str, list[int]]] = defaultdict(
        lambda: defaultdict(list))

    for e in events:
        if e["type"] != "tool_call":
            continue
        name = e.get("tool_name", "")
        inp = e.get("tool_input", {})
        line = e["line"]

        if name in ("ReadFile", "Read"):
            files[inp.get("path", "?")]["read"].append(line)
        elif name in ("WriteFile", "Write"):
            files[inp.get("path", "?")]["write"].append(line)
        elif name in ("StrReplaceFile", "Edit"):
            files[inp.get("path", "?")]["edit"].append(line)
        elif name == "Grep":
            path = inp.get("path", "(cwd)")
            files[path]["grep"].append(line)
        elif name == "Glob":
            path = inp.get("path", "(cwd)")
            pattern = inp.get("pattern", "")
            files[f"{path} [{pattern}]"]["glob"].append(line)

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
            line_refs += f"…+{len(all_lines) - 5}"
        out.append(f"  [{ops_str:>20}]  {path}  ({line_refs})")
    out.append(f"\nTotal: {len(files)} files/paths")
    return "\n".join(out)


def compute_commands(events: list[dict]) -> str:
    """Extract all Shell commands."""
    lines = ["SHELL COMMANDS", "=" * 40]
    for e in events:
        if e["type"] == "tool_call" and e.get("tool_name") == "Shell":
            cmd = e["tool_input"].get("command", "")
            desc = e["tool_input"].get("description", "")
            bg = " [bg]" if e["tool_input"].get("run_in_background") else ""
            batch_tag = ""
            if e.get("batch_size", 1) > 1:
                batch_tag = f" [parallel {e['batch_index']}/{e['batch_size']}]"
            header = f"L{e['line']:<4} [{short_time(e['ts'])}]{batch_tag}{bg}"
            if desc:
                header += f" — {desc}"
            lines.append(header)
            lines.append(f"  $ {cmd}")
            lines.append("")
    return "\n".join(lines)


def compute_edits(events: list[dict], verbose: bool = False) -> str:
    """Summarize all StrReplaceFile calls."""
    lines = ["EDITS", "=" * 40]
    per_file: dict[str, dict[str, int]] = defaultdict(
        lambda: {"count": 0, "added": 0, "removed": 0})
    total_count = 0

    for e in events:
        if not (e["type"] == "tool_call" and e.get(
                "tool_name") in ("StrReplaceFile", "Edit")):
            continue
        inp = e.get("tool_input", {})
        path = inp.get("path", "?")
        edits = inp.get("edit", [])
        if not isinstance(edits, list):
            edits = [edits] if edits else []

        count = len(edits)
        total_count += count
        per_file[path]["count"] += count

        for edit in edits:
            if not isinstance(edit, dict):
                continue
            old_text = edit.get("old", "")
            new_text = edit.get("new", "")
            # Rough line count estimate
            old_lines = old_text.count("\n") + 1 if old_text else 0
            new_lines = new_text.count("\n") + 1 if new_text else 0
            per_file[path]["removed"] += old_lines
            per_file[path]["added"] += new_lines

            if verbose:
                old_preview = old_text
                new_preview = new_text
            else:
                old_preview = old_text[:100] + \
                    ("..." if len(old_text) > 100 else "")
                new_preview = new_text[:100] + \
                    ("..." if len(new_text) > 100 else "")
            lines.append(f"L{e['line']:<4} [{short_time(e['ts'])}] {path}")
            lines.append(f"  - {old_preview!r}")
            lines.append(f"  + {new_preview!r}")
            lines.append("")

    if total_count == 0:
        lines.append("(no edits found)")
        return "\n".join(lines)

    lines.append("PER-FILE SUMMARY")
    lines.append("-" * 40)
    lines.append(f"  {'edits':>5}  {'+lines':>7}  {'-lines':>7}  path")
    rows = sorted(per_file.items(
    ), key=lambda kv: (-kv[1]["count"], -(kv[1]["added"] + kv[1]["removed"])))
    for path, stat in rows:
        lines.append(
            f"  {stat['count']:>5}  {stat['added']:>7}  {stat['removed']:>7}  {path}")
    lines.append("")
    lines.append(f"TOTAL: {total_count} edits across {len(per_file)} files")
    return "\n".join(lines)


def compute_writes(events: list[dict], verbose: bool = False) -> str:
    """Summarize all WriteFile calls."""
    lines = ["WRITES", "=" * 40]
    count = 0
    for e in events:
        if e["type"] == "tool_call" and e.get(
                "tool_name") in ("WriteFile", "Write"):
            inp = e.get("tool_input", {})
            path = inp.get("path", "?")
            content = inp.get("content", "")
            count += 1
            lines.append(
                f"L{e['line']:<4} [{short_time(e['ts'])}] {path} ({len(content)} chars)")
            if verbose:
                lines.append(content)
            else:
                preview = content[:200] + ("..." if len(content) > 200 else "")
                lines.append(f"  {preview}")
            lines.append("")
    lines.append(f"Total: {count} files written")
    return "\n".join(lines)


def compute_cache(meta_events: list[dict], rates: dict | None = None,
                  events: list[dict] | None = None) -> str:
    """Compute token usage from StatusUpdate / usage.record events. Optionally estimate billing."""
    # Default rates depend on the session's first event timestamp.
    # Override with --rates "fresh=0.95,cache_read=0.16,output=4.00"
    session_default_rates = _rates_for_session(events or [], meta_events)
    override_rates = rates or {}

    lines = ["CACHE / TOKEN USAGE", "=" * 40]
    total_input_other = 0
    total_output = 0
    total_cache_read = 0
    total_cache_create = 0
    count = 0
    total_cost = 0.0
    detected_model: str | None = None
    detected_rates: dict | None = None
    # Accumulated per-record so the category breakdown reconciles with TOTAL
    # even when a session switches model mid-flight (observed: kimi-for-coding
    # -> k3, 24s apart). Re-deriving these from summed tokens against one
    # session-wide rate silently drifts from the per-record total.
    cost_by_cat = {"fresh": 0.0, "cache_read": 0.0, "cache_create": 0.0, "output": 0.0}

    for m in meta_events:
        if m["type"] != "status_update":
            continue
        tu = m.get("token_usage")
        if not tu:
            continue
        count += 1
        io = tu.get("input_other", 0) or 0
        out = tu.get("output", 0) or 0
        cr = tu.get("input_cache_read", 0) or 0
        cc = tu.get("input_cache_creation", 0) or 0
        total_input_other += io
        total_output += out
        total_cache_read += cr
        total_cache_create += cc

        # kimi-code usage.record carries its own model; legacy falls back to
        # session timestamp heuristic, or --rates if provided.
        if override_rates:
            rec_rates = override_rates
        elif m.get("model"):
            rec_rates = _rates_for_model(m["model"])
            if detected_model is None:
                detected_model, detected_rates = m["model"], rec_rates
        else:
            rec_rates = session_default_rates
            if detected_model is None:
                detected_model, detected_rates = _label_for_rates(rec_rates), rec_rates

        cost_by_cat["fresh"] += io * rec_rates["fresh"] / 1_000_000
        cost_by_cat["cache_read"] += cr * rec_rates["cache_read"] / 1_000_000
        cost_by_cat["cache_create"] += cc * rec_rates["cache_create"] / 1_000_000
        cost_by_cat["output"] += out * rec_rates["output"] / 1_000_000

    total_cost = sum(cost_by_cat.values())

    if count == 0:
        lines.append("(no StatusUpdate with token_usage found)")
        return "\n".join(lines)

    total_in = total_input_other + total_cache_read + total_cache_create
    hit = (total_cache_read / total_in * 100.0) if total_in else 0.0

    lines.append(f"StatusUpdates: {count}")
    lines.append(f"  input total:     {total_in:>12,}")
    lines.append(f"    fresh:         {total_input_other:>12,}")
    if total_cache_create:
        lines.append(f"    cache_create:  {total_cache_create:>12,}")
    lines.append(f"    cache_read:    {total_cache_read:>12,}")
    lines.append(f"  output:          {total_output:>12,}")
    lines.append(f"  hit rate:        {hit:5.1f}%")

    # Billing estimate. Categories come straight from the per-record tally
    # above, so they always sum to TOTAL.
    fresh_cost = cost_by_cat["fresh"]
    cache_read_cost = cost_by_cat["cache_read"]
    cache_create_cost = cost_by_cat["cache_create"]
    output_cost = cost_by_cat["output"]
    display_model = detected_model or _label_for_rates(session_default_rates)
    display_rates = override_rates or detected_rates or session_default_rates
    lines.append("")
    lines.append("ESTIMATED BILLING")
    lines.append(f"  Detected model: {display_model}")
    lines.append(f"  Rates (USD per 1M tokens): fresh=${display_rates['fresh']:.2f}  "
                 f"cache_read=${display_rates['cache_read']:.2f}  "
                 f"cache_create=${display_rates['cache_create']:.2f}  "
                 f"output=${display_rates['output']:.2f}")
    lines.append(f"  {'Category':<20} {'Cost':>10}")
    lines.append(f"  {'-' * 20} {'-' * 10}")
    if fresh_cost:
        lines.append(f"  {'Fresh input':<20} ${fresh_cost:>9.2f}")
    if cache_read_cost:
        lines.append(f"  {'Cache read':<20} ${cache_read_cost:>9.2f}")
    if cache_create_cost:
        lines.append(f"  {'Cache create':<20} ${cache_create_cost:>9.2f}")
    if output_cost:
        lines.append(f"  {'Output':<20} ${output_cost:>9.2f}")
    lines.append(f"  {'-' * 20} {'-' * 10}")
    lines.append(f"  {'TOTAL':<20} ${total_cost:>9.2f}")
    if count:
        lines.append(
            f"  {'PER TURN':<20} ${total_cost / count:>9.4f}  ({count} turns)")

    return "\n".join(lines)


def compute_context_growth(meta_events: list[dict]) -> str:
    """Context size over time from StatusUpdate events."""
    lines = ["CONTEXT GROWTH", "=" * 40]
    usage_records: list[dict] = []

    for m in meta_events:
        if m["type"] != "status_update":
            continue
        tu = m.get("token_usage")
        if not tu:
            continue
        input_other = tu.get("input_other", 0) or 0
        cache_read = tu.get("input_cache_read", 0) or 0
        cache_create = tu.get("input_cache_creation", 0) or 0
        output = tu.get("output", 0) or 0
        total_input = input_other + cache_read + cache_create
        usage_records.append({
            "ts": m["ts"],
            "line": m["line"],
            "input": total_input,
            "output": output,
        })

    if not usage_records:
        lines.append("(no StatusUpdate with token_usage found)")
        return "\n".join(lines)

    # Dedupe by keeping only the last record per line (some files have
    # duplicate StatusUpdates)
    seen_lines: set[int] = set()
    deduped: list[dict] = []
    for rec in reversed(usage_records):
        if rec["line"] not in seen_lines:
            seen_lines.add(rec["line"])
            deduped.append(rec)
    deduped.reverse()

    lines.append(
        f"  {'#':>4}  {'time':<19}  {'L#':>5}  {'input':>10}  {'output':>8}  {'delta':>10}"
    )
    prev_input = 0
    for idx, t in enumerate(deduped, 1):
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
        f"Total: {len(deduped)} snapshots, final context: {deduped[-1]['input']:,} input tokens")
    return "\n".join(lines)


def _tool_call_excerpt(call: dict, limit: int = 80) -> str:
    """One-line excerpt for a tool call."""
    inp = call.get("tool_input", {}) or {}
    name = call.get("tool_name", "")
    if name == "Shell":
        text = inp.get("command", "")
    elif name in ("ReadFile", "Read", "WriteFile", "Write", "StrReplaceFile", "Edit"):
        text = inp.get("path", "")
    elif name == "Grep":
        text = f"{inp.get('pattern', '')}  in {inp.get('path', '(cwd)')}"
    elif name == "Glob":
        text = f"{inp.get('pattern', '')}  in {inp.get('path', '(cwd)')}"
    elif name == "FetchURL":
        text = inp.get("url", "")
    elif name == "Agent":
        text = inp.get("description", "") or inp.get("prompt", "")[:80]
    else:
        text = json.dumps(inp)[:200]
    text = text.replace("\n", " ")
    if len(text) > limit:
        text = text[:limit] + "..."
    return text


def compute_latency(events: list[dict]) -> str:
    """Rank tool calls by wall-clock latency (result_ts - call_ts)."""
    lines = ["TOOL LATENCY", "=" * 40]

    buckets_spec = [
        ("[0, 100ms)", 0.0, 0.1),
        ("[100ms, 1s)", 0.1, 1.0),
        ("[1s, 5s)", 1.0, 5.0),
        ("[5s, 30s)", 5.0, 30.0),
        ("[30s, 2m)", 30.0, 120.0),
        ("[2m, inf)", 120.0, float("inf")),
    ]

    def _bucket_label(value: float) -> str | None:
        for label, lo, hi in buckets_spec:
            if lo <= value < hi:
                return label
        return None

    per_tool_hist: dict[str, Counter] = defaultdict(Counter)
    per_tool_totals: dict[str, dict[str, float]] = defaultdict(
        lambda: {"count": 0, "total": 0.0, "max": 0.0}
    )
    all_pairs: list[tuple[float, dict, dict]] = []

    for e in events:
        if e["type"] != "tool_call":
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
        name = e.get("tool_name", "?")
        per_tool_totals[name]["count"] += 1
        per_tool_totals[name]["total"] += dur
        if dur > per_tool_totals[name]["max"]:
            per_tool_totals[name]["max"] = dur
        lbl = _bucket_label(dur)
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
    for name in sorted(per_tool_totals.keys(),
                       key=lambda n: -per_tool_totals[n]["total"]):
        t = per_tool_totals[name]
        avg = t["total"] / t["count"] if t["count"] else 0.0
        hist = per_tool_hist[name]
        buckets = "  ".join(
            f"{hist.get(lbl, 0):>13}" for lbl, _, _ in buckets_spec)
        lines.append(
            f"  {name:<18}  {int(t['count']):>5}  "
            f"{avg:>7.2f}s  {t['max']:>7.2f}s   {buckets}"
        )
    lines.append("")

    lines.append("SLOWEST 20 CALLS:")
    lines.append(
        f"  {'time':<19}  {'L#':>5}  {'dur':>9}  {'tool':<14}  detail")
    for dur, call, result in sorted(all_pairs, key=lambda x: -x[0])[:20]:
        name = call.get("tool_name", "?")
        lines.append(
            f"  {short_time(call['ts']):<19}  "
            f"L{call['line']:<4}  "
            f"{dur:>8.2f}s  {name:<14}  {_tool_call_excerpt(call)}"
        )

    return "\n".join(lines)


def compute_reply_latency(events: list[dict], top_n: int = 20) -> str:
    """Wall-clock gap between each user message and the assistant's first response event.

    A SteerInput (mid-turn user intervention) is treated as a terminator only
    — it stops the clock on the preceding user_message but does NOT anchor a
    new latency measurement, mirroring how parse_session.py handles
    `[Request interrupted by user]`.
    """
    lines = ["REPLY LATENCY", "=" * 40]
    lines.append(
        "(gap from each user message to the assistant's first response event)")
    lines.append("")

    assistant_side_types = {"assistant_text", "thinking", "tool_call"}
    pairs: list[tuple[float, dict]] = []
    n = len(events)
    for i, e in enumerate(events):
        if e["type"] != "user_message":
            continue
        t_user = parse_timestamp(e["ts"])
        if not t_user:
            continue
        found = None
        superseded = False
        for j in range(i + 1, n):
            nxt = events[j]
            if nxt["type"] in assistant_side_types:
                found = nxt
                break
            if nxt["type"] == "steer_input":
                # Mid-turn user steer terminates the latency window the same
                # way an `[Request interrupted by user]` does in Claude.
                found = nxt
                break
            if nxt["type"] == "user_message":
                superseded = True
                break
        if superseded or not found:
            continue
        t_reply = parse_timestamp(found["ts"])
        if not t_reply:
            continue
        dur = (t_reply - t_user).total_seconds()
        if dur < 0:
            continue
        pairs.append((dur, e))

    if not pairs:
        lines.append("(no user messages with following assistant events)")
        return "\n".join(lines)

    durations = sorted(d for d, _ in pairs)
    count = len(durations)
    total = sum(durations)
    mean = total / count
    mn = durations[0]
    mx = durations[-1]
    median = durations[count // 2] if count % 2 == 1 else (
        durations[count // 2 - 1] + durations[count // 2]) / 2.0
    p95_idx = max(0, min(count - 1, int(round(0.95 * (count - 1)))))
    p95 = durations[p95_idx]

    lines.append(
        f"count: {count}   min: {mn:.2f}s   max: {mx:.2f}s   "
        f"median: {median:.2f}s   p95: {p95:.2f}s   mean: {mean:.2f}s"
    )
    lines.append("")

    buckets_spec = [
        ("[0, 0.5s)", 0.0, 0.5),
        ("[0.5s, 1s)", 0.5, 1.0),
        ("[1s, 2s)", 1.0, 2.0),
        ("[2s, 5s)", 2.0, 5.0),
        ("[5s, 10s)", 5.0, 10.0),
        ("[10s, 30s)", 10.0, 30.0),
        ("[30s, 2m)", 30.0, 120.0),
        ("[2m, inf)", 120.0, float("inf")),
    ]

    def _bucket_label(value: float) -> str | None:
        for label, lo, hi in buckets_spec:
            if lo <= value < hi:
                return label
        return None

    hist: Counter = Counter()
    for d in durations:
        lbl = _bucket_label(d)
        if lbl:
            hist[lbl] += 1
    lines.append("HISTOGRAM:")
    for label, _, _ in buckets_spec:
        c = hist.get(label, 0)
        bar = "#" * min(c, 60)
        lines.append(f"  {label:<13}  {c:>5}  {bar}")
    lines.append("")

    lines.append(f"SLOWEST {top_n}:")
    lines.append(f"  {'time':<19}  {'L#':>5}  {'latency':>9}  preview")
    slowest = sorted(pairs, key=lambda x: -x[0])[:top_n]
    for dur, ev in slowest:
        preview = (ev.get("detail", "") or "").replace("\n", " ").strip()
        if len(preview) > 60:
            preview = preview[:57] + "..."
        lines.append(
            f"  {short_time(ev['ts']):<19}  "
            f"L{ev['line']:<4}  "
            f"{dur:>8.2f}s  {preview}"
        )

    return "\n".join(lines)


def compute_shell_errors(events: list[dict]) -> str:
    """Shell calls with stderr, interrupted, or non-zero exit."""
    lines = ["SHELL ERRORS", "=" * 40]
    found = 0
    for e in events:
        if e["type"] != "tool_call" or e.get("tool_name") != "Shell":
            continue
        result = e.get("paired_result") or {}
        detail = result.get("detail", "") or ""
        is_error = bool(result.get("is_error"))
        # Kimi Shell results: check for error indicators in detail
        exit_match = re.match(r"Exit code (\d+)", detail)
        exit_code = int(exit_match.group(1)) if exit_match else None
        nonzero_exit = exit_code is not None and exit_code != 0
        has_error_text = bool(
            re.search(
                r"error|Error|ERROR|Traceback|FAILED|exception|Exception",
                detail))

        if not (nonzero_exit or is_error or has_error_text):
            continue
        found += 1
        inp = e.get("tool_input") or {}
        cmd_excerpt = (inp.get("command") or "").replace("\n", " ")[:120]
        status_parts = []
        if exit_code is not None:
            status_parts.append(f"exit={exit_code}")
        if is_error and exit_code is None:
            status_parts.append("is_error")
        status = " ".join(status_parts) or "error"
        lines.append(f"L{e['line']:<4} [{short_time(e['ts'])}]  {status}")
        lines.append(f"    $ {cmd_excerpt}")
        if detail:
            detail_preview = detail.replace("\n", " ")[:200]
            lines.append(f"    out: {detail_preview}")
    if not found:
        lines.append("(no shell errors found)")
    else:
        lines.append("")
        lines.append(f"Total: {found} shell errors")
    return "\n".join(lines)


def compute_tool_orphans(events: list[dict]) -> str:
    """Find tool calls without a matching tool_result."""
    lines = ["TOOL ORPHANS", "=" * 40]
    orphans = []
    for e in events:
        if e["type"] != "tool_call":
            continue
        if e.get("paired_result") is None:
            orphans.append(e)

    if not orphans:
        lines.append("(no orphaned tool calls — all have matching results)")
        lines.append("")
        lines.append("0 orphaned calls")
        return "\n".join(lines)

    for e in orphans:
        name = e.get("tool_name", "?")
        excerpt = _tool_call_excerpt(e, limit=80)
        lines.append(
            f"L{e['line']:<4} [{short_time(e['ts'])}] {name:<14}  {excerpt}")
    lines.append("")
    lines.append(
        f"{len(orphans)} orphaned call{'s' if len(orphans) != 1 else ''}")
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
        "approval_request": "APPROVE",
        "approval_response": "APPR-RES",
        "subagent_event": "SUBAGENT",
        "steer_input": "STEER",
        "plan_display": "PLAN",
        "media": "MEDIA",
        "parse_error": "ERROR",
    }

    preview_limit = 200 if verbose else 80
    for e in events:
        label = type_labels.get(e["type"], e["type"].upper())
        if e["type"] == "tool_call":
            name = e.get("tool_name", "?")
            label = f"TOOL:{name}"
        elif e["type"] == "tool_result":
            call = e.get("paired_call") or {}
            name = call.get("tool_name", "?")
            err = "[ERR]" if e.get("is_error") else ""
            label = f"RESULT:{name}{err}"

        if e["type"] == "tool_call":
            detail = _tool_call_detail(e).replace("\n", " ")
        else:
            detail = (e.get("detail") or "").replace("\n", " ")
        if len(detail) > preview_limit:
            detail = detail[:preview_limit] + "..."

        lines.append(
            f"L{e['line']:<4} [{short_time(e['ts'])}] {label:<20} {detail}")

    lines.append("")
    lines.append(f"{len(events)} events")
    return "\n".join(lines)


def compute_turns(events: list[dict], meta_events: list[dict]) -> str:
    # Every subscript of `current_turn` below sits behind an explicit
    # `is not None` on the same condition. pylint infers the name over the
    # whole function -- where it is also assigned None -- so it reports each
    # one anyway.
    # pylint: disable=unsupported-assignment-operation,unsubscriptable-object
    """Per-turn overview from TurnBegin/TurnEnd + StatusUpdate."""
    lines = ["TURNS", "=" * 40]

    # Build list of turns with their begin/end lines
    turns: list[dict] = []
    current_turn: dict | None = None
    for m in meta_events:
        if m["type"] == "turn_begin":
            if current_turn:
                turns.append(current_turn)
            current_turn = {
                "begin_line": m["line"],
                "begin_ts": m["ts"],
                "end_line": None,
                "end_ts": None,
                "steps": 0,
                "status_updates": [],
            }
        # `is not None` rather than a bare truth test: it is the same
        # condition, and it is the one a static analyser narrows through.
        elif m["type"] == "turn_end" and current_turn is not None:
            current_turn["end_line"] = m["line"]
            current_turn["end_ts"] = m["ts"]
            turns.append(current_turn)
            current_turn = None
        elif m["type"] == "step_begin" and current_turn is not None:
            current_turn["steps"] += 1
        elif m["type"] == "status_update" and current_turn is not None:
            current_turn["status_updates"].append(m)

    if current_turn:
        turns.append(current_turn)

    if not turns:
        lines.append("(no turn boundaries found)")
        return "\n".join(lines)

    lines.append(
        f"  {'#':>4}  {'begin':<19}  {'L_begin':>7}  {'end':<19}  {'L_end':>7}  {'steps':>5}  {'last_input':>10}  {'last_output':>8}"
    )
    for idx, t in enumerate(turns, 1):
        # Get the last status_update for this turn
        last_su = t["status_updates"][-1] if t["status_updates"] else None
        if last_su:
            tu = last_su.get("token_usage") or {}
            input_other = tu.get("input_other", 0) or 0
            cache_read = tu.get("input_cache_read", 0) or 0
            cache_create = tu.get("input_cache_creation", 0) or 0
            total_input = input_other + cache_read + cache_create
            output = tu.get("output", 0) or 0
        else:
            total_input = 0
            output = 0
        begin_time = short_time(t["begin_ts"])
        end_time = short_time(t["end_ts"]) if t["end_ts"] else "(open)"
        end_line_str = f"L{t['end_line']}" if t['end_line'] else "---"
        lines.append(
            f"  {idx:>4}  {begin_time:<19}  L{t['begin_line']:<4}  "
            f"{end_time:<19}  {end_line_str:>7}  "
            f"{t['steps']:>5}  {total_input:>10,}  {output:>8,}"
        )

    lines.append("")
    lines.append(f"Total: {len(turns)} turns")
    return "\n".join(lines)


def compute_subagents(events: list[dict], meta_events: list[dict]) -> str:
    """Subagent events summary."""
    lines = ["SUBAGENTS", "=" * 40]
    subagent_events = [e for e in events if e["type"] == "subagent_event"]
    if not subagent_events:
        lines.append("(no subagent events found)")
        return "\n".join(lines)

    by_parent: dict[str, list[dict]] = defaultdict(list)
    for e in subagent_events:
        by_parent[e.get("parent_tool_call_id", "?")].append(e)

    lines.append(f"Total subagent events: {len(subagent_events)}")
    lines.append("")
    for parent_id, evs in sorted(by_parent.items()):
        lines.append(f"Parent tool_call_id: {parent_id}  ({len(evs)} events)")
        for e in evs:
            sub_evt = e.get("event", {})
            sub_type = sub_evt.get("type", "?")
            lines.append(
                f"  L{e['line']:<4} [{short_time(e['ts'])}] "
                f"agent={e.get('agent_id', '?')}  type={e.get('subagent_type', '?')}  "
                f"event={sub_type}"
            )
        lines.append("")
    return "\n".join(lines)


def filter_by_path(events: list[dict], pattern: str) -> list[dict]:
    """Filter events to only those touching a file path matching pattern."""
    pattern_lower = pattern.lower()
    result = []
    for e in events:
        if e["type"] == "tool_call":
            inp = e.get("tool_input", {})
            file_path = inp.get("path", "") or inp.get("file_path", "")
            command = inp.get("command", "")
            if pattern_lower in file_path.lower() or pattern_lower in command.lower():
                result.append(e)
                continue
        elif e["type"] == "tool_result":
            paired = e.get("paired_call")
            if paired:
                inp = paired.get("tool_input", {})
                file_path = inp.get("path", "") or inp.get("file_path", "")
                command = inp.get("command", "")
                if pattern_lower in file_path.lower() or pattern_lower in command.lower():
                    result.append(e)
                    continue
        # Non-tool events pass through
        if e["type"] in ("user_message", "assistant_text", "thinking"):
            result.append(e)
    return result


def filter_time(events: list[dict], after: str |
                None, before: str | None) -> list[dict]:
    """Filter events by time range."""
    def parse_time_arg(t: str):
        m = re.match(r"^(\d+)([dhw])$", t)
        if m:
            n, unit = int(m.group(1)), m.group(2)
            mul = {"d": 86400, "h": 3600, "w": 86400 * 7}[unit]
            return ("absolute", datetime.now(
                timezone.utc) - timedelta(seconds=n * mul))
        if "-" in t and ":" not in t.split("-")[-1]:
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

    def passes(e, mode, val, op):
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
        events = [
            e for e in events if passes(
                e,
                mode,
                val,
                lambda a,
                b: a >= b)]
    if before:
        mode, val = parse_time_arg(before)
        events = [
            e for e in events if passes(
                e,
                mode,
                val,
                lambda a,
                b: a <= b)]
    return events


def filter_line_range(events: list[dict], range_str: str) -> list[dict]:
    """Filter events by JSONL line range."""
    start, end = range_str.split("-")
    start, end = int(start), int(end)
    return [e for e in events if start <= e["line"] <= end]


ERROR_PATTERNS = re.compile(
    r"error|Error|ERROR|Traceback|FAILED|exception|Exception|"
    r"AttributeError|NameError|ImportError|TypeError|ValueError|"
    r"KeyError|IndexError|FileNotFoundError|ModuleNotFoundError|"
    r"column .* does not exist|no such file|permission denied|"
    r"Exit code [1-9]",
    re.IGNORECASE,
)


def main():
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8")
            except (OSError, ValueError):
                pass

    parser = argparse.ArgumentParser(
        description="Parse Kimi Code CLI wire.jsonl for investigation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="Print version and exit")
    parser.add_argument(
        "wire",
        nargs="?",
        help="Path to wire.jsonl file, session directory, or directory (recursively joins all */wire.jsonl).")

    type_group = parser.add_argument_group(
        "event type filters (union if multiple)")
    type_group.add_argument("--tools", action="store_true", help="Tool calls")
    type_group.add_argument(
        "--results",
        action="store_true",
        help="Tool results only")
    type_group.add_argument(
        "--user",
        action="store_true",
        help="User messages only")
    type_group.add_argument(
        "--assistant",
        action="store_true",
        help="Assistant text only")
    type_group.add_argument(
        "--thinking",
        action="store_true",
        help="Thinking blocks only")
    type_group.add_argument(
        "--conversation",
        action="store_true",
        help="User + assistant text only")
    type_group.add_argument(
        "--pairs",
        action="store_true",
        help="Tool calls paired with results")
    type_group.add_argument(
        "--errors",
        action="store_true",
        help="Failed tool results and API-provider errors")

    parser.add_argument(
        "--tool",
        metavar="NAME",
        help="Filter by tool name (comma-separated for multiple)")

    analysis_group = parser.add_argument_group("analysis modes")
    analysis_group.add_argument(
        "--stats",
        action="store_true",
        help="Session statistics")
    analysis_group.add_argument(
        "--timeline",
        action="store_true",
        help="Chronological interleave of ALL events")
    analysis_group.add_argument(
        "--commands",
        action="store_true",
        help="All Shell commands")
    analysis_group.add_argument(
        "--files",
        action="store_true",
        help="All files touched")
    analysis_group.add_argument(
        "--writes",
        action="store_true",
        help="All WriteFile calls")
    analysis_group.add_argument(
        "--edits",
        action="store_true",
        help="All StrReplaceFile calls")
    analysis_group.add_argument(
        "--cache",
        action="store_true",
        help="Token totals from StatusUpdate + billing estimate")
    parser.add_argument("--rates", metavar="SPEC", help=(
        "Override pricing rates for --cache billing estimate. "
        'Format: "fresh=N,cache_read=N,cache_create=N,output=N" (USD per 1M tokens). '
        "Default: Kimi K2.6 rates (fresh=$0.95, cache_read=$0.16, output=$4.00)."
    ))
    analysis_group.add_argument(
        "--context-growth",
        action="store_true",
        help="Context size over time")
    analysis_group.add_argument(
        "--latency",
        action="store_true",
        help="Rank tool calls by wall-clock latency")
    analysis_group.add_argument(
        "--reply-latency",
        action="store_true",
        help="Gap from user message to assistant response")
    analysis_group.add_argument(
        "--shell-errors",
        action="store_true",
        help="Shell calls with errors")
    analysis_group.add_argument(
        "--tool-orphans",
        action="store_true",
        help="Tool calls without matching results")
    analysis_group.add_argument(
        "--turns",
        action="store_true",
        help="Per-turn overview")
    analysis_group.add_argument(
        "--subagents",
        action="store_true",
        help="Subagent events summary")

    parser.add_argument(
        "--path",
        metavar="PATTERN",
        help="Filter events by file path (substring match)")
    parser.add_argument("--range", metavar="N-M", help="JSONL line range")
    parser.add_argument(
        "--after",
        metavar="WHEN",
        help="Events at/after WHEN. HH:MM:SS, YYYY-MM-DD, or Nd/Nh/Nw.")
    parser.add_argument(
        "--before",
        metavar="WHEN",
        help="Events at/before WHEN.")
    parser.add_argument(
        "--grep",
        metavar="PATTERN",
        help="Regex filter on formatted output (block-level)")

    parser.add_argument(
        "--full",
        action="store_true",
        help="Disable truncation")
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show full content")
    parser.add_argument("-o", "--output", metavar="FILE", help="Write to file")
    parser.add_argument(
        "--with-source",
        action="store_true",
        help="Show source file basename")
    parser.add_argument(
        "--timing",
        action="store_true",
        help="Print per-phase wall time to stderr")

    args = parser.parse_args()

    if args.version:
        print(f"parse_wire.py {__version__}")
        sys.exit(0)

    if not args.wire:
        parser.error("the following arguments are required: wire")

    # Resolve wire path
    wire_path = args.wire
    wire_path_obj = Path(wire_path)

    if not wire_path_obj.exists():
        # Try as session hash under ~/.kimi-code/sessions
        sessions_root = Path.home() / ".kimi-code" / "sessions"
        if sessions_root.is_dir():
            # Could be a session hash or a uuid under it
            candidates = []
            for session_dir in sessions_root.iterdir():
                if not session_dir.is_dir():
                    continue
                if session_dir.name == wire_path:
                    # It's a session hash — find wire.jsonl inside
                    for sub in session_dir.iterdir():
                        if sub.is_dir():
                            w = _wire_file(sub)
                            if w:
                                candidates.append(str(w))
                else:
                    # Check if wire_path is a subdir uuid
                    w = _wire_file(session_dir / wire_path)
                    if w:
                        candidates.append(str(w))
            if len(candidates) == 1:
                wire_path = candidates[0]
                wire_path_obj = Path(wire_path)
            elif len(candidates) > 1:
                print(
                    f"Ambiguous '{wire_path}' — matches {len(candidates)} files:",
                    file=sys.stderr)
                for c in candidates:
                    print(f"  {c}", file=sys.stderr)
                sys.exit(1)

    if not wire_path_obj.exists():
        print(f"File not found: {wire_path}", file=sys.stderr)
        sys.exit(1)

    timing_enabled = args.timing
    timings: list[tuple[str, float]] = []

    def _tick(label: str, t0: float):
        if timing_enabled:
            timings.append((label, time.perf_counter() - t0))

    events: list[dict] = []
    meta_events: list[dict] = []

    if wire_path_obj.is_dir():
        t0 = time.perf_counter()
        files: list[str] = []
        for dp, _, fns in os.walk(wire_path, followlinks=True):
            for fn in fns:
                if fn in ("wire.jsonl", "wire.jsonl.xz"):
                    files.append(os.path.join(dp, fn))
        files.sort()
        _tick(f"walk ({len(files)} files)", t0)
        if not files:
            print(f"No wire.jsonl files under {wire_path}", file=sys.stderr)
            sys.exit(1)
        t0 = time.perf_counter()
        for f in files:
            ev, me = extract_events(f)
            events.extend(ev)
            meta_events.extend(me)
        _tick(
            f"extract_events ({len(events):,} ev / {len(meta_events):,} meta)", t0)
        t0 = time.perf_counter()
        events.sort(key=lambda e: e.get("ts") or "￿")
        meta_events.sort(key=lambda m: m.get("ts") or "￿")
        _tick("sort by ts", t0)
    else:
        t0 = time.perf_counter()
        events, meta_events = extract_events(wire_path)
        _tick(
            f"extract_events ({len(events):,} ev / {len(meta_events):,} meta)", t0)

    t0 = time.perf_counter()
    annotate_parallel(events)
    _tick("annotate_parallel", t0)
    t0 = time.perf_counter()
    link_pairs(events)
    _tick("link_pairs", t0)

    if args.full:
        args.verbose = True

    # Apply filters
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
            if (e["type"] == "tool_call" and e.get("tool_name", "") in tool_names)
            or (e["type"] == "tool_result" and e.get("paired_call", {}).get("tool_name", "") in tool_names)
            or e["type"] not in ("tool_call", "tool_result")
        ]

    # Mode dispatch
    output = None
    if args.stats:
        output = compute_stats(events, meta_events)
    elif args.files:
        output = compute_files(events)
    elif args.commands:
        output = compute_commands(events)
    elif args.edits:
        output = compute_edits(events, verbose=args.verbose)
    elif args.writes:
        output = compute_writes(events, verbose=args.verbose)
    elif args.cache:
        rate_dict = {}
        if args.rates:
            for part in args.rates.split(","):
                if "=" in part:
                    k, v = part.split("=", 1)
                    try:
                        rate_dict[k.strip()] = float(v.strip())
                    except ValueError:
                        pass
        output = compute_cache(meta_events, rates=rate_dict, events=events)
    elif args.context_growth:
        output = compute_context_growth(meta_events)
    elif args.latency:
        output = compute_latency(events)
    elif args.reply_latency:
        output = compute_reply_latency(events)
    elif args.shell_errors:
        output = compute_shell_errors(events)
    elif args.tool_orphans:
        output = compute_tool_orphans(events)
    elif args.turns:
        output = compute_turns(events, meta_events)
    elif args.subagents:
        output = compute_subagents(events, meta_events)
    elif args.timeline:
        output = compute_timeline(events, verbose=args.verbose)
    elif args.pairs:
        output = "\n".join(
            format_pair(
                ev,
                ev.get("paired_result"),
                verbose=args.verbose,
                with_source=args.with_source)
            for ev in events if ev["type"] == "tool_call"
        )
    else:
        # Type filter fallback
        type_flags = {
            "tools": args.tools,
            "results": args.results,
            "user": args.user,
            "assistant": args.assistant,
            "thinking": args.thinking,
            "conversation": args.conversation,
            "errors": args.errors,
        }
        active_filters = {k for k, v in type_flags.items() if v}

        if active_filters:
            allowed_types: set[str] = set()
            if "tools" in active_filters:
                allowed_types.add("tool_call")
            if "results" in active_filters:
                allowed_types.add("tool_result")
            if "user" in active_filters:
                allowed_types.add("user_message")
            if "assistant" in active_filters:
                allowed_types.add("assistant_text")
            if "thinking" in active_filters:
                allowed_types.add("thinking")
            if "conversation" in active_filters:
                allowed_types |= {"user_message", "assistant_text"}
            if "errors" in active_filters:
                allowed_types |= {"tool_result", "api_error"}

            events = [e for e in events if e["type"] in allowed_types]

            if ("errors" in active_filters
                    and "results" not in active_filters):
                # Trust the explicit is_error boolean set by Kimi on every
                # ToolResult. Earlier versions OR'd a regex match (`error|
                # Error|ERROR|...`) over the detail text as a fallback, which
                # flooded with false positives because most tool outputs
                # (CLAUDE.md, source code, docs) legitimately contain the
                # word "error". kimi-dash's tool-error-rate query uses the
                # same boolean-only criterion (`is_error IS NOT NULL`); the
                # two surfaces now agree.
                events = [
                    e for e in events
                    if e["type"] != "tool_result" or e.get("is_error")
                ]

        formatted_lines = [
            format_event(
                e,
                verbose=args.verbose,
                with_source=args.with_source) for e in events]
        output = "\n".join(formatted_lines)

    # --grep block-level filter
    if args.grep and output:
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

    if output is None:
        # No mode predicate matched. Silent-empty stdout makes it look like
        # a successful run with no events; raise instead so the caller knows
        # they need a mode flag.
        print(
            "ERROR: no mode selected. Pick one or more flags (e.g. --stats, "
            "--timeline, --conversation). See --help.",
            file=sys.stderr,
        )
        sys.exit(2)
    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
        print(f"Output written to {args.output}")
    else:
        print(output)

    if timing_enabled and timings:
        total = sum(t for _, t in timings)
        print(
            f"\n--- timing (parse_wire.py {__version__}) ---",
            file=sys.stderr)
        for tlabel, t in timings:
            print(f"  {tlabel:<40} {t:>8.3f}s", file=sys.stderr)
        print(f"  {'TOTAL':<40} {total:>8.3f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
