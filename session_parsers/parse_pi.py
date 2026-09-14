#!/usr/bin/env python3
"""Pi session JSONL parser — investigation utility for Pi coding-agent sessions.

Fourth sibling to the provider-owned transcript parsers (`parse_session.py`
for Claude Code, `parse_wire.py` for Kimi Code, `parse_codex.py` for Codex).
Target format: `~/.pi/agent/sessions/<munged-cwd>/<timestamp>_<uuid>.jsonl`,
one JSON object per line. Entry `type` is one of `session` (identity, once),
`message` (role `user`, `assistant`, or `toolResult`), `custom_message`
(role-less: monitor deliveries, extension steers), `model_change`,
`thinking_level_change`, and `compaction`.

WHY THE TOKEN ARITHMETIC LOOKS PARANOID
---------------------------------------
Three properties of this format make the naive sum wrong, each verified
exactly against a live corpus on 2026-08-29:

1. `usage.reasoning` is NOT inside `usage.output`, and NOT inside
   `usage.totalTokens`. A real assistant message reads input 132, output 88,
   cacheRead 2048, reasoning 69, totalTokens 2268 — and 132+88+2048 is
   exactly 2268. So `totalTokens` silently drops reasoning, and a report
   that adds reasoning to totalTokens while also adding output double-counts
   nothing but must still add reasoning to output on the response side.
   The parser's rule: response = output + reasoning; grand total =
   input + output + cacheRead + cacheWrite + reasoning.
2. `usage` is PER REQUEST, and one user turn is MANY requests: an agent loop
   emits one assistant message per tool round, each carrying its own usage.
   Per-turn rollups that take the turn's first assistant message undercount
   by the number of tool rounds. The parser never groups usage by turn; it
   sums messages, and labels context size as per-request.
3. Tool results are top-level `toolResult` messages paired to assistant
   content parts by `toolCallId` — not content inside a user message, and
   not carrying usage of their own. Orphan checks therefore run BOTH ways:
   a call with no result and a result with no call are different failures.

Errors ride inside ordinary messages: an assistant message carries
`stopReason` (`stop`, `toolUse`, `aborted`, `error`) and an `errorMessage`;
a tool result carries `isError`. There is no separate error entry type, so
error modes must scan messages rather than look for error records.

`usage.cost` is pi's own cost estimate and is reported as pi stated it, never
re-derived; `--tokens` sums it per model.

Read-only. Safe against a live session file being appended to.
"""
import argparse
import glob as _glob
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

__version__ = "1.0.0"

DEFAULT_ROOT = os.path.expanduser("~/.pi/agent/sessions")

TRUNCATE = 96


# ---------------------------------------------------------------------------
# time and text helpers
# ---------------------------------------------------------------------------

def parse_ts(raw):
    """ISO-8601 string -> aware datetime; None when absent or unparseable."""
    if not raw:
        return None
    try:
        dt = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def fmt_ts(dt):
    return dt.strftime("%m-%d %H:%M:%S") if dt else "?"


def fmt_full(dt):
    return dt.strftime("%Y-%m-%d %H:%M:%S %z") if dt else "(none)"


def fmt_dur(seconds):
    if seconds is None:
        return "?"
    seconds = int(seconds)
    if seconds < 90:
        return "%ds" % seconds
    if seconds < 5400:
        return "%dm%02ds" % (seconds // 60, seconds % 60)
    return "%dh%02dm" % (seconds // 3600, (seconds % 3600) // 60)


def clip(text, full=False, width=TRUNCATE):
    text = " ".join(str(text or "").split())
    if full or len(text) <= width:
        return text
    return text[:width - 3] + "..."


# ---------------------------------------------------------------------------
# extraction: one entry shape -> flat normalized events
# ---------------------------------------------------------------------------

def _parts_text(content):
    """Plain text of a message content that may be a string or a part list."""
    if isinstance(content, str):
        return content
    out = []
    for part in content or []:
        if not isinstance(part, dict):
            continue
        if part.get("type") == "text":
            out.append(part.get("text") or "")
        elif part.get("type") == "thinking":
            out.append(part.get("thinking") or "")
    return "\n".join(t for t in out if t)


def _response(usage):
    """Response size of one request: output plus reasoning, never either
    alone (trap 1)."""
    return int(usage.get("output") or 0) + int(usage.get("reasoning") or 0)


def _grand_total(usage):
    """All tokens one request touched: pi's totalTokens EXCLUDES reasoning
    (trap 1), so the grand total adds it back."""
    return sum(int(usage.get(k) or 0) for k in
               ("input", "output", "cacheRead", "cacheWrite", "reasoning"))


def _context_after(usage):
    """Window occupancy after this request: prompt (input + both caches)
    plus the response it produced."""
    return sum(int(usage.get(k) or 0) for k in
               ("input", "cacheRead", "cacheWrite")) + _response(usage)


def extract_events(files, line_grep=None):
    """Parse session files into flat events, in file then line order."""
    events = []
    for path in files:
        try:
            handle = open(path, encoding="utf-8", errors="replace")
        except OSError:
            continue
        with handle:
            for lineno, line in enumerate(handle, 1):
                if line_grep and line_grep not in line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    events.append({"kind": "other", "file": path, "line": lineno,
                                   "ts": None, "note": "unparseable line"})
                    continue
                if not isinstance(entry, dict):
                    continue
                events.extend(_entry_events(entry, path, lineno))
    return events


def _entry_events(entry, path, lineno):
    etype = entry.get("type")
    ts = parse_ts(entry.get("timestamp"))
    base = {"file": path, "line": lineno, "ts": ts}

    if etype == "session":
        return [dict(base, kind="meta", session_id=entry.get("id"),
                     cwd=entry.get("cwd"), version=entry.get("version"))]

    if etype == "model_change":
        return [dict(base, kind="model", model_id=entry.get("modelId"),
                     provider=entry.get("provider"))]

    if etype == "thinking_level_change":
        return [dict(base, kind="thinking_level",
                     level=entry.get("thinkingLevel"))]

    if etype == "compaction":
        return [dict(base, kind="compact",
                     summary_tokens=entry.get("tokensBefore"),
                     detail=clip(entry.get("summary") or entry.get("detail") or ""))]

    if etype == "custom_message":
        return [dict(base, kind="custom", custom_type=entry.get("customType"),
                     text=_parts_text(entry.get("content")),
                     display=entry.get("display"))]

    if etype != "message":
        return [dict(base, kind="other", note="type=%s" % etype)]

    message = entry.get("message") or {}
    role = message.get("role")
    events = []
    if role == "user":
        events.append(dict(base, kind="user", text=_parts_text(message.get("content"))))
    elif role == "assistant":
        events.append(_assistant_event(base, message))
        for part in message.get("content") or []:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "thinking":
                events.append(dict(base, kind="thinking",
                                   chars=len(part.get("thinking") or ""),
                                   signature=part.get("thinkingSignature")))
            elif part.get("type") == "toolCall":
                events.append(dict(base, kind="tool_call", call_id=part.get("id"),
                                   tool=part.get("name"),
                                   arguments=part.get("arguments"),
                                   response_id=message.get("responseId")))
    elif role == "toolResult":
        events.append(dict(base, kind="tool_result", call_id=message.get("toolCallId"),
                           tool=message.get("toolName"), ok=not message.get("isError"),
                           error=message.get("errorMessage"),
                           text=_parts_text(message.get("content"))))
    else:
        events.append(dict(base, kind="other", note="role=%s" % role))
    return events


def _assistant_event(base, message):
    usage = message.get("usage") or {}
    stop = message.get("stopReason")
    event = dict(base, kind="assistant",
                 text=_parts_text(message.get("content")),
                 model=message.get("model"), provider=message.get("provider"),
                 stop_reason=stop,
                 error=message.get("errorMessage") if stop in ("error", "aborted") else None,
                 usage=usage or None,
                 response=_response(usage) if usage else None,
                 total=_grand_total(usage) if usage else None,
                 context_after=_context_after(usage) if usage else None)
    if usage:
        event["cost"] = (usage.get("cost") or {}).get("total")
    return event


def resolve_target(target):
    """A session .jsonl, a session id (matched in DEFAULT_ROOT), or a
    directory (recursively joined, sorted by name)."""
    path = Path(target)
    if path.is_dir():
        return sorted(str(p) for p in path.rglob("*.jsonl"))
    if path.is_file():
        return [str(path)]
    # A bare session id: search the default root for a file carrying it.
    hits = sorted(str(p) for p in Path(DEFAULT_ROOT).rglob("*.jsonl")
                  if target in p.name)
    if hits:
        return hits
    hits = sorted(_glob.glob(target))
    if hits:
        return hits
    raise SystemExit("error: no session file matches %r" % target)


# ---------------------------------------------------------------------------
# pairing: tool calls <-> results (trap 3)
# ---------------------------------------------------------------------------

def link_pairs(events):
    """Attach each tool_result onto its tool_call event; classify orphans.

    Returns (paired_count, unmatched_calls, orphan_results): the unmatched
    sets are different failures — an unexecuted call versus a result whose
    call line was lost.
    """
    by_id = {}
    for ev in events:
        if ev["kind"] == "tool_call" and ev.get("call_id"):
            by_id[ev["call_id"]] = ev
    results = [e for e in events if e["kind"] == "tool_result"]
    call_ids = set(by_id)
    paired = 0
    for result in results:
        call = by_id.pop(result.get("call_id"), None)
        if call is not None:
            call["result"] = result
            paired += 1
    # What remains in by_id are calls no result answered; orphan results are
    # those whose id never named a call at all.
    orphan_results = [r for r in results if r.get("call_id") not in call_ids]
    return paired, by_id, orphan_results


# ---------------------------------------------------------------------------
# filters and modes
# ---------------------------------------------------------------------------

def apply_filters(events, args):
    """Union of the requested event-type filters; no filter = everything."""
    wanted = []
    if args.tools:
        wanted.append("tool_call")
    if args.results:
        wanted.append("tool_result")
    if args.user:
        wanted.append("user")
    if args.assistant:
        wanted.append("assistant")
    if args.thinking:
        wanted.append("thinking")
    if args.custom:
        wanted.append("custom")
    if not wanted:
        return events
    return [e for e in events if e["kind"] in wanted]


def label(ev, args):
    kind = ev["kind"]
    if kind == "user":
        return "user: %s" % clip(ev.get("text"), args.full)
    if kind == "assistant":
        head = "asst: %s" % clip(ev.get("text"), args.full) if ev.get("text") else "asst:"
        if ev.get("stop_reason") and ev["stop_reason"] != "stop":
            head += "  [%s]" % ev["stop_reason"]
        if ev.get("response") is not None:
            head += "  (%s tok)" % format_int(ev["response"])
        return head
    if kind == "thinking":
        return "think: %s chars" % format_int(ev.get("chars") or 0)
    if kind == "tool_call":
        return "call: %s %s" % (ev.get("tool"),
                                clip(json.dumps(ev.get("arguments") or {}), args.full))
    if kind == "tool_result":
        flag = "ok" if ev.get("ok") else "ERROR"
        return "res[%s]: %s" % (flag, clip(ev.get("text"), args.full))
    if kind == "custom":
        return "cust[%s]: %s" % (ev.get("custom_type"), clip(ev.get("text"), args.full))
    if kind == "model":
        return "model -> %s (%s)" % (ev.get("model_id"), ev.get("provider"))
    if kind == "thinking_level":
        return "thinking -> %s" % ev.get("level")
    if kind == "compact":
        return "COMPACT: %s" % ev.get("detail")
    return "other: %s" % ev.get("note", "")


def format_int(n):
    return "{:,}".format(int(n or 0))


def mode_stats(events, args, emit):
    stamped = [e for e in events if e["ts"]]
    kinds = Counter(e["kind"] for e in events)
    emit("SESSION STATISTICS")
    emit("=" * 40)
    emit("Start:      %s" % fmt_full(stamped[0]["ts"] if stamped else None))
    emit("End:        %s" % fmt_full(stamped[-1]["ts"] if stamped else None))
    if len(stamped) > 1:
        span = (stamped[-1]["ts"] - stamped[0]["ts"]).total_seconds()
        emit("Duration:   %s  (wall)" % fmt_dur(span))
    files = sorted({e["file"] for e in events})
    emit("Files:      %d" % len(files))
    ids = {e.get("session_id") for e in events if e["kind"] == "meta"}
    ids.discard(None)
    emit("Session ids:%d" % len(ids))
    emit("User msgs:  %d" % kinds.get("user", 0))
    emit("Asst msgs:  %d  (%d requests carried usage)" % (
        kinds.get("assistant", 0),
        sum(1 for e in events if e["kind"] == "assistant" and e.get("usage"))))
    emit("Thinking:   %d parts" % kinds.get("thinking", 0))
    results = [e for e in events if e["kind"] == "tool_result"]
    _, unmatched_calls, orphan_results = link_pairs(events)
    emit("Tool calls: %d  (results: %d, unpaired calls: %d, orphan results: %d)" % (
        sum(1 for e in events if e["kind"] == "tool_call"),
        len(results), len(unmatched_calls), len(orphan_results)))
    errors = [e for e in events if e["kind"] == "assistant" and e.get("stop_reason") in ("error", "aborted")]
    emit("Turn errors:%d aborted, %d errored" % (
        sum(1 for e in errors if e["stop_reason"] == "aborted"),
        sum(1 for e in errors if e["stop_reason"] == "error")))
    failed = [r for r in results if not r.get("ok")]
    emit("Tool errors:%d of %d results failed" % (len(failed), len(results)))
    emit("Compactions:%d" % kinds.get("compact", 0))

    models = Counter(e.get("model") for e in events
                     if e["kind"] == "assistant" and e.get("model"))
    if models:
        emit("")
        emit("MODELS SERVING:")
        for name, count in models.most_common():
            emit("  %-28s %d responses" % (name, count))
    return 0


def mode_tokens(events, args, emit):
    """Token and cost totals per model. Response = output + reasoning (trap
    1); the grand total never uses pi's totalTokens because it drops
    reasoning."""
    per_model = {}
    for ev in events:
        if ev["kind"] != "assistant" or not ev.get("usage"):
            continue
        bucket = per_model.setdefault(ev.get("model") or "(unknown)", {
            "requests": 0, "input": 0, "output": 0, "reasoning": 0,
            "cache_read": 0, "cache_write": 0, "response": 0, "total": 0,
            "cost": 0.0})
        usage = ev["usage"]
        bucket["requests"] += 1
        for key in ("input", "output", "reasoning", "cache_read", "cache_write"):
            bucket[key] += int(usage.get({"cache_read": "cacheRead",
                                          "cache_write": "cacheWrite"}.get(key, key)) or 0)
        bucket["response"] += ev["response"] or 0
        bucket["total"] += ev["total"] or 0
        bucket["cost"] += ev.get("cost") or 0.0

    emit("TOKENS AND COST (per model, per request sums)")
    emit("=" * 48)
    if not per_model:
        emit("(no assistant usage in this selection)")
        return 0
    grand = Counter()
    for model, bucket in sorted(per_model.items()):
        emit("%s  —  %s requests" % (model, format_int(bucket["requests"])))
        emit("  input %s  (+cache read %s, write %s)" % (
            format_int(bucket["input"]), format_int(bucket["cache_read"]),
            format_int(bucket["cache_write"])))
        emit("  response %s  (output %s + reasoning %s)" % (
            format_int(bucket["response"]), format_int(bucket["output"]),
            format_int(bucket["reasoning"])))
        emit("  grand total %s   cost %s (pi estimate)" % (
            format_int(bucket["total"]),
            "$%.4f" % bucket["cost"] if bucket["cost"] else "n/a"))
        for key in ("requests", "input", "output", "reasoning", "cache_read",
                    "cache_write", "response", "total"):
            grand[key] += bucket[key]
        grand["cost"] += bucket["cost"]
    if len(per_model) > 1:
        emit("")
        emit("ALL MODELS: response %s, grand total %s, cost %s" % (
            format_int(grand["response"]), format_int(grand["total"]),
            "$%.4f" % grand["cost"] if grand["cost"] else "n/a"))
    return 0


def mode_timeline(events, args, emit):
    for ev in apply_filters(events, args):
        emit("%s  %s" % (fmt_ts(ev["ts"]), label(ev, args)))
    return 0


def mode_context_growth(events, args, emit):
    """Window occupancy after each request — the compaction-relevant series.
    Per request, never per turn (trap 2)."""
    rows = [e for e in events if e["kind"] == "assistant" and e.get("context_after")]
    emit("CONTEXT GROWTH (window occupancy after each request)")
    emit("=" * 52)
    if not rows:
        emit("(no usage-bearing assistant messages)")
        return 0
    first = rows[0]["ts"]
    for ev in rows:
        offset = fmt_dur((ev["ts"] - first).total_seconds()) if ev["ts"] and first else "?"
        emit("+%-7s %-12s %s tok  model %s" % (
            offset, fmt_ts(ev["ts"]), format_int(ev["context_after"]),
            (ev.get("model") or "?").split("/")[-1]))
    peak = max(ev["context_after"] for ev in rows)
    emit("")
    emit("peak: %s tok; final: %s tok" % (
        format_int(peak), format_int(rows[-1]["context_after"])))
    return 0


def mode_tool_errors(events, args, emit):
    failed = [r for r in events if r["kind"] == "tool_result" and not r.get("ok")]
    emit("FAILED TOOL RESULTS: %d" % len(failed))
    emit("=" * 40)
    for r in failed:
        emit("%s  %s (%s)" % (fmt_ts(r["ts"]), r.get("tool"), r.get("call_id")))
        body = r.get("error") or r.get("text") or ""
        if body:
            emit("    %s" % clip(body, args.full, width=TRUNCATE * 2))
    return 0


def mode_orphans(events, args, emit):
    """Trap 3: both orphan directions are different failures."""
    paired, unmatched_calls, orphan_results = link_pairs(events)
    emit("UNPAIRED TOOL TRAFFIC")
    emit("=" * 40)
    emit("calls with no result: %d" % len(unmatched_calls))
    for call in unmatched_calls.values():
        emit("  %s  %s %s" % (fmt_ts(call["ts"]), call.get("tool"),
                              clip(json.dumps(call.get("arguments") or {}))))
    emit("results with no call: %d" % len(orphan_results))
    for r in orphan_results:
        emit("  %s  %s (%s)" % (fmt_ts(r["ts"]), r.get("tool"), r.get("call_id")))
    emit("paired:            %d" % paired)
    return 0


def mode_compaction(events, args, emit):
    rows = [e for e in events if e["kind"] == "compact"]
    emit("COMPACTIONS: %d" % len(rows))
    emit("=" * 40)
    for ev in rows:
        emit("%s  %s" % (fmt_ts(ev["ts"]), ev.get("detail") or "(no detail)"))
    return 0


def mode_models(events, args, emit):
    emit("MODEL AND THINKING LEVEL CHANGES")
    emit("=" * 40)
    rows = [e for e in events if e["kind"] in ("model", "thinking_level")]
    if not rows:
        emit("(none recorded)")
    for ev in rows:
        if ev["kind"] == "model":
            emit("%s  model          %s (%s)" % (
                fmt_ts(ev["ts"]), ev.get("model_id"), ev.get("provider")))
        else:
            emit("%s  thinking level %s" % (fmt_ts(ev["ts"]), ev.get("level")))
    return 0


def mode_errors(events, args, emit):
    """Turn-level failures: assistant stopReason error/aborted (no separate
    error entry exists)."""
    rows = [e for e in events
            if e["kind"] == "assistant" and e.get("stop_reason") in ("error", "aborted")]
    emit("TURN FAILURES: %d" % len(rows))
    emit("=" * 40)
    for ev in rows:
        emit("%s  [%s]  %s" % (fmt_ts(ev["ts"]), ev.get("stop_reason"),
                               clip(ev.get("error") or ev.get("text") or "", args.full)))
    return 0


def mode_identity(events, args, emit):
    emit("SESSION IDENTITY")
    emit("=" * 40)
    for ev in (e for e in events if e["kind"] == "meta"):
        emit("id:      %s" % ev.get("session_id"))
        emit("cwd:     %s" % ev.get("cwd"))
        emit("version: %s" % ev.get("version"))
        emit("started: %s" % fmt_full(ev["ts"]))
        emit("file:    %s" % ev["file"])
    stamped = [e for e in events if e["ts"]]
    if stamped:
        emit("last:    %s" % fmt_full(stamped[-1]["ts"]))
    return 0


def mode_last_assistant(events, args, emit):
    last = None
    for e in events:
        if e["kind"] == "assistant" and e.get("text"):
            last = e
    emit("LAST ASSISTANT TEXT")
    emit("=" * 40)
    if last is None:
        emit("(none)")
    else:
        emit("%s  (%s)" % (fmt_full(last["ts"]), last.get("model") or "?"))
        emit(last.get("text") if args.full else clip(last["text"], width=2000))
    return 0


MODES = {
    "stats": mode_stats,
    "tokens": mode_tokens,
    "timeline": mode_timeline,
    "context-growth": mode_context_growth,
    "tool-errors": mode_tool_errors,
    "orphans": mode_orphans,
    "compaction": mode_compaction,
    "models": mode_models,
    "errors": mode_errors,
    "identity": mode_identity,
    "last-assistant": mode_last_assistant,
}


def build_parser():
    ap = argparse.ArgumentParser(
        prog="parse_pi.py",
        description="Parse Pi coding-agent session JSONL for investigation",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session", nargs="?",
                    help="session .jsonl, a session id, or a directory "
                         "(recursively joined, sorted by name)")
    ap.add_argument("--version", action="store_true",
                    help="print version and exit")

    g = ap.add_argument_group("event type filters (union if multiple)")
    g.add_argument("--tools", action="store_true", help="tool calls")
    g.add_argument("--results", action="store_true", help="tool results only")
    g.add_argument("--user", action="store_true", help="user messages only")
    g.add_argument("--assistant", action="store_true", help="assistant text only")
    g.add_argument("--thinking", action="store_true", help="thinking parts only")
    g.add_argument("--custom", action="store_true", help="custom messages only")

    m = ap.add_argument_group("analysis modes")
    m.add_argument("--stats", action="store_true", help="session statistics")
    m.add_argument("--tokens", action="store_true",
                   help="token totals + pi cost estimate, per model")
    m.add_argument("--timeline", action="store_true", help="chronological one-liners")
    m.add_argument("--context-growth", action="store_true",
                   help="window occupancy after each request")
    m.add_argument("--tool-errors", action="store_true", help="failed tool results")
    m.add_argument("--orphans", action="store_true",
                   help="tool calls and results with no partner")
    m.add_argument("--compaction", action="store_true", help="compaction entries")
    m.add_argument("--models", action="store_true",
                   help="model and thinking-level changes")
    m.add_argument("--errors", action="store_true",
                   help="aborted and errored turns")
    m.add_argument("--identity", action="store_true", help="session identity header")
    m.add_argument("--last-assistant", action="store_true",
                   help="final assistant text")

    ap.add_argument("--full", action="store_true",
                    help="do not clip long text")
    ap.add_argument("--line-grep", metavar="SUBSTRING",
                    help="only parse lines containing this substring")
    return ap


def selected_mode(args):
    for name in MODES:
        if getattr(args, name.replace("-", "_")):
            return name
    return None


def run_segment(events, args, stream):
    mode = selected_mode(args)
    emit = print if stream is sys.stdout else (lambda *a, **k: print(*a, file=stream, **k))
    if mode:
        MODES[mode](events, args, emit)
    else:
        for ev in apply_filters(events, args):
            emit("%s  %s" % (fmt_ts(ev["ts"]), label(ev, args)))
    return 0


def split_segments(argv):
    segments = [[]]
    for token in argv:
        if token == "--and":
            segments.append([])
        else:
            segments[-1].append(token)
    return segments


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    segments = split_segments(argv)
    ap = build_parser()

    head = ap.parse_args(segments[0])
    if head.version:
        print("parse_pi.py %s" % __version__)
        return 0
    if not head.session:
        ap.error("a session file, session id, or directory is required")

    files = resolve_target(head.session)
    events = extract_events(files, head.line_grep)

    rc = 0
    for index, segment in enumerate(segments):
        if index == 0:
            args = head
        else:
            if any(tok == "--line-grep" or tok.startswith("--line-grep=")
                   for tok in segment):
                print("error: --line-grep belongs to the first segment",
                      file=sys.stderr)
                rc = rc or 2
                continue
            args = ap.parse_args([head.session] + segment)
            if args.version:
                print("parse_pi.py %s" % __version__)
                continue
        if len(segments) > 1:
            print("===== %s =====" % (selected_mode(args) or "events"))
        try:
            run_segment(events, args, sys.stdout)
        except SystemExit as exc:
            print("segment error: %s" % exc, file=sys.stderr)
            rc = rc or 2
        except Exception as exc:  # one bad segment must not kill the chain
            print("segment error: %s: %s" % (type(exc).__name__, exc),
                  file=sys.stderr)
            rc = rc or 2
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
