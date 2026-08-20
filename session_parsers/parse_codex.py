#!/usr/bin/env python3
"""Codex rollout JSONL parser — investigation utility for Codex CLI sessions.

Third sibling to the provider-owned transcript parsers (`parse_session.py` for
Claude Code, `parse_wire.py` for Kimi Code). This parser lives in the neutral
runtime so every selected harness can inspect Codex transcripts without a
duplicate provider-owned copy.

Target format: `~/.codex/sessions/YYYY/MM/DD/rollout-<ts>-<uuid>.jsonl`, one JSON
object per line, each `{timestamp, type, payload}`. Record `type` is one of
`session_meta`, `turn_context`, `event_msg`, `response_item`, `world_state`,
`compacted`, `inter_agent_communication_metadata`; `event_msg`/`response_item`
carry a second discriminator in `payload.type`.

WHY THE COST MODES LOOK PARANOID
--------------------------------
Four properties of this format make the naive token sum wrong, each verified
against a real 46-file corpus (2026-08-05):

1. `info.total_token_usage` is a MONOTONIC CUMULATIVE counter, and a forked
   thread INHERITS its parent's running total. 8 of 46 files opened at 43M-70M
   tokens already on the clock. Summing the final value per file inflated that
   corpus by 1.18x. This parser always differences consecutive cumulative
   snapshots and never trusts a single total.
2. Duplicate `token_count` events repeat the previous `last_token_usage`
   verbatim (47 consecutive repeats in one file). Summing `last_token_usage`
   double-counts. A zero cumulative delta means no new request happened, so
   differencing drops the duplicates for free.
3. `cached_input_tokens` is a SUBSET of `input_tokens`, not an addend. Observed
   cache rates run 90-98%, so adding rather than subtracting overstates fresh
   input by ~40x.
4. `token_count` payloads carry NO model field and NO agent id, while a single
   session mixes models and folds subagent usage into the same counter. Model
   attribution therefore comes from the most recent `turn_context` /
   `thread_settings_applied` record preceding each token_count line.

Pricing is ALWAYS computed at public API rates, even though these sessions ran
on a ChatGPT subscription (`rate_limits.plan_type` is plus/pro, and
`credits.has_credits` is false). The subscription is a billing arrangement, not
a measure of what the work costs; the API rate is the comparable number. Use
`--rates` to override the table.

Read-only. Safe against a live rollout file being appended to.
"""
import argparse
import glob as _glob
import json
import math
import os
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone

__version__ = "1.0.2"

DEFAULT_ROOT = os.path.expanduser("~/.codex/sessions")

# --------------------------------------------------------------------------
# Pricing
# --------------------------------------------------------------------------
# Public OpenAI API list rates, USD per 1M tokens, as of the 2026-07-30 price
# cut (Terra -20%, Luna -80%; Sol held at its launch rate). Cache reads keep the
# 90% discount; explicit cache writes bill at 1.25x uncached input.
SOL_RATES = {"input": 5.00, "cached": 0.50, "cache_write": 6.25, "output": 30.00}
TERRA_RATES = {"input": 2.00, "cached": 0.20, "cache_write": 2.50, "output": 12.00}
LUNA_RATES = {"input": 0.20, "cached": 0.02, "cache_write": 0.25, "output": 1.20}

# Requests whose input exceeds this threshold bill the WHOLE request at the
# long-context meter: 2x input, 1.5x output.
LONG_CONTEXT_THRESHOLD = 272_000
LONG_CONTEXT_INPUT_MULT = 2.0
LONG_CONTEXT_OUTPUT_MULT = 1.5

# Unknown gpt-5.6-* variants bill at the flagship rate. A wrong overcount is
# visible and arguable; a wrong undercount silently understates the bill.
FALLBACK_RATES = SOL_RATES

_RATE_LABELS = ((SOL_RATES, "gpt-5.6-sol"),
                (TERRA_RATES, "gpt-5.6-terra"),
                (LUNA_RATES, "gpt-5.6-luna"))


def rates_for_model(model):
    """Map a turn_context/thread_settings model string onto a rate table."""
    m = (model or "").lower()
    if "terra" in m:
        return TERRA_RATES
    if "luna" in m:
        return LUNA_RATES
    if "sol" in m:
        return SOL_RATES
    return FALLBACK_RATES


def rate_label(rates):
    for known, label in _RATE_LABELS:
        if rates is known:
            return label
    return "(custom rates)"


def parse_rates(spec):
    """--rates "input=5,cached=0.5,cache_write=6.25,output=30" -> dict."""
    out = {}
    for chunk in (spec or "").split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" not in chunk:
            raise ValueError("bad --rates term %r (want key=value)" % chunk)
        key, _, value = chunk.partition("=")
        key = key.strip()
        if key not in ("input", "cached", "cache_write", "output"):
            raise ValueError("unknown --rates key %r" % key)
        parsed = float(value)
        if not math.isfinite(parsed) or parsed < 0:
            raise ValueError("--rates %s must be a finite non-negative number" % key)
        out[key] = parsed
    if not out:
        raise ValueError("--rates given but empty")
    return out


def price_request(usage, rates):
    """Cost in USD for ONE request's usage delta.

    `usage` keys mirror the transcript: input_tokens is the TOTAL prompt size,
    with cached_input_tokens and cache_write_input_tokens as subsets of it.
    """
    total_in = max(0, usage.get("input_tokens", 0))
    cached = max(0, usage.get("cached_input_tokens", 0))
    written = max(0, usage.get("cache_write_input_tokens", 0))
    out = max(0, usage.get("output_tokens", 0))
    fresh = max(0, total_in - cached - written)
    in_mult = LONG_CONTEXT_INPUT_MULT if total_in > LONG_CONTEXT_THRESHOLD else 1.0
    out_mult = LONG_CONTEXT_OUTPUT_MULT if total_in > LONG_CONTEXT_THRESHOLD else 1.0
    cost = (fresh * rates["input"] * in_mult
            + cached * rates["cached"] * in_mult
            + written * rates["cache_write"] * in_mult
            + out * rates["output"] * out_mult) / 1_000_000.0
    return cost, fresh, total_in > LONG_CONTEXT_THRESHOLD


# --------------------------------------------------------------------------
# Target resolution
# --------------------------------------------------------------------------
def resolve_target(target):
    """A rollout file, a session id, or a directory -> sorted file list."""
    if not target:
        raise SystemExit("error: no target given")
    if os.path.isfile(target):
        return [target]
    if os.path.isdir(target):
        files = sorted(_glob.glob(os.path.join(target, "**", "*.jsonl"),
                                  recursive=True))
        if not files:
            raise SystemExit("error: no .jsonl under %s" % target)
        return files
    # Bare session id (full uuid or a unique prefix).
    hits = sorted(_glob.glob(os.path.join(DEFAULT_ROOT, "**",
                                          "*%s*.jsonl" % _glob.escape(target)),
                             recursive=True))
    if not hits:
        raise SystemExit("error: no rollout file matches %r (looked under %s)"
                         % (target, DEFAULT_ROOT))
    return hits


# --------------------------------------------------------------------------
# Time helpers
# --------------------------------------------------------------------------
def parse_ts(raw):
    if not raw:
        return None
    try:
        text = raw.replace("Z", "+00:00")
        dt = datetime.fromisoformat(text)
    except (ValueError, AttributeError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def parse_when(spec, reference=None):
    """HH:MM:SS | YYYY-MM-DD | YYYY-MM-DDTHH:MM:SS | Nd/Nh/Nw -> datetime."""
    spec = (spec or "").strip()
    now = reference or datetime.now(timezone.utc)
    rel = re.fullmatch(r"(\d+)([dhw])", spec)
    if rel:
        n = int(rel.group(1))
        unit = rel.group(2)
        delta = {"d": timedelta(days=n), "h": timedelta(hours=n),
                 "w": timedelta(weeks=n)}[unit]
        return now - delta
    if re.fullmatch(r"\d{2}:\d{2}:\d{2}", spec):
        base = now.date()
        hh, mm, ss = (int(x) for x in spec.split(":"))
        return datetime(base.year, base.month, base.day, hh, mm, ss,
                        tzinfo=timezone.utc)
    got = parse_ts(spec)
    if got is None:
        raise SystemExit("error: cannot parse time %r" % spec)
    return got


def fmt_ts(dt):
    return dt.strftime("%H:%M:%S") if dt else "--:--:--"


def fmt_full(dt):
    return dt.strftime("%Y-%m-%d %H:%M:%S UTC") if dt else "(no timestamp)"


def fmt_dur(seconds):
    if seconds is None:
        return "-"
    seconds = float(seconds)
    if seconds < 60:
        return "%.1fs" % seconds
    if seconds < 3600:
        return "%dm %02ds" % (seconds // 60, seconds % 60)
    return "%dh %02dm" % (seconds // 3600, (seconds % 3600) // 60)


# --------------------------------------------------------------------------
# JS argument scraping
# --------------------------------------------------------------------------
# The single custom tool is `exec`, whose input is a JS program calling
# tools.<api>({...}). Keys are inconsistently quoted (`cmd:"x"` next to
# `"workdir":"/root"`), so a JSON parse of the object literal is not available;
# scan for the key and read the following quoted string with JS escapes.
_API_RE = re.compile(r"tools\.([A-Za-z_][A-Za-z_0-9]*)\s*\(")


def exec_apis(src):
    """Every tools.<api> called by an exec payload, in first-seen order."""
    seen = []
    for name in _API_RE.findall(src or ""):
        if name not in seen:
            seen.append(name)
    return seen


def js_string_value(src, key):
    """Value of `key:"..."` / `"key":"..."` in a JS object literal, or None."""
    if not src:
        return None
    pattern = re.compile(r'["\']?%s["\']?\s*:\s*(["\'])' % re.escape(key))
    match = pattern.search(src)
    if not match:
        return None
    quote = match.group(1)
    i = match.end()
    buf = []
    while i < len(src):
        ch = src[i]
        if ch == "\\" and i + 1 < len(src):
            nxt = src[i + 1]
            simple = {"b": "\b", "f": "\f", "n": "\n", "r": "\r",
                      "t": "\t", "v": "\v", "0": "\0", "\\": "\\",
                      "'": "'", '"': '"', "/": "/"}
            if nxt in simple:
                buf.append(simple[nxt])
                i += 2
                continue
            width = 2 if nxt == "x" else 4 if nxt == "u" else 0
            digits = src[i + 2:i + 2 + width]
            if width and len(digits) == width and re.fullmatch(
                    r"[0-9A-Fa-f]{%d}" % width, digits):
                buf.append(chr(int(digits, 16)))
                i += 2 + width
                continue
            # JavaScript accepts legacy escapes, but preserving an unknown
            # sequence is safer for shell regexes and Windows paths than
            # silently deleting its backslash.
            buf.append("\\" + nxt)
            i += 2
            continue
        if ch == quote:
            return "".join(buf)
        buf.append(ch)
        i += 1
    return "".join(buf)


def output_text(payload):
    """Flatten a *_call_output payload into one string."""
    out = payload.get("output")
    if isinstance(out, str):
        return out
    if isinstance(out, list):
        parts = []
        for chunk in out:
            if isinstance(chunk, dict):
                if chunk.get("text"):
                    parts.append(chunk["text"])
            elif isinstance(chunk, str):
                parts.append(chunk)
        return "\n".join(parts)
    if isinstance(out, dict):
        return json.dumps(out)
    return ""


def content_text(content):
    """Flatten a response_item message/agent_message content list."""
    if isinstance(content, str):
        return content
    parts = []
    for chunk in content or []:
        if isinstance(chunk, dict):
            if chunk.get("text"):
                parts.append(chunk["text"])
            elif chunk.get("type") == "encrypted_content":
                parts.append("<encrypted>")
        elif isinstance(chunk, str):
            parts.append(chunk)
    return "\n".join(parts)


# --------------------------------------------------------------------------
# Extraction
# --------------------------------------------------------------------------
def extract_events(files, line_grep=None):
    """Read rollout files into a flat, timestamp-sorted event list."""
    literal = None
    regex = None
    if line_grep:
        if re.search(r"[.^$*+?{}\[\]\\|()]", line_grep):
            regex = re.compile(line_grep)
        else:
            literal = line_grep

    events = []
    for path in files:
        src = os.path.basename(path)
        if src.endswith(".jsonl"):
            src = src[:-6]
        # Model in force at each point; token_count carries none of its own.
        current_model = None
        try:
            handle = open(path, "r", encoding="utf-8", errors="replace")
        except OSError as exc:
            print("warning: cannot read %s (%s)" % (path, exc), file=sys.stderr)
            continue
        with handle:
            for lineno, raw in enumerate(handle, 1):
                if literal is not None and literal not in raw:
                    continue
                if regex is not None and not regex.search(raw):
                    continue
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    rec = json.loads(raw)
                except (ValueError, TypeError):
                    continue
                if not isinstance(rec, dict):
                    continue
                payload = rec.get("payload")
                payload = payload if isinstance(payload, dict) else {}
                ev = {
                    "line": lineno,
                    "src": src,
                    "path": path,
                    "ts": parse_ts(rec.get("timestamp")),
                    "rtype": rec.get("type"),
                    "ptype": payload.get("type"),
                    "payload": payload,
                    "model": current_model,
                }
                kind, extra = classify(rec.get("type"), payload)
                ev["kind"] = kind
                ev.update(extra)
                if kind == "model":
                    current_model = ev.get("model_name") or current_model
                    ev["model"] = current_model
                events.append(ev)
    events.sort(key=lambda e: (e["ts"] or datetime.min.replace(
        tzinfo=timezone.utc), e["src"], e["line"]))
    return events


def classify(rtype, payload):
    """Map a raw record onto a normalized kind plus kind-specific fields."""
    ptype = payload.get("type")

    if rtype == "session_meta":
        return "meta", {
            "session_id": payload.get("session_id"),
            "forked_from": payload.get("forked_from_id"),
            "cwd": payload.get("cwd"),
            "cli_version": payload.get("cli_version"),
            "originator": payload.get("originator"),
        }

    if rtype == "turn_context":
        return "model", {
            "model_name": payload.get("model"),
            "effort": payload.get("effort"),
            "turn_id": payload.get("turn_id"),
            "cwd": payload.get("cwd"),
        }

    if rtype == "compacted":
        return "compact", {"replaced": len(payload.get("replacement_history") or [])}

    if rtype == "world_state":
        return "world", {}

    if rtype == "inter_agent_communication_metadata":
        return "inter_agent", {"trigger_turn": payload.get("trigger_turn")}

    if rtype == "event_msg":
        if ptype == "user_message":
            return "user", {"text": payload.get("message") or ""}
        if ptype == "agent_message":
            return "assistant", {"text": payload.get("message") or "",
                                 "phase": payload.get("phase")}
        if ptype == "agent_reasoning":
            return "thinking", {"text": payload.get("text") or ""}
        if ptype == "token_count":
            info = payload.get("info") or {}
            return "token", {
                "total": info.get("total_token_usage") or {},
                "last": info.get("last_token_usage") or {},
                "context_window": info.get("model_context_window"),
                "rate_limits": payload.get("rate_limits") or {},
            }
        if ptype == "task_started":
            return "turn_start", {"turn_id": payload.get("turn_id"),
                                  "context_window": payload.get("model_context_window")}
        if ptype == "task_complete":
            return "turn_end", {
                "turn_id": payload.get("turn_id"),
                "duration_ms": payload.get("duration_ms"),
                "ttft_ms": payload.get("time_to_first_token_ms"),
                "text": payload.get("last_agent_message") or "",
            }
        if ptype == "turn_aborted":
            return "turn_abort", {"turn_id": payload.get("turn_id"),
                                  "reason": payload.get("reason"),
                                  "duration_ms": payload.get("duration_ms")}
        if ptype == "patch_apply_end":
            changes = payload.get("changes") or {}
            return "patch", {
                "call_id": payload.get("call_id"),
                "turn_id": payload.get("turn_id"),
                "ok": bool(payload.get("success")),
                "stderr": payload.get("stderr") or "",
                "changes": changes,
                "paths": sorted(changes.keys()),
            }
        if ptype == "sub_agent_activity":
            return "subagent", {
                "agent_path": payload.get("agent_path"),
                "thread_id": payload.get("agent_thread_id"),
                "event_kind": payload.get("kind"),
                "call_id": payload.get("event_id"),
            }
        if ptype == "mcp_tool_call_end":
            inv = payload.get("invocation") or {}
            dur = payload.get("duration") or {}
            secs = dur.get("secs", 0) + dur.get("nanos", 0) / 1e9 if dur else None
            result = payload.get("result")
            ok = True
            if isinstance(result, dict):
                ok = "Err" not in result
            return "mcp", {
                "call_id": payload.get("call_id"),
                "server": inv.get("server"),
                "mcp_tool": inv.get("tool"),
                "args": inv.get("arguments"),
                "duration_s": secs,
                "ok": ok,
            }
        if ptype == "web_search_end":
            return "websearch", {"call_id": payload.get("call_id"),
                                 "query": payload.get("query"),
                                 "hits": len(payload.get("results") or [])}
        if ptype == "context_compacted":
            return "compact_evt", {}
        if ptype == "thread_goal_updated":
            goal = payload.get("goal") or {}
            return "goal", {"objective": goal.get("objective"),
                            "status": goal.get("status")}
        if ptype == "thread_settings_applied":
            settings = payload.get("thread_settings") or {}
            return "model", {"model_name": settings.get("model"),
                             "effort": None, "turn_id": None,
                             "cwd": settings.get("cwd")}
        return "other", {}

    if rtype == "response_item":
        if ptype == "custom_tool_call":
            src_text = payload.get("input") or ""
            apis = exec_apis(src_text)
            return "tool_call", {
                "call_id": payload.get("call_id"),
                "tool": payload.get("name"),
                "api": apis[0] if apis else payload.get("name"),
                "apis": apis,
                "input": src_text,
                "cmd": js_string_value(src_text, "cmd"),
                "workdir": js_string_value(src_text, "workdir"),
                "turn_id": (payload.get("internal_chat_message_metadata_passthrough")
                            or {}).get("turn_id"),
            }
        if ptype == "function_call":
            return "tool_call", {
                "call_id": payload.get("call_id"),
                "tool": payload.get("name"),
                "api": payload.get("name"),
                "apis": [payload.get("name")],
                "input": payload.get("arguments") or "",
                "cmd": None,
                "workdir": None,
                "turn_id": (payload.get("internal_chat_message_metadata_passthrough")
                            or {}).get("turn_id"),
            }
        if ptype in ("custom_tool_call_output", "function_call_output"):
            text = output_text(payload)
            head = text.split("\n", 1)[0] if text else ""
            failed = head.startswith("Script failed") or head.startswith("collab spawn failed")
            backgrounded = head.startswith("Script running with cell ID")
            return "tool_result", {
                "call_id": payload.get("call_id"),
                "text": text,
                "ok": not failed,
                "bg": backgrounded,
                "head": head,
            }
        if ptype == "reasoning":
            summary = payload.get("summary") or []
            return "reasoning_raw", {"summary": len(summary)}
        if ptype == "agent_message":
            return "agent_msg", {
                "author": payload.get("author"),
                "recipient": payload.get("recipient"),
                "text": content_text(payload.get("content")),
            }
        if ptype == "message":
            return "message", {"role": payload.get("role"),
                               "text": content_text(payload.get("content"))}
        return "other", {}

    return "other", {}


def backfill_models(events):
    """Attribute leading events in a file that precede its first model record.

    A forked rollout replays the parent thread's history — including its
    token_count records — before the new thread emits a turn_context, so the
    first few hundred requests in a fork have no model in front of them. Where
    the file declares EXACTLY ONE model overall there is only one answer, so
    fill it in. Where a file mixes models the prefix stays unattributed rather
    than guessing which side of the switch it belonged to.
    """
    declared = defaultdict(set)
    for ev in events:
        if ev["kind"] == "model" and ev.get("model_name"):
            declared[source_key(ev)].add(ev["model_name"])
    single = {key: next(iter(names)) for key, names in declared.items()
              if len(names) == 1}
    filled = 0
    for ev in events:
        key = source_key(ev)
        if ev.get("model") is None and key in single:
            ev["model"] = single[key]
            ev["model_backfilled"] = True
            filled += 1
    return filled


def link_pairs(events):
    """Attach each tool_result to its tool_call by call_id."""
    calls = {}
    for ev in events:
        if ev["kind"] == "tool_call" and ev.get("call_id"):
            calls.setdefault((source_key(ev), ev["call_id"]), ev)
    for ev in events:
        pair_key = (source_key(ev), ev.get("call_id"))
        if ev["kind"] == "tool_result" and pair_key in calls:
            call = calls[pair_key]
            call["result"] = ev
            ev["call"] = call
    return calls


# --------------------------------------------------------------------------
# Per-request usage: difference the cumulative counter
# --------------------------------------------------------------------------
USAGE_KEYS = ("input_tokens", "cached_input_tokens", "cache_write_input_tokens",
              "output_tokens", "reasoning_output_tokens", "total_tokens")


def source_key(ev):
    """Stable per-rollout identity; `src` alone is only a display basename."""
    return ev.get("path") or ev["src"]


def request_usage(events):
    """Yield one record per REAL request, per source file.

    Differences consecutive `total_token_usage` snapshots within a file. This
    is what makes the numbers trustworthy: a fork's inherited baseline never
    enters a delta, and a duplicate token_count produces a zero delta that is
    dropped rather than counted twice.
    """
    per_file = defaultdict(list)
    for ev in events:
        if ev["kind"] == "token":
            per_file[source_key(ev)].append(ev)

    out = []
    for _key, evs in per_file.items():
        evs.sort(key=lambda e: e["line"])
        prev = None
        for ev in evs:
            total = ev.get("total") or {}
            if not total:
                continue
            if prev is None:
                # Baseline this file inherited from a parent thread = the first
                # cumulative snapshot minus the request that produced it.
                last = ev.get("last") or {}
                prev = {k: total.get(k, 0) - last.get(k, 0) for k in USAGE_KEYS}
            delta = {k: total.get(k, 0) - prev.get(k, 0) for k in USAGE_KEYS}
            prev = {k: total.get(k, 0) for k in USAGE_KEYS}
            if all(delta.get(k, 0) <= 0 for k in USAGE_KEYS):
                continue  # duplicate token_count, or a non-advancing snapshot
            out.append({
                "src": ev["src"],
                "path": ev.get("path"),
                "line": ev["line"],
                "ts": ev["ts"],
                "model": ev.get("model"),
                "usage": delta,
                "cumulative": total,
                "rate_limits": ev.get("rate_limits") or {},
            })
    out.sort(key=lambda r: (r["ts"] or datetime.min.replace(tzinfo=timezone.utc),
                            r["src"], r["line"]))
    return out


# --------------------------------------------------------------------------
# Filtering / formatting
# --------------------------------------------------------------------------
KIND_FILTERS = {
    "tools": {"tool_call"},
    "results": {"tool_result"},
    "user": {"user"},
    "assistant": {"assistant"},
    "thinking": {"thinking"},
    "conversation": {"user", "assistant"},
}

TRUNCATE = 300


def clip(text, full=False, width=TRUNCATE):
    text = (text or "").replace("\r", "")
    if full:
        return text
    text = " ".join(text.split())
    return text if len(text) <= width else text[:width] + " ...[+%d]" % (len(text) - width)


def event_path_blob(ev):
    """Text a --path filter should match against for this event."""
    bits = []
    for key in ("cmd", "input", "workdir"):
        if ev.get(key):
            bits.append(str(ev[key]))
    if ev.get("paths"):
        bits.extend(ev["paths"])
    return "\n".join(bits)


def apply_filters(events, args):
    out = events
    if args.range:
        match = re.fullmatch(r"(\d+)-(\d+)", args.range)
        if not match:
            raise SystemExit("error: --range wants N-M")
        lo, hi = int(match.group(1)), int(match.group(2))
        out = [e for e in out if lo <= e["line"] <= hi]
    if args.after:
        when = parse_when(args.after)
        out = [e for e in out if e["ts"] and e["ts"] >= when]
    if args.before:
        when = parse_when(args.before)
        out = [e for e in out if e["ts"] and e["ts"] <= when]
    if args.path:
        needle = args.path
        out = [e for e in out if needle in event_path_blob(e)]
    if args.tool:
        wanted = {t.strip() for t in args.tool.split(",") if t.strip()}
        kept = []
        for e in out:
            names = set()
            if e.get("tool"):
                names.add(e["tool"])
            names.update(x for x in (e.get("apis") or []) if x)
            if e.get("mcp_tool"):
                names.add(e["mcp_tool"])
            call = e.get("call")
            if call:
                if call.get("tool"):
                    names.add(call["tool"])
                names.update(x for x in (call.get("apis") or []) if x)
            if names & wanted:
                kept.append(e)
        out = kept
    return out


def label(ev, args):
    src = " %s" % ev["src"] if args.with_source else ""
    return "L%-6d%s %s" % (ev["line"], src, fmt_ts(ev["ts"]))


def format_event(ev, args):
    head = label(ev, args)
    kind = ev["kind"]
    full = args.full
    if kind == "user":
        return "%s  USER\n%s" % (head, clip(ev.get("text"), full))
    if kind == "assistant":
        phase = ev.get("phase") or "-"
        return "%s  ASSISTANT [%s]\n%s" % (head, phase, clip(ev.get("text"), full))
    if kind == "thinking":
        return "%s  THINKING\n%s" % (head, clip(ev.get("text"), full))
    if kind == "tool_call":
        detail = ev.get("cmd") or ev.get("input") or ""
        return "%s  TOOL %s(%s)  wd=%s\n%s" % (
            head, ev.get("tool"), ",".join(ev.get("apis") or []),
            ev.get("workdir") or "-", clip(detail, full))
    if kind == "tool_result":
        flag = "OK " if ev.get("ok") else "ERR"
        if ev.get("bg"):
            flag = "BG "
        return "%s  RESULT %s\n%s" % (head, flag, clip(ev.get("text"), full))
    if kind == "patch":
        flag = "OK" if ev.get("ok") else "FAILED"
        return "%s  PATCH %s  %s" % (head, flag, ", ".join(ev.get("paths") or []))
    if kind == "mcp":
        return "%s  MCP %s.%s  %s  %s" % (
            head, ev.get("server"), ev.get("mcp_tool"),
            "ok" if ev.get("ok") else "ERR", fmt_dur(ev.get("duration_s")))
    if kind == "websearch":
        return "%s  WEBSEARCH (%d hits)  %s" % (head, ev.get("hits") or 0,
                                                ev.get("query"))
    if kind == "subagent":
        return "%s  SUBAGENT %-9s %s" % (head, ev.get("event_kind"),
                                         ev.get("agent_path"))
    if kind == "agent_msg":
        return "%s  AGENT-MSG %s -> %s\n%s" % (head, ev.get("author"),
                                               ev.get("recipient"),
                                               clip(ev.get("text"), full))
    if kind == "model":
        return "%s  MODEL %s effort=%s" % (head, ev.get("model_name"),
                                           ev.get("effort"))
    if kind == "turn_start":
        return "%s  TURN-START %s" % (head, ev.get("turn_id"))
    if kind == "turn_end":
        return "%s  TURN-END %s  %s" % (head, ev.get("turn_id"),
                                        fmt_dur((ev.get("duration_ms") or 0) / 1000.0))
    if kind == "turn_abort":
        return "%s  TURN-ABORT %s  reason=%s" % (head, ev.get("turn_id"),
                                                 ev.get("reason"))
    if kind == "compact":
        return "%s  COMPACTED (replaced %d items)" % (head, ev.get("replaced") or 0)
    if kind == "compact_evt":
        return "%s  CONTEXT-COMPACTED" % head
    if kind == "goal":
        return "%s  GOAL [%s] %s" % (head, ev.get("status"), ev.get("objective"))
    if kind == "meta":
        fork = ("  forked_from=%s" % ev["forked_from"]) if ev.get("forked_from") else ""
        return "%s  SESSION %s  cwd=%s  cli=%s%s" % (
            head, ev.get("session_id"), ev.get("cwd"), ev.get("cli_version"), fork)
    if kind == "token":
        total = ev.get("total") or {}
        return "%s  TOKENS cum_total=%s" % (head, f"{total.get('total_tokens', 0):,}")
    if kind == "message":
        return "%s  MSG[%s]\n%s" % (head, ev.get("role"), clip(ev.get("text"), full))
    return "%s  %s/%s" % (head, ev.get("rtype"), ev.get("ptype"))


# --------------------------------------------------------------------------
# Modes
# --------------------------------------------------------------------------
def mode_stats(events, args, emit):
    stamped = [e for e in events if e["ts"]]
    start = stamped[0]["ts"] if stamped else None
    end = stamped[-1]["ts"] if stamped else None
    kinds = Counter(e["kind"] for e in events)
    emit("SESSION STATISTICS")
    emit("=" * 40)
    emit("Start:      %s" % fmt_full(start))
    emit("End:        %s" % fmt_full(end))
    if start and end:
        emit("Duration:   %s  (wall)" % fmt_dur((end - start).total_seconds()))
    files = sorted({source_key(e) for e in events})
    emit("Files:      %d" % len(files))
    sessions = {e.get("session_id") for e in events if e["kind"] == "meta"}
    sessions.discard(None)
    forks = {e.get("forked_from") for e in events if e["kind"] == "meta"}
    forks.discard(None)
    emit("Session ids:%d  (declared fork parents: %d)" % (len(sessions), len(forks)))
    emit("Events:     %d" % len(events))
    emit("User msgs:  %d" % kinds.get("user", 0))
    emit("Asst msgs:  %d" % kinds.get("assistant", 0))
    emit("Reasoning:  %d" % kinds.get("thinking", 0))
    emit("Tool calls: %d" % kinds.get("tool_call", 0))
    results = [e for e in events if e["kind"] == "tool_result"]
    emit("Results:    %d (%d failed)" % (len(results),
                                         sum(1 for r in results if not r.get("ok"))))
    emit("Patches:    %d (%d failed)" % (
        kinds.get("patch", 0),
        sum(1 for e in events if e["kind"] == "patch" and not e.get("ok"))))
    emit("Compactions:%d" % kinds.get("compact", 0))

    turns = [e for e in events if e["kind"] == "turn_end"]
    aborts = [e for e in events if e["kind"] == "turn_abort"]
    emit("Turns:      %d completed, %d aborted" % (len(turns), len(aborts)))

    models = Counter(e.get("model_name") for e in events if e["kind"] == "model")
    if models:
        emit("")
        emit("MODELS DECLARED:")
        for name, count in models.most_common():
            emit("  %-24s %d turn_context/settings records" % (name, count))

    apis = Counter()
    for ev in events:
        if ev["kind"] == "tool_call":
            for api in ev.get("apis") or [ev.get("tool")]:
                apis[api] += 1
    if apis:
        emit("")
        emit("TOOL USAGE:")
        for name, count in apis.most_common():
            emit("  %-24s %d" % (name, count))

    reqs = request_usage(events)
    if reqs:
        emit("")
        emit("BILLED REQUESTS: %d  (see --cache for tokens and cost)" % len(reqs))
    return 0


def mode_timeline(events, args, emit):
    for ev in events:
        if ev["kind"] in ("world", "other", "reasoning_raw", "inter_agent"):
            continue
        text = format_event(ev, args)
        emit(text.replace("\n", " | ") if not args.full else text)
    return 0


def mode_commands(events, args, emit):
    n = 0
    for ev in events:
        if ev["kind"] != "tool_call":
            continue
        cmd = ev.get("cmd")
        if not cmd:
            continue
        n += 1
        result = ev.get("result")
        tag = ""
        if result is not None:
            if result.get("bg"):
                tag = "  [backgrounded]"
            elif not result.get("ok"):
                tag = "  [FAILED]"
        emit("%s  wd=%s%s" % (label(ev, args), ev.get("workdir") or "-", tag))
        emit(clip(cmd, args.full, 2000))
        emit("")
    emit("%d commands" % n)
    return 0


def _patch_stats(diff):
    plus = minus = 0
    for line in (diff or "").splitlines():
        if line.startswith("+") and not line.startswith("+++"):
            plus += 1
        elif line.startswith("-") and not line.startswith("---"):
            minus += 1
    return plus, minus


def mode_files(events, args, emit):
    touched = defaultdict(Counter)
    for ev in events:
        if ev["kind"] != "patch":
            continue
        for path, change in (ev.get("changes") or {}).items():
            kind = change.get("type") if isinstance(change, dict) else "?"
            touched[path][kind] += 1
    if not touched:
        emit("no patched files")
        return 0
    emit("%-70s %s" % ("path", "changes"))
    for path in sorted(touched):
        counts = ", ".join("%s x%d" % (k, v) for k, v in touched[path].most_common())
        emit("%-70s %s" % (path, counts))
    emit("")
    emit("%d distinct paths" % len(touched))
    return 0


def _patch_mode(events, args, emit, wanted, title):
    n = 0
    for ev in events:
        if ev["kind"] != "patch":
            continue
        for path, change in sorted((ev.get("changes") or {}).items()):
            if not isinstance(change, dict) or change.get("type") not in wanted:
                continue
            n += 1
            diff = change.get("unified_diff") or ""
            plus, minus = _patch_stats(diff)
            emit("%s  %s  %s  (+%d/-%d)%s" % (
                label(ev, args), (change.get("type") or "?").upper(), path, plus, minus,
                "" if ev.get("ok") else "  [PATCH FAILED]"))
            if diff:
                emit(clip(diff, args.full, 600))
            emit("")
    emit("%d %s" % (n, title))
    return 0


def mode_writes(events, args, emit):
    return _patch_mode(events, args, emit, {"add"}, "file creations")


def mode_edits(events, args, emit):
    return _patch_mode(events, args, emit, {"update", "delete"}, "file edits/deletes")


def mode_cache(events, args, emit):
    reqs = request_usage(events)
    if not reqs:
        emit("no token_count records")
        return 0
    override = parse_rates(args.rates) if args.rates else None

    counts = defaultdict(Counter)   # model -> token/request tallies (ints)
    costs = defaultdict(float)      # model -> USD
    for req in reqs:
        model = req.get("model") or "(unattributed)"
        rates = dict(rates_for_model(req.get("model")))
        if override:
            rates.update(override)
        cost, fresh, long_ctx = price_request(req["usage"], rates)
        tally = counts[model]
        tally["n"] += 1
        tally["fresh"] += fresh
        tally["long"] += 1 if long_ctx else 0
        for key in USAGE_KEYS:
            tally[key] += req["usage"].get(key, 0)
        costs[model] += cost

    emit("TOKEN USAGE  (per-request deltas of the cumulative counter)")
    emit("=" * 72)
    total_cost = 0.0
    grand = Counter()
    for model in sorted(counts):
        tally = counts[model]
        total_in = tally["input_tokens"]
        emit("")
        emit("MODEL %s   (%d requests)" % (model, tally["n"]))
        emit("  input total:         %14s" % f"{total_in:,}")
        emit("    fresh:             %14s" % f"{tally['fresh']:,}")
        emit("    cache read:        %14s" % f"{tally['cached_input_tokens']:,}")
        emit("    cache write:       %14s" % f"{tally['cache_write_input_tokens']:,}")
        emit("  output:              %14s" % f"{tally['output_tokens']:,}")
        emit("    of which reasoning:%14s" % f"{tally['reasoning_output_tokens']:,}")
        if total_in:
            emit("  cache hit rate:      %13.1f%%"
                 % (100.0 * tally["cached_input_tokens"] / total_in))
        if tally["long"]:
            emit("  long-context requests (>%s input): %d"
                 % (f"{LONG_CONTEXT_THRESHOLD:,}", tally["long"]))
        base = rates_for_model(None if model == "(unattributed)" else model)
        rates = dict(base)
        if override:
            rates.update(override)
        emit("  rates USD/1M: in=%.2f cached=%.2f write=%.2f out=%.2f  [%s]" % (
            rates["input"], rates["cached"], rates["cache_write"], rates["output"],
            "--rates override" if override else rate_label(base)))
        emit("  COST:                %14s" % ("$%.4f" % costs[model]))
        total_cost += costs[model]
        grand.update(tally)

    emit("")
    emit("-" * 72)
    emit("TOTAL   requests=%d  input=%s (fresh %s)  output=%s" % (
        grand["n"], f"{grand['input_tokens']:,}", f"{grand['fresh']:,}",
        f"{grand['output_tokens']:,}"))
    emit("TOTAL COST (%s): $%.4f"
         % ("--rates override" if override else "API list rates", total_cost))
    if grand["n"]:
        emit("  per request: $%.4f" % (total_cost / grand["n"]))
    if not override:
        emit("")
        emit("Priced at public API rates by design. These sessions ran on a")
        emit("ChatGPT subscription, so this is what the same work would have")
        emit("cost through the API — not an invoice line.")
    return 0


def mode_context_growth(events, args, emit):
    reqs = request_usage(events)
    if not reqs:
        emit("no token_count records")
        return 0
    emit("%-6s %-8s %-16s %14s %12s %12s" % ("#", "time", "model", "context in",
                                             "output", "delta in"))
    prev = None
    for i, req in enumerate(reqs, 1):
        size = (req["cumulative"] or {}).get("input_tokens", 0)
        # Context fed to THIS request = its own input delta is not the context;
        # the request's prompt size is the delta's input_tokens.
        prompt = req["usage"].get("input_tokens", 0)
        delta = "" if prev is None else "%+d" % (prompt - prev)
        prev = prompt
        emit("%-6d %-8s %-16s %14s %12s %12s" % (
            i, fmt_ts(req["ts"]), (req.get("model") or "-")[:16],
            f"{prompt:,}", f"{req['usage'].get('output_tokens', 0):,}", delta))
        del size
    return 0


def mode_turns(events, args, emit):
    starts = {}
    rows = []
    for ev in events:
        if ev["kind"] == "turn_start":
            starts[ev.get("turn_id")] = ev
        elif ev["kind"] in ("turn_end", "turn_abort"):
            begin = starts.get(ev.get("turn_id"))
            rows.append((ev, begin))
    if not rows:
        emit("no turn records")
        return 0
    emit("%-8s %-38s %10s %10s  %s" % ("time", "turn_id", "duration", "ttft", "state"))
    for ev, _begin in rows:
        state = "complete" if ev["kind"] == "turn_end" else "ABORTED(%s)" % ev.get("reason")
        emit("%-8s %-38s %10s %10s  %s" % (
            fmt_ts(ev["ts"]), ev.get("turn_id"),
            fmt_dur((ev.get("duration_ms") or 0) / 1000.0),
            fmt_dur((ev.get("ttft_ms") or 0) / 1000.0) if ev.get("ttft_ms") else "-",
            state))
    done = sum(1 for e, _ in rows if e["kind"] == "turn_end")
    emit("")
    emit("%d turns (%d completed, %d aborted)" % (len(rows), done, len(rows) - done))
    return 0


def mode_latency(events, args, emit):
    rows = []
    for ev in events:
        if ev["kind"] != "tool_call":
            continue
        result = ev.get("result")
        if not result or not ev["ts"] or not result["ts"]:
            continue
        rows.append(((result["ts"] - ev["ts"]).total_seconds(), ev, result))
    rows.sort(key=lambda r: -r[0])
    if not rows:
        emit("no paired tool calls")
        return 0
    emit("%10s  %-8s  %s" % ("latency", "time", "call"))
    for secs, ev, _result in rows:
        emit("%10s  %-8s  L%-6d %s(%s) %s" % (
            fmt_dur(secs), fmt_ts(ev["ts"]), ev["line"], ev.get("tool"),
            ",".join(ev.get("apis") or []),
            clip(ev.get("cmd") or ev.get("input"), False, 90)))
    return 0


def mode_reply_latency(events, args, emit):
    # `prompt` is only ever the `pending` event the elif above proved is not
    # None; pylint infers it over the whole function, where it is also None.
    # pylint: disable=unsubscriptable-object
    rows = []
    pending = None
    for ev in events:
        if ev["kind"] == "user":
            pending = ev
        elif pending is not None and ev["kind"] in ("thinking", "assistant", "tool_call"):
            prompt = pending
            if ev["ts"] and prompt["ts"]:
                rows.append(((ev["ts"] - prompt["ts"]).total_seconds(), prompt, ev))
            pending = None
    if not rows:
        emit("no user->response pairs")
        return 0
    emit("%10s  %-8s  %s" % ("gap", "time", "prompt"))
    for secs, user, _first in rows:
        emit("%10s  %-8s  L%-6d %s" % (fmt_dur(secs), fmt_ts(user["ts"]),
                                       user["line"], clip(user.get("text"), False, 90)))
    total = sum(r[0] for r in rows)
    emit("")
    emit("%d replies, mean %s" % (len(rows), fmt_dur(total / len(rows))))
    return 0


def mode_shell_errors(events, args, emit):
    n = 0
    for ev in events:
        if ev["kind"] == "tool_result" and not ev.get("ok"):
            n += 1
            call = ev.get("call")
            emit("%s  FAILED" % label(ev, args))
            if call:
                emit("  cmd: %s" % clip(call.get("cmd") or call.get("input"), False, 400))
            emit("  out: %s" % clip(ev.get("text"), args.full, 600))
            emit("")
        elif ev["kind"] == "patch" and not ev.get("ok"):
            n += 1
            emit("%s  PATCH FAILED  %s" % (label(ev, args), ", ".join(ev.get("paths") or [])))
            if ev.get("stderr"):
                emit("  %s" % clip(ev["stderr"], args.full, 400))
            emit("")
    emit("%d failures" % n)
    return 0


def mode_tool_orphans(events, args, emit):
    n = 0
    for ev in events:
        if ev["kind"] == "tool_call" and not ev.get("result"):
            n += 1
            emit("%s  NO RESULT  %s(%s)  %s" % (
                label(ev, args), ev.get("tool"), ",".join(ev.get("apis") or []),
                clip(ev.get("cmd") or ev.get("input"), False, 120)))
    emit("")
    emit("%d tool calls without a result" % n)
    return 0


def mode_subagents(events, args, emit):
    lifecycles = defaultdict(list)
    for ev in events:
        if ev["kind"] == "subagent":
            lifecycles[ev.get("agent_path") or "(unnamed)"].append(ev)
    spawns = [e for e in events if e["kind"] == "tool_call"
              and e.get("tool") in ("spawn_agent", "wait_agent", "send_message",
                                    "interrupt_agent", "followup_task")]
    msgs = [e for e in events if e["kind"] == "agent_msg"]

    if not lifecycles and not spawns and not msgs:
        emit("no subagent activity")
        return 0
    emit("SUBAGENT ACTIVITY")
    emit("=" * 60)
    for path in sorted(lifecycles):
        evs = lifecycles[path]
        kinds = Counter(e.get("event_kind") for e in evs)
        span = ""
        stamped = [e["ts"] for e in evs if e["ts"]]
        if len(stamped) > 1:
            span = "  span=%s" % fmt_dur((max(stamped) - min(stamped)).total_seconds())
        emit("%-44s %s%s" % (path, dict(kinds), span))
    emit("")
    emit("multi-agent tool calls: %s" % dict(Counter(e.get("tool") for e in spawns)))
    emit("inter-agent messages:   %d" % len(msgs))
    return 0


def mode_forks(events, args, emit):
    """Lineage report — the reason per-file cost sums lie."""
    metas = [e for e in events if e["kind"] == "meta"]
    if not metas:
        emit("no session_meta records")
        return 0
    by_file = {}
    for ev in metas:
        by_file.setdefault(source_key(ev), ev)
    by_id = Counter(ev.get("session_id") for ev in by_file.values())
    children = defaultdict(list)
    for key, ev in by_file.items():
        if ev.get("forked_from"):
            children[ev["forked_from"]].append(key)

    emit("FORK / LINEAGE REPORT")
    emit("=" * 60)
    emit("rollout files:      %d" % len(by_file))
    emit("distinct session ids:%d" % len(by_id))
    shared = {sid: n for sid, n in by_id.items() if n > 1}
    if shared:
        emit("")
        emit("session ids spanning MORE THAN ONE FILE (a fork keeps the parent id):")
        for sid, n in sorted(shared.items(), key=lambda kv: -kv[1]):
            emit("  %s  in %d files" % (sid, n))
    roots = [key for key, ev in by_file.items() if not ev.get("forked_from")]
    emit("")
    emit("root (non-forked) files: %d" % len(roots))
    for key in sorted(roots):
        emit("  %s" % by_file[key]["src"])
    if children:
        emit("")
        emit("fork parents:")
        for parent, kids in sorted(children.items(), key=lambda kv: -len(kv[1])):
            emit("  %s  <- %d file(s)" % (parent, len(kids)))

    inherited = []
    for key in sorted(by_file):
        evs = [e for e in events if e["kind"] == "token" and source_key(e) == key]
        if not evs:
            continue
        evs.sort(key=lambda e: e["line"])
        first = evs[0]
        base = (first.get("total") or {}).get("total_tokens", 0) - \
               (first.get("last") or {}).get("total_tokens", 0)
        if base > 0:
            inherited.append((by_file[key]["src"], base))
    if inherited:
        emit("")
        emit("files whose cumulative counter STARTS non-zero (inherited from parent):")
        for src, base in sorted(inherited, key=lambda r: -r[1]):
            emit("  %-40s baseline %s tokens" % (src, f"{base:,}"))
        emit("")
        emit("Summing final total_token_usage per file would double-count these")
        emit("baselines. --cache differences the counter, so it does not.")
    return 0


def mode_rate_limits(events, args, emit):
    rows = []
    for ev in events:
        if ev["kind"] != "token":
            continue
        rl = ev.get("rate_limits") or {}
        primary = rl.get("primary") or {}
        if primary.get("used_percent") is None:
            continue
        rows.append((ev["ts"], primary.get("used_percent"),
                     primary.get("window_minutes"), rl.get("plan_type")))
    if not rows:
        emit("no rate_limits records")
        return 0
    emit("%-20s %8s  %-10s %s" % ("time", "used", "window", "plan"))
    last = None
    for ts, pct, window, plan in rows:
        key = (round(pct, 1), plan)
        if key == last:
            continue
        last = key
        emit("%-20s %7.1f%%  %-10s %s" % (
            fmt_full(ts), pct,
            "%dm" % window if window else "-", plan))
    emit("")
    emit("plans seen: %s" % dict(Counter(r[3] for r in rows)))
    return 0


def mode_mcp(events, args, emit):
    n = 0
    for ev in events:
        if ev["kind"] != "mcp":
            continue
        n += 1
        emit("%s  %s.%s  %s  %s" % (
            label(ev, args), ev.get("server"), ev.get("mcp_tool"),
            "ok" if ev.get("ok") else "ERR", fmt_dur(ev.get("duration_s"))))
        if ev.get("args"):
            emit("  args: %s" % clip(json.dumps(ev["args"]), args.full, 300))
    emit("")
    emit("%d MCP calls" % n)
    return 0


def mode_websearch(events, args, emit):
    n = 0
    for ev in events:
        if ev["kind"] != "websearch":
            continue
        n += 1
        emit("%s  (%d hits)  %s" % (label(ev, args), ev.get("hits") or 0,
                                    ev.get("query")))
    emit("")
    emit("%d web searches" % n)
    return 0


def mode_compact(events, args, emit):
    n = 0
    for ev in events:
        if ev["kind"] in ("compact", "compact_evt"):
            n += 1
            emit(format_event(ev, args))
    emit("")
    emit("%d compaction events" % n)
    return 0


def mode_pairs(events, args, emit):
    for ev in events:
        if ev["kind"] != "tool_call":
            continue
        emit(format_event(ev, args))
        result = ev.get("result")
        if result is None:
            emit("  -> (no result)")
        else:
            emit("  -> %s" % format_event(result, args))
        emit("")
    return 0


def mode_errors(events, args, emit):
    return mode_shell_errors(events, args, emit)


def mode_last_assistant_message(events, args, emit):
    for ev in reversed(events):
        if ev["kind"] == "assistant" and (ev.get("text") or "").strip():
            emit(ev["text"])
            return 0
    return 0


MODES = {
    "stats": mode_stats,
    "timeline": mode_timeline,
    "commands": mode_commands,
    "files": mode_files,
    "writes": mode_writes,
    "edits": mode_edits,
    "cache": mode_cache,
    "context_growth": mode_context_growth,
    "turns": mode_turns,
    "latency": mode_latency,
    "reply_latency": mode_reply_latency,
    "shell_errors": mode_shell_errors,
    "tool_orphans": mode_tool_orphans,
    "subagents": mode_subagents,
    "forks": mode_forks,
    "rate_limits": mode_rate_limits,
    "mcp": mode_mcp,
    "websearch": mode_websearch,
    "compact": mode_compact,
    "pairs": mode_pairs,
    "errors": mode_errors,
    "last_assistant_message": mode_last_assistant_message,
}

EVENT_FILTER_FLAGS = ("tools", "results", "user", "assistant", "thinking",
                      "conversation")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def build_parser():
    ap = argparse.ArgumentParser(
        prog="parse_codex.py",
        description="Parse Codex CLI rollout JSONL for investigation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=EPILOG)
    ap.add_argument("rollout", nargs="?",
                    help="rollout .jsonl, a session id, or a directory "
                         "(recursively joined, sorted by timestamp)")
    ap.add_argument("--version", action="store_true",
                    help="print version and exit")

    g = ap.add_argument_group("event type filters (union if multiple)")
    g.add_argument("--tools", action="store_true", help="tool calls")
    g.add_argument("--results", action="store_true", help="tool results only")
    g.add_argument("--user", action="store_true", help="user messages only")
    g.add_argument("--assistant", action="store_true", help="assistant text only")
    g.add_argument("--thinking", action="store_true", help="reasoning summaries only")
    g.add_argument("--conversation", action="store_true", help="user + assistant text")

    m = ap.add_argument_group("analysis modes")
    m.add_argument("--stats", action="store_true", help="session statistics")
    m.add_argument("--timeline", action="store_true", help="chronological one-liners")
    m.add_argument("--commands", action="store_true",
                   help="shell commands from tools.exec_command")
    m.add_argument("--files", action="store_true", help="files touched by patches")
    m.add_argument("--writes", action="store_true", help="file creations (patch add)")
    m.add_argument("--edits", action="store_true",
                   help="file edits/deletes with unified diffs")
    m.add_argument("--cache", action="store_true",
                   help="token totals + API-rate cost, per model")
    m.add_argument("--context-growth", action="store_true",
                   help="prompt size per request over time")
    m.add_argument("--turns", action="store_true",
                   help="per-turn duration, time-to-first-token, aborts")
    m.add_argument("--latency", action="store_true",
                   help="rank tool calls by wall-clock latency")
    m.add_argument("--reply-latency", action="store_true",
                   help="gap from user message to first response event")
    m.add_argument("--shell-errors", action="store_true",
                   help="failed exec calls and failed patches")
    m.add_argument("--tool-orphans", action="store_true",
                   help="tool calls with no matching output")
    m.add_argument("--subagents", action="store_true",
                   help="sub-agent lifecycle and inter-agent messages")
    m.add_argument("--forks", action="store_true",
                   help="lineage report: shared session ids, inherited counters")
    m.add_argument("--rate-limits", action="store_true",
                   help="plan type and rate-limit utilization over time")
    m.add_argument("--mcp", action="store_true", help="MCP tool calls")
    m.add_argument("--websearch", action="store_true", help="web searches")
    m.add_argument("--compact", action="store_true", help="compaction events")
    m.add_argument("--pairs", action="store_true", help="tool calls with results")
    m.add_argument("--errors", action="store_true", help="failures only")
    m.add_argument("--last-assistant-message", action="store_true",
                   help="last assistant text, verbatim")

    ap.add_argument("--tool", metavar="NAME",
                    help="filter by tool/API name (comma-separated), e.g. "
                         "exec_command, apply_patch, spawn_agent")
    ap.add_argument("--rates", metavar="SPEC",
                    help='override pricing: "input=5,cached=0.5,'
                         'cache_write=6.25,output=30" (USD per 1M tokens)')
    ap.add_argument("--path", metavar="PATTERN",
                    help="filter events by file path / command substring")
    ap.add_argument("--range", metavar="N-M", help="JSONL line range")
    ap.add_argument("--after", metavar="WHEN",
                    help="events at/after WHEN (HH:MM:SS, ISO, or Nd/Nh/Nw)")
    ap.add_argument("--before", metavar="WHEN", help="events at/before WHEN")
    ap.add_argument("--grep", metavar="PATTERN",
                    help="regex filter on formatted output (block-level)")
    ap.add_argument("--line-grep", metavar="PATTERN",
                    help="raw-line prefilter, applied before JSON parsing "
                         "(only worth it in directory mode)")
    ap.add_argument("--full", "-v", "--verbose", action="store_true",
                    help="disable truncation")
    ap.add_argument("-o", "--output", metavar="FILE", help="write to file")
    ap.add_argument("--with-source", action="store_true",
                    help="show source file basename on every event")
    ap.add_argument("--timing", action="store_true",
                    help="print per-phase wall time to stderr")
    return ap


EPILOG = """
Codex rollout parser — sibling to parse_session.py (Claude) and parse_wire.py
(Kimi). Reads ~/.codex/sessions/YYYY/MM/DD/rollout-*.jsonl.

COST
    --cache prices every request at PUBLIC API LIST RATES, always, including
    sessions that ran on a ChatGPT subscription. Per-request usage comes from
    differencing the cumulative token counter, never from summing snapshots:
    forks inherit their parent's running total and duplicate token_count
    events repeat the previous reading, so both a per-file sum and a
    last_token_usage sum overstate the bill.

    Rates default by model (Sol / Terra / Luna, post-2026-07-30 pricing);
    requests over 272,000 input tokens bill the whole request at the
    long-context meter (2x input, 1.5x output). Override with --rates.

    Model attribution comes from the most recent turn_context, because
    token_count records carry no model of their own.

FORKS
    --forks shows which rollout files share a session id and which start with
    an inherited counter. Read it before quoting any per-file total.

Examples:
    parse_codex.py <file|id|dir> --stats
    parse_codex.py <file> --cache
    parse_codex.py ~/.codex/sessions --forks
    parse_codex.py <file> --commands --after 10:25:00
    parse_codex.py <file> --tools --tool apply_patch
    parse_codex.py <file> --edits --full
    parse_codex.py <file> --cache -o cost.txt --and --turns
"""


def selected_mode(args):
    chosen = [name for name in MODES if getattr(args, name, False)]
    if len(chosen) > 1:
        raise SystemExit("error: pick ONE analysis mode (got %s)" %
                         ", ".join("--" + c.replace("_", "-") for c in chosen))
    return chosen[0] if chosen else None


def selected_filters(args):
    return {name for name in EVENT_FILTER_FLAGS if getattr(args, name, False)}


def run_segment(events, args, stream):
    lines = []

    def emit(text=""):
        lines.append(text)

    mode = selected_mode(args)
    view = apply_filters(events, args)

    if mode:
        MODES[mode](view, args, emit)
    else:
        kinds = set()
        for name in selected_filters(args):
            kinds |= KIND_FILTERS[name]
        if not kinds:
            kinds = {"user", "assistant", "thinking", "tool_call", "tool_result",
                     "patch", "mcp", "websearch", "subagent", "compact", "meta",
                     "model", "goal", "agent_msg", "turn_start", "turn_end",
                     "turn_abort"}
        for ev in view:
            if ev["kind"] in kinds:
                emit(format_event(ev, args))
                emit("")

    text = "\n".join(lines)
    if args.grep:
        pattern = re.compile(args.grep)
        blocks = text.split("\n\n")
        text = "\n\n".join(b for b in blocks if pattern.search(b))

    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
        print("wrote %s (%d lines)" % (args.output, len(text.splitlines())),
              file=stream)
    else:
        print(text, file=stream)
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
        print("parse_codex.py %s" % __version__)
        return 0
    if not head.rollout:
        ap.error("a rollout file, session id, or directory is required")

    t0 = time.time()
    files = resolve_target(head.rollout)
    events = extract_events(files, head.line_grep)
    backfill_models(events)
    link_pairs(events)
    if head.timing:
        print("load: %.2fs  (%d files, %d events)" %
              (time.time() - t0, len(files), len(events)), file=sys.stderr)

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
            args = ap.parse_args([head.rollout] + segment)
            if args.version:
                print("parse_codex.py %s" % __version__)
                continue
        if len(segments) > 1 and not args.output:
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
    if head.timing:
        print("total: %.2fs" % (time.time() - t0), file=sys.stderr)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
