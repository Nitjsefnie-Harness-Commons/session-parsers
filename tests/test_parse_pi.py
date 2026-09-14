"""Suite for parse_pi.py — the Pi coding-agent session parser.

The fixtures are synthetic entries in the real schema, because every number
here is hand-computed against the three traps the module's docstring names:
`usage.reasoning` lives outside both `output` and `totalTokens`; usage is per
request, not per turn; tool results are top-level `toolResult` messages paired
by `toolCallId`. Each trap gets a test whose expected value would differ under
the naive reading.
"""
import contextlib
import io
import json
import os

import _util

_PP = _util.load(os.path.join(_util.SCRIPTS, "parse_pi.py"))

# pylint: disable=protected-access

T0 = "2026-08-29T10:00:00.000Z"
T1 = "2026-08-29T10:00:05.000Z"
T2 = "2026-08-29T10:00:12.000Z"
T3 = "2026-08-29T10:02:00.000Z"

# The live example from the module docstring: 132 + 88 + 2048 = 2268 exactly,
# so totalTokens has dropped the 69 reasoning tokens.
USAGE = {"input": 132, "output": 88, "cacheRead": 2048, "cacheWrite": 0,
         "reasoning": 69, "totalTokens": 2268, "cost": {"total": 0.0123}}


def _session(ts=T0, sid="sess-1"):
    return {"type": "session", "timestamp": ts, "id": sid, "cwd": "/work",
            "version": "0.9.0"}


def _user(ts, text):
    return {"type": "message", "timestamp": ts,
            "message": {"role": "user", "content": text}}


_LIVE = object()  # "the docstring's usage"; None means a usage-less message


def _assistant(ts, parts, usage=_LIVE, stop="stop", error=None,
               model="anthropic/claude-x", response_id="resp-1"):
    if usage is _LIVE:
        usage = USAGE
    msg = {"role": "assistant", "content": parts, "model": model,
           "provider": "anthropic", "stopReason": stop, "usage": usage,
           "responseId": response_id}
    if error is not None:
        msg["errorMessage"] = error
    return {"type": "message", "timestamp": ts, "message": msg}


def _call(call_id, name, arguments):
    return {"type": "toolCall", "id": call_id, "name": name,
            "arguments": arguments}


def _result(ts, call_id, text, is_error=False, error=None, tool="bash"):
    msg = {"role": "toolResult", "toolCallId": call_id, "toolName": tool,
           "content": [{"type": "text", "text": text}], "isError": is_error}
    if error is not None:
        msg["errorMessage"] = error
    return {"type": "message", "timestamp": ts, "message": msg}


def _write(tmpdir, name, records):
    path = os.path.join(tmpdir, name)
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(rec if isinstance(rec, str) else json.dumps(rec))
            fh.write("\n")
    return path


def _corpus():
    """One session exercising every entry type and both orphan directions."""
    return [
        _session(),
        {"type": "model_change", "timestamp": T0, "modelId": "claude-x",
         "provider": "anthropic"},
        {"type": "thinking_level_change", "timestamp": T0,
         "thinkingLevel": "high"},
        _user(T0, "list the files"),
        _assistant(T1, [
            {"type": "thinking", "thinking": "plan it", "thinkingSignature": "sig"},
            {"type": "text", "text": "Listing."},
            _call("c1", "bash", {"command": "ls"}),
            _call("c2", "bash", {"command": "never answered"}),
        ], stop="toolUse"),
        _result(T2, "c1", "a.py b.py"),
        _result(T2, "ghost", "no such call", is_error=True, error="lost"),
        {"type": "custom_message", "timestamp": T2, "customType": "monitor",
         "content": [{"type": "text", "text": "deliver"}], "display": True},
        {"type": "compaction", "timestamp": T2, "tokensBefore": 5000,
         "summary": "squashed"},
        _assistant(T3, [{"type": "text", "text": "Done."}],
                   usage=dict(USAGE, input=200, reasoning=0, cost={}),
                   model="openai/gpt-y"),
        _assistant(T3, [], usage=None, stop="error", error="boom"),
        _assistant(T3, [], usage=None, stop="aborted", error="ctrl-c"),
        {"type": "weird", "timestamp": T3},
        {"type": "message", "timestamp": T3, "message": {"role": "system"}},
        "{not json",
        "[1, 2]",
    ]


def _events(tmp, records=None, line_grep=None):
    path = _write(tmp, "s.jsonl", _corpus() if records is None else records)
    return path, _PP.extract_events([path], line_grep)


def _capture(mode, events, tmp, **flags):
    args = _PP.build_parser().parse_args([])
    for key, value in flags.items():
        setattr(args, key, value)
    lines = []
    _PP.MODES[mode](events, args, lines.append)
    return "\n".join(lines)


# ---------------------------------------------------------------- helpers
def test_parse_ts_accepts_z_naive_and_refuses_garbage(tmp):
    aware = _PP.parse_ts(T0)
    assert aware.tzinfo is not None
    naive = _PP.parse_ts("2026-08-29T10:00:00")
    assert naive.utcoffset().total_seconds() == 0
    assert _PP.parse_ts("") is None
    assert _PP.parse_ts("yesterday") is None


def test_fmt_dur_picks_the_unit_by_magnitude(tmp):
    assert _PP.fmt_dur(None) == "?"
    assert _PP.fmt_dur(45) == "45s"
    assert _PP.fmt_dur(125) == "2m05s"
    assert _PP.fmt_dur(7300) == "2h01m"


def test_fmt_ts_and_fmt_full_render_absence(tmp):
    assert _PP.fmt_ts(None) == "?"
    assert _PP.fmt_full(None) == "(none)"
    assert _PP.fmt_ts(_PP.parse_ts(T0)) == "08-29 10:00:00"
    assert _PP.fmt_full(_PP.parse_ts(T0)).startswith("2026-08-29 10:00:00")


def test_clip_collapses_whitespace_and_truncates(tmp):
    assert _PP.clip("a  b\n c") == "a b c"
    long = "x" * 200
    assert _PP.clip(long) == "x" * 93 + "..."
    assert _PP.clip(long, full=True) == long
    assert _PP.clip(long, width=10) == "xxxxxxx..."
    assert _PP.clip(None) == ""


def test_format_int_groups_thousands(tmp):
    assert _PP.format_int(1234567) == "1,234,567"
    assert _PP.format_int(None) == "0"


# ---------------------------------------------------------------- trap 1
def test_response_adds_reasoning_to_output(tmp):
    assert _PP._response(USAGE) == 88 + 69


def test_grand_total_adds_reasoning_back_past_total_tokens(tmp):
    assert _PP._grand_total(USAGE) == 2268 + 69
    assert _PP._grand_total(USAGE) != USAGE["totalTokens"]


def test_context_after_is_prompt_plus_response(tmp):
    assert _PP._context_after(USAGE) == 132 + 2048 + 0 + 88 + 69


def test_missing_usage_keys_count_as_zero(tmp):
    assert _PP._response({}) == 0
    assert _PP._grand_total({"input": None}) == 0


# ---------------------------------------------------------------- extraction
def test_parts_text_accepts_string_and_part_lists(tmp):
    assert _PP._parts_text("plain") == "plain"
    parts = [{"type": "text", "text": "a"}, "junk",
             {"type": "thinking", "thinking": "b"}, {"type": "image"},
             {"type": "text", "text": ""}]
    assert _PP._parts_text(parts) == "a\nb"
    assert _PP._parts_text(None) == ""


def test_every_entry_type_becomes_its_own_event_kind(tmp):
    _, evs = _events(tmp)
    kinds = [e["kind"] for e in evs]
    assert kinds.count("meta") == 1
    assert kinds.count("model") == 1
    assert kinds.count("thinking_level") == 1
    assert kinds.count("user") == 1
    assert kinds.count("assistant") == 4
    assert kinds.count("thinking") == 1
    assert kinds.count("tool_call") == 2
    assert kinds.count("tool_result") == 2
    assert kinds.count("custom") == 1
    assert kinds.count("compact") == 1
    # unknown type, unknown role, and the unparseable line; the JSON array
    # line is a non-dict and is dropped without an event.
    assert kinds.count("other") == 3


def test_meta_event_carries_identity(tmp):
    _, evs = _events(tmp)
    meta = next(e for e in evs if e["kind"] == "meta")
    assert meta["session_id"] == "sess-1"
    assert meta["cwd"] == "/work"
    assert meta["version"] == "0.9.0"
    assert meta["line"] == 1


def test_assistant_event_derives_all_three_token_figures(tmp):
    _, evs = _events(tmp)
    first = next(e for e in evs if e["kind"] == "assistant")
    assert first["response"] == 157
    assert first["total"] == 2337
    assert first["context_after"] == 2337
    assert first["cost"] == 0.0123
    assert first["stop_reason"] == "toolUse"
    assert first["error"] is None
    assert first["text"] == "plan it\nListing."


def test_assistant_without_usage_has_no_token_figures(tmp):
    _, evs = _events(tmp)
    errored = [e for e in evs if e["kind"] == "assistant" and e["stop_reason"] == "error"]
    assert len(errored) == 1
    assert errored[0]["usage"] is None
    assert errored[0]["response"] is None
    assert errored[0]["total"] is None
    assert "cost" not in errored[0]
    assert errored[0]["error"] == "boom"


def test_error_message_is_kept_only_for_failed_stops(tmp):
    ok = _assistant(T1, [], stop="stop", error="stale text")
    _, evs = _events(tmp, [ok])
    assert evs[0]["error"] is None


def test_tool_call_and_result_events_carry_pairing_keys(tmp):
    _, evs = _events(tmp)
    call = next(e for e in evs if e["kind"] == "tool_call")
    assert call["call_id"] == "c1"
    assert call["tool"] == "bash"
    assert call["arguments"] == {"command": "ls"}
    assert call["response_id"] == "resp-1"
    results = [e for e in evs if e["kind"] == "tool_result"]
    assert results[0]["ok"] is True and results[0]["text"] == "a.py b.py"
    assert results[1]["ok"] is False and results[1]["error"] == "lost"


def test_compaction_custom_and_change_events(tmp):
    _, evs = _events(tmp)
    compact = next(e for e in evs if e["kind"] == "compact")
    assert compact["summary_tokens"] == 5000 and compact["detail"] == "squashed"
    custom = next(e for e in evs if e["kind"] == "custom")
    assert custom["custom_type"] == "monitor" and custom["text"] == "deliver"
    model = next(e for e in evs if e["kind"] == "model")
    assert model["model_id"] == "claude-x" and model["provider"] == "anthropic"
    level = next(e for e in evs if e["kind"] == "thinking_level")
    assert level["level"] == "high"


def test_unparseable_line_is_an_other_event_with_its_line_number(tmp):
    _, evs = _events(tmp)
    notes = [(e["line"], e["note"]) for e in evs if e["kind"] == "other"]
    assert (15, "unparseable line") in notes
    assert (13, "type=weird") in notes
    assert (14, "role=system") in notes


def test_line_grep_skips_lines_before_parsing(tmp):
    _, evs = _events(tmp, line_grep="toolResult")
    assert {e["kind"] for e in evs} == {"tool_result"}


def test_unreadable_file_is_skipped(tmp):
    evs = _PP.extract_events([os.path.join(tmp, "missing.jsonl")])
    assert evs == []


def test_files_are_read_in_the_order_given(tmp):
    a = _write(tmp, "a.jsonl", [_user(T0, "first")])
    b = _write(tmp, "b.jsonl", [_user(T1, "second")])
    evs = _PP.extract_events([b, a])
    assert [e["text"] for e in evs] == ["second", "first"]
    assert evs[0]["file"] == b


# ---------------------------------------------------------------- trap 3
def test_link_pairs_separates_the_two_orphan_directions(tmp):
    _, evs = _events(tmp)
    paired, unmatched, orphans = _PP.link_pairs(evs)
    assert paired == 1
    assert list(unmatched) == ["c2"]
    assert [r["call_id"] for r in orphans] == ["ghost"]
    call = next(e for e in evs if e.get("call_id") == "c1")
    assert call["result"]["text"] == "a.py b.py"


def test_link_pairs_ignores_calls_without_ids(tmp):
    evs = [{"kind": "tool_call", "call_id": None},
           {"kind": "tool_result", "call_id": None}]
    paired, unmatched, orphans = _PP.link_pairs(evs)
    assert paired == 0 and unmatched == {} and len(orphans) == 1


# ---------------------------------------------------------------- filters
def test_apply_filters_is_a_union_and_defaults_to_everything(tmp):
    _, evs = _events(tmp)
    args = _PP.build_parser().parse_args([])
    assert _PP.apply_filters(evs, args) is evs
    args = _PP.build_parser().parse_args(["--user", "--custom"])
    assert {e["kind"] for e in _PP.apply_filters(evs, args)} == {"user", "custom"}
    for flag, kind in (("--tools", "tool_call"), ("--results", "tool_result"),
                       ("--assistant", "assistant"), ("--thinking", "thinking")):
        args = _PP.build_parser().parse_args([flag])
        assert {e["kind"] for e in _PP.apply_filters(evs, args)} == {kind}, flag


def test_label_renders_every_kind(tmp):
    _, evs = _events(tmp)
    args = _PP.build_parser().parse_args([])
    by_kind = {}
    for e in evs:
        by_kind.setdefault(e["kind"], e)
    out = {k: _PP.label(e, args) for k, e in by_kind.items()}
    assert out["user"] == "user: list the files"
    assert out["assistant"] == "asst: plan it Listing.  [toolUse]  (157 tok)"
    assert out["thinking"] == "think: 7 chars"
    assert out["tool_call"] == 'call: bash {"command": "ls"}'
    assert out["tool_result"] == "res[ok]: a.py b.py"
    assert out["custom"] == "cust[monitor]: deliver"
    assert out["model"] == "model -> claude-x (anthropic)"
    assert out["thinking_level"] == "thinking -> high"
    assert out["compact"] == "COMPACT: squashed"
    assert out["other"].startswith("other: ")
    bare = _PP.label({"kind": "assistant", "text": "", "stop_reason": "stop"}, args)
    assert bare == "asst:"
    failed = _PP.label({"kind": "tool_result", "ok": False, "text": "x"}, args)
    assert failed == "res[ERROR]: x"


# ---------------------------------------------------------------- modes
def test_stats_counts_every_category(tmp):
    _, evs = _events(tmp)
    out = _capture("stats", evs, tmp)
    assert "Start:      2026-08-29 10:00:00" in out
    assert "Duration:   2m00s" in out
    assert "Files:      1" in out
    assert "Session ids:1" in out
    assert "User msgs:  1" in out
    assert "Asst msgs:  4  (2 requests carried usage)" in out
    assert "Thinking:   1 parts" in out
    assert "Tool calls: 2  (results: 2, unpaired calls: 1, orphan results: 1)" in out
    assert "Turn errors:1 aborted, 1 errored" in out
    assert "Tool errors:1 of 2 results failed" in out
    assert "Compactions:1" in out
    assert "anthropic/claude-x" in out and "1 responses" in out


def test_stats_on_an_empty_selection(tmp):
    out = _capture("stats", [], tmp)
    assert "Start:      (none)" in out
    assert "Duration" not in out
    assert "MODELS SERVING" not in out


def test_tokens_sums_per_model_and_across_models(tmp):
    _, evs = _events(tmp)
    out = _capture("tokens", evs, tmp)
    assert "anthropic/claude-x  —  1 requests" in out
    assert "input 132  (+cache read 2,048, write 0)" in out
    assert "response 157  (output 88 + reasoning 69)" in out
    assert "grand total 2,337   cost $0.0123 (pi estimate)" in out
    assert "openai/gpt-y  —  1 requests" in out
    assert "grand total 2,336   cost n/a" in out
    assert "ALL MODELS: response 245, grand total 4,673, cost $0.0123" in out


def test_tokens_with_no_usage_says_so(tmp):
    out = _capture("tokens", [{"kind": "user", "ts": None}], tmp)
    assert "(no assistant usage in this selection)" in out


def test_tokens_buckets_an_unnamed_model(tmp):
    _, evs = _events(tmp, [_assistant(T1, [], model=None)])
    out = _capture("tokens", evs, tmp)
    assert "(unknown)  —  1 requests" in out
    assert "ALL MODELS" not in out


def test_timeline_honours_filters(tmp):
    _, evs = _events(tmp)
    out = _capture("timeline", evs, tmp, user=True)
    assert out == "08-29 10:00:00  user: list the files"


def test_context_growth_reports_offsets_and_peak(tmp):
    _, evs = _events(tmp)
    out = _capture("context-growth", evs, tmp)
    assert "+0s      08-29 10:00:05 2,337 tok  model claude-x" in out
    assert "+1m55s   08-29 10:02:00 2,336 tok  model gpt-y" in out
    assert "peak: 2,337 tok; final: 2,336 tok" in out


def test_context_growth_without_usage(tmp):
    out = _capture("context-growth", [], tmp)
    assert "(no usage-bearing assistant messages)" in out


def test_tool_errors_lists_failed_results_with_bodies(tmp):
    _, evs = _events(tmp)
    out = _capture("tool-errors", evs, tmp)
    assert "FAILED TOOL RESULTS: 1" in out
    assert "bash (ghost)" in out
    assert "    lost" in out


def test_orphans_reports_both_directions(tmp):
    _, evs = _events(tmp)
    out = _capture("orphans", evs, tmp)
    assert "calls with no result: 1" in out
    assert 'bash {"command": "never answered"}' in out
    assert "results with no call: 1" in out
    assert "bash (ghost)" in out
    assert "paired:            1" in out


def test_compaction_mode_lists_entries(tmp):
    _, evs = _events(tmp)
    out = _capture("compaction", evs, tmp)
    assert "COMPACTIONS: 1" in out
    assert "08-29 10:00:12  squashed" in out
    bare = _capture("compaction", [{"kind": "compact", "ts": None, "detail": ""}], tmp)
    assert "(no detail)" in bare


def test_models_mode_lists_changes_or_none(tmp):
    _, evs = _events(tmp)
    out = _capture("models", evs, tmp)
    assert "model          claude-x (anthropic)" in out
    assert "thinking level high" in out
    assert "(none recorded)" in _capture("models", [], tmp)


def test_errors_mode_lists_turn_failures(tmp):
    _, evs = _events(tmp)
    out = _capture("errors", evs, tmp)
    assert "TURN FAILURES: 2" in out
    assert "[error]  boom" in out
    assert "[aborted]  ctrl-c" in out


def test_identity_mode_prints_the_header(tmp):
    path, evs = _events(tmp)
    out = _capture("identity", evs, tmp)
    assert "id:      sess-1" in out
    assert "cwd:     /work" in out
    assert "version: 0.9.0" in out
    assert "file:    %s" % path in out
    assert "last:    2026-08-29 10:02:00" in out


def test_last_assistant_picks_the_final_text_bearing_message(tmp):
    _, evs = _events(tmp)
    out = _capture("last-assistant", evs, tmp)
    assert "(openai/gpt-y)" in out
    assert out.endswith("Done.")
    assert "(none)" in _capture("last-assistant", [], tmp)
    full = _capture("last-assistant", evs, tmp, full=True)
    assert full.endswith("Done.")


# ---------------------------------------------------------------- targets
def test_resolve_target_accepts_file_directory_and_glob(tmp):
    sub = os.path.join(tmp, "nested")
    os.makedirs(sub)
    b = _write(sub, "b.jsonl", [])
    a = _write(tmp, "a.jsonl", [])
    assert _PP.resolve_target(a) == [a]
    assert _PP.resolve_target(tmp) == [a, b]
    assert _PP.resolve_target(os.path.join(tmp, "*.jsonl")) == [a]


def test_resolve_target_searches_the_default_root_by_id(tmp):
    root = os.path.join(tmp, "root")
    os.makedirs(os.path.join(root, "proj"))
    hit = _write(os.path.join(root, "proj"), "20260829_abc-uuid.jsonl", [])
    saved = _PP.DEFAULT_ROOT
    _PP.DEFAULT_ROOT = root
    try:
        assert _PP.resolve_target("abc-uuid") == [hit]
        try:
            _PP.resolve_target("nothing-like-this")
        except SystemExit as exc:
            assert "no session file matches" in str(exc)
        else:
            raise AssertionError("an unmatched target must exit")
    finally:
        _PP.DEFAULT_ROOT = saved


# ---------------------------------------------------------------- CLI
def _main(argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        rc = _PP.main(argv)
    return rc, out.getvalue(), err.getvalue()


def test_version_flag_names_the_tool(tmp):
    rc, out, _ = _main(["--version"])
    assert rc == 0
    assert out.strip() == "parse_pi.py %s" % _PP.__version__


def test_missing_session_is_a_usage_error(tmp):
    try:
        _main([])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("argparse must reject a missing session")


def test_default_run_is_a_filtered_event_listing(tmp):
    path, _ = _events(tmp)
    rc, out, err = _main([path, "--tools"])
    assert rc == 0 and err == ""
    lines = out.splitlines()
    assert len(lines) == 2
    assert all("call: bash" in line for line in lines)


def test_and_chains_segments_under_headers(tmp):
    path, _ = _events(tmp)
    rc, out, _ = _main([path, "--stats", "--and", "--identity", "--and", "--user"])
    assert rc == 0
    assert "===== stats =====" in out
    assert "===== identity =====" in out
    assert "===== events =====" in out
    assert "user: list the files" in out


def test_line_grep_is_refused_in_a_later_segment(tmp):
    path, _ = _events(tmp)
    rc, out, err = _main([path, "--identity", "--and", "--line-grep", "x", "--stats"])
    assert rc == 2
    assert "--line-grep belongs to the first segment" in err
    assert "===== identity =====" in out


def test_version_in_a_later_segment_prints_and_continues(tmp):
    path, _ = _events(tmp)
    rc, out, _ = _main([path, "--identity", "--and", "--version"])
    assert rc == 0
    assert "parse_pi.py %s" % _PP.__version__ in out


def test_a_crashing_segment_does_not_kill_the_chain(tmp):
    path, _ = _events(tmp)
    saved = _PP.MODES["errors"]
    _PP.MODES["errors"] = lambda *a: (_ for _ in ()).throw(RuntimeError("bad"))
    try:
        rc, out, err = _main([path, "--errors", "--and", "--identity"])
    finally:
        _PP.MODES["errors"] = saved
    assert rc == 2
    assert "segment error: RuntimeError: bad" in err
    assert "id:      sess-1" in out


def test_a_segment_raising_system_exit_is_reported_not_propagated(tmp):
    path, _ = _events(tmp)
    saved = _PP.MODES["errors"]
    _PP.MODES["errors"] = lambda *a: (_ for _ in ()).throw(SystemExit("stop"))
    try:
        rc, _, err = _main([path, "--errors"])
    finally:
        _PP.MODES["errors"] = saved
    assert rc == 2
    assert "segment error: stop" in err


def test_split_segments_and_selected_mode(tmp):
    assert _PP.split_segments(["a", "--and", "b", "c", "--and"]) == [["a"], ["b", "c"], []]
    args = _PP.build_parser().parse_args(["--context-growth"])
    assert _PP.selected_mode(args) == "context-growth"
    assert _PP.selected_mode(_PP.build_parser().parse_args([])) is None


def test_run_segment_writes_to_the_given_stream(tmp):
    _, evs = _events(tmp)
    args = _PP.build_parser().parse_args(["--user"])
    sink = io.StringIO()
    assert _PP.run_segment(evs, args, sink) == 0
    assert sink.getvalue().strip() == "08-29 10:00:00  user: list the files"


def main():
    return _util.runner(_util.collect(globals()), tmp_prefix="parsepi_")


if __name__ == "__main__":
    raise SystemExit(main())
