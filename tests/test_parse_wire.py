"""Suite for parse_wire.py — the Kimi wire.jsonl parser.

Fixtures are synthetic records in the real schema rather than real
transcripts: wire.jsonl carries per-record usage counters whose failure mode
is a plausible-looking wrong number, so the arithmetic here is hand-computed
and exact.
"""

# The module's leading-underscore helpers are the units under test here:
# rate selection, format detection, torn-line salvage and result rendering are
# reachable from the CLI only through a whole parse, which is exactly the
# indirection that let a wrong number look plausible. Naming them directly is
# the point of this suite, not an encapsulation slip.
# pylint: disable=protected-access
import json
import lzma
import os

import _util

_PW = _util.load(_util.script("parse_wire.py"))

# A fixed instant well past every rate cutoff, in milliseconds, which is the
# unit kimi-code stamps its records with. Tests that care about a cutoff state
# their own timestamp instead.
T0_MS = 1_784_300_000_000


def _write(tmp, records, name="wire.jsonl"):
    """Write records as one JSON object per line and return the path."""
    path = os.path.join(tmp, name)
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")
    return path


def _write_raw(tmp, text, name="wire.jsonl"):
    path = os.path.join(tmp, name)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    return path


def _kc_meta(ts_ms=T0_MS):
    return {"type": "metadata", "created_at": ts_ms,
            "protocol_version": "1.1"}


def _loop(ts_ms, event):
    return {"type": "context.append_loop_event", "time": ts_ms,
            "event": event}


def _usage_record(ts_ms, model, **usage):
    return {"type": "usage.record", "time": ts_ms, "model": model,
            "usage": usage}


def _legacy(ts, msg_type, payload):
    return {"timestamp": ts,
            "message": {"type": msg_type, "payload": payload}}


def _types(evs):
    return [e["type"] for e in evs]


def _total(report):
    for line in report.splitlines():
        if line.strip().startswith("TOTAL"):
            return float(line.split("$")[1])
    raise AssertionError(f"no TOTAL line in report:\n{report}")


def _category_cost(report, label):
    for line in report.splitlines():
        if line.strip().startswith(label):
            return float(line.split("$")[1].split()[0])
    raise AssertionError(f"no {label!r} cost line in report:\n{report}")


# ------------------------------------------------- kimi-code tool results
def test_missing_result_renders_empty_not_the_word_none(tmp):
    """A tool.result with no `result` key is an absence, not the text None.

    `_kc_result_detail(None)` used to stringify to 'None', which in parser
    output is indistinguishable from a tool whose output genuinely was that
    word.
    """
    assert _PW._kc_result_detail(None) == ""


def test_result_without_output_renders_empty(tmp):
    assert _PW._kc_result_detail({}) == ""
    assert _PW._kc_result_detail({"isError": True}) == ""


def test_result_output_string_and_list_shapes(tmp):
    assert _PW._kc_result_detail({"output": "txt"}) == "txt"
    assert _PW._kc_result_detail(
        {"output": [{"text": "a"}, {"text": "b"}]}) == "a\nb"
    # A non-dict element is shown rather than dropped: the shape is unexpected
    # and hiding it would lose the only evidence of that.
    assert _PW._kc_result_detail({"output": ["a", 7]}) == "a\n7"


def test_unexpected_non_dict_result_is_still_shown(tmp):
    """The stringify branch stays for shapes that are genuinely unexpected."""
    assert _PW._kc_result_detail("boom") == "boom"
    assert _PW._kc_result_detail(7) == "7"


# ------------------------------------------------------ format detection
def test_metadata_alone_separates_the_two_formats(tmp):
    """`created_at` is the only field that tells the two metadata lines apart.

    Everything downstream forks on this verdict, and both formats put a
    `metadata` record first, so a wrong answer here mis-parses the whole file
    rather than one record.
    """
    assert _PW._detect_format([json.dumps(_kc_meta()) + "\n"]) == "kimi-code"
    assert _PW._detect_format(
        [json.dumps({"type": "metadata", "session_id": "s"}) + "\n"]) == "legacy"


def test_envelope_and_message_types_decide_when_metadata_is_absent(tmp):
    kc = json.dumps({"type": "usage.record", "time": T0_MS}) + "\n"
    legacy = json.dumps(_legacy("2026-06-01T10:00:00Z", "ToolCall", {})) + "\n"
    assert _PW._detect_format([kc]) == "kimi-code"
    assert _PW._detect_format([legacy]) == "legacy"
    # Blank and undecidable lines are skipped, not treated as a verdict.
    assert _PW._detect_format(["\n", "  \n", kc]) == "kimi-code"
    assert _PW._detect_format([]) == "legacy"


# ----------------------------------------------- kimi-code event extraction
def test_kimi_code_extracts_every_event_type_it_recognises(tmp):
    path = _write(tmp, [
        _kc_meta(),
        {"type": "turn.prompt", "time": T0_MS + 1000,
         "input": [{"type": "text", "text": "hello"}]},
        {"type": "turn.steer", "time": T0_MS + 2000,
         "input": [{"type": "text", "text": "stop"}]},
        _loop(T0_MS + 3000, {"type": "step.begin", "turnId": "t1", "step": 1}),
        _loop(T0_MS + 4000, {"type": "content.part", "turnId": "t1",
                             "part": {"type": "think", "think": "hmm"}}),
        _loop(T0_MS + 5000, {"type": "content.part", "turnId": "t1",
                             "part": {"type": "text", "text": "answering"}}),
        _loop(T0_MS + 6000, {"type": "tool.call", "turnId": "t1",
                             "name": "Shell", "toolCallId": "c1",
                             "args": {"command": "ls"}}),
        _loop(T0_MS + 7000, {"type": "tool.result", "turnId": "t1",
                             "toolCallId": "c1",
                             "result": {"output": "a.txt"}}),
        _loop(T0_MS + 8000, {"type": "step.end", "turnId": "t1",
                             "finishReason": "stop"}),
    ])
    events, meta = _PW.extract_events(path)

    assert _types(events) == ["user_message", "steer_input", "thinking",
                              "assistant_text", "tool_call", "tool_result"]
    assert events[0]["detail"] == "hello"
    assert events[2]["detail"] == "hmm"
    assert events[4]["tool_name"] == "Shell"
    assert events[4]["tool_input"] == {"command": "ls"}
    assert events[5]["detail"] == "a.txt"
    assert events[5]["is_error"] is False
    # Milliseconds in, seconds out: a factor-1000 slip here would push every
    # event to 1970 or to the year 58,000, which no other assertion notices.
    assert events[0]["ts"] == (T0_MS + 1000) / 1000.0
    assert all(e["src"] == path for e in events)
    assert _types(meta) == ["metadata", "turn_begin", "step_begin",
                            "step_end", "turn_end"]


def test_appended_messages_split_into_text_and_tool_calls(tmp):
    path = _write(tmp, [
        _kc_meta(),
        {"type": "context.append_message", "time": T0_MS,
         "message": {"role": "user",
                     "content": [{"type": "text", "text": "do it"}]}},
        {"type": "context.append_message", "time": T0_MS,
         "message": {"role": "assistant",
                     "content": [{"type": "text", "text": "on it"}],
                     "toolCalls": [
                         {"type": "function", "id": "c1", "name": "Shell",
                          "arguments": '{"command": "ls"}'},
                         {"type": "function", "id": "c2",
                          "function": {"name": "ReadFile",
                                       "arguments": '{"path": "/tmp/x"}'}}]}},
        {"type": "context.append_message", "time": T0_MS,
         "message": {"role": "tool", "toolCallId": "c1", "isError": True,
                     "content": [{"type": "text", "text": "boom"}]}},
    ])
    events, _meta = _PW.extract_events(path)

    assert _types(events) == ["user_message", "assistant_text", "tool_call",
                              "tool_call", "tool_result"]
    # v1.1 flat shape and v1.0 nested shape must both resolve.
    assert [(e["tool_name"], e["tool_call_id"], e["tool_input"])
            for e in events if e["type"] == "tool_call"] == [
        ("Shell", "c1", {"command": "ls"}),
        ("ReadFile", "c2", {"path": "/tmp/x"})]
    assert events[4]["is_error"] is True
    assert events[4]["detail"] == "boom"


def test_media_parts_and_unknown_loop_events_are_kept(tmp):
    path = _write(tmp, [
        _kc_meta(),
        _loop(T0_MS, {"type": "content.part", "turnId": "t1",
                      "part": {"type": "image_url",
                               "image_url": {"url": "https://x/y.png"}}}),
        _loop(T0_MS, {"type": "goal.update", "turnId": "t1"}),
        {"type": "goal.set", "time": T0_MS, "goal": "ship"},
    ])
    events, meta = _PW.extract_events(path)

    assert _types(events) == ["media"]
    assert "https://x/y.png" in events[0]["detail"]
    # An unrecognised record is filed, never dropped: the catch-alls are what
    # keep a new kimi-code event type visible before anyone writes a branch.
    assert _types(meta) == ["metadata", "loop_goal.update", "goal.set"]


def test_tool_call_arguments_survive_every_shape_they_arrive_in(tmp):
    assert _PW._kc_args_to_input(None) == {}
    assert _PW._kc_args_to_input("") == {}
    assert _PW._kc_args_to_input({"a": 1}) == {"a": 1}
    assert _PW._kc_args_to_input('{"a": 1}') == {"a": 1}
    # Unparseable arguments are preserved verbatim rather than discarded.
    assert _PW._kc_args_to_input("{not json") == {"_raw": "{not json"}
    assert _PW._kc_args_to_input(7) == {"_raw": 7}
    assert _PW._kc_parse_tool_call({"type": "custom"}) == ("", None, "")


# --------------------------------------------------------- turn accounting
def test_a_new_turn_id_closes_the_previous_turn(tmp):
    """Turn boundaries are inferred from step.begin, not announced.

    kimi-code emits no turn-end record, so the parser closes the open turn
    when a step arrives carrying a different turnId, and again at EOF. A
    missed close merges two turns into one and takes their step and token
    counts with it.
    """
    path = _write(tmp, [
        _kc_meta(),
        _loop(T0_MS, {"type": "step.begin", "turnId": "t1", "step": 1}),
        _loop(T0_MS + 1000, {"type": "step.begin", "turnId": "t1", "step": 2}),
        _usage_record(T0_MS + 2000, "k3", inputOther=10, output=5),
        _loop(T0_MS + 3000, {"type": "step.begin", "turnId": "t2", "step": 1}),
    ])
    _events, meta = _PW.extract_events(path)

    assert _types(meta) == ["metadata", "turn_begin", "step_begin",
                            "step_begin", "status_update", "turn_end",
                            "turn_begin", "step_begin", "turn_end"]
    report = _PW.compute_turns([], meta)
    assert "Total: 2 turns" in report
    # Rows are the right-aligned numbered ones; the header's index column is
    # "#", so it never matches.
    body = [ln for ln in report.splitlines() if ln[:6].strip().isdigit()]
    assert len(body) == 2
    assert body[0].split()[-2:] == ["10", "5"]     # turn 1 saw the usage record
    assert body[1].split()[-2:] == ["0", "0"]      # turn 2 saw none


def test_the_last_turn_is_closed_at_end_of_file(tmp):
    path = _write(tmp, [
        _kc_meta(),
        _loop(T0_MS, {"type": "step.begin", "turnId": "t1", "step": 1}),
    ])
    _events, meta = _PW.extract_events(path)
    assert _types(meta) == ["metadata", "turn_begin", "step_begin", "turn_end"]
    assert meta[-1]["line"] == 2       # the line count, not a record's line


# ------------------------------------------------------- llm.error records
def test_llm_error_keeps_the_fields_the_status_plugin_reads(tmp):
    """The Kimi Discord status plugin classifies on `kind`.

    It never sees the wire file, only these events, so dropping or renaming a
    field here surfaces as a wrong bot presence rather than a parser error.
    """
    path = _write(tmp, [
        _kc_meta(),
        {"type": "llm.error", "time": T0_MS, "kind": "rate_limit",
         "retryable": True, "statusCode": 429, "errorName": "RateLimitError",
         "model": "kimi-k3", "message": "slow down"},
        {"type": "llm.error", "time": T0_MS + 1000},
    ])
    events, _meta = _PW.extract_events(path)

    assert _types(events) == ["api_error", "api_error"]
    assert events[0]["kind"] == "rate_limit"
    assert events[0]["retryable"] is True
    assert events[0]["status_code"] == 429
    assert events[0]["error_name"] == "RateLimitError"
    assert events[0]["model"] == "kimi-k3"
    assert events[0]["detail"] == "slow down"
    # A bare error still classifies: "other", not a crash and not a blank.
    assert events[1]["kind"] == "other"
    assert events[1]["retryable"] is False
    assert events[1]["detail"] == ""
    assert "API errors: 2" in _PW.compute_stats(events, [])


# ------------------------------------------------------- usage accounting
def test_usage_record_maps_camel_case_counters_onto_the_shared_shape(tmp):
    path = _write(tmp, [
        _kc_meta(),
        _usage_record(T0_MS, "k3", inputOther=11, output=22,
                      inputCacheRead=33, inputCacheCreation=44),
        _usage_record(T0_MS + 1000, "k3"),
    ])
    _events, meta = _PW.extract_events(path)
    updates = [m for m in meta if m["type"] == "status_update"]

    assert updates[0]["token_usage"] == {
        "input_other": 11, "output": 22,
        "input_cache_read": 33, "input_cache_creation": 44}
    assert updates[0]["model"] == "k3"
    # A record with no usage counters is zeros, not missing keys: compute_cache
    # adds these unconditionally.
    assert updates[1]["token_usage"] == {
        "input_other": 0, "output": 0,
        "input_cache_read": 0, "input_cache_creation": 0}


def test_wire_model_string_prices_the_record(tmp):
    path = _write(tmp, [
        _kc_meta(),
        _usage_record(T0_MS, "k3", inputOther=1_000_000,
                      inputCacheRead=2_000_000, inputCacheCreation=500_000,
                      output=100_000),
    ])
    _events, meta = _PW.extract_events(path)
    report = _PW.compute_cache(meta)

    # 1M fresh @ $3.00 + 2M cache_read @ $0.30 + 0.5M cache_create @ $0
    # + 0.1M output @ $15.00
    assert _category_cost(report, "Fresh input") == 3.00
    assert _category_cost(report, "Cache read") == 0.60
    assert _category_cost(report, "Output") == 1.50
    assert _total(report) == 5.10
    assert "Detected model: k3" in report
    assert "input total:        3,500,000" in report


def test_a_mid_session_model_switch_reconciles_with_the_total(tmp):
    """Per-record rates, not one session-wide rate applied to summed tokens.

    A session that switches model mid-flight is observed in the wild, and
    re-deriving the category breakdown from the totals silently drifts from
    the per-record sum — the breakdown and the TOTAL would disagree.
    """
    path = _write(tmp, [
        _kc_meta(),
        _usage_record(T0_MS, "kimi-for-coding",
                      inputOther=1_000_000, output=1_000_000),
        _usage_record(T0_MS + 24_000, "k3",
                      inputOther=1_000_000, output=1_000_000),
    ])
    _events, meta = _PW.extract_events(path)
    report = _PW.compute_cache(meta)

    # k2.6 rates for the ambiguous id, k3 rates for the named one.
    assert _category_cost(report, "Fresh input") == 0.95 + 3.00
    assert _category_cost(report, "Output") == 4.00 + 15.00
    assert _total(report) == 22.95
    assert (_category_cost(report, "Fresh input")
            + _category_cost(report, "Output")) == _total(report)
    assert "PER TURN" in report and "(2 turns)" in report


def test_an_unrecognised_model_is_never_promoted_to_k3_by_date(tmp):
    """The date rung applies only to records with NO model string at all.

    k3 is ~3x the k2 rates, so a wrong promotion overcounts by 3x. A wrong
    undercount is the deliberate choice.
    """
    assert _PW._rates_for_model("k3") is _PW.K3_RATES
    assert _PW._rates_for_model("kimi-code/k3") is _PW.K3_RATES
    assert _PW._rates_for_model("moonshot/kimi-k3-preview") is _PW.K3_RATES
    assert _PW._rates_for_model("kimi-k2-7-code") is _PW.K2_7_CODE_RATES
    assert _PW._rates_for_model("kimi-for-coding") is _PW.K2_6_RATES
    assert _PW._rates_for_model(None) is _PW.K2_6_RATES


def test_a_record_with_no_model_is_priced_by_the_sessions_first_timestamp(tmp):
    before = _PW.MODEL_CUTOFF_EPOCH - 1
    between = _PW.K3_CUTOFF_EPOCH - 1
    after = _PW.K3_CUTOFF_EPOCH

    assert _PW._rates_for_session([{"ts": before}], []) is _PW.K2_6_RATES
    assert _PW._rates_for_session([{"ts": between}], []) is _PW.K2_7_CODE_RATES
    assert _PW._rates_for_session([{"ts": after}], []) is _PW.K3_RATES
    # No timestamp anywhere: the middle rung, not the most expensive one.
    assert _PW._rates_for_session([], []) is _PW.K2_7_CODE_RATES
    # meta_events are consulted when the event list carries no timestamp.
    assert _PW._rates_for_session([{"ts": ""}], [{"ts": before}]) is \
        _PW.K2_6_RATES


def test_legacy_status_updates_price_through_the_session_default(tmp):
    """Legacy StatusUpdate carries no model, so the date heuristic decides."""
    ts = "2026-08-01T00:00:00Z"          # past the k3 cutoff
    path = _write(tmp, [
        _legacy(ts, "StatusUpdate",
                {"token_usage": {"input_other": 1_000_000, "output": 0}}),
    ])
    events, meta = _PW.extract_events(path)
    report = _PW.compute_cache(meta, events=events)

    assert _total(report) == 3.00
    assert "Detected model: kimi-k3" in report


def test_explicit_rates_override_every_other_source(tmp):
    path = _write(tmp, [
        _kc_meta(),
        _usage_record(T0_MS, "k3", inputOther=1_000_000, output=1_000_000),
    ])
    _events, meta = _PW.extract_events(path)
    rates = {"fresh": 1.0, "cache_read": 0.0,
             "cache_create": 0.0, "output": 2.0}
    assert _total(_PW.compute_cache(meta, rates=rates)) == 3.00


def test_a_session_with_no_usage_records_says_so(tmp):
    assert "(no StatusUpdate with token_usage found)" in _PW.compute_cache([])
    assert "(no StatusUpdate with token_usage found)" in \
        _PW.compute_context_growth([])


def test_context_growth_deduplicates_repeated_lines_and_shows_deltas(tmp):
    meta = [
        {"type": "status_update", "line": 1, "ts": T0_MS / 1000.0,
         "token_usage": {"input_other": 100, "input_cache_read": 900,
                         "input_cache_creation": 0, "output": 10}},
        {"type": "status_update", "line": 1, "ts": T0_MS / 1000.0,
         "token_usage": {"input_other": 100, "input_cache_read": 900,
                         "input_cache_creation": 0, "output": 20}},
        {"type": "status_update", "line": 2, "ts": T0_MS / 1000.0 + 1,
         "token_usage": {"input_other": 200, "input_cache_read": 1_300,
                         "input_cache_creation": 0, "output": 30}},
    ]
    report = _PW.compute_context_growth(meta)
    assert "Total: 2 snapshots" in report
    assert "final context: 1,500 input tokens" in report
    assert "+500" in report          # 1,500 - 1,000, the second snapshot


# ---------------------------------------------------------- legacy format
def test_legacy_records_extract_into_the_same_event_shapes(tmp):
    ts = "2026-06-01T10:00:00Z"
    path = _write(tmp, [
        _legacy(ts, "TurnBegin", {"user_input": [{"text": "hi"}]}),
        _legacy(ts, "ContentPart", {"type": "think", "think": "hmm"}),
        _legacy(ts, "ContentPart", {"type": "text", "text": "sure"}),
        _legacy(ts, "ToolCall",
                {"id": "c1", "function": {"name": "Shell",
                                          "arguments": '{"command": "ls"}'}}),
        _legacy(ts, "ToolResult",
                {"tool_call_id": "c1",
                 "return_value": {"output": "a.txt", "is_error": False}}),
        _legacy(ts, "SubagentEvent",
                {"parent_tool_call_id": "c1", "agent_id": "a1",
                 "subagent_type": "explore", "event": {"type": "start"}}),
        _legacy(ts, "TurnEnd", {}),
    ])
    events, meta = _PW.extract_events(path)

    assert _types(events) == ["user_message", "thinking", "assistant_text",
                              "tool_call", "tool_result", "subagent_event"]
    assert events[0]["detail"] == "hi"
    assert events[3]["tool_input"] == {"command": "ls"}
    assert events[4]["detail"] == "a.txt"
    assert _types(meta) == ["turn_begin", "turn_end"]
    assert "explore" in _PW.compute_subagents(events, meta)


def test_legacy_tool_arguments_that_are_not_json_are_preserved(tmp):
    path = _write(tmp, [
        _legacy("2026-06-01T10:00:00Z", "ToolCall",
                {"id": "c1", "function": {"name": "Shell",
                                          "arguments": "{truncated"}}),
    ])
    events, _meta = _PW.extract_events(path)
    assert events[0]["tool_input"] == {"_raw": "{truncated"}


# ------------------------------------------- damaged and compressed files
def test_a_torn_record_is_salvaged_rather_than_dropped(tmp):
    """Concurrent writers leave two objects on one line, the first truncated.

    Dropping the line loses every event in it; the salvage marks what it
    recovered so a consumer can tell a partial record from a whole one.
    """
    good = json.dumps({"type": "usage.record", "time": T0_MS,
                       "model": "k3", "usage": {"output": 5}})
    torn = '{"type": "llm.error", "time": %d, "kind": "timeo' % T0_MS
    objs = list(_PW._iter_json_objects(torn + good))

    assert len(objs) == 2
    assert objs[0]["_torn_truncated"] is True
    assert objs[0]["kind"] == "timeo"
    assert objs[1]["model"] == "k3"


def test_an_unsalvageable_line_becomes_a_parse_error_event(tmp):
    path = _write_raw(tmp, json.dumps(_kc_meta()) + "\nnot json at all\n")
    events, _meta = _PW.extract_events(path)
    assert _types(events) == ["parse_error"]
    assert "line 2" in events[0]["detail"]


def test_an_xz_archived_wire_file_reads_like_a_plain_one(tmp):
    """The R2 archive stores wire.jsonl.xz; the parser reads either form."""
    path = os.path.join(tmp, "wire.jsonl.xz")
    with lzma.open(path, "wt", encoding="utf-8") as fh:
        fh.write(json.dumps(_kc_meta()) + "\n")
        fh.write(json.dumps({"type": "turn.prompt", "time": T0_MS,
                             "input": [{"type": "text", "text": "hi"}]}) + "\n")
    events, _meta = _PW.extract_events(path)
    assert _types(events) == ["user_message"]
    assert events[0]["detail"] == "hi"


# ---------------------------------------------------- pairing and reports
def test_calls_pair_with_results_and_unpaired_calls_are_orphans(tmp):
    path = _write(tmp, [
        _kc_meta(),
        _loop(T0_MS, {"type": "tool.call", "name": "Shell",
                      "toolCallId": "c1", "args": {"command": "ls"}}),
        _loop(T0_MS + 500, {"type": "tool.call", "name": "ReadFile",
                            "toolCallId": "c2", "args": {"path": "/tmp/x"}}),
        _loop(T0_MS + 9000, {"type": "tool.call", "name": "Shell",
                             "toolCallId": "c3", "args": {"command": "id"}}),
        _loop(T0_MS + 10_000, {"type": "tool.result", "toolCallId": "c1",
                               "result": {"output": "a.txt"}}),
    ])
    events, _meta = _PW.extract_events(path)
    _PW.annotate_parallel(events)
    _PW.link_pairs(events)

    calls = [e for e in events if e["type"] == "tool_call"]
    assert calls[0]["paired_result"]["detail"] == "a.txt"
    # c1 and c2 are half a second apart -- one dispatch; c3 is nine seconds
    # later and is its own.
    assert [c.get("batch_size", 1) for c in calls] == [2, 2, 1]

    orphans = _PW.compute_tool_orphans(events)
    assert "2 orphaned calls" in orphans
    assert "ReadFile" in orphans and "id" in orphans


def test_stats_counts_each_event_class(tmp):
    path = _write(tmp, [
        _kc_meta(),
        {"type": "turn.prompt", "time": T0_MS,
         "input": [{"type": "text", "text": "hi"}]},
        _loop(T0_MS + 1000, {"type": "step.begin", "turnId": "t1", "step": 1}),
        _loop(T0_MS + 2000, {"type": "content.part", "turnId": "t1",
                             "part": {"type": "text", "text": "sure"}}),
        _loop(T0_MS + 3000, {"type": "tool.call", "name": "Shell",
                             "toolCallId": "c1", "args": {"command": "ls"}}),
        _loop(T0_MS + 4000, {"type": "tool.result", "toolCallId": "c1",
                             "result": {"isError": True, "output": "boom"}}),
    ])
    events, meta = _PW.extract_events(path)
    report = _PW.compute_stats(events, meta)

    assert "User msgs:  1" in report
    assert "Asst msgs:  1" in report
    assert "Tool calls: 1" in report
    assert "Results:    1 (1 errors)" in report
    assert "Turns:      1" in report
    assert "Steps:      1" in report
    assert "Shell                   1" in report


def test_timestamps_parse_from_every_form_the_formats_use(tmp):
    epoch = _PW.parse_timestamp(1_784_300_000)
    iso_z = _PW.parse_timestamp("2026-07-16T14:45:55Z")
    numeric_string = _PW.parse_timestamp("1784300000")

    assert epoch is not None and epoch.tzinfo is not None
    assert iso_z is not None and iso_z.hour == 14
    assert numeric_string == epoch
    assert _PW.parse_timestamp("") is None
    assert _PW.parse_timestamp("not a time") is None
    assert _PW.short_time("") == "????-??-?? ??:??:??"


def main():
    return _util.runner(_util.collect(globals()), tmp_prefix="parsewire_")


if __name__ == "__main__":
    raise SystemExit(main())
