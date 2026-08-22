"""Suite for parse_session.py's ingest and formatting layer.

Every report this parser produces reads events built by this layer: a raw
JSONL line becomes one or more records, a record becomes events, and an event
becomes a printed line. None of it was tested. Its failure modes are silent
rather than loud -- a usage max merged the wrong way misprices a whole
session, a torn line dropped instead of salvaged loses a record with no
diagnostic, and a batch split on the wrong side of the 2s boundary reports two
parallel batches where the reader saw one.

So the assertions below are on produced VALUES: the merged token counts, the
salvaged record, the batch index, the exact formatted line. Fixtures are
synthetic JSONL written into the temp dir each test is handed and read back
through the real file path, which keeps the suite off any machine's own
transcripts -- CI runners have none.
"""

# The leading-underscore helpers ARE the units here: record dispatch, torn-line
# salvage, usage merging and event formatting are otherwise reachable only
# through a whole parse plus a report, which is the indirection that let this
# layer go untested. Naming them directly is the point of the suite.
# pylint: disable=protected-access
import json
import lzma
import os
import re

import _util

_PS = _util.load(_util.script("parse_session.py"))

TS = "2026-07-30T12:00:{:02d}Z"


# --- fixture builders -------------------------------------------------------

def _write(tmp, name, records):
    """Write `records` (dicts, or raw strings taken verbatim) as one JSONL."""
    path = os.path.join(tmp, name)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for rec in records:
            fh.write(rec if isinstance(rec, str) else json.dumps(rec))
            fh.write("\n")
    return path


def _user_rec(text, uuid="u1", second=1):
    return {"type": "user", "uuid": uuid, "timestamp": TS.format(second),
            "message": {"role": "user", "content": text}}


def _assistant_rec(blocks, uuid="a1", second=2, model="claude-x",
                   usage=None, request_id="req1"):
    msg = {"role": "assistant", "model": model, "content": blocks}
    if usage is not None:
        msg["usage"] = usage
    return {"type": "assistant", "uuid": uuid, "timestamp": TS.format(second),
            "requestId": request_id, "message": msg}


def _result_rec(tool_use_id, text, uuid="r1", second=3, is_error=False,
                tool_use_result=None):
    return {"type": "user", "uuid": uuid, "timestamp": TS.format(second),
            "toolUseResult": tool_use_result,
            "message": {"role": "user", "content": [{
                "type": "tool_result", "tool_use_id": tool_use_id,
                "content": text, "is_error": is_error}]}}


def _types(events):
    return [e["type"] for e in events]


def _of_type(events, etype):
    return [e for e in events if e["type"] == etype]


# --- _open_text -------------------------------------------------------------

def test_open_text_reads_a_plain_utf8_file(tmp):
    path = _write(tmp, "plain.jsonl", [{"k": "café — 日"}])

    with _PS._open_text(path) as fh:
        body = fh.read()

    # Decoded as UTF-8 regardless of the platform's preferred encoding: on a
    # Windows runner the default is cp1252 and these characters would mojibake.
    assert json.loads(body)["k"] == "café — 日", body


def test_open_text_transparently_decompresses_xz(tmp):
    raw = json.dumps({"k": "compressed"}) + "\n"
    path = os.path.join(tmp, "archived.jsonl.xz")
    with lzma.open(path, "wb") as fh:
        fh.write(raw.encode("utf-8"))

    with _PS._open_text(path) as fh:
        body = fh.read()

    assert body == raw, body


# --- parse_timestamp / short_time -------------------------------------------

def test_parse_timestamp_reads_the_z_suffix_as_utc(tmp):
    dt = _PS.parse_timestamp("2026-07-30T12:34:56Z")

    assert (dt.year, dt.month, dt.day) == (2026, 7, 30)
    assert (dt.hour, dt.minute, dt.second) == (12, 34, 56)
    # `Z` must survive as a real offset: a naive datetime here would raise
    # when subtracted from an aware one during latency arithmetic.
    assert dt.utcoffset().total_seconds() == 0, dt.tzinfo


def test_parse_timestamp_returns_none_rather_than_raising(tmp):
    assert _PS.parse_timestamp("") is None
    assert _PS.parse_timestamp(None) is None
    assert _PS.parse_timestamp("not-a-timestamp") is None
    assert _PS.parse_timestamp("2026-13-45T99:99:99Z") is None


def test_short_time_renders_a_19_char_column(tmp):
    out = _PS.short_time("2026-07-30T12:34:56Z")

    assert out == "2026-07-30 12:34:56", out
    # The docstring promises width 19 and tabular callers reserve exactly that.
    assert len(out) == 19, out


def test_short_time_pads_an_unparseable_stamp_to_the_same_width(tmp):
    out = _PS.short_time("garbage")

    assert out == "????-??-?? ??:??:??", out
    assert len(out) == len(_PS.short_time("2026-07-30T12:34:56Z")), out


# --- the _meta_* handlers and their dispatch --------------------------------

def test_meta_queue_operation_captures_operation_and_content(tmp):
    meta = _PS._meta_queue_operation(
        {"operation": "enqueue", "content": "do the thing"}, 7, TS.format(1))

    assert meta["type"] == "queue-operation"
    assert meta["line"] == 7
    assert meta["ts"] == TS.format(1)
    assert meta["operation"] == "enqueue"
    assert meta["content"] == "do the thing"


def test_meta_system_captures_subtype_and_keeps_the_raw_record(tmp):
    obj = {"subtype": "local_command", "content": "/stats", "extra": 1}

    meta = _PS._meta_system(obj, 3, TS.format(2))

    assert meta["subtype"] == "local_command"
    assert meta["content"] == "/stats"
    # `raw` is what compute_hooks and friends re-read for fields the handler
    # did not lift, so it must be the record itself, not a copy of the lift.
    assert meta["raw"] is obj


def test_meta_attachment_tolerates_a_null_attachment(tmp):
    present = _PS._meta_attachment(
        {"attachment": {"type": "hook_event"}}, 1, TS.format(1))
    missing = _PS._meta_attachment({}, 2, TS.format(1))
    explicit_null = _PS._meta_attachment({"attachment": None}, 3, TS.format(1))

    assert present["attachment_type"] == "hook_event"
    # `"attachment": null` is a real shape in transcripts; the `or {}` is what
    # keeps it from becoming an AttributeError on None.get.
    assert missing["attachment_type"] == ""
    assert explicit_null["attachment_type"] == ""


def test_meta_permission_mode_and_last_prompt_default_to_empty(tmp):
    mode = _PS._meta_permission_mode({"permissionMode": "plan"}, 1, TS.format(1))
    mode_missing = _PS._meta_permission_mode({}, 2, TS.format(1))
    prompt = _PS._meta_last_prompt({"lastPrompt": "hello"}, 3, TS.format(1))
    prompt_missing = _PS._meta_last_prompt({}, 4, TS.format(1))

    assert mode["permissionMode"] == "plan"
    assert mode_missing["permissionMode"] == ""
    assert prompt["type"] == "last-prompt"
    assert prompt["content"] == "hello"
    assert prompt_missing["content"] == ""


def test_meta_handlers_dispatch_by_record_type(tmp):
    records = [
        {"type": "queue-operation", "operation": "dequeue"},
        {"type": "system", "subtype": "turn_duration"},
        {"type": "attachment", "attachment": {"type": "hook_event"}},
        {"type": "permission-mode", "permissionMode": "acceptEdits"},
        {"type": "last-prompt", "lastPrompt": "p"},
    ]
    meta_events = []

    _PS._process_record_objs(records, 5, [], meta_events, {}, None)

    # Every dispatch-table entry must emit a meta event whose `type` is the
    # record type it was keyed on -- a mismatch makes a whole mode see nothing.
    assert _types(meta_events) == [
        "queue-operation", "system", "attachment", "permission-mode",
        "last-prompt"]
    assert set(_PS._META_HANDLERS) == set(_types(meta_events))
    assert all(m["line"] == 5 for m in meta_events)


# --- _build_tool_call_event / _tool_call_detail -----------------------------

def test_build_tool_call_event_carries_name_input_and_id(tmp):
    event = _PS._build_tool_call_event(
        {"name": "Bash", "id": "tu9", "input": {"command": "ls -la"}},
        4, TS.format(1))

    assert event["type"] == "tool_call"
    assert event["tool_name"] == "Bash"
    assert event["tool_use_id"] == "tu9"
    assert event["tool_input"] == {"command": "ls -la"}
    assert event["line"] == 4
    # Detail is deliberately left blank and rendered lazily; a non-empty value
    # here would defeat the json.dumps deferral that meta-only modes rely on.
    assert event["detail"] == ""


def test_build_agent_spawn_summarises_the_agent(tmp):
    event = _PS._build_tool_call_event({
        "name": "Agent", "id": "tu1",
        "input": {"name": "scout", "model": "opus", "run_in_background": True,
                  "team_name": "alpha", "subagent_type": "explore",
                  "prompt": "go look"},
    }, 2, TS.format(1))

    assert event["type"] == "agent_spawn"
    assert event["agent_name"] == "scout"
    assert event["agent_model"] == "opus"
    assert event["agent_bg"] is True
    assert event["agent_team"] == "alpha"
    assert event["agent_subagent_type"] == "explore"
    assert event["agent_prompt"] == "go look"
    assert event["detail"] == (
        "Agent(scout, model=opus, bg=True, team=alpha, subagent_type=explore)"
    ), event["detail"]


def test_build_agent_spawn_fills_placeholders_for_missing_fields(tmp):
    event = _PS._build_tool_call_event(
        {"name": "Agent", "id": "tu1", "input": {}}, 1, TS.format(1))

    assert event["agent_name"] == "?"
    assert event["agent_model"] == "(default)"
    assert event["agent_bg"] is False
    assert event["agent_team"] == "(none)"
    assert event["agent_subagent_type"] == "(default)"
    assert event["agent_prompt"] == ""


def test_tool_call_detail_renders_then_caches_the_render(tmp):
    event = _PS._build_tool_call_event(
        {"name": "Read", "id": "tu1", "input": {"file_path": "a.py"}},
        1, TS.format(1))

    first = _PS._tool_call_detail(event)

    assert first == 'Read({"file_path": "a.py"})', first
    assert event["detail"] == first, "the render was not memoised onto the event"
    # Memoised, not recomputed: a later mutation of tool_input must not change
    # a detail some earlier report line already printed.
    event["tool_input"]["file_path"] = "b.py"
    assert _PS._tool_call_detail(event) == first


def test_tool_call_detail_returns_a_prebuilt_detail_untouched(tmp):
    spawn = _PS._build_tool_call_event(
        {"name": "Agent", "id": "tu1", "input": {"name": "scout"}},
        1, TS.format(1))

    assert _PS._tool_call_detail(spawn) == spawn["detail"]
    assert "Agent(scout," in _PS._tool_call_detail(spawn)


# --- _parse_user_content ----------------------------------------------------

def test_user_string_content_becomes_one_message_carrying_the_uuid(tmp):
    out = _PS._parse_user_content("hello there", None, 3, TS.format(1), "uu1")

    assert len(out) == 1
    assert out[0]["type"] == "user_message"
    assert out[0]["detail"] == "hello there"
    assert out[0]["uuid"] == "uu1"
    assert out[0]["line"] == 3


def test_user_tool_result_joins_a_list_payload_and_keeps_the_raw_result(tmp):
    payload = [{"type": "text", "text": "line one"},
               {"type": "text", "text": "line two"},
               "bare string"]
    tur = {"stdout": "x"}

    out = _PS._parse_user_content(
        [{"type": "tool_result", "tool_use_id": "tu1", "content": payload,
          "is_error": True}], tur, 4, TS.format(1))

    assert len(out) == 1
    result = out[0]
    assert result["type"] == "tool_result"
    assert result["tool_use_id"] == "tu1"
    assert result["is_error"] is True
    # Blocks join with newlines, and a non-dict block falls back to str().
    assert result["detail"] == "line one\nline two\nbare string", result["detail"]
    assert result["toolUseResult"] is tur


def test_user_tool_result_stringifies_a_non_list_payload(tmp):
    out = _PS._parse_user_content(
        [{"type": "tool_result", "tool_use_id": "tu1", "content": 42}],
        None, 1, TS.format(1))

    assert out[0]["detail"] == "42", out[0]["detail"]
    assert out[0]["is_error"] is False, "a result with no is_error must read as clean"


def test_user_content_mixes_text_blocks_and_results_in_order(tmp):
    out = _PS._parse_user_content([
        {"type": "text", "text": "before"},
        {"type": "tool_result", "tool_use_id": "tu1", "content": "out"},
        {"type": "image", "source": {}},
        {"type": "text", "text": "after"},
    ], None, 1, TS.format(1))

    # An unknown block type is dropped rather than emitted as an empty event.
    assert _types(out) == ["user_message", "tool_result", "user_message"]
    assert [e["detail"] for e in out] == ["before", "out", "after"]


def test_user_content_of_an_unexpected_shape_yields_nothing(tmp):
    assert _PS._parse_user_content(None, None, 1, TS.format(1)) == []
    assert _PS._parse_user_content({"type": "text"}, None, 1, TS.format(1)) == []
    assert _PS._parse_user_content([], None, 1, TS.format(1)) == []


# --- _extract_server_result_text --------------------------------------------

def test_server_result_text_lifts_the_text_field_of_a_result_dict(tmp):
    text, is_error = _PS._extract_server_result_text(
        {"type": "advisor_result", "text": "the advice"})

    assert text == "the advice"
    assert is_error is False


def test_server_result_text_flags_every_error_marker(tmp):
    by_type = _PS._extract_server_result_text(
        {"type": "advisor_tool_result_error", "error_code": "rate_limit"})
    by_code = _PS._extract_server_result_text({"type": "x", "error_code": "boom"})
    by_flag = _PS._extract_server_result_text({"type": "x", "is_error": True})
    clean = _PS._extract_server_result_text({"type": "advisor_result", "text": ""})

    # Three independent markers, because the shape differs per server tool.
    assert by_type[1] is True
    assert by_code[1] is True
    assert by_flag[1] is True
    assert clean[1] is False


def test_server_result_text_dumps_a_dict_that_has_no_text(tmp):
    text, is_error = _PS._extract_server_result_text(
        {"type": "advisor_tool_result_error", "error_code": "rate_limit"})

    assert json.loads(text) == {"type": "advisor_tool_result_error",
                                "error_code": "rate_limit"}
    assert is_error is True


def test_server_result_text_joins_a_list_of_search_results(tmp):
    text, is_error = _PS._extract_server_result_text(
        [{"text": "first"}, {"url": "http://x"}, "third"])

    parts = text.split("\n")
    assert parts[0] == "first"
    # A dict with no text is dumped whole rather than dropped.
    assert json.loads(parts[1]) == {"url": "http://x"}
    assert parts[2] == "third"
    assert is_error is False


def test_server_result_text_stringifies_a_plain_payload(tmp):
    assert _PS._extract_server_result_text("mcp output") == ("mcp output", False)
    assert _PS._extract_server_result_text(7) == ("7", False)


# --- _parse_assistant_content -----------------------------------------------

def test_assistant_content_tags_every_event_with_the_responding_model(tmp):
    out = _PS._parse_assistant_content([
        {"type": "text", "text": "hi"},
        {"type": "thinking", "thinking": "hmm"},
        {"type": "tool_use", "id": "tu1", "name": "Bash", "input": {}},
    ], 2, TS.format(1), model="claude-opus-x")

    assert _types(out) == ["assistant_text", "thinking", "tool_call"]
    # --reply-latency-words attributes a reply to a model off this tag; an
    # untagged event would be attributed to whatever ran last.
    assert {e["model"] for e in out} == {"claude-opus-x"}


def test_assistant_content_drops_an_empty_thinking_block(tmp):
    out = _PS._parse_assistant_content(
        [{"type": "thinking", "thinking": ""}, {"type": "thinking"}],
        1, TS.format(1))

    assert out == []


def test_assistant_content_routes_a_server_tool_use_through_the_call_path(tmp):
    out = _PS._parse_assistant_content([{
        "type": "server_tool_use", "id": "srv1", "name": "web_search",
        "input": {"query": "q"},
    }], 1, TS.format(1))

    # A provider-executed call must land as a normal tool_call so --tools,
    # --stats and --latency see it at all.
    assert _types(out) == ["tool_call"]
    assert out[0]["tool_name"] == "web_search"
    assert out[0]["tool_use_id"] == "srv1"


def test_assistant_content_emits_a_server_tool_result_as_a_tool_result(tmp):
    out = _PS._parse_assistant_content([{
        "type": "web_search_tool_result", "tool_use_id": "srv1",
        "content": {"type": "advisor_tool_result_error", "error_code": "e"},
    }], 6, TS.format(1))

    assert _types(out) == ["tool_result"]
    # It must carry the originating id, or the call shows up as an orphan.
    assert out[0]["tool_use_id"] == "srv1"
    assert out[0]["is_error"] is True
    assert out[0]["toolUseResult"] is None


def test_assistant_content_of_an_unexpected_shape_yields_nothing(tmp):
    assert _PS._parse_assistant_content("plain text", 1, TS.format(1)) == []
    assert _PS._parse_assistant_content(None, 1, TS.format(1)) == []


# --- _merge_usage_max / _flatten_usage --------------------------------------

def test_merge_usage_keeps_the_larger_counter_in_each_direction(tmp):
    a = {"input_tokens": 100, "output_tokens": 5,
         "cache_read_input_tokens": 900, "cache_creation_input_tokens": 10}
    b = {"input_tokens": 100, "output_tokens": 42,
         "cache_read_input_tokens": 300, "cache_creation_input_tokens": 10}

    merged = _PS._merge_usage_max(a, b)

    # Streaming records report a lagging output_tokens; the aggregate wins
    # regardless of which side it arrived on.
    assert merged["output_tokens"] == 42
    assert merged["cache_read_input_tokens"] == 900
    assert merged["input_tokens"] == 100
    assert merged["cache_creation_input_tokens"] == 10
    assert a["output_tokens"] == 5, "inputs must not be mutated"


def test_merge_usage_fills_a_counter_missing_from_the_first_record(tmp):
    merged = _PS._merge_usage_max({"input_tokens": 7},
                                  {"output_tokens": 9, "input_tokens": None})

    assert merged["output_tokens"] == 9
    # An explicit null must not overwrite a real count with None.
    assert merged["input_tokens"] == 7


def test_merge_usage_merges_nested_cache_creation_key_by_key(tmp):
    a = {"cache_creation": {"ephemeral_5m_input_tokens": 500,
                            "ephemeral_1h_input_tokens": 10}}
    b = {"cache_creation": {"ephemeral_5m_input_tokens": 20,
                            "ephemeral_1h_input_tokens": 800,
                            "ephemeral_new_input_tokens": 3}}

    merged = _PS._merge_usage_max(a, b)

    assert merged["cache_creation"] == {"ephemeral_5m_input_tokens": 500,
                                        "ephemeral_1h_input_tokens": 800,
                                        "ephemeral_new_input_tokens": 3}
    assert a["cache_creation"]["ephemeral_1h_input_tokens"] == 10


def test_merge_usage_adopts_a_nested_dict_the_first_record_lacks(tmp):
    merged = _PS._merge_usage_max({"server_tool_use": None},
                                  {"server_tool_use": {"web_search_requests": 4}})

    assert merged["server_tool_use"] == {"web_search_requests": 4}
    merged["server_tool_use"]["web_search_requests"] = 99
    assert merged["server_tool_use"]["web_search_requests"] == 99


def test_merge_usage_keeps_unknown_keys_and_copies_new_ones(tmp):
    merged = _PS._merge_usage_max(
        {"service_tier": "standard"},
        {"service_tier": "batch", "future_field": [1, 2]})

    # Forward-compat: an unrecognised key is preserved from `a` when present,
    # so a new API field is never silently dropped.
    assert merged["service_tier"] == "standard"
    assert merged["future_field"] == [1, 2]


def test_merge_usage_refuses_a_bool_in_a_token_counter(tmp):
    merged = _PS._merge_usage_max({"output_tokens": 5}, {"output_tokens": True})

    # isinstance(True, int) is True, so a naive `>` would let a JSON `true`
    # replace a real count. The explicit type() check is what prevents it.
    assert merged["output_tokens"] == 5


def test_flatten_usage_sums_the_iterations_array(tmp):
    usage = {"input_tokens": 10, "output_tokens": 1,
             "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0,
             "iterations": [
                 {"input_tokens": 10, "output_tokens": 1,
                  "cache_read_input_tokens": 5},
                 {"input_tokens": 30, "output_tokens": 7,
                  "cache_read_input_tokens": 5},
             ]}

    out = _PS._flatten_usage(usage)

    # The top-level fields are a partial snapshot of the last iteration only;
    # billing needs the sum across all of them.
    assert out["input_tokens"] == 40
    assert out["output_tokens"] == 8
    assert out["cache_read_input_tokens"] == 10
    assert out["cache_creation_input_tokens"] == 0
    assert usage["input_tokens"] == 10, "the input usage dict must not be mutated"


def test_flatten_usage_sums_nested_dicts_across_iterations(tmp):
    out = _PS._flatten_usage({"iterations": [
        {"cache_creation": {"ephemeral_5m_input_tokens": 100},
         "server_tool_use": {"web_search_requests": 1}},
        {"cache_creation": {"ephemeral_5m_input_tokens": 25,
                            "ephemeral_1h_input_tokens": 7},
         "server_tool_use": {"web_search_requests": 2}},
        {"cache_creation": None},
    ]})

    # Nested counters SUM (unlike the max merge across requestId records).
    assert out["cache_creation"] == {"ephemeral_5m_input_tokens": 125,
                                     "ephemeral_1h_input_tokens": 7}
    assert out["server_tool_use"] == {"web_search_requests": 3}


def test_flatten_usage_passes_a_plain_record_straight_through(tmp):
    usage = {"input_tokens": 10, "output_tokens": 2}

    assert _PS._flatten_usage(usage) is usage
    assert _PS._flatten_usage({"input_tokens": 3, "iterations": []})["input_tokens"] == 3


def test_flatten_usage_treats_a_missing_counter_as_zero(tmp):
    out = _PS._flatten_usage({"iterations": [{"output_tokens": 4}, {}]})

    assert out["output_tokens"] == 4
    assert out["input_tokens"] == 0


# --- _close_torn_prefix / _iter_json_objects --------------------------------

def test_close_torn_prefix_closes_nested_containers(tmp):
    obj = _PS._close_torn_prefix('{"a": {"b": [1, 2')

    assert obj == {"a": {"b": [1, 2]}, "_torn_truncated": True}


def test_close_torn_prefix_closes_a_string_cut_mid_value(tmp):
    obj = _PS._close_torn_prefix('{"type": "system", "content": "half a mess')

    assert obj["content"] == "half a mess"
    # Consumers key off this marker to know the record is partial.
    assert obj["_torn_truncated"] is True


def test_close_torn_prefix_ignores_brackets_inside_a_string(tmp):
    obj = _PS._close_torn_prefix('{"cmd": "echo {[ ok')

    # A brace inside a quoted value is data, not nesting: counting it would
    # append a spurious `}` and break the round-trip.
    assert obj == {"cmd": "echo {[ ok", "_torn_truncated": True}


def test_close_torn_prefix_closes_only_what_is_still_open(tmp):
    nested = _PS._close_torn_prefix('{"a": {"b": 1}, "c": "cut')
    array = _PS._close_torn_prefix('{"a": [1, 2], "c": "cut')

    # Containers that already closed inside the prefix must not collect a
    # second closer -- one spurious `}` invalidates the whole salvage.
    assert nested == {"a": {"b": 1}, "c": "cut", "_torn_truncated": True}
    assert array == {"a": [1, 2], "c": "cut", "_torn_truncated": True}


def test_close_torn_prefix_does_not_end_a_string_on_an_escaped_quote(tmp):
    obj = _PS._close_torn_prefix('{"a": "he said \\"hi\\" then')

    # Treating the escaped quote as the end of the string would leave the
    # tracker outside a string it is still inside, and every brace in the
    # remaining text would then be counted as nesting.
    assert obj == {"a": 'he said "hi" then', "_torn_truncated": True}


def test_close_torn_prefix_gives_up_on_an_unsalvageable_cut(tmp):
    # Cut mid-key, mid-number and mid-escape: closing braces cannot repair any
    # of these, and a wrong guess would invent a record that never existed.
    assert _PS._close_torn_prefix('{"a": {"b') is None
    assert _PS._close_torn_prefix('{"a": 12, "b":') is None
    assert _PS._close_torn_prefix('{"a": "x\\') is None
    assert _PS._close_torn_prefix('') is None


def test_close_torn_prefix_leaves_a_non_dict_unmarked(tmp):
    obj = _PS._close_torn_prefix('[1, 2, 3')

    assert obj == [1, 2, 3], "a torn array must round-trip without a dict marker"


def test_iter_json_objects_yields_both_objects_on_a_concatenated_line(tmp):
    objs = list(_PS._iter_json_objects('{"a": 1}{"b": 2}'))

    assert objs == [{"a": 1}, {"b": 2}]


def test_iter_json_objects_tolerates_whitespace_between_records(tmp):
    objs = list(_PS._iter_json_objects('  {"a": 1}  \t {"b": 2}   '))

    assert objs == [{"a": 1}, {"b": 2}]


def test_iter_json_objects_salvages_a_torn_record_then_resyncs(tmp):
    line = '{"type": "system", "content": "cut off{"type": "user", "uuid": "u2"}'

    objs = list(_PS._iter_json_objects(line))

    # The salvaged header keeps the fields that made it to disk, and the
    # complete record that got appended is still returned in full.
    assert len(objs) == 2, objs
    assert objs[0] == {"type": "system", "content": "cut off",
                       "_torn_truncated": True}
    assert objs[1] == {"type": "user", "uuid": "u2"}


def test_iter_json_objects_skips_an_unbalanced_leading_brace(tmp):
    objs = list(_PS._iter_json_objects('}}{"b": 2}'))

    assert objs == [{"b": 2}]


def test_iter_json_objects_yields_nothing_when_nothing_decodes(tmp):
    assert list(_PS._iter_json_objects('{"broken')) == []
    assert list(_PS._iter_json_objects('not json at all')) == []
    assert list(_PS._iter_json_objects('')) == []
    assert list(_PS._iter_json_objects('   ')) == []


# --- _process_record_objs ---------------------------------------------------

def test_process_record_objs_reports_an_undecodable_line_as_a_parse_error(tmp):
    events = []

    _PS._process_record_objs([], 12, events, [], {}, None)

    assert len(events) == 1
    assert events[0]["type"] == "parse_error"
    assert events[0]["line"] == 12
    # The line number is in the text because that is how a reader finds it.
    assert "line 12" in events[0]["detail"], events[0]["detail"]


def test_process_record_objs_skips_noise_record_types(tmp):
    events, meta = [], []

    _PS._process_record_objs(
        [{"type": "progress", "message": {"role": "user", "content": "x"}},
         {"type": "file-history-snapshot"}], 1, events, meta, {}, None)

    assert events == []
    assert meta == []


def test_process_record_objs_dedupes_by_uuid_when_a_set_is_given(tmp):
    rec = _user_rec("only once", uuid="dup1")
    seen = set()
    events = []

    _PS._process_record_objs([rec], 1, events, [], {}, seen)
    _PS._process_record_objs([rec], 2, events, [], {}, seen)

    assert len(events) == 1, _types(events)
    assert seen == {"dup1"}


def test_process_record_objs_keeps_a_record_with_no_uuid(tmp):
    rec = {"type": "user", "timestamp": TS.format(1),
           "message": {"role": "user", "content": "legacy"}}
    seen = set()
    events = []

    _PS._process_record_objs([rec], 1, events, [], {}, seen)
    _PS._process_record_objs([rec], 2, events, [], {}, seen)

    # Dedup keys on uuid; a record without one must not be collapsed away.
    assert len(events) == 2


# --- extract_events (through real files) ------------------------------------

def test_extract_events_parses_a_written_transcript(tmp):
    path = _write(tmp, "s.jsonl", [
        _user_rec("do it", uuid="u1", second=1),
        _assistant_rec([{"type": "text", "text": "on it"},
                        {"type": "thinking", "thinking": "plan"},
                        {"type": "tool_use", "id": "tu1", "name": "Bash",
                         "input": {"command": "ls"}}],
                       uuid="a1", second=2,
                       usage={"input_tokens": 11, "output_tokens": 3}),
        _result_rec("tu1", "a b c", uuid="r1", second=3),
        {"type": "system", "subtype": "local_command", "content": "/stats",
         "timestamp": TS.format(4)},
    ])

    events, meta = _PS.extract_events(path)

    assert _types(events) == ["user_message", "assistant_text", "thinking",
                              "tool_call", "tool_result"]
    # Line numbers are 1-based over the raw file and are how every report
    # points a reader back at the transcript.
    assert [e["line"] for e in events] == [1, 2, 2, 2, 3]
    assert _of_type(events, "tool_call")[0]["tool_name"] == "Bash"
    assert _of_type(events, "tool_result")[0]["detail"] == "a b c"
    assert _types(meta) == ["assistant_usage", "system"]
    assert meta[0]["usage"]["input_tokens"] == 11
    assert meta[0]["model"] == "claude-x"


def test_extract_events_tags_every_event_with_its_source_path(tmp):
    path = _write(tmp, "src.jsonl", [
        _user_rec("hi"),
        {"type": "system", "subtype": "x", "timestamp": TS.format(1)}])

    events, meta = _PS.extract_events(path)

    assert [e["src"] for e in events] == [path]
    assert [m["src"] for m in meta] == [path]


def test_extract_events_records_a_bad_line_and_keeps_reading(tmp):
    path = _write(tmp, "bad.jsonl", [
        _user_rec("first", uuid="u1", second=1),
        '{"type": "user", "message": {"role": "user", "content": ',
        "this line is not json at all",
        _user_rec("last", uuid="u2", second=2),
    ])

    events, _meta = _PS.extract_events(path)

    # A truncated record must not end the parse: the records after it are the
    # whole reason the file was opened.
    assert _types(events) == ["user_message", "parse_error", "parse_error",
                              "user_message"]
    assert [e["line"] for e in events] == [1, 2, 3, 4]
    assert events[0]["detail"] == "first"
    assert events[3]["detail"] == "last"


def test_extract_events_salvages_a_torn_line_from_a_real_file(tmp):
    torn = ('{"type": "system", "subtype": "local_command", "content": "cut'
            + json.dumps(_user_rec("survivor", uuid="u9", second=2)))
    path = _write(tmp, "torn.jsonl", [_user_rec("before", uuid="u1"), torn])

    events, meta = _PS.extract_events(path)

    assert _types(events) == ["user_message", "user_message"]
    assert events[1]["detail"] == "survivor", "the appended record was lost"
    # The truncated header is still salvaged as a meta record, flagged partial.
    assert [m["type"] for m in meta] == ["system"]
    assert meta[0]["content"] == "cut"
    assert meta[0]["raw"]["_torn_truncated"] is True
    assert not _of_type(events, "parse_error")


def test_extract_events_reads_an_xz_transcript(tmp):
    path = os.path.join(tmp, "archived.jsonl.xz")
    body = "\n".join(json.dumps(r) for r in
                     [_user_rec("compressed", uuid="u1"),
                      _user_rec("second", uuid="u2", second=2)]) + "\n"
    with lzma.open(path, "wb") as fh:
        fh.write(body.encode("utf-8"))

    events, _meta = _PS.extract_events(path)

    assert [e["detail"] for e in events] == ["compressed", "second"]


def test_extract_events_handles_an_empty_file_and_empty_lines(tmp):
    empty = _write(tmp, "empty.jsonl", [])
    blanks = _write(tmp, "blanks.jsonl", ["", "", _user_rec("after the gap")])

    assert _PS.extract_events(empty) == ([], [])
    events, _meta = _PS.extract_events(blanks)
    # An empty line is skipped outright -- no parse_error row -- and the
    # records after it keep their true line numbers.
    assert _types(events) == ["user_message"]
    assert events[0]["line"] == 3


def test_extract_events_merges_usage_across_records_of_one_request(tmp):
    path = _write(tmp, "stream.jsonl", [
        _assistant_rec([{"type": "text", "text": "part"}], uuid="a1", second=1,
                       request_id="req1",
                       usage={"input_tokens": 500, "output_tokens": 2,
                              "cache_read_input_tokens": 40}),
        _assistant_rec([{"type": "text", "text": "rest"}], uuid="a2", second=2,
                       request_id="req1",
                       usage={"input_tokens": 500, "output_tokens": 137,
                              "cache_read_input_tokens": 40}),
    ])

    _events, meta = _PS.extract_events(path)

    usage_events = [m for m in meta if m["type"] == "assistant_usage"]
    # One API call split across two JSONL records is ONE priced turn, and the
    # aggregate output count is the later one -- counting both would double
    # the input tokens and undercount the output.
    assert len(usage_events) == 1, usage_events
    assert usage_events[0]["usage"]["output_tokens"] == 137
    assert usage_events[0]["usage"]["input_tokens"] == 500


def test_extract_events_keeps_records_that_carry_no_request_id(tmp):
    path = _write(tmp, "legacy.jsonl", [
        _assistant_rec([], uuid="a1", second=1, request_id="",
                       usage={"output_tokens": 1}),
        _assistant_rec([], uuid="a2", second=2, request_id="",
                       usage={"output_tokens": 2}),
    ])

    _events, meta = _PS.extract_events(path)

    # Without a requestId there is nothing to merge on, so both stand.
    assert [m["usage"]["output_tokens"] for m in meta] == [1, 2]


def test_extract_events_line_filter_skips_lines_before_parsing(tmp):
    path = _write(tmp, "filter.jsonl", [
        _user_rec("chatter", uuid="u1", second=1),
        _assistant_rec([{"type": "tool_use", "id": "tu1", "name": "Bash",
                         "input": {"command": "ls"}}], uuid="a1", second=2,
                       usage=None),
        _user_rec("more chatter", uuid="u2", second=3),
    ])

    events, _meta = _PS.extract_events(path, line_filter="Bash")

    assert _types(events) == ["tool_call"]
    # The filter drops lines, never renumbers them: L2 must still say 2.
    assert events[0]["line"] == 2


def test_extract_events_line_filter_accepts_a_regex_and_a_compiled_pattern(tmp):
    path = _write(tmp, "regex.jsonl", [
        _user_rec("chatter", uuid="u1", second=1),
        _assistant_rec([{"type": "tool_use", "id": "tu1", "name": "Bash",
                         "input": {}}], uuid="a1", second=2, usage=None),
        _assistant_rec([{"type": "tool_use", "id": "tu2", "name": "Write",
                         "input": {}}], uuid="a2", second=3, usage=None),
    ])

    as_regex, _ = _PS.extract_events(path, line_filter='"Bash"|"Write"')
    compiled, _ = _PS.extract_events(path, line_filter=re.compile(r'"name": "Bash"'))

    # A pattern with metacharacters takes the regex branch; an already-compiled
    # pattern is used as given.
    assert [e["tool_name"] for e in as_regex] == ["Bash", "Write"]
    assert [e["tool_name"] for e in compiled] == ["Bash"]


def test_extract_events_dedupes_the_same_uuid_across_two_files(tmp):
    shared = _user_rec("shared record", uuid="same-uuid", second=1)
    main = _write(tmp, "main.jsonl", [shared,
                                      _user_rec("main only", uuid="m1", second=2)])
    side = _write(tmp, "agent.jsonl", [dict(shared, isSidechain=True),
                                       _user_rec("side only", uuid="s1", second=3)])
    seen = set()

    first, _ = _PS.extract_events(main, seen_uuids=seen)
    second, _ = _PS.extract_events(side, seen_uuids=seen)

    # A subagent file re-records the parent's rows under a different wrapper;
    # the inner uuid is what makes them the same event.
    assert [e["detail"] for e in first] == ["shared record", "main only"]
    assert [e["detail"] for e in second] == ["side only"]

    fresh_a, _ = _PS.extract_events(main)
    fresh_b, _ = _PS.extract_events(side)
    # Passing no set disables the cross-file pass entirely.
    assert len(fresh_a) + len(fresh_b) == 4


# --- annotate_parallel ------------------------------------------------------

def _call(second, line, name="Read", microseconds=0):
    ts = f"2026-07-30T12:00:{second:02d}.{microseconds:06d}Z"
    return {"line": line, "type": "tool_call", "ts": ts, "tool_name": name,
            "tool_use_id": f"tu{line}", "tool_input": {}, "detail": ""}


def test_annotate_parallel_groups_calls_dispatched_together(tmp):
    events = [_call(1, 1), _call(1, 2), _call(1, 3), _call(30, 4)]

    _PS.annotate_parallel(events)

    assert [e["batch_index"] for e in events] == [1, 2, 3, 1]
    assert [e["batch_size"] for e in events] == [3, 3, 3, 1]


def test_annotate_parallel_splits_exactly_at_the_two_second_boundary(tmp):
    just_inside = [_call(0, 1), _call(1, 2, microseconds=999999)]
    on_the_line = [_call(0, 1), _call(2, 2)]

    _PS.annotate_parallel(just_inside)
    _PS.annotate_parallel(on_the_line)

    # The window is `< 2.0`, so 1.999999s batches and 2.0s starts a new one.
    assert [e["batch_size"] for e in just_inside] == [2, 2]
    assert [e["batch_size"] for e in on_the_line] == [1, 1]


def test_annotate_parallel_ignores_non_tool_events_between_calls(tmp):
    events = [_call(1, 1),
              {"line": 2, "type": "user_message", "ts": TS.format(1),
               "detail": "x"},
              _call(1, 3)]

    _PS.annotate_parallel(events)

    # Batching is over tool calls only; an interleaved message must not split
    # a batch that the timestamps say was dispatched at once.
    assert events[0]["batch_size"] == 2
    assert events[2]["batch_index"] == 2
    assert "batch_size" not in events[1]


def test_annotate_parallel_batches_an_agent_spawn_with_a_tool_call(tmp):
    spawn = dict(_call(1, 2), type="agent_spawn", agent_name="scout")
    events = [_call(1, 1), spawn]

    _PS.annotate_parallel(events)

    assert spawn["batch_size"] == 2
    assert spawn["batch_index"] == 2


def test_annotate_parallel_starts_a_new_batch_on_an_unusable_timestamp(tmp):
    events = [_call(1, 1), dict(_call(1, 2), ts="")]

    _PS.annotate_parallel(events)

    # An unparseable stamp proves nothing about proximity, so the safe answer
    # is a separate batch rather than a claimed one.
    assert [e["batch_size"] for e in events] == [1, 1]


def test_annotate_parallel_leaves_an_event_list_with_no_calls_alone(tmp):
    events = [{"line": 1, "type": "user_message", "ts": TS.format(1),
               "detail": "x"}]

    _PS.annotate_parallel(events)
    _PS.annotate_parallel([])

    assert events == [{"line": 1, "type": "user_message", "ts": TS.format(1),
                       "detail": "x"}]


# --- link_pairs -------------------------------------------------------------

def _result(tool_use_id, line=9, detail="ok", is_error=False):
    return {"line": line, "type": "tool_result", "ts": TS.format(5),
            "tool_use_id": tool_use_id, "detail": detail, "is_error": is_error}


def test_link_pairs_links_a_call_and_its_result_both_ways(tmp):
    call = _call(1, 1)
    result = _result("tu1")

    _PS.link_pairs([call, result])

    assert result["paired_call"] is call
    assert call["paired_result"] is result


def test_link_pairs_leaves_an_unmatched_result_and_call_unlinked(tmp):
    call = _call(1, 1)
    orphan = _result("tu-nonexistent")

    _PS.link_pairs([call, orphan])

    # --tool-orphans reads exactly this absence; inventing an empty dict here
    # would make every dropped call look answered.
    assert "paired_call" not in orphan
    assert "paired_result" not in call


def test_link_pairs_ignores_events_without_a_tool_use_id(tmp):
    call = dict(_call(1, 1), tool_use_id="")
    result = dict(_result(""), tool_use_id="")

    _PS.link_pairs([call, result])

    assert "paired_call" not in result
    assert "paired_result" not in call


# --- format_event / format_pair / _event_side_label -------------------------

def _event(etype, detail, line=7, ts=None, **over):
    ev = {"line": line, "type": etype, "ts": ts or TS.format(1), "detail": detail}
    ev.update(over)
    return ev


def test_format_event_renders_each_conversational_type(tmp):
    stamp = "[2026-07-30 12:00:01]"

    assert _PS.format_event(_event("user_message", "hi")) == \
        f"L7    {stamp} USER: hi"
    assert _PS.format_event(_event("assistant_text", "ok")) == \
        f"L7    {stamp} ASSISTANT: ok"
    assert _PS.format_event(_event("thinking", "hmm")) == \
        f"L7    {stamp} THINKING: hmm"
    assert _PS.format_event(_event("tool_result", "out")) == \
        f"L7    {stamp} RESULT: out"


def test_format_event_pads_the_line_number_to_a_fixed_column(tmp):
    narrow = _PS.format_event(_event("user_message", "hi", line=7))
    wide = _PS.format_event(_event("user_message", "hi", line=12345))

    # Left-aligned in width 4 so short line numbers still line up; a longer
    # number overflows rather than being truncated.
    assert narrow.startswith("L7    ["), narrow
    assert wide.startswith("L12345 ["), wide


def test_format_event_renders_a_tool_call_from_its_input(tmp):
    call = _PS._build_tool_call_event(
        {"name": "Bash", "id": "tu1", "input": {"command": "ls"}},
        7, TS.format(1))

    out = _PS.format_event(call)

    assert out == 'L7    [2026-07-30 12:00:01] TOOL: Bash({"command": "ls"})', out


def test_format_event_tags_a_member_of_a_parallel_batch(tmp):
    solo = _event("tool_call", "Read(...)", batch_size=1, batch_index=1)
    member = _event("tool_call", "Read(...)", batch_size=3, batch_index=2)

    # A batch of one is not a batch and must not be tagged.
    assert "parallel" not in _PS.format_event(solo)
    assert "[parallel 2/3]" in _PS.format_event(member)
    assert "TOOL [parallel 2/3]:" in _PS.format_event(member)


def test_format_event_marks_an_errored_result(tmp):
    clean = _PS.format_event(_event("tool_result", "out", is_error=False))
    failed = _PS.format_event(_event("tool_result", "out", is_error=True))

    assert "[ERROR]" not in clean
    assert "RESULT [ERROR]: out" in failed


def test_format_event_renders_a_parse_error_without_a_timestamp(tmp):
    out = _PS.format_event(_event("parse_error", "Invalid JSON at line 4", ts=""))

    # A torn line has no usable timestamp, so the slot carries the marker
    # instead of a row of question marks.
    assert out == "L7    [ERROR] Invalid JSON at line 4", out


def test_format_event_falls_back_to_the_type_name_for_an_unknown_event(tmp):
    out = _PS.format_event(_event("assistant_usage", "42 tokens"))

    assert out == "L7    [2026-07-30 12:00:01] assistant_usage: 42 tokens", out


def test_format_event_with_source_strips_the_transcript_suffix(tmp):
    plain = _event("user_message", "hi", src=os.path.join("d", "abc123.jsonl"))
    archived = _event("user_message", "hi",
                      src=os.path.join("d", "abc123.jsonl.xz"))

    # Both spellings of the same session must produce the SAME tag, or folder
    # mode reports one session under two names.
    assert "<abc123> " in _PS.format_event(plain, with_source=True)
    assert "<abc123> " in _PS.format_event(archived, with_source=True)
    # The tag sits after L<n> so block detection on ^L keeps working.
    assert _PS.format_event(plain, with_source=True).startswith("L7    <abc123> [")
    assert "<" not in _PS.format_event(plain)


def test_format_event_omits_the_source_tag_when_no_source_is_known(tmp):
    out = _PS.format_event(_event("user_message", "hi"), with_source=True)

    assert out == "L7    [2026-07-30 12:00:01] USER: hi", out


def test_format_event_verbose_appends_the_agent_prompt(tmp):
    spawn = _PS._build_tool_call_event({
        "name": "Agent", "id": "tu1",
        "input": {"name": "scout", "prompt": "go look at the thing"},
    }, 7, TS.format(1))

    quiet = _PS.format_event(spawn)
    loud = _PS.format_event(spawn, verbose=True)

    assert quiet.startswith("L7    [2026-07-30 12:00:01] AGENT_SPAWN: Agent(scout,")
    assert "go look at the thing" not in quiet
    assert "go look at the thing" in loud
    # The header states the prompt length so a reader can see truncation.
    assert "PROMPT (20 chars):" in loud, loud


def test_format_event_verbose_adds_nothing_for_a_promptless_spawn(tmp):
    spawn = _PS._build_tool_call_event(
        {"name": "Agent", "id": "tu1", "input": {"name": "scout"}},
        7, TS.format(1))

    assert _PS.format_event(spawn, verbose=True) == _PS.format_event(spawn)


def test_format_pair_previews_the_result_under_the_call(tmp):
    call = _event("tool_call", "Bash(...)", line=7)
    result = _event("tool_result", "the output", line=9)

    out = _PS.format_pair(call, result)
    lines = out.split("\n")

    assert lines[0] == _PS.format_event(call)
    assert lines[1] == "  └─ RESULT (L9, 10 chars): the output", lines[1]


def test_format_pair_truncates_a_long_preview_at_200_chars(tmp):
    call = _event("tool_call", "Bash(ls)")
    exact = _PS.format_pair(call, _event("tool_result", "x" * 200, line=9))
    over = _PS.format_pair(call, _event("tool_result", "y" * 201, line=9))

    # The boundary is `> 200`, so exactly 200 chars is shown whole.
    assert exact.endswith("x" * 200), exact[-20:]
    assert "..." not in exact
    assert over.endswith("y" * 200 + "..."), over[-20:]
    # The char count always reports the FULL body, not the preview.
    assert "(L9, 201 chars):" in over


def test_format_pair_verbose_shows_the_whole_body_on_its_own_line(tmp):
    call = _event("tool_call", "Bash(ls)")
    result = _event("tool_result", "z" * 500, line=9, is_error=True)

    out = _PS.format_pair(call, result, verbose=True)

    assert "z" * 500 in out
    assert "..." not in out
    assert "RESULT [ERROR] (L9, 500 chars):\n" in out, out[:200]


def test_format_pair_says_so_when_the_result_is_missing(tmp):
    out = _PS.format_pair(_event("tool_call", "Bash(...)"), None)

    # An orphaned call must read as unanswered rather than as a blank result.
    assert out.endswith("\n  └─ RESULT: (not found)"), out


def test_format_pair_passes_verbose_and_source_down_to_the_call_line(tmp):
    call = _PS._build_tool_call_event({
        "name": "Agent", "id": "tu1",
        "input": {"name": "scout", "prompt": "look"},
    }, 7, TS.format(1))
    call["src"] = os.path.join("d", "sess.jsonl")

    out = _PS.format_pair(call, None, verbose=True, with_source=True)

    assert "<sess> " in out
    assert "PROMPT (4 chars):" in out


def test_event_side_label_names_the_tool_behind_each_event(tmp):
    call = _event("tool_call", "", tool_name="Bash")
    spawn = _event("agent_spawn", "", agent_name="scout")

    assert _PS._event_side_label(call) == "TOOL Bash"
    assert _PS._event_side_label(spawn) == "AGENT scout"
    assert _PS._event_side_label(_event("user_message", "hi")) == "USER"
    assert _PS._event_side_label(_event("assistant_text", "hi")) == "ASSISTANT"
    assert _PS._event_side_label(_event("thinking", "hi")) == "THINKING"
    # An unlabelled type falls through to its own name rather than "?".
    assert _PS._event_side_label(_event("assistant_usage", "")) == "assistant_usage"


def test_event_side_label_reaches_through_a_result_to_its_call(tmp):
    call = _event("tool_call", "", tool_name="Edit")
    linked = _event("tool_result", "out", paired_call=call)
    failed = _event("tool_result", "out", paired_call=call, is_error=True)
    orphan = _event("tool_result", "out")
    agent_result = _event("tool_result", "out",
                          paired_call={"agent_name": "scout"})

    assert _PS._event_side_label(linked) == "RESULT Edit"
    assert _PS._event_side_label(failed) == "RESULT Edit [ERR]"
    # An unpaired result still labels, because the gap report prints it either
    # way and a KeyError there would kill the whole mode.
    assert _PS._event_side_label(orphan) == "RESULT ?"
    assert _PS._event_side_label(agent_result) == "RESULT scout"


def test_format_and_label_agree_on_a_transcript_parsed_end_to_end(tmp):
    path = _write(tmp, "e2e.jsonl", [
        _user_rec("run it", uuid="u1", second=1),
        _assistant_rec([{"type": "tool_use", "id": "tu1", "name": "Bash",
                         "input": {"command": "ls"}},
                        {"type": "tool_use", "id": "tu2", "name": "Read",
                         "input": {"file_path": "a.py"}}],
                       uuid="a1", second=2, usage=None),
        _result_rec("tu2", "file body", uuid="r1", second=3),
    ])

    events, _meta = _PS.extract_events(path)
    _PS.annotate_parallel(events)
    _PS.link_pairs(events)
    calls = _of_type(events, "tool_call")

    # The full ingest path, then the formatting the reports print: the two
    # tool_uses in one record share a timestamp, so they are one batch of 2,
    # and only the second one was answered.
    assert [c["batch_index"] for c in calls] == [1, 2]
    assert "[parallel 1/2]" in _PS.format_event(calls[0], with_source=True)
    assert "<e2e> " in _PS.format_event(calls[0], with_source=True)
    assert _PS.format_pair(calls[0], calls[0].get("paired_result")).endswith(
        "RESULT: (not found)")
    assert "(L3, 9 chars): file body" in _PS.format_pair(
        calls[1], calls[1]["paired_result"])
    assert _PS._event_side_label(calls[1]["paired_result"]) == "RESULT Read"


def main():
    return _util.runner(_util.collect(globals()), tmp_prefix="extract_")


if __name__ == "__main__":
    raise SystemExit(main())
