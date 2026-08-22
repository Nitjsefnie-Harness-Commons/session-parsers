"""Suite for parse_session.py's timing passes: tool latency, reply latency,
turn tables and context growth.

These passes are arithmetic — bucket edges, means, medians, p95, per-word
effect sizes, per-turn deltas — and a wrong number still renders as a number,
so nothing here asserts merely that a report came back. Every assertion is on
a value the report states: the count in a histogram bucket, the mean on a
summary line, the delta in a growth row, the order of the rows.

The classifiers feeding reply latency get the most attention. A hook-injected
system reminder, a `[Request interrupted by user]` marker and a replayed
re-dispatch prompt must all be kept out of "how long did the model take to
answer a human", and each exclusion is pinned differentially: the same events
with the exclusion defeated produce a visibly different number.

Synthetic input throughout — event dicts in the shapes the parser emits, and,
where a pass reads source files itself, real JSONL written into the test's own
tmp dir. CI runners have no transcripts to borrow.
"""
# pylint: disable=protected-access
import datetime as dt
import json
import os
import re

import _util

_PS = _util.load(_util.script("parse_session.py"))

_BASE = dt.datetime(2026, 7, 30, 12, 0, 0, tzinfo=dt.timezone.utc)


def _ts(offset):
    """ISO timestamp `offset` seconds after a fixed base — never the clock."""
    return (_BASE + dt.timedelta(seconds=offset)).isoformat().replace("+00:00", "Z")


# --- shared report readers --------------------------------------------------

def _line(report, prefix):
    """The first report line starting with `prefix`, stripped."""
    for ln in report.splitlines():
        if ln.strip().startswith(prefix):
            return ln.strip()
    raise AssertionError(f"no {prefix!r} line in report:\n{report}")


def _section_rows(report, header_prefix):
    """Non-blank lines following `header_prefix`, up to the next blank line."""
    out = []
    started = False
    for ln in report.splitlines():
        if not started:
            started = ln.strip().startswith(header_prefix)
            continue
        if not ln.strip():
            break
        out.append(ln)
    return out


# --- _bucket_label ----------------------------------------------------------

_SPEC = [("low", 0.0, 1.0), ("high", 1.0, 2.0)]


def test_bucket_label_is_lower_inclusive_upper_exclusive(tmp):
    assert _PS._bucket_label(0.0, _SPEC) == "low"
    assert _PS._bucket_label(0.999, _SPEC) == "low"
    # The edge value belongs to the bucket it opens, not the one it closes.
    assert _PS._bucket_label(1.0, _SPEC) == "high"
    assert _PS._bucket_label(1.999, _SPEC) == "high"


def test_bucket_label_returns_none_outside_every_bucket(tmp):
    assert _PS._bucket_label(2.0, _SPEC) is None
    assert _PS._bucket_label(-0.5, _SPEC) is None


def test_bucket_label_edges_match_exactly_one_bucket(tmp):
    """Any value may match at most one bucket, edges included.

    Overlapping intervals would double-count a duration in the histogram
    while the total count stayed right, so the table would not visibly
    disagree with itself.
    """
    spec = [("a", 0.0, 0.1), ("b", 0.1, 1.0), ("c", 1.0, float("inf"))]
    for value in (0.0, 0.05, 0.1, 0.5, 1.0, 1e9):
        matches = [lbl for lbl, lo, hi in spec if lo <= value < hi]
        assert len(matches) == 1, (value, matches)
        assert _PS._bucket_label(value, spec) == matches[0], value


# --- compute_latency --------------------------------------------------------

def _call(name, start, end, line=1, etype="tool_call", **over):
    """A tool call paired with a result `end - start` seconds later."""
    event = {
        "type": etype, "ts": _ts(start), "line": line,
        "tool_use_id": f"tu{line}", "tool_input": {},
        "paired_result": {"ts": _ts(end)},
    }
    if name is not None:
        event["tool_name"] = name
    event.update(over)
    return event


_LAT_BUCKETS = ["[0, 100ms)", "[100ms, 1s)", "[1s, 5s)",
                "[5s, 30s)", "[30s, 2m)", "[2m, inf)"]


def _tool_row(report, tool):
    """(n, avg, max, [6 bucket counts]) from the PER-TOOL HISTOGRAM row.

    The row is `name n avg max b0..b5`; only the histogram rows start with a
    bare tool name (slowest-call rows start with a date, the header with
    'tool'), so first-token matching is unambiguous.
    """
    for ln in report.splitlines():
        parts = ln.split()
        if len(parts) == 10 and parts[0] == tool:
            return int(parts[1]), parts[2], parts[3], [int(p) for p in parts[4:]]
    raise AssertionError(f"no histogram row for {tool!r}:\n{report}")


def test_latency_bucket_edges_land_in_the_opening_bucket(tmp):
    """A duration exactly on an edge counts once, in the upper bucket.

    Every bucket lower bound is used as a duration, so the [0, 100ms) bucket
    must come back empty and every other bucket must hold exactly one call.
    """
    events = [_call("Bash", 0, edge, line=i)
              for i, edge in enumerate([0.1, 1.0, 5.0, 30.0, 120.0], start=1)]

    report = _PS.compute_latency(events)

    # The column order is positional in the row, so pin the header labels
    # first — otherwise the counts below would be read against the wrong
    # columns if a bucket were ever renamed or reordered.
    header = _line(report, "tool")
    assert [lbl for lbl in _LAT_BUCKETS if lbl in header] == _LAT_BUCKETS, header
    assert [header.index(lbl) for lbl in _LAT_BUCKETS] == \
        sorted(header.index(lbl) for lbl in _LAT_BUCKETS), header

    n, _avg, _mx, buckets = _tool_row(report, "Bash")
    assert n == 5
    assert dict(zip(_LAT_BUCKETS, buckets)) == {
        "[0, 100ms)": 0, "[100ms, 1s)": 1, "[1s, 5s)": 1,
        "[5s, 30s)": 1, "[30s, 2m)": 1, "[2m, inf)": 1}, buckets


def test_latency_sub_edge_duration_falls_in_the_first_bucket(tmp):
    _n, _avg, _mx, buckets = _tool_row(
        _PS.compute_latency([_call("Bash", 0, 0.099, line=1)]), "Bash")

    assert dict(zip(_LAT_BUCKETS, buckets))["[0, 100ms)"] == 1, buckets
    assert sum(buckets) == 1, buckets


def test_latency_reports_per_tool_count_average_and_max(tmp):
    events = [_call("Bash", 0, 1, line=1), _call("Bash", 0, 3, line=2)]

    n, avg, mx, _buckets = _tool_row(_PS.compute_latency(events), "Bash")

    assert (n, avg, mx) == (2, "2.00s", "3.00s")


def test_latency_averages_only_over_the_tools_own_calls(tmp):
    """A second tool's durations must not leak into the first tool's mean."""
    events = [_call("Bash", 0, 1, line=1), _call("Read", 0, 9, line=2)]

    report = _PS.compute_latency(events)

    assert _tool_row(report, "Bash")[1] == "1.00s"
    assert _tool_row(report, "Read")[1] == "9.00s"


def test_latency_orders_tools_by_total_time_spent(tmp):
    """Ranking by total, not by call count: Read is slower in aggregate even
    though Bash was called more often."""
    events = [_call("Bash", 0, 1, line=1), _call("Bash", 0, 1, line=2),
              _call("Bash", 0, 1, line=3), _call("Read", 0, 10, line=4)]

    rows = [ln.split()[0] for ln in _PS.compute_latency(events).splitlines()
            if len(ln.split()) == 10 and ln.split()[0] in ("Bash", "Read")]

    assert rows == ["Read", "Bash"], rows


def test_latency_clamps_a_negative_duration_to_zero(tmp):
    """A result timestamped before its call (clock skew / compaction) must not
    produce a negative latency or vanish from the counts."""
    n, avg, mx, buckets = _tool_row(
        _PS.compute_latency([_call("Bash", 5, 1, line=1)]), "Bash")

    assert (n, avg, mx) == (1, "0.00s", "0.00s")
    assert dict(zip(_LAT_BUCKETS, buckets))["[0, 100ms)"] == 1, buckets
    assert sum(buckets) == 1, buckets


def test_latency_slowest_list_is_capped_at_twenty(tmp):
    events = [_call("Bash", 0, i, line=i) for i in range(1, 26)]

    rows = _section_rows(_PS.compute_latency(events), "SLOWEST 20 CALLS:")[1:]

    assert len(rows) == 20, len(rows)
    assert "25.00s" in rows[0], rows[0]
    assert "6.00s" in rows[-1], rows[-1]
    # The five fastest calls are the ones dropped.
    assert not any("  5.00s" in r for r in rows), rows


def test_latency_names_an_untitled_agent_spawn_agent(tmp):
    """agent_spawn carries no tool_name of its own in older records."""
    events = [_call(None, 0, 4, line=1, etype="agent_spawn")]

    assert _tool_row(_PS.compute_latency(events), "Agent")[0] == 1


def test_latency_skips_calls_with_an_unparseable_timestamp(tmp):
    events = [_call("Bash", 0, 1, line=1, ts="not-a-timestamp")]

    assert "(no paired tool call/result events)" in _PS.compute_latency(events)


def test_latency_skips_calls_with_no_paired_result(tmp):
    events = [_call("Bash", 0, 1, line=1, paired_result=None)]

    assert "(no paired tool call/result events)" in _PS.compute_latency(events)


def test_latency_ignores_non_tool_events(tmp):
    """A user message carrying a paired_result is not a tool call."""
    events = [{"type": "user_message", "ts": _ts(0), "line": 1,
               "detail": "hi", "paired_result": {"ts": _ts(9)}}]

    assert "(no paired tool call/result events)" in _PS.compute_latency(events)


# --- reply-latency classifiers ----------------------------------------------

def test_is_instrumentation_user_msg_matches_every_prefix(tmp):
    for prefix in _PS._INSTRUMENTATION_USER_PREFIXES:
        event = {"type": "user_message", "detail": prefix + "payload"}
        assert _PS._is_instrumentation_user_msg(event), prefix


def test_is_instrumentation_user_msg_tolerates_leading_whitespace(tmp):
    event = {"type": "user_message", "detail": "\n  <bash-stdout>ok"}

    assert _PS._is_instrumentation_user_msg(event)


def test_is_instrumentation_user_msg_rejects_other_input(tmp):
    assert not _PS._is_instrumentation_user_msg(
        {"type": "user_message", "detail": "please fix the parser"})
    assert not _PS._is_instrumentation_user_msg({"type": "user_message"})
    # Type matters: the same text on an assistant event is not instrumentation.
    assert not _PS._is_instrumentation_user_msg(
        {"type": "assistant_text", "detail": "<bash-input>ls"})


def test_is_interrupt_user_msg_matches_both_marker_spellings(tmp):
    for text in ("[Request interrupted by user]",
                 "[Request interrupted by user for tool use]",
                 "  [Request interrupted by user]"):
        assert _PS._is_interrupt_user_msg(
            {"type": "user_message", "detail": text}), text


def test_is_interrupt_user_msg_rejects_other_input(tmp):
    assert not _PS._is_interrupt_user_msg(
        {"type": "user_message", "detail": "the request was interrupted"})
    assert not _PS._is_interrupt_user_msg(
        {"type": "assistant_text", "detail": "[Request interrupted by user]"})


def test_is_word_drops_short_numeric_and_stopword_tokens(tmp):
    assert _PS._is_word("parser")
    assert _PS._is_word("abc")
    assert _PS._is_word("reply-latency")
    assert _PS._is_word("min_count")
    assert not _PS._is_word("ab")           # under three characters
    assert not _PS._is_word("800px")        # digit — dimension/hex junk
    assert not _PS._is_word("71e3af8")
    assert not _PS._is_word("the")          # stopword
    assert not _PS._is_word("don't")        # contraction stopword, kept whole


def test_is_human_prompt_rejects_synthetic_user_records(tmp):
    for marker in ("<task-notification>", "<observed_from_primary_session>",
                   "<system-reminder>", "<post-tool-use>", "<bash-input>",
                   "<command-name>"):
        assert not _PS._is_human_prompt({"detail": marker + "x"}), marker


def test_is_human_prompt_accepts_typed_text_and_rejects_empty(tmp):
    assert _PS._is_human_prompt({"detail": "  refactor the parser"})
    assert not _PS._is_human_prompt({"detail": ""})
    assert not _PS._is_human_prompt({"detail": None})
    assert not _PS._is_human_prompt({})


# --- _replayed_user_message_lines -------------------------------------------

def _u(text, at, line, uuid="", src="a.jsonl", **over):
    event = {"type": "user_message", "ts": _ts(at), "line": line,
             "detail": text, "uuid": uuid, "src": src}
    event.update(over)
    return event


def _a(text, at, line, model="m1", src="a.jsonl", **over):
    event = {"type": "assistant_text", "ts": _ts(at), "line": line,
             "detail": text, "model": model, "src": src}
    event.update(over)
    return event


def test_replayed_lines_flags_a_duplicate_uuid_at_a_later_line(tmp):
    events = [_u("prompt", 0, 1, uuid="A"), _u("prompt", 0, 7, uuid="A")]

    assert _PS._replayed_user_message_lines(events) == {("a.jsonl", 7)}


def test_replayed_lines_ignores_a_duplicate_uuid_on_the_same_line(tmp):
    """One record with several text blocks emits several events at one line —
    that is not a re-dispatch replay."""
    events = [_u("block one", 0, 1, uuid="A"), _u("block two", 0, 1, uuid="A")]

    assert _PS._replayed_user_message_lines(events) == set()


def test_replayed_lines_ignores_records_without_a_uuid(tmp):
    events = [_u("prompt", 0, 1), _u("prompt", 0, 7)]

    assert _PS._replayed_user_message_lines(events) == set()


def test_replayed_lines_ignores_non_user_events(tmp):
    events = [_a("reply", 0, 1, uuid="A"), _a("reply", 0, 7, uuid="A")]

    assert _PS._replayed_user_message_lines(events) == set()


def test_replayed_lines_separates_sources(tmp):
    """Two session files may legitimately carry the same uuid — the dedup key
    is (src, uuid), so that is not a replay within either file."""
    events = [_u("prompt", 0, 1, uuid="A", src="a.jsonl"),
              _u("prompt", 0, 7, uuid="A", src="b.jsonl")]

    assert _PS._replayed_user_message_lines(events) == set()


# --- compute_reply_latency --------------------------------------------------

def _convo(specs, gap=100):
    """Alternating human prompt / assistant reply events.

    `specs` is (prompt, latency_seconds, model); turns are spaced by `gap` so
    each anchor's terminator is its own reply and not the next prompt.
    """
    events = []
    at = 0.0
    line = 1
    for prompt, dur, model in specs:
        events.append(_u(prompt, at, line, uuid=f"u{line}"))
        line += 1
        events.append(_a("ok", at + dur, line, model=model))
        line += 1
        at += dur + gap
    return events


def _summary(report):
    """The count/min/max/median/p95/mean values as a dict of strings."""
    toks = _line(report, "count:").split()
    return {k.rstrip(":"): v for k, v in zip(toks[::2], toks[1::2])}


_REPLY_BUCKETS = ["[0, 0.5s)", "[0.5s, 1s)", "[1s, 2s)", "[2s, 5s)",
                  "[5s, 10s)", "[10s, 30s)", "[30s, 2m)", "[2m, inf)"]


def _reply_buckets(report):
    counts = {}
    for ln in report.splitlines():
        stripped = ln.strip()
        for label in _REPLY_BUCKETS:
            if stripped.startswith(label):
                counts[label] = int(stripped[len(label):].split()[0])
    return counts


def test_reply_latency_summary_statistics_for_an_odd_count(tmp):
    events = _convo([(f"prompt {i}", i, "m1") for i in range(1, 6)])

    report = _PS.compute_reply_latency(events)

    # median = the middle of [1,2,3,4,5]; p95 index = round(0.95 * 4) = 4.
    assert _line(report, "count:") == (
        "count: 5   min: 1.00s   max: 5.00s   "
        "median: 3.00s   p95: 5.00s   mean: 3.00s")


def test_reply_latency_median_of_an_even_count_averages_the_middle_pair(tmp):
    events = _convo([(f"prompt {i}", i, "m1") for i in range(1, 5)])

    assert _summary(_PS.compute_reply_latency(events)) == {
        "count": "4", "min": "1.00s", "max": "4.00s",
        "median": "2.50s", "p95": "4.00s", "mean": "2.50s"}


def test_reply_latency_histogram_edges_land_in_the_opening_bucket(tmp):
    """Each bucket's lower bound is used as a latency, so [0, 0.5s) stays
    empty and every other bucket holds exactly one reply."""
    edges = [0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 120.0]
    events = _convo([(f"prompt {i}", edge, "m1")
                     for i, edge in enumerate(edges)], gap=300)

    counts = _reply_buckets(_PS.compute_reply_latency(events))

    assert counts == {"[0, 0.5s)": 0, "[0.5s, 1s)": 1, "[1s, 2s)": 1,
                      "[2s, 5s)": 1, "[5s, 10s)": 1, "[10s, 30s)": 1,
                      "[30s, 2m)": 1, "[2m, inf)": 1}, counts


def test_reply_latency_instrumentation_never_anchors(tmp):
    """`<bash-input>` and friends arrive in bursts of three sharing a
    timestamp and are flagged 'DO NOT respond'; anchoring on them would
    triple-count every local shell command."""
    events = [_u("<bash-input>ls", 0, 1, uuid="u1"),
              _u("<bash-stdout>a b c", 0, 2, uuid="u2"),
              _u("<bash-stderr>", 0, 3, uuid="u3"),
              _a("ok", 9, 4)]

    report = _PS.compute_reply_latency(events)

    assert "(no user messages with following assistant events)" in report


def test_reply_latency_instrumentation_does_not_break_the_walk(tmp):
    """Instrumentation between a prompt and its reply is stepped over, not
    treated as a superseding user message."""
    events = [_u("please run it", 0, 1, uuid="u1"),
              _u("<bash-input>ls", 1, 2, uuid="u2"),
              _a("ok", 4, 3)]

    assert _summary(_PS.compute_reply_latency(events))["count"] == "1"
    assert _summary(_PS.compute_reply_latency(events))["max"] == "4.00s"


def test_reply_latency_interrupt_terminates_but_does_not_anchor(tmp):
    """The interrupt stops the clock at 7s on the prompt before it; it must
    not then start a new 13s measurement of its own."""
    events = [_u("do the thing", 0, 1, uuid="u1"),
              _u("[Request interrupted by user]", 7, 2, uuid="u2"),
              _a("late reply", 20, 3)]

    summary = _summary(_PS.compute_reply_latency(events))

    assert summary["count"] == "1"
    assert summary["max"] == "7.00s"


def test_reply_latency_replayed_prompt_is_invisible(tmp):
    """A re-dispatch appends the original user record again, with the same
    uuid and its ORIGINAL timestamp, at a later line. Anchoring on it invents
    a latency as long as everything that happened in between.
    """
    def build(second_uuid):
        return [_u("original prompt", 0, 1, uuid="u1"),
                _a("reply", 2, 2),
                _u("original prompt", 0, 3, uuid=second_uuid),
                _a("much later reply", 100, 4)]

    replayed = _summary(_PS.compute_reply_latency(build("u1")))
    distinct = _summary(_PS.compute_reply_latency(build("u2")))

    assert replayed == {"count": "1", "min": "2.00s", "max": "2.00s",
                        "median": "2.00s", "p95": "2.00s", "mean": "2.00s"}
    # Control: with a distinct uuid the same events yield the 100s phantom.
    assert distinct["count"] == "2"
    assert distinct["max"] == "100.00s"


def test_reply_latency_a_later_prompt_supersedes_an_unanswered_one(tmp):
    """The user typed again before any reply arrived; only the second prompt
    was actually answered, so only it may be measured."""
    events = [_u("first attempt", 0, 1, uuid="u1"),
              _u("second attempt", 3, 2, uuid="u2"),
              _a("reply", 10, 3)]

    report = _PS.compute_reply_latency(events)

    assert _summary(report)["count"] == "1"
    assert _summary(report)["max"] == "7.00s"
    assert "first attempt" not in report


def test_reply_latency_drops_a_negative_duration(tmp):
    """A reply timestamped before its prompt is a compaction artifact; the
    pass drops it rather than clamping it to zero."""
    events = [_u("prompt", 10, 1, uuid="u1"), _a("reply", 5, 2)]

    assert "(no user messages with following assistant events)" in \
        _PS.compute_reply_latency(events)


def test_reply_latency_skips_a_prompt_with_no_terminator(tmp):
    events = [_u("prompt", 0, 1, uuid="u1"),
              {"type": "tool_result", "ts": _ts(1), "line": 2,
               "detail": "ok", "src": "a.jsonl"}]

    assert "(no user messages with following assistant events)" in \
        _PS.compute_reply_latency(events)


def test_reply_latency_walks_past_tool_results(tmp):
    """A tool_result is neither terminator nor superseder — the walk continues
    to the assistant event behind it."""
    events = [_u("prompt", 0, 1, uuid="u1"),
              {"type": "tool_result", "ts": _ts(1), "line": 2,
               "detail": "ok", "src": "a.jsonl"},
              _a("reply", 6, 3)]

    assert _summary(_PS.compute_reply_latency(events))["max"] == "6.00s"


def test_reply_latency_thinking_counts_as_the_first_response(tmp):
    """Latency is time-to-first-token, and a thinking block is the model
    starting to answer."""
    events = [_u("prompt", 0, 1, uuid="u1"),
              {"type": "thinking", "ts": _ts(2), "line": 2,
               "detail": "...", "src": "a.jsonl"},
              _a("reply", 30, 3)]

    assert _summary(_PS.compute_reply_latency(events))["max"] == "2.00s"


def test_reply_latency_top_n_truncates_and_sorts_descending(tmp):
    events = _convo([(f"prompt {i}", i, "m1") for i in range(1, 6)])

    rows = _section_rows(
        _PS.compute_reply_latency(events, top_n=2), "SLOWEST 2:")[1:]

    assert len(rows) == 2, rows
    assert "5.00s" in rows[0] and "prompt 5" in rows[0], rows[0]
    assert "4.00s" in rows[1] and "prompt 4" in rows[1], rows[1]


def test_reply_latency_truncates_a_long_prompt_preview(tmp):
    events = _convo([("z" * 200, 3, "m1")])

    row = _section_rows(_PS.compute_reply_latency(events), "SLOWEST 20:")[1]

    assert "z" * 57 + "..." in row, row
    assert "z" * 58 not in row, row


def test_reply_latency_skips_an_anchor_with_an_unparseable_timestamp(tmp):
    events = [_u("prompt", 0, 1, uuid="u1", ts="not-a-timestamp"),
              _a("reply", 5, 2)]

    assert "(no user messages with following assistant events)" in \
        _PS.compute_reply_latency(events)


def test_reply_latency_skips_a_terminator_with_an_unparseable_timestamp(tmp):
    events = [_u("prompt", 0, 1, uuid="u1"), _a("reply", 5, 2, ts="")]

    assert "(no user messages with following assistant events)" in \
        _PS.compute_reply_latency(events)


def test_reply_latency_walk_ignores_events_from_another_source_file(tmp):
    """Even walking a merged list, the terminator search stays inside the
    anchor's own file — otherwise a neighbouring session's first event closes
    the window and reports a latency that never happened."""
    events = [_u("prompt", 0, 1, uuid="u1", src="a.jsonl"),
              _a("someone else's reply", 5, 1, src="b.jsonl"),
              _a("the real reply", 30, 2, src="a.jsonl")]

    assert _summary(_PS.compute_reply_latency(events))["max"] == "30.00s"


def test_reply_latency_walk_steps_over_a_replayed_prompt(tmp):
    """A replay sitting between a prompt and its reply must not supersede the
    prompt — that would silently discard a genuine measurement."""
    def build(fourth_uuid):
        return [_u("context prompt", 0, 1, uuid="B"),
                _a("ok", 1, 2),
                _u("real prompt", 2, 3, uuid="A"),
                _u("context prompt", 3, 4, uuid=fourth_uuid),
                _a("answer", 10, 5)]

    replayed = _PS.compute_reply_latency(build("B"))
    distinct = _PS.compute_reply_latency(build("C"))

    assert _summary(replayed)["count"] == "2"
    assert "8.00s  real prompt" in replayed, replayed
    # Control: a genuinely new prompt there DOES supersede, losing the 8s.
    assert "real prompt" not in distinct, distinct


def test_reply_latency_reports_nothing_for_an_empty_session(tmp):
    report = _PS.compute_reply_latency([])

    assert "(no user messages with following assistant events)" in report
    assert report.startswith("REPLY LATENCY")


# --- _prompt_preview --------------------------------------------------------

def test_prompt_preview_stops_at_the_first_newline(tmp):
    assert _PS._prompt_preview("  hello world\nsecond line") == "hello world"


def test_prompt_preview_honours_the_limit(tmp):
    assert _PS._prompt_preview("x" * 150) == "x" * 100
    assert _PS._prompt_preview("abcdef", limit=3) == "abc"


def test_prompt_preview_rstrips_and_survives_empty_input(tmp):
    # A prompt whose first line is padding must not render as trailing space.
    assert _PS._prompt_preview("hi   \nthere") == "hi"
    assert _PS._prompt_preview("") == ""
    assert _PS._prompt_preview(None) == ""


# --- _word_effect_rows ------------------------------------------------------

def _wpairs(specs):
    return [(dur, {"detail": text}) for text, dur in specs]


_WORD_SPECS = [("refactor the parser", 10.0), ("refactor module", 10.0),
               ("document the parser", 2.0), ("document module", 2.0)]


def test_word_effect_rows_effect_is_with_mean_minus_without_mean(tmp):
    rows, count, mean = _PS._word_effect_rows(_wpairs(_WORD_SPECS), min_count=2)

    assert (count, mean) == (4, 6.0)
    by_word = {r[1]: r for r in rows}
    assert set(by_word) == {"refactor", "document", "parser", "module"}, by_word
    # refactor rides the two 10s prompts, so with=10, without=2, effect=+8.
    eff, _w, n, with_mean, without_mean, preview = by_word["refactor"]
    assert (eff, n, with_mean, without_mean) == (8.0, 2, 10.0, 2.0)
    assert preview == "refactor the parser"
    assert by_word["document"][0] == -8.0
    # parser and module split the fast and slow prompts evenly: no effect.
    assert by_word["parser"][0] == 0.0
    assert by_word["module"][0] == 0.0


def test_word_effect_rows_drops_a_word_below_min_count(tmp):
    rows, _count, _mean = _PS._word_effect_rows(_wpairs(_WORD_SPECS),
                                                min_count=3)

    assert rows == []


def test_word_effect_rows_drops_a_word_present_in_every_prompt(tmp):
    """A word in all N prompts has no 'without' side to contrast against, so
    its effect would be a division by zero."""
    pairs = _wpairs([("parser fast", 1.0), ("parser slow", 9.0)])

    words = {r[1] for r in _PS._word_effect_rows(pairs, min_count=1)[0]}

    assert "parser" not in words, words
    assert words == {"fast", "slow"}, words


def test_word_effect_rows_ignores_stopwords_and_numeric_tokens(tmp):
    pairs = _wpairs([("the parser 800px 71e3af8 alpha", 1.0),
                     ("the module alpha", 9.0)])

    words = {r[1] for r in _PS._word_effect_rows(pairs, min_count=1)[0]}

    assert words == {"parser", "module"}, words


def test_word_effect_rows_strips_an_injected_system_reminder(tmp):
    """A hook-injected reminder rides inside the user's prompt; its words are
    the harness's, not the human's, and must not be attributed."""
    pairs = _wpairs([
        ("refactor parser <system-reminder>zebra quokka</system-reminder>", 1.0),
        ("document parser", 9.0)])

    words = {r[1] for r in _PS._word_effect_rows(pairs, min_count=1)[0]}

    assert "zebra" not in words and "quokka" not in words, words
    assert "refactor" in words, words


def test_word_effect_rows_counts_a_repeated_word_once_per_prompt(tmp):
    """The tokenizer runs over a set(), so five mentions in one prompt are one
    observation — otherwise a rambling prompt would dominate the ranking."""
    pairs = _wpairs([("parser parser parser parser alpha", 1.0),
                     ("module alpha", 9.0)])

    by_word = {r[1]: r for r in _PS._word_effect_rows(pairs, min_count=1)[0]}

    assert by_word["parser"][2] == 1, by_word["parser"]


# --- compute_reply_latency_words --------------------------------------------

_WORD_ROW_RE = re.compile(
    r"^\s*([+-])\s*([\d.]+)s\s+(\S+)\s+n=(\d+)\s+([\d.]+)/\s*([\d.]+)s")


def _word_row(report, word):
    """(effect, n, with_mean, without_mean) from the row for `word`."""
    for ln in report.splitlines():
        m = _WORD_ROW_RE.match(ln)
        if m and m.group(3) == word:
            sign = 1.0 if m.group(1) == "+" else -1.0
            return (sign * float(m.group(2)), int(m.group(4)),
                    float(m.group(5)), float(m.group(6)))
    raise AssertionError(f"no word row for {word!r}:\n{report}")


def _word_rows(report):
    return {m.group(3) for m in
            (_WORD_ROW_RE.match(ln) for ln in report.splitlines()) if m}


def _model_convo(specs, model, start_line=1, start_at=0.0, gap=100):
    events = []
    at = start_at
    line = start_line
    for prompt, dur in specs:
        events.append(_u(prompt, at, line, uuid=f"u{line}"))
        line += 1
        events.append(_a("ok", at + dur, line, model=model))
        line += 1
        at += dur + gap
    return events


def test_reply_latency_words_needs_twice_min_count_samples(tmp):
    events = _model_convo(_WORD_SPECS[:3], "alpha")

    report = _PS.compute_reply_latency_words(events, min_count=2)

    assert ("(not enough samples: 3 human-prompt replies, "
            "need >= 4 for min_count=2)") in report, report


def test_reply_latency_words_reports_effects_within_a_model(tmp):
    events = _model_convo(_WORD_SPECS, "alpha")

    report = _PS.compute_reply_latency_words(events, min_count=2)

    assert ("### alpha  —  4 replies, mean 6.00s, 4 words qualified"
            in report), report
    assert _word_row(report, "refactor") == (8.0, 2, 10.0, 2.0)
    assert _word_row(report, "document") == (-8.0, 2, 2.0, 10.0)


def test_reply_latency_words_omits_stopwords_from_the_ranking(tmp):
    """'the' appears in exactly two of the four prompts — enough for
    min_count — and is dropped only because it is a stopword."""
    events = _model_convo(_WORD_SPECS, "alpha")

    report = _PS.compute_reply_latency_words(events, min_count=2)

    assert _word_rows(report) == {"refactor", "document", "parser", "module"}


def test_reply_latency_words_splits_by_model_and_skips_thin_ones(tmp):
    alpha = _model_convo(_WORD_SPECS, "alpha")
    beta = _model_convo(_WORD_SPECS[:2], "beta", start_line=100,
                        start_at=10000.0)

    report = _PS.compute_reply_latency_words(alpha + beta, min_count=2)

    assert ("(6 human-prompt replies across 2 model(s); effect computed "
            "WITHIN each model, min_count=2)") in report, report
    assert "### alpha" in report, report
    assert "### beta" not in report, report
    assert ("(skipped 1 model(s) with < 4 human-prompt replies: beta [2])"
            in report), report


def test_reply_latency_words_excludes_synthetic_anchors(tmp):
    """A task-notification is a real measured reply but not a human prompt:
    --reply-latency counts five, the word ranking only four."""
    events = _model_convo(_WORD_SPECS, "alpha")
    events += _model_convo([("<task-notification>done</task-notification>", 3.0)],
                           "alpha", start_line=100, start_at=10000.0)

    assert _summary(_PS.compute_reply_latency(events))["count"] == "5"
    assert "(4 human-prompt replies across 1 model(s)" in \
        _PS.compute_reply_latency_words(events, min_count=2)


def test_reply_latency_words_top_n_truncates_each_direction(tmp):
    events = _model_convo(_WORD_SPECS, "alpha")

    report = _PS.compute_reply_latency_words(events, top_n=1, min_count=2)

    # The two sub-tables run together with no blank line between them, so
    # they are split on their headers rather than on whitespace.
    lines = report.splitlines()
    start = next(i for i, ln in enumerate(lines)
                 if ln.strip() == "-> SLOWER (top 1):")
    middle = next(i for i, ln in enumerate(lines)
                  if ln.strip() == "-> FASTER (top 1):")
    picked = [[m.group(3) for m in
               (_WORD_ROW_RE.match(ln) for ln in chunk) if m]
              for chunk in (lines[start:middle], lines[middle:])]
    # Four words qualified; top_n=1 keeps only the extreme of each direction.
    assert picked == [["refactor"], ["document"]], picked


def test_reply_latency_words_reports_no_qualifying_words(tmp):
    """Every word appears in every prompt, so nothing has a contrast."""
    events = _model_convo([("parser run", float(i)) for i in range(1, 5)],
                          "alpha")

    report = _PS.compute_reply_latency_words(events, min_count=2)

    assert "(no words reached min_count=2)" in report, report


# --- compute_reply_latency_word ---------------------------------------------

def _side(report, prefix):
    """(n, mean) from a `WITH "x":` / `WITHOUT       :` line."""
    ln = _line(report, prefix)
    return (int(re.search(r"n=(\d+)", ln).group(1)),
            float(re.search(r"mean=\s*([\d.]+)s", ln).group(1)))


def test_reply_latency_word_splits_with_and_without(tmp):
    events = _model_convo(_WORD_SPECS, "alpha")

    report = _PS.compute_reply_latency_word(events, "refactor")

    assert _side(report, 'WITH "refactor"') == (2, 10.0)
    assert _side(report, "WITHOUT") == (2, 2.0)
    assert "effect        :  +8.00s (slower with the word)" in report, report
    assert '(total: "refactor" appeared in 2 human prompt(s))' in report


def test_reply_latency_word_reports_a_faster_word(tmp):
    events = _model_convo(_WORD_SPECS, "alpha")

    report = _PS.compute_reply_latency_word(events, "document")

    assert "effect        :  -8.00s (faster with the word)" in report, report


def test_reply_latency_word_applies_no_stopword_filter(tmp):
    """The aggregate ranking drops 'the'; an explicitly requested word is
    matched anyway — that asymmetry is the documented contract."""
    events = _model_convo(_WORD_SPECS, "alpha")

    assert "the" not in _word_rows(
        _PS.compute_reply_latency_words(events, min_count=2))
    assert _side(_PS.compute_reply_latency_word(events, "the"),
                 'WITH "the"') == (2, 6.0)


def test_reply_latency_word_matches_whole_tokens_only(tmp):
    """'parse' must not match the token 'parser'."""
    events = _model_convo(_WORD_SPECS, "alpha")

    report = _PS.compute_reply_latency_word(events, "parse")

    assert '(word "parse" did not appear in any human prompt)' in report


def test_reply_latency_word_normalises_case_and_padding(tmp):
    events = _model_convo(_WORD_SPECS, "alpha")

    report = _PS.compute_reply_latency_word(events, "  REFACTOR  ")

    assert _side(report, 'WITH "refactor"') == (2, 10.0)


def test_reply_latency_word_rejects_an_empty_word(tmp):
    report = _PS.compute_reply_latency_word(_model_convo(_WORD_SPECS, "alpha"), "")

    assert "(no word given)" in report


def test_reply_latency_word_reports_no_human_prompts(tmp):
    events = _model_convo([("<system-reminder>x</system-reminder>", 1.0)],
                          "alpha")

    assert "(no human-prompt replies)" in \
        _PS.compute_reply_latency_word(events, "parser")


def test_reply_latency_word_caps_examples_at_three(tmp):
    events = _model_convo([(f"parser run {i}", float(i)) for i in range(1, 6)],
                          "alpha")

    report = _PS.compute_reply_latency_word(events, "parser")

    assert len([ln for ln in report.splitlines() if "e.g. [" in ln]) == 3


# --- src_paths / after / before ---------------------------------------------

def _jsonl(path, records):
    with open(path, "w", encoding="utf-8") as handle:
        for rec in records:
            handle.write(json.dumps(rec) + "\n")
    return path


def _user_rec(text, at, uuid):
    return {"type": "user", "uuid": uuid, "timestamp": _ts(at),
            "message": {"role": "user", "content": text}}


def _asst_rec(text, at, uuid, model="m1"):
    return {"type": "assistant", "uuid": uuid, "timestamp": _ts(at),
            "requestId": "req-" + uuid,
            "message": {"role": "assistant", "model": model,
                        "content": [{"type": "text", "text": text}]}}


def _pair_file(tmp, name, prompts):
    """A session file holding one user/assistant pair per (prompt, at, dur)."""
    records = []
    for i, (prompt, at, dur) in enumerate(prompts):
        records.append(_user_rec(prompt, at, f"{name}-u{i}"))
        records.append(_asst_rec("ok", at + dur, f"{name}-a{i}"))
    return _jsonl(os.path.join(tmp, name), records)


def test_reply_latency_reads_multiple_src_paths(tmp):
    """Two source files means folder mode: pairs come from the files, and the
    caller's `events` list is not consulted at all."""
    file_a = _pair_file(tmp, "a.jsonl", [("first", 0, 2)])
    file_b = _pair_file(tmp, "b.jsonl", [("second", 3600, 3)])

    report = _PS.compute_reply_latency([], src_paths=[file_a, file_b])

    assert _summary(report) == {
        "count": "2", "min": "2.00s", "max": "3.00s",
        "median": "2.50s", "p95": "3.00s", "mean": "2.50s"}


def test_reply_latency_never_bridges_across_session_files(tmp):
    """File A ends on an unanswered prompt and file B opens with an assistant
    event an hour later. Merging the two streams would report a 3600s reply;
    per-file walking reports none.
    """
    file_a = _jsonl(os.path.join(tmp, "a.jsonl"),
                    [_user_rec("dangling prompt", 0, "a-u0")])
    file_b = _jsonl(os.path.join(tmp, "b.jsonl"),
                    [_asst_rec("unrelated", 3600, "b-a0")])

    report = _PS.compute_reply_latency([], src_paths=[file_a, file_b])

    assert "(no user messages with following assistant events)" in report
    assert "3600.00s" not in report


def test_reply_latency_after_and_before_scope_each_file(tmp):
    file_a = _pair_file(tmp, "a.jsonl", [("early", 0, 2), ("late", 7200, 5)])
    file_b = _pair_file(tmp, "b.jsonl", [("later still", 10800, 1)])
    paths = [file_a, file_b]

    unfiltered = _PS.compute_reply_latency([], src_paths=paths)
    after = _PS.compute_reply_latency([], src_paths=paths,
                                      after="2026-07-30T13:00:00")
    before = _PS.compute_reply_latency([], src_paths=paths,
                                       before="2026-07-30T13:00:00")

    assert _summary(unfiltered)["count"] == "3"
    # base is 12:00:00Z, so only the 14:00 and 15:00 pairs survive `after`.
    assert _summary(after)["count"] == "2"
    assert _summary(after)["max"] == "5.00s"
    assert _summary(before) == {
        "count": "1", "min": "2.00s", "max": "2.00s",
        "median": "2.00s", "p95": "2.00s", "mean": "2.00s"}


def test_reply_latency_skips_an_unreadable_src_path(tmp):
    """A corpus can name a file that has since been removed; the pass must
    keep going rather than raise out of the whole report."""
    file_a = _pair_file(tmp, "a.jsonl", [("first", 0, 2)])
    missing = os.path.join(tmp, "gone.jsonl")

    report = _PS.compute_reply_latency([], src_paths=[file_a, missing])

    assert _summary(report)["count"] == "1"


def test_reply_latency_single_src_path_uses_the_event_list(tmp):
    """One source file is not folder mode: `events` already carries the
    range/path/tool filters applied upstream, so it wins over a re-read."""
    file_a = _pair_file(tmp, "a.jsonl", [("from the file", 0, 99)])
    events = _convo([("from the list", 4, "m1")])

    report = _PS.compute_reply_latency(events, src_paths=[file_a])

    assert _summary(report)["max"] == "4.00s"
    assert "from the file" not in report


def test_collect_reply_latency_pairs_dispatches_on_path_count(tmp):
    file_a = _pair_file(tmp, "a.jsonl", [("from the file", 0, 8)])
    file_b = _pair_file(tmp, "b.jsonl", [("also file", 3600, 6)])
    events = _convo([("from the list", 4, "m1")])

    one = _PS._collect_reply_latency_pairs(events, [file_a], None, None)
    two = _PS._collect_reply_latency_pairs(events, [file_a, file_b], None, None)

    assert [d for d, _, _ in one] == [4.0]
    assert sorted(d for d, _, _ in two) == [6.0, 8.0]


def test_reply_latency_tags_the_responding_model_from_the_file(tmp):
    file_a = _pair_file(tmp, "a.jsonl", [("refactor the parser", 0, 10),
                                         ("refactor module", 200, 10),
                                         ("document the parser", 400, 2),
                                         ("document module", 600, 2)])
    file_b = _pair_file(tmp, "b.jsonl", [("refactor the parser", 5000, 1)])

    report = _PS.compute_reply_latency_words([], min_count=2,
                                             src_paths=[file_a, file_b])

    assert "### m1  —  5 replies" in report, report


# --- _top_turns_table -------------------------------------------------------

def _turn_row(**over):
    row = {"first_ts": _ts(0), "first_line": 1, "requestId": "req",
           "cost": 0.0, "output": 0, "read": 0, "create": 0,
           "eph1h": 0, "eph5": 0, "fresh": 0}
    row.update(over)
    return row


def test_top_turns_table_sorts_descending_by_the_named_key(tmp):
    rows = [_turn_row(requestId="r5", output=5),
            _turn_row(requestId="r30", output=30),
            _turn_row(requestId="r10", output=10)]

    out = _PS._top_turns_table("TOP", rows, "output", ["output"])

    assert [ln.split()[3] for ln in out[3:]] == ["r30", "r10", "r5"], out


def test_top_turns_table_sorts_by_the_key_it_was_given_not_the_first_column(tmp):
    """The sort key and the displayed columns are independent arguments."""
    rows = [_turn_row(requestId="a", output=100, read=1),
            _turn_row(requestId="b", output=1, read=100)]

    out = _PS._top_turns_table("TOP", rows, "read", ["output", "c_read"])

    assert [ln.split()[3] for ln in out[3:]] == ["b", "a"], out


def test_top_turns_table_truncates_to_ten_rows(tmp):
    rows = [_turn_row(requestId=f"r{i}", output=i) for i in range(15)]

    out = _PS._top_turns_table("TOP", rows, "output", ["output"])

    assert len(out) == 13, out          # blank, title, header, 10 data rows
    assert out[3].split()[3] == "r14"
    assert out[-1].split()[3] == "r5"


def test_top_turns_table_returns_nothing_for_no_rows(tmp):
    assert _PS._top_turns_table("TOP", [], "output", ["output"]) == []


def test_top_turns_table_positive_guard_suppresses_an_all_zero_table(tmp):
    """A table of zeros is noise; the guard is what keeps it out of a report
    for a session that never used, say, 1h cache."""
    rows = [_turn_row(requestId="r", output=0)]

    assert _PS._top_turns_table("TOP", rows, "output", ["output"],
                                require_positive_top=True) == []
    assert _PS._top_turns_table("TOP", rows, "output", ["output"]) != []


def test_top_turns_table_positive_guard_passes_on_a_positive_top(tmp):
    rows = [_turn_row(requestId="r", output=1), _turn_row(requestId="z")]

    out = _PS._top_turns_table("TOP", rows, "output", ["output"],
                               require_positive_top=True)

    assert len(out) == 5, out


def test_top_turns_table_renders_columns_in_the_requested_order(tmp):
    rows = [_turn_row(requestId="r", read=7, fresh=9)]

    out = _PS._top_turns_table("TOP", rows, "read", ["c_read", "fresh"])

    header = out[2].split()
    assert header[3:] == ["c_read", "fresh", "$"], header
    assert out[3].split()[4:6] == ["7", "9"], out[3]


def test_top_turns_table_groups_thousands_and_formats_cost(tmp):
    rows = [_turn_row(requestId="r", read=1234567, cost=0.5)]

    out = _PS._top_turns_table("TOP", rows, "read", ["c_read"])

    assert "1,234,567" in out[3], out[3]
    assert "$ 0.500" in out[3], out[3]


# --- compute_turns ----------------------------------------------------------

def _usage_meta(uuid, line, at, **usage):
    return {"type": "assistant_usage", "uuid": uuid, "line": line,
            "ts": _ts(at), "model": "m1", "requestId": "req-" + uuid,
            "usage": usage}


def _turn_meta(parent, line, at, duration_ms, messages=None):
    return {"type": "system", "subtype": "turn_duration", "line": line,
            "ts": _ts(at),
            "raw": {"parentUuid": parent, "durationMs": duration_ms,
                    "messageCount": messages}}


def _turn_rows(report):
    """Whitespace-split data rows of the TURNS table."""
    return [ln.split() for ln in report.splitlines()
            if ln.startswith("  2026-")]


def test_turns_reports_the_absence_of_turn_duration_records(tmp):
    report = _PS.compute_turns([], [])

    assert "(no system/turn_duration records found" in report
    assert "--context-growth" in report, "the report must name the alternative"


def test_turns_joins_usage_by_parent_uuid(tmp):
    meta = [_usage_meta("a1", 1, 0, input_tokens=100,
                        cache_creation_input_tokens=200,
                        cache_read_input_tokens=700, output_tokens=42),
            _turn_meta("a1", 2, 1, 1500, messages=7)]

    report = _PS.compute_turns([], meta)

    assert _line(report, "Total turns:") == \
        "Total turns: 1  total compute: 1.5s"
    # date, time, L#, durMs, msgs, requestId, input, output
    assert _turn_rows(report) == [
        ["2026-07-30", "12:00:01", "L2", "1500", "7", "req-a1", "1,000", "42"]]


def test_turns_sorts_by_duration_descending(tmp):
    meta = []
    for i, ms in enumerate([100, 5000, 900], start=1):
        meta.append(_usage_meta(f"a{i}", i * 2 - 1, i, output_tokens=i))
        meta.append(_turn_meta(f"a{i}", i * 2, i, ms))

    rows = _turn_rows(_PS.compute_turns([], meta))

    assert [r[3] for r in rows] == ["5000", "900", "100"], rows


def test_turns_totals_compute_time_across_turns(tmp):
    meta = [_usage_meta("a1", 1, 0), _turn_meta("a1", 2, 0, 1500),
            _usage_meta("a2", 3, 1), _turn_meta("a2", 4, 1, 2500)]

    assert _line(_PS.compute_turns([], meta), "Total turns:") == \
        "Total turns: 2  total compute: 4.0s"


def test_turns_survives_a_turn_with_no_matching_usage(tmp):
    """The parent record can be missing (a truncated log); the turn still has
    a duration worth reporting, with zeros for tokens."""
    report = _PS.compute_turns([], [_turn_meta("nobody", 1, 0, 1500)])

    rows = _turn_rows(report)
    assert len(rows) == 1, rows
    # requestId renders empty, so the row splits to 7 tokens ending 0 / 0.
    assert rows[0] == ["2026-07-30", "12:00:00", "L1", "1500", "0", "0", "0"]


def test_turns_defaults_a_missing_duration_to_zero(tmp):
    meta = [{"type": "system", "subtype": "turn_duration", "line": 1,
             "ts": _ts(0), "raw": {"parentUuid": "a1"}}]

    assert _line(_PS.compute_turns([], meta), "Total turns:") == \
        "Total turns: 1  total compute: 0.0s"


def test_turns_ignores_other_system_subtypes(tmp):
    meta = [{"type": "system", "subtype": "something_else", "line": 1,
             "ts": _ts(0), "raw": {}}]

    assert "(no system/turn_duration records found" in _PS.compute_turns([], meta)


# --- compute_context_growth -------------------------------------------------

def _growth_usage(line, at, fresh=0, create=0, read=0, output=0,
                  model="m1", **extra):
    usage = {"input_tokens": fresh, "cache_creation_input_tokens": create,
             "cache_read_input_tokens": read, "output_tokens": output}
    usage.update(extra)
    return {"type": "assistant_usage", "line": line, "ts": _ts(at),
            "model": model, "usage": usage}


def _growth_rows(report):
    return [ln.split() for ln in report.splitlines()
            if re.match(r"^\s+\d+\s+2026-", ln)]


def test_context_growth_reports_the_absence_of_usage_records(tmp):
    assert "(no assistant_usage records found)" in \
        _PS.compute_context_growth([], [])


def test_context_growth_numbers_turns_and_reports_the_delta(tmp):
    events = [_u("one", 0, 1), _u("two", 2, 3), _u("three", 4, 5)]
    meta = [_growth_usage(2, 1, read=1000, output=10),
            _growth_usage(4, 3, read=3000, output=20),
            _growth_usage(6, 5, read=3500, output=30)]

    report = _PS.compute_context_growth(events, meta)

    rows = _growth_rows(report)
    assert [(r[0], r[3], r[4], r[5], r[6]) for r in rows] == [
        ("1", "L2", "1,000", "10", "+1,000"),
        ("2", "L4", "3,000", "20", "+2,000"),
        ("3", "L6", "3,500", "30", "+500")], rows
    assert _line(report, "Total:") == \
        "Total: 3 turns, final context: 3,500 input tokens"


def test_context_growth_sums_fresh_create_and_read_as_input(tmp):
    events = [_u("one", 0, 1)]
    meta = [_growth_usage(2, 1, fresh=7, create=30, read=900, output=1)]

    assert _growth_rows(_PS.compute_context_growth(events, meta))[0][4] == "937"


def test_context_growth_keeps_the_last_usage_before_a_boundary(tmp):
    """A turn emits several API calls; the turn's context size is the last
    one, not the first."""
    events = [_u("one", 0, 1), _u("two", 4, 4)]
    meta = [_growth_usage(2, 1, read=1000), _growth_usage(3, 2, read=1500),
            _growth_usage(5, 4, read=2000)]

    report = _PS.compute_context_growth(events, meta)

    rows = _growth_rows(report)
    assert [r[4] for r in rows] == ["1,500", "2,000"], rows
    assert "1,000" not in report, report


def test_context_growth_skips_synthetic_usage_records(tmp):
    """`<synthetic>` records carry all-zero usage and would otherwise become
    the representative data point for the turn they end."""
    events = [_u("one", 0, 1)]
    real = _growth_usage(2, 1, read=5000)
    synthetic = _growth_usage(3, 2, model="<synthetic>")

    with_synthetic = _PS.compute_context_growth(events, [real, synthetic])
    without = _PS.compute_context_growth(events, [real])

    assert _line(with_synthetic, "Total:") == _line(without, "Total:")
    assert _line(without, "Total:") == \
        "Total: 1 turns, final context: 5,000 input tokens"


def test_context_growth_drops_zero_input_turns_without_corrupting_deltas(tmp):
    """A refused/interrupted turn reports 0 input. Keeping it would reset
    prev_input to 0 and report the next turn's ENTIRE context as its delta."""
    events = [_u("one", 0, 1), _u("two", 2, 3), _u("three", 4, 5)]
    meta = [_growth_usage(2, 1, read=1000), _growth_usage(4, 3),
            _growth_usage(6, 5, read=1500)]

    report = _PS.compute_context_growth(events, meta)

    rows = _growth_rows(report)
    assert [(r[4], r[6]) for r in rows] == [("1,000", "+1,000"),
                                            ("1,500", "+500")], rows
    assert "+1,500" not in report, report


def test_context_growth_reports_when_every_turn_had_zero_input(tmp):
    events = [_u("one", 0, 1)]

    assert "(no turns with usage data)" in \
        _PS.compute_context_growth(events, [_growth_usage(2, 1)])


def test_context_growth_takes_the_peak_iteration_not_their_sum(tmp):
    """With sub-calls, top-level cache_read is the billing rollup across
    separate API requests. Context is the largest single request."""
    events = [_u("one", 0, 1)]
    iters = [{"cache_read_input_tokens": 1000},
             {"cache_read_input_tokens": 1000}]
    peaked = _growth_usage(2, 1, read=2000, iterations=iters)
    plain = _growth_usage(2, 1, read=2000)

    assert _growth_rows(_PS.compute_context_growth(events, [peaked]))[0][4] == \
        "1,000"
    assert _growth_rows(_PS.compute_context_growth(events, [plain]))[0][4] == \
        "2,000"


def test_context_growth_ignores_a_single_iteration_block(tmp):
    """One iteration is not a multi-call turn, so the top-level rollup stands."""
    events = [_u("one", 0, 1)]
    meta = [_growth_usage(2, 1, read=2000,
                          iterations=[{"cache_read_input_tokens": 5}])]

    assert _growth_rows(_PS.compute_context_growth(events, meta))[0][4] == \
        "2,000"


def test_context_growth_instrumentation_is_not_a_turn_boundary(tmp):
    """`<bash-input>` bursts fire mid-turn; treating one as a boundary splits
    a single turn in two."""
    meta = [_growth_usage(2, 1, read=1000), _growth_usage(4, 3, read=2000)]
    real = [_u("one", 0, 1), _u("two", 2, 3)]
    instrumented = [_u("one", 0, 1), _u("<bash-input>ls", 2, 3)]

    assert _line(_PS.compute_context_growth(real, meta), "Total:") == \
        "Total: 2 turns, final context: 2,000 input tokens"
    assert _line(_PS.compute_context_growth(instrumented, meta), "Total:") == \
        "Total: 1 turns, final context: 2,000 input tokens"


def test_context_growth_interrupt_marker_is_not_a_turn_boundary(tmp):
    meta = [_growth_usage(2, 1, read=1000), _growth_usage(4, 3, read=2000)]
    events = [_u("one", 0, 1),
              _u("[Request interrupted by user]", 2, 3)]

    assert _line(_PS.compute_context_growth(events, meta), "Total:") == \
        "Total: 1 turns, final context: 2,000 input tokens"


def test_context_growth_ignores_meta_events_that_are_not_usage(tmp):
    """The meta stream also carries hooks, system records and queue
    operations; only assistant_usage has tokens to plot."""
    events = [_u("one", 0, 1)]
    meta = [{"type": "system", "subtype": "turn_duration", "line": 2,
             "ts": _ts(0), "raw": {}},
            {"type": "attachment", "line": 3, "ts": _ts(0), "raw": {}},
            _growth_usage(4, 1, read=1000)]

    assert _line(_PS.compute_context_growth(events, meta), "Total:") == \
        "Total: 1 turns, final context: 1,000 input tokens"


def test_context_growth_only_user_messages_are_turn_boundaries(tmp):
    """An assistant event at the same line position must not split a turn."""
    meta = [_growth_usage(2, 1, read=1000), _growth_usage(4, 3, read=2000)]
    events = [_u("one", 0, 1), _a("mid-turn reply", 2, 3)]

    assert _line(_PS.compute_context_growth(events, meta), "Total:") == \
        "Total: 1 turns, final context: 2,000 input tokens"


def test_context_growth_orders_usage_records_by_timestamp(tmp):
    """Meta events can arrive out of order after a merge; the table is
    chronological regardless."""
    events = [_u("one", 0, 1), _u("two", 2, 3)]
    meta = [_growth_usage(4, 3, read=2000), _growth_usage(2, 1, read=1000)]

    rows = _growth_rows(_PS.compute_context_growth(events, meta))

    assert [r[4] for r in rows] == ["1,000", "2,000"], rows


def main():
    return _util.runner(_util.collect(globals()), tmp_prefix="latency_")


if __name__ == "__main__":
    raise SystemExit(main())
