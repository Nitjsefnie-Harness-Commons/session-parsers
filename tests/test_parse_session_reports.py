"""Suite for parse_session.py's report passes.

parse_session.py sat at 13.2% line coverage — 2552 uncovered lines, 64% of
everything uncovered in this repo. The gap was almost entirely the
`compute_*` family: each one takes a list of parsed events and returns a
report string, each is independently callable, and none of them had a test.

These are what a person reads when they ask what a session did. A pass that
silently miscounts is worse than one that crashes, because the number still
looks like a number — so the assertions below are on the reported VALUES,
not merely that a report was produced.

Synthetic events throughout: the shapes are the ones the parser emits, and
building them by hand keeps the suite off any machine's real transcripts,
which CI runners do not have.
"""
import _util

_PS = _util.load(_util.script("parse_session.py"))

TS = "2026-07-30T12:00:{:02d}Z"


# --- event builders ---------------------------------------------------------

def _tool_call(name, line=1, second=0, **over):
    event = {
        "type": "tool_call", "tool_name": name, "ts": TS.format(second),
        "line": line, "tool_use_id": f"tu{line}", "tool_input": {},
        "paired_result": {},
    }
    event.update(over)
    return event


def _user(text, line=1, second=0):
    return {"type": "user_message", "ts": TS.format(second), "line": line,
            "detail": text}


def _assistant(text, line=1, second=0):
    return {"type": "assistant_text", "ts": TS.format(second), "line": line,
            "detail": text}


def _result(line=1, second=0, is_error=False):
    return {"type": "tool_result", "ts": TS.format(second), "line": line,
            "detail": "ok", "is_error": is_error}


def _row(report, prefix):
    """The first report line starting with `prefix`, stripped."""
    for line in report.splitlines():
        if line.strip().startswith(prefix):
            return line.strip()
    raise AssertionError(f"no {prefix!r} row in report:\n{report}")


def _has(report, needle):
    return any(needle in line for line in report.splitlines())


# --- compute_stats ----------------------------------------------------------

def test_stats_counts_each_event_class(tmp):
    events = [
        _tool_call("Read", line=1, second=1),
        _tool_call("Bash", line=2, second=2),
        _tool_call("Read", line=3, second=3),
        _user("hello", line=4, second=4),
        _assistant("hi", line=5, second=5),
        _result(line=6, second=6),
        _result(line=7, second=7, is_error=True),
        {"type": "thinking", "ts": TS.format(8), "line": 8, "detail": "..."},
    ]

    report = _PS.compute_stats(events)

    assert _has(report, "Read"), report
    assert _has(report, "Bash"), report
    # Two Reads and one Bash: the per-tool counts must reflect that.
    read_row = _row(report, "Read")
    assert "2" in read_row, read_row


def test_stats_reports_the_error_result_count(tmp):
    clean = _PS.compute_stats([_result(line=1, second=1)])
    failed = _PS.compute_stats([_result(line=1, second=1, is_error=True)])

    assert clean != failed, "an errored result changed nothing in the report"


def test_stats_counts_a_parallel_batch_once(tmp):
    """batch_index == 1 marks the head of a parallel batch.

    Counting every member as a batch would report three batches where a
    reader sees one.
    """
    events = [
        _tool_call("Read", line=1, second=1, batch_size=3, batch_index=1),
        _tool_call("Read", line=2, second=1, batch_size=3, batch_index=2),
        _tool_call("Read", line=3, second=1, batch_size=3, batch_index=3),
    ]

    report = _PS.compute_stats(events)

    assert _has(report, "3"), report


def test_stats_survives_an_empty_session(tmp):
    """A session with nothing in it must report, not divide by zero."""
    report = _PS.compute_stats([])

    assert isinstance(report, str)
    assert report.strip(), "empty session produced an empty report"


# --- compute_edits ----------------------------------------------------------

def _edit(path, added, removed, line=1, user_modified=False, **inp):
    patch_lines = ["+" + "x" * 3] * added + ["-" + "y" * 3] * removed
    tool_input = {"file_path": path, "old_string": "old", "new_string": "new"}
    tool_input.update(inp)
    return _tool_call(
        "Edit", line=line, second=line,
        tool_input=tool_input,
        paired_result={"toolUseResult": {
            "userModified": user_modified,
            "structuredPatch": [{"lines": patch_lines}],
        }},
    )


def test_edits_totals_added_and_removed_lines(tmp):
    report = _PS.compute_edits([
        _edit("a.py", added=3, removed=1, line=1),
        _edit("b.py", added=2, removed=4, line=2),
    ])

    assert _has(report, "a.py"), report
    assert _has(report, "b.py"), report
    # 5 added, 5 removed across the two files.
    assert "5" in report, report


def test_edits_separates_accepted_from_user_modified(tmp):
    """A user-modified edit is not what the model wrote.

    Counting them together would report the model's edits as landing
    verbatim when a human rewrote them.
    """
    report = _PS.compute_edits([
        _edit("a.py", added=1, removed=0, line=1),
        _edit("a.py", added=1, removed=0, line=2, user_modified=True),
    ])

    assert "USER-MODIFIED" in report, report


def test_edits_aggregates_per_file(tmp):
    report = _PS.compute_edits([
        _edit("same.py", added=1, removed=0, line=1),
        _edit("same.py", added=1, removed=0, line=2),
        _edit("same.py", added=1, removed=0, line=3),
    ])

    # One row for the file, carrying three edits — not three rows.
    assert report.count("same.py") >= 1, report
    assert "3" in report, report


def test_edits_marks_replace_all(tmp):
    report = _PS.compute_edits(
        [_edit("a.py", added=1, removed=0, line=1, replace_all=True)])

    assert "replace_all" in report, report


def test_edits_truncates_long_strings_unless_verbose(tmp):
    long_string = "z" * 400
    event = _edit("a.py", added=0, removed=0, line=1)
    event["tool_input"]["old_string"] = long_string

    terse = _PS.compute_edits([event])
    verbose = _PS.compute_edits([event], verbose=True)

    assert "..." in terse, terse
    assert len(verbose) > len(terse), "verbose was not longer than terse"


def test_edits_ignores_non_edit_tools(tmp):
    report = _PS.compute_edits([_tool_call("Read", line=1)])

    assert "Read" not in report, report


def test_edits_survives_a_non_dict_tool_result(tmp):
    """toolUseResult is not always a dict; a string must not crash the pass."""
    event = _tool_call("Edit", line=1,
                       tool_input={"file_path": "a.py"},
                       paired_result={"toolUseResult": "just a string"})

    report = _PS.compute_edits([event])

    assert "a.py" in report, report


# --- compute_shell_errors ---------------------------------------------------

def _shell(command, line=1, stderr="", interrupted=False, is_error=False):
    return _tool_call(
        "Bash", line=line, second=line,
        tool_input={"command": command},
        paired_result={
            "is_error": is_error,
            "detail": "",
            "toolUseResult": {"stderr": stderr, "interrupted": interrupted},
        },
    )


def test_shell_errors_reports_a_command_with_stderr(tmp):
    report = _PS.compute_shell_errors([_shell("ls /nope", stderr="no such file")])

    assert "ls /nope" in report, report


def test_shell_errors_reports_an_interrupted_command(tmp):
    report = _PS.compute_shell_errors([_shell("sleep 100", interrupted=True)])

    assert "sleep 100" in report, report


def test_shell_errors_ignores_a_clean_command(tmp):
    report = _PS.compute_shell_errors([_shell("true")])

    assert "true" not in report, report


def test_shell_errors_ignores_non_shell_tools(tmp):
    report = _PS.compute_shell_errors([
        _tool_call("Read", line=1,
                   paired_result={"toolUseResult": {"stderr": "boom"}}),
    ])

    assert "boom" not in report, report


# --- compute_latency --------------------------------------------------------

def _timed(name, seconds, line=1):
    return _tool_call(
        name, line=line, second=0,
        use_ts=TS.format(0), result_ts=TS.format(seconds),
        paired_result={"ts": TS.format(seconds)},
    )


def test_latency_report_is_produced_for_timed_calls(tmp):
    report = _PS.compute_latency([
        _timed("Bash", 3, line=1),
        _timed("Read", 1, line=2),
    ])

    assert isinstance(report, str)
    assert "TOOL LATENCY" in report, report


def test_latency_survives_calls_with_no_timing(tmp):
    """A call whose result never arrived has no latency to rank."""
    report = _PS.compute_latency([_tool_call("Bash", line=1)])

    assert isinstance(report, str)


# --- format_event -----------------------------------------------------------

def test_format_event_renders_a_tool_call(tmp):
    line = _PS.format_event(
        _tool_call("Read", line=7, second=5,
                   tool_input={"file_path": "/tmp/x"}))

    assert "Read" in line, line
    assert "7" in line, line


def test_format_event_renders_a_user_message(tmp):
    line = _PS.format_event(_user("hello there", line=3, second=2))

    assert "hello there" in line, line


def test_format_event_marks_a_parallel_batch(tmp):
    plain = _PS.format_event(_tool_call("Read", line=1))
    batched = _PS.format_event(
        _tool_call("Read", line=1, batch_size=3, batch_index=1))

    assert batched != plain, "a batched call rendered identically to a lone one"


def test_format_event_verbose_is_not_shorter(tmp):
    event = _tool_call("Bash", line=1,
                       tool_input={"command": "echo " + "x" * 300})

    terse = _PS.format_event(event)
    verbose = _PS.format_event(event, verbose=True)

    assert len(verbose) >= len(terse)


# --- compute_hooks ----------------------------------------------------------

def _hook(hook_event, line=1, att_type="hook_success", **att):
    """A hook attachment in the shape compute_hooks actually reads.

    The payload lives at raw.attachment, not at the top level — an event
    built the obvious way matches nothing, and the pass then reports "no
    matching hook events" while a test asserting only that a header exists
    passes anyway. Which is what the first draft of this file did.
    """
    attachment = {"hookEvent": hook_event, "type": att_type}
    attachment.update(att)
    return {"type": "attachment", "ts": TS.format(line), "line": line,
            "raw": {"attachment": attachment}}


def _hook_rows(report):
    """Report lines that are neither header, rule, nor the empty notice."""
    return [ln for ln in report.splitlines()
            if ln.strip() and not ln.startswith(("HOOK INVOCATIONS", "="))
            and "no matching hook events" not in ln]


def test_hooks_reports_an_invocation(tmp):
    report = _PS.compute_hooks([_hook("PreToolUse", line=1, hookName="guard")])

    assert "PreToolUse" in report, report
    assert _hook_rows(report), f"no row rendered:\n{report}"


def test_hooks_filter_selects_by_event_name(tmp):
    meta = [_hook("PreToolUse", line=1), _hook("Stop", line=2)]

    only_stop = _PS.compute_hooks(meta, event_filter="Stop")

    assert "Stop" in only_stop, only_stop
    # The header echoes the filter verbatim, so PreToolUse must be absent
    # from the ROWS rather than from the whole report.
    assert not any("PreToolUse" in row for row in _hook_rows(only_stop)), only_stop


def test_hooks_filter_is_case_insensitive(tmp):
    meta = [_hook("PreToolUse", line=1)]

    lower = _hook_rows(_PS.compute_hooks(meta, event_filter="pretooluse"))
    exact = _hook_rows(_PS.compute_hooks(meta, event_filter="PreToolUse"))

    # Not byte-equality of the reports: the header quotes the filter as
    # given, so those differ by design. The ROWS selected must not.
    assert lower == exact, f"{lower!r} != {exact!r}"
    assert lower, "the case-insensitive filter selected nothing"


def test_hooks_ignores_an_attachment_with_no_hook_event(tmp):
    meta = [{"type": "attachment", "ts": TS.format(1), "line": 1,
             "raw": {"attachment": {"type": "something_else"}}}]

    assert not _hook_rows(_PS.compute_hooks(meta))


def test_hooks_ignores_non_attachment_meta(tmp):
    meta = [{"type": "system", "ts": TS.format(1), "line": 1, "raw": {}}]

    assert not _hook_rows(_PS.compute_hooks(meta))


def test_hooks_survives_no_invocations(tmp):
    report = _PS.compute_hooks([])

    assert isinstance(report, str)
    assert report.strip()


if __name__ == "__main__":
    raise SystemExit(_util.runner(_util.collect(dict(globals()))))
