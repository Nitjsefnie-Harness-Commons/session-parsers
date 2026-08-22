"""Suite for parse_session.py's per-artifact and diagnostic report passes.

The `compute_*` family splits in two. One half answers "what did this session
touch" — the file roll-up, the shell-command list, the writes, the agent
launches, the hook timeline, the orphaned calls. The other half is heuristic:
it FLAGS things (tmp-script churn, redirect-then-read, pointless
backgrounding, cache outliers), and a heuristic that never fires is
indistinguishable from a clean session unless a test pins both directions.

So every heuristic here is asserted twice — once on input that must be
flagged, once on input that must not — and the threshold cases sit exactly on
their boundary, because `>=` and `>` produce identical reports on every input
except that one.

Synthetic events throughout, in the shapes `_build_tool_call_event`,
`_meta_system`, `_meta_queue_operation` and the assistant-usage record
actually emit. Nothing here touches the filesystem or any real transcript:
CI runners have neither.
"""
import datetime

import _util

_PS = _util.load(_util.script("parse_session.py"))

# The private helpers this suite covers, bound once. Looked up through
# `vars()` rather than by attribute: they are module-level functions of a
# single-file tool, and pylint reads the attribute form as reaching past a
# class's encapsulation, which is not what is happening here.
_tool_call_excerpt = vars(_PS)["_tool_call_excerpt"]
_hook_preview = vars(_PS)["_hook_preview"]
_render_hook_success = vars(_PS)["_render_hook_success"]
_render_hook_blocking_error = vars(_PS)["_render_hook_blocking_error"]
_render_hook_permission_decision = vars(_PS)["_render_hook_permission_decision"]
_render_hook_additional_context = vars(_PS)["_render_hook_additional_context"]

_EPOCH = datetime.datetime(2026, 7, 30, 12, 0, 0)


def _ts(second=0):
    """ISO timestamp `second` seconds past the suite's fixed epoch.

    Fixed, not `utcnow()`: a duration assertion against a wall clock is a
    test that fails on a slow CI runner and nowhere else.
    """
    stamp = _EPOCH + datetime.timedelta(seconds=second)
    return stamp.strftime("%Y-%m-%dT%H:%M:%SZ")


# --- event builders ---------------------------------------------------------

def _tool_call(name, line=1, second=None, **over):
    event = {
        "type": "tool_call", "tool_name": name,
        "ts": _ts(line if second is None else second),
        "line": line, "tool_use_id": f"tu{line}", "tool_input": {},
        "paired_result": {},
    }
    event.update(over)
    return event


def _agent(line=1, second=None, detail="", **over):
    """An agent_spawn event, as `_build_tool_call_event` emits for `Agent`."""
    event = {
        "type": "agent_spawn", "tool_name": "Agent",
        "ts": _ts(line if second is None else second),
        "line": line, "tool_use_id": f"tu{line}", "tool_input": {},
        "agent_name": "worker", "agent_model": "(default)", "agent_bg": False,
        "agent_team": "(none)", "agent_prompt": "", "agent_subagent_type": "impl",
        "paired_result": {"detail": detail, "ts": _ts(line), "line": line},
        "detail": "Agent(worker)",
    }
    event.update(over)
    return event


def _user(text, line=1):
    return {"type": "user_message", "ts": _ts(line), "line": line,
            "detail": text}


def _assistant(text, line=1):
    return {"type": "assistant_text", "ts": _ts(line), "line": line,
            "detail": text}


def _system(subtype, line=1, content="", **raw):
    obj = {"subtype": subtype, "content": content}
    obj.update(raw)
    return {"line": line, "type": "system", "ts": _ts(line),
            "subtype": subtype, "content": content, "raw": obj}


def _queue_op(content, line=1, operation="enqueue"):
    return {"line": line, "type": "queue-operation", "ts": _ts(line),
            "operation": operation, "content": content,
            "raw": {"operation": operation, "content": content}}


def _attachment(line=1, **att):
    return {"line": line, "type": "attachment", "ts": _ts(line),
            "attachment_type": att.get("type", ""),
            "raw": {"attachment": att}}


def _usage(line=1, model="claude-opus-5", request_id="", **counts):
    usage = {
        "input_tokens": counts.get("fresh", 0),
        "cache_creation_input_tokens": counts.get("create", 0),
        "cache_read_input_tokens": counts.get("read", 0),
        "output_tokens": counts.get("output", 0),
    }
    return {"line": line, "type": "assistant_usage", "ts": _ts(line),
            "model": model, "requestId": request_id, "uuid": f"u{line}",
            "usage": usage}


# --- report readers ---------------------------------------------------------

def _row(report, prefix):
    """The first report line starting with `prefix`, stripped."""
    for line in report.splitlines():
        if line.strip().startswith(prefix):
            return line.strip()
    raise AssertionError(f"no {prefix!r} row in report:\n{report}")


def _has(report, needle):
    return needle in report


def _section(report, header_prefix):
    """Lines after the line starting with `header_prefix`, up to a blank one.

    The anomaly and churn reports emit several independently-flagging
    sections; asserting a token is "in the report" would let a hit in one
    section satisfy an assertion about another.
    """
    lines = report.splitlines()
    for i, line in enumerate(lines):
        if line.startswith(header_prefix):
            out = []
            for tail in lines[i + 1:]:
                if not tail.strip():
                    break
                out.append(tail)
            return out
    raise AssertionError(f"no {header_prefix!r} section in:\n{report}")


# --- compute_files ----------------------------------------------------------

def test_files_rolls_up_the_ops_per_path(tmp):
    report = _PS.compute_files([
        _tool_call("Read", line=1, tool_input={"file_path": "a.py"}),
        _tool_call("Read", line=2, tool_input={"file_path": "a.py"}),
        _tool_call("Edit", line=3, tool_input={"file_path": "a.py"}),
        _tool_call("Write", line=4, tool_input={"file_path": "b.py"}),
    ])

    row = _row(report, "[")
    # One row for a.py carrying both ops with their counts, not three rows.
    assert "edit(1), read(2)" in row, row
    assert "a.py" in row, row
    assert "(L1,L2,L3)" in row, row
    assert _has(report, "Total: 2 files/paths"), report


def test_files_truncates_the_line_refs_past_five(tmp):
    report = _PS.compute_files(
        [_tool_call("Read", line=n, tool_input={"file_path": "a.py"})
         for n in range(1, 8)])

    row = _row(report, "[")
    assert "L1,L2,L3,L4,L5…+2" in row, row


def test_files_labels_a_pathless_grep_as_cwd(tmp):
    report = _PS.compute_files(
        [_tool_call("Grep", line=1, tool_input={"pattern": "x"})])

    assert _has(report, "(cwd)"), report
    assert _has(report, "grep(1)"), report


def test_files_keys_a_glob_by_path_and_pattern(tmp):
    """Two globs over one directory with different patterns are two searches.

    Keying on the directory alone would merge them into one row and lose
    which pattern was actually run.
    """
    report = _PS.compute_files([
        _tool_call("Glob", line=1, tool_input={"path": "src", "pattern": "*.py"}),
        _tool_call("Glob", line=2, tool_input={"path": "src", "pattern": "*.md"}),
    ])

    assert _has(report, "src [*.py]"), report
    assert _has(report, "src [*.md]"), report
    assert _has(report, "Total: 2 files/paths"), report


def test_files_ignores_paths_named_inside_shell_commands(tmp):
    """Shell text is deliberately not scanned for paths — too unreliable."""
    report = _PS.compute_files(
        [_tool_call("Bash", line=1, tool_input={"command": "cat secret.py"})])

    assert _has(report, "Total: 0 files/paths"), report
    assert not _has(report, "secret.py"), report


def test_files_falls_back_to_a_placeholder_path(tmp):
    """A Read with no file_path still counts as a path touched."""
    report = _PS.compute_files([_tool_call("Read", line=1, tool_input={})])

    assert _row(report, "[").endswith("]  ?  (L1)"), report
    assert _has(report, "Total: 1 files/paths"), report


def test_files_survives_an_empty_session(tmp):
    report = _PS.compute_files([])

    assert _has(report, "Total: 0 files/paths"), report


# --- compute_commands -------------------------------------------------------

def test_commands_marks_a_background_call(tmp):
    fg = _PS.compute_commands(
        [_tool_call("Bash", line=1, tool_input={"command": "make"})])
    bg = _PS.compute_commands(
        [_tool_call("Bash", line=1,
                    tool_input={"command": "make", "run_in_background": True})])

    assert not _has(fg, "[bg]"), fg
    assert _has(bg, "[bg]"), bg


def test_commands_uses_the_powershell_prompt_marker(tmp):
    """A mixed-shell session must not present PowerShell as Bash."""
    report = _PS.compute_commands([
        _tool_call("Bash", line=1, tool_input={"command": "ls"}),
        _tool_call("PowerShell", line=2, tool_input={"command": "Get-ChildItem"}),
    ])

    assert _has(report, "  $ ls"), report
    assert _has(report, "PS> Get-ChildItem"), report


def test_commands_tags_a_parallel_batch_member(tmp):
    report = _PS.compute_commands([
        _tool_call("Bash", line=1, tool_input={"command": "a"},
                   batch_size=3, batch_index=2),
    ])

    assert _has(report, "[parallel 2/3]"), report


def test_commands_appends_the_description(tmp):
    report = _PS.compute_commands([
        _tool_call("Bash", line=1,
                   tool_input={"command": "make", "description": "build it"}),
    ])

    assert _has(report, "— build it"), report


def test_commands_ignores_non_shell_tools(tmp):
    report = _PS.compute_commands(
        [_tool_call("Read", line=1, tool_input={"file_path": "a.py"})])

    assert not _has(report, "a.py"), report


# --- compute_writes ---------------------------------------------------------

def test_writes_reports_the_size_and_the_total(tmp):
    report = _PS.compute_writes([
        _tool_call("Write", line=3,
                   tool_input={"file_path": "a.py", "content": "abcde"}),
        _tool_call("Write", line=4,
                   tool_input={"file_path": "b.py", "content": ""}),
    ])

    assert _has(report, "a.py (5 chars)"), report
    assert _has(report, "b.py (0 chars)"), report
    assert _has(report, "Total: 2 files written"), report


def test_writes_truncates_the_preview_unless_verbose(tmp):
    body = "z" * 300
    event = _tool_call("Write", line=1,
                       tool_input={"file_path": "a.py", "content": body})

    terse = _PS.compute_writes([event])
    verbose = _PS.compute_writes([event], verbose=True)

    assert "z" * 200 + "..." in terse, terse
    assert body in verbose, "verbose dropped part of the written content"
    assert body not in terse


def test_writes_reports_zero_for_a_session_with_no_writes(tmp):
    report = _PS.compute_writes([_tool_call("Read", line=1)])

    assert _has(report, "Total: 0 files written"), report


# --- _tool_call_excerpt -----------------------------------------------------

def test_tool_call_excerpt_picks_the_field_each_tool_carries(tmp):
    excerpt = _tool_call_excerpt

    assert excerpt(_tool_call("Bash", tool_input={"command": "ls -l"})) == "ls -l"
    assert excerpt(_tool_call("PowerShell",
                              tool_input={"command": "gci"})) == "gci"
    assert excerpt(_tool_call("Read", tool_input={"file_path": "a.py"})) == "a.py"
    assert excerpt(_tool_call("WebFetch",
                              tool_input={"url": "https://x/y"})) == "https://x/y"
    assert excerpt(_tool_call(
        "Grep", tool_input={"pattern": "p"})) == "p  in (cwd)"
    assert excerpt(_tool_call(
        "Glob", tool_input={"pattern": "*.py", "path": "src"})) == "*.py  in src"


def test_tool_call_excerpt_names_the_agent_for_a_spawn(tmp):
    assert _tool_call_excerpt(_agent(agent_name="reviewer")) == "agent=reviewer"


def test_tool_call_excerpt_falls_back_to_the_raw_input(tmp):
    """An unknown tool still has to render something identifying."""
    out = _tool_call_excerpt(
        _tool_call("SomethingNew", tool_input={"k": "v"}), limit=200)

    assert '"k"' in out and '"v"' in out, out


def test_tool_call_excerpt_truncates_at_the_limit(tmp):
    out = _tool_call_excerpt(
        _tool_call("Bash", tool_input={"command": "x" * 100}), limit=80)

    assert out == "x" * 80 + "...", out


def test_tool_call_excerpt_flattens_newlines(tmp):
    """The excerpt is placed on one report line; an embedded \\n would split it."""
    out = _tool_call_excerpt(
        _tool_call("Bash", tool_input={"command": "a\nb"}))

    assert out == "a b", out


def test_tool_call_excerpt_survives_a_null_tool_input(tmp):
    out = _tool_call_excerpt(_tool_call("Read", tool_input=None))

    assert out == "", out


# --- compute_agent_tasks ----------------------------------------------------

_NOTIF = ("task-notification <tool-use-id>{tu}</tool-use-id>"
          "<task-id>{task}</task-id><status>{status}</status>"
          "<summary>{summary}</summary>")


def test_agent_tasks_reports_a_session_with_no_spawns(tmp):
    report = _PS.compute_agent_tasks([_tool_call("Read", line=1)], [])

    assert _has(report, "(no Agent spawns found)"), report


def test_agent_tasks_correlates_a_notification_by_tool_use_id(tmp):
    spawn = _agent(line=1, second=0, agent_bg=True, agent_prompt="do it",
                   tool_use_id="tuA",
                   paired_result={"detail": "agentId: ag7 output_file: out.md"})
    notif = _queue_op(_NOTIF.format(tu="tuA", task="ag7", status="completed",
                                    summary="wrote the thing"), line=9)
    notif["ts"] = _ts(42)

    report = _PS.compute_agent_tasks([spawn], [notif])

    assert _has(report, "Total agent launches: 1"), report
    assert _has(report, "agent_id=ag7"), report
    assert _has(report, "prompt=5 chars"), report
    # 42s between the spawn ts and the notification ts.
    assert _has(report, "duration=42.0s"), report
    assert _has(report, "status=completed"), report
    assert _has(report, "summary: wrote the thing"), report


def test_agent_tasks_falls_back_to_the_task_id(tmp):
    """A notification carrying only <task-id> still has to find its launch."""
    spawn = _agent(line=1, agent_bg=True, tool_use_id="tuB",
                   paired_result={"detail": "agentId: ag9"})
    notif = _queue_op(
        "task-notification <task-id>ag9</task-id><status>failed</status>",
        line=5)

    report = _PS.compute_agent_tasks([spawn], [notif])

    assert _has(report, "status=failed"), report
    assert not _has(report, "(no completion notification found)"), report


def test_agent_tasks_treats_a_foreground_result_as_the_completion(tmp):
    """A foreground agent has no notification — its tool_result IS the finish.

    Without that leg every FG launch reads as never having completed.
    """
    spawn = _agent(line=1, second=0, agent_bg=False,
                   paired_result={"detail": "agentId: ag1", "ts": _ts(3),
                                  "line": 2})

    report = _PS.compute_agent_tasks([spawn], [])

    assert _has(report, "FG  type=impl"), report
    assert _has(report, "status=completed"), report
    assert _has(report, "duration=3.0s"), report
    assert _has(report, "(FG result, 12 chars)"), report


def test_agent_tasks_flags_an_unnotified_background_launch(tmp):
    spawn = _agent(line=1, agent_bg=True,
                   paired_result={"detail": "agentId: ag2"})

    report = _PS.compute_agent_tasks([spawn], [])

    assert _has(report, "bg  type=impl"), report
    assert _has(report, "(no completion notification found)"), report


def test_agent_tasks_links_the_read_of_the_output_file(tmp):
    spawn = _agent(line=1, agent_bg=True, tool_use_id="tuC",
                   paired_result={"detail": "output_file: /out/report.md"})
    read = _tool_call("Read", line=7,
                      tool_input={"file_path": "/out/report.md"},
                      paired_result={"detail": "hello"})

    linked = _PS.compute_agent_tasks([spawn, read], [])
    unread = _PS.compute_agent_tasks([spawn], [])

    assert _has(linked, "output_file: /out/report.md"), linked
    assert _has(linked, "read L7"), linked
    assert _has(linked, "result=5 chars"), linked
    assert _has(unread, "(output file never read)"), unread


def test_agent_tasks_keeps_only_the_first_read_of_an_output_file(tmp):
    """A re-read is not a second delivery; the first read is when it landed."""
    spawn = _agent(line=1, agent_bg=True, tool_use_id="tuD",
                   paired_result={"detail": "output_file: /out/r.md"})
    reads = [_tool_call("Read", line=n, tool_input={"file_path": "/out/r.md"},
                        paired_result={"detail": "x"}) for n in (4, 6)]

    report = _PS.compute_agent_tasks([spawn] + reads, [])

    assert _has(report, "read L4"), report
    assert not _has(report, "read L6"), report


def test_agent_tasks_ignores_meta_that_matches_no_launch(tmp):
    """Real transcripts interleave unrelated meta; none of it may bind here."""
    spawn = _agent(line=1, agent_bg=True, tool_use_id="tuE",
                   paired_result={"detail": "agentId: ag1"})
    meta = [
        _system("api_error", line=2),
        _queue_op("an ordinary queue op", line=3),
        _queue_op("task-notification <task-id>unknown</task-id>", line=4),
    ]

    report = _PS.compute_agent_tasks([spawn], meta)

    assert _has(report, "(no completion notification found)"), report
    assert not _has(report, "status="), report


# --- hook renderers ---------------------------------------------------------

def test_hook_preview_flattens_and_reports_the_dropped_length(tmp):
    short = _hook_preview("a\nb", n=200)
    long_out = _hook_preview("y" * 250, n=200)

    assert short == "a\\nb", short
    # The count of what was dropped is the point: a bare "..." hides scale.
    assert long_out == "y" * 200 + "... [50 more chars]", long_out


def test_hook_preview_survives_a_none_body(tmp):
    assert _hook_preview(None) == ""


def _hook(att_type, hook_event="PreToolUse", line=1, **att):
    att.update({"type": att_type, "hookEvent": hook_event})
    return _attachment(line=line, **att)


def test_hook_success_renders_exit_duration_and_stream_sizes(tmp):
    out = _render_hook_success(
        {"command": "guard.py", "exitCode": 0, "durationMs": 12,
         "stdout": "o" * 1500, "stderr": "e"}, False)

    assert out[0] == "  exit=0  duration=12ms  stdout=1,500B  stderr=1B", out
    assert "  cmd: guard.py" in out, out
    assert any(ln.startswith("  stdout: ") for ln in out), out
    assert "  stderr: e" in out, out


def test_hook_success_verbose_emits_the_whole_stdout(tmp):
    att = {"exitCode": 0, "durationMs": 1, "stdout": "one\ntwo", "stderr": ""}

    terse = _render_hook_success(att, False)
    verbose = _render_hook_success(att, True)

    assert "  stdout: one\\ntwo" in terse, terse
    assert "  stdout:" in verbose and "    one" in verbose and "    two" in verbose


def test_hook_success_falls_back_to_inline_content(tmp):
    """Some hook records carry `content` instead of `stdout`.

    Rendering only stdout would show a 0B hook that actually said something.
    """
    with_content = _render_hook_success(
        {"exitCode": 0, "durationMs": 1, "content": "ctx"}, False)
    with_stdout = _render_hook_success(
        {"exitCode": 0, "durationMs": 1, "stdout": "so", "content": "ctx"}, False)

    assert "content=3B" in with_content[0], with_content
    assert "  content: ctx" in with_content, with_content
    # stdout present: content is already accounted for, so it is not repeated.
    assert not any(ln.startswith("  content: ") for ln in with_stdout), with_stdout


def test_hook_success_verbose_expands_the_inline_content(tmp):
    """The content fallback has its own verbose leg, separate from stdout's."""
    att = {"exitCode": 0, "durationMs": 1, "content": "one\ntwo"}

    terse = _render_hook_success(att, False)
    verbose = _render_hook_success(att, True)

    assert "  content: one\\ntwo" in terse, terse
    assert "  content:" in verbose and "    one" in verbose and "    two" in verbose


def test_hook_blocking_error_renders_both_payload_shapes(tmp):
    nested = _render_hook_blocking_error(
        {"blockingError": {"blockingError": "denied", "command": "gate.py"}},
        False)
    flat = _render_hook_blocking_error({"blockingError": "denied"}, False)

    assert "  cmd: gate.py" in nested, nested
    assert "  error: denied" in nested, nested
    # A bare string payload must not be dropped just for lacking the wrapper.
    assert "  error: denied" in flat, flat


def test_hook_blocking_error_verbose_keeps_the_line_breaks(tmp):
    att = {"blockingError": {"blockingError": "line1\nline2"}}

    terse = _render_hook_blocking_error(att, False)
    verbose = _render_hook_blocking_error(att, True)

    assert "  error: line1\\nline2" in terse, terse
    assert "    line1" in verbose and "    line2" in verbose, verbose


def test_hook_blocking_error_renders_nothing_when_empty(tmp):
    assert _render_hook_blocking_error({}, False) == []


def test_hook_permission_decision_falls_back_to_message(tmp):
    with_reason = _render_hook_permission_decision(
        {"decision": "deny", "reason": "not allowed"}, False)
    with_message = _render_hook_permission_decision(
        {"decision": "allow", "message": "fine"}, False)
    bare = _render_hook_permission_decision({}, False)

    assert with_reason == ["  decision=deny", "  reason: not allowed"]
    assert with_message == ["  decision=allow", "  reason: fine"]
    assert bare == ["  decision=?"]


def test_hook_additional_context_sums_the_segments(tmp):
    out = _render_hook_additional_context(
        {"content": ["ab", "cde"]}, False)

    assert out[0] == "  segments=2  total=5B", out
    assert "  [0] (2B) ab" in out, out
    assert "  [1] (3B) cde" in out, out


def test_hook_additional_context_verbose_expands_each_segment(tmp):
    att = {"content": ["x\ny"]}

    terse = _render_hook_additional_context(att, False)
    verbose = _render_hook_additional_context(att, True)

    assert "  [0] (3B) x\\ny" in terse, terse
    assert "  [0] (3B):" in verbose and "    x" in verbose and "    y" in verbose


def test_hook_additional_context_handles_a_plain_string(tmp):
    terse = _render_hook_additional_context({"content": "hello"}, False)
    verbose = _render_hook_additional_context({"content": "hello"}, True)

    assert terse == ["  content=5B", "  hello"], terse
    assert verbose == ["  content=5B", "    hello"], verbose


def test_hook_additional_context_ignores_an_absent_content(tmp):
    assert _render_hook_additional_context({}, False) == []


def test_hooks_summarizes_durations_blocks_and_denies(tmp):
    meta = [
        _hook("hook_success", line=1, durationMs=10, hookName="guard"),
        _hook("hook_success", line=2, durationMs=30, hookName="guard"),
        _hook("hook_blocking_error", "Stop", line=3,
              blockingError={"blockingError": "nope"}),
        _hook("hook_permission_decision", "PermissionRequest", line=4,
              decision="Deny"),
    ]

    report = _PS.compute_hooks(meta)

    assert _has(report, "Total hook attachments: 4"), report
    # avg over the two hook_success rows only — the other two carry no duration.
    assert _has(report, "n=2  total=40ms  avg=20.0ms"), report
    assert _has(report, "hook_blocking_error total: 1"), report
    # The decision compare is case-insensitive: "Deny" is a deny.
    assert _has(report, "hook_permission_decision denies: 1"), report
    # Both successes came from the same hook, so its tally reads 2, not 1.
    assert _row(report, "guard").split() == ["guard", "2"], report


def test_hooks_routes_each_attachment_to_its_renderer(tmp):
    meta = [
        _hook("hook_success", line=1, exitCode=2, durationMs=5, stdout="hi"),
        _hook("hook_additional_context", "SessionStart", line=2,
              content=["seg"]),
        _hook("unknown_hook_shape", "Stop", line=3),
    ]

    report = _PS.compute_hooks(meta)

    assert _has(report, "exit=2"), report
    assert _has(report, "segments=1"), report
    # An unrecognised attachment type still gets its header row, no renderer.
    assert _has(report, "unknown_hook_shape"), report


def test_hooks_verbose_expands_the_bodies(tmp):
    meta = [_hook("hook_success", line=1, durationMs=1, exitCode=0,
                  stdout="alpha\nbeta")]

    terse = _PS.compute_hooks(meta)
    verbose = _PS.compute_hooks(meta, verbose=True)

    assert _has(terse, "stdout: alpha\\nbeta"), terse
    assert _has(verbose, "    alpha"), verbose
    assert len(verbose) > len(terse)


def test_hooks_shows_the_tool_use_id_when_present(tmp):
    report = _PS.compute_hooks(
        [_hook("hook_success", line=1, durationMs=1, toolUseID="tu42")])

    assert _has(report, "toolUseID=tu42"), report


# --- compute_compact --------------------------------------------------------

def test_compact_reports_the_trigger_and_pretokens(tmp):
    report = _PS.compute_compact([
        _system("compact_boundary", line=12, content="summary line",
                compactMetadata={"trigger": "auto", "preTokens": 1234567}),
    ])

    assert _has(report, "trigger=auto"), report
    assert _has(report, "preTokens=1,234,567"), report
    assert _has(report, "content: summary line"), report


def test_compact_defaults_a_missing_metadata_block(tmp):
    report = _PS.compute_compact([_system("compact_boundary", line=1)])

    assert _has(report, "trigger=?"), report
    assert _has(report, "preTokens=0"), report


def test_compact_reports_a_session_that_never_compacted(tmp):
    report = _PS.compute_compact([_system("api_error", line=1)])

    assert _has(report, "(no compact_boundary events)"), report


# --- compute_api_errors -----------------------------------------------------

def test_api_errors_unwraps_the_doubly_nested_error(tmp):
    """The real payload nests error.error.error; the outer levels say nothing."""
    report = _PS.compute_api_errors([
        _system("api_error", line=4,
                error={"status": 529,
                       "error": {"error": {"type": "overloaded_error",
                                           "message": "Overloaded"}}},
                retryInMs=1500.0, retryAttempt=2, maxRetries=5),
    ])

    assert _has(report, "status=529"), report
    assert _has(report, "type=overloaded_error"), report
    assert _has(report, "attempt=2/5"), report
    assert _has(report, "retryIn=1500ms"), report
    assert _has(report, "msg: Overloaded"), report


def test_api_errors_falls_back_through_the_nesting_levels(tmp):
    report = _PS.compute_api_errors([
        _system("api_error", line=1, error={"type": "outer_only"}),
    ])

    assert _has(report, "type=outer_only"), report
    assert _has(report, "status=?"), report


def test_api_errors_omits_retry_when_it_is_not_a_number(tmp):
    report = _PS.compute_api_errors([
        _system("api_error", line=1, error={}, retryInMs="soon"),
    ])

    assert not _has(report, "retryIn="), report


def test_api_errors_reports_a_clean_session(tmp):
    report = _PS.compute_api_errors([_system("local_command", line=1)])

    assert _has(report, "(no api_error events)"), report


# --- compute_slash ----------------------------------------------------------

def test_slash_reports_the_command_and_its_args(tmp):
    report = _PS.compute_slash([
        _system("local_command", line=3,
                content="<command-name>/loop</command-name>"
                        "<command-args>  5m  </command-args>"),
        _system("local_command", line=4,
                content="<command-name>/clear</command-name>"
                        "<command-args></command-args>"),
    ])

    assert _has(report, "/loop"), report
    assert _has(report, "args=5m"), report
    # Empty args must not render a dangling "args=" tag.
    assert _row(report, "L4").endswith("/clear"), report


def test_slash_ignores_a_local_command_with_no_command_name(tmp):
    report = _PS.compute_slash(
        [_system("local_command", line=1, content="just some stdout")])

    assert _has(report, "(no slash commands found)"), report


def test_slash_ignores_meta_of_another_subtype(tmp):
    report = _PS.compute_slash([
        _system("api_error", line=1,
                content="<command-name>/nope</command-name>"),
        _queue_op("<command-name>/also-nope</command-name>", line=2),
    ])

    assert _has(report, "(no slash commands found)"), report


def test_slash_reports_a_session_with_none(tmp):
    report = _PS.compute_slash([])

    assert _has(report, "(no slash commands found)"), report


# --- compute_last_assistant_message -----------------------------------------

def test_last_assistant_message_joins_the_whole_final_turn(tmp):
    """A closing reply split by a tool call must come back whole."""
    out = _PS.compute_last_assistant_message([
        _assistant("from an earlier turn", line=1),
        _user("go on", line=2),
        _assistant("first part", line=3),
        _tool_call("Read", line=4),
        _assistant("second part", line=5),
    ])

    assert out == "first part\nsecond part", repr(out)


def test_last_assistant_message_is_raw_with_no_banner(tmp):
    """A Stop hook matches on this text; a banner is something to strip off."""
    out = _PS.compute_last_assistant_message([_assistant("  done  ", line=1)])

    assert out == "done", repr(out)


def test_last_assistant_message_is_empty_for_a_tool_only_turn(tmp):
    out = _PS.compute_last_assistant_message([
        _user("go", line=1),
        _tool_call("Bash", line=2),
    ])

    assert out == "", repr(out)


def test_last_assistant_message_is_empty_for_no_events(tmp):
    assert _PS.compute_last_assistant_message([]) == ""


# --- compute_tool_orphans ---------------------------------------------------

def test_tool_orphans_flags_a_call_whose_result_never_arrived(tmp):
    report = _PS.compute_tool_orphans([
        _tool_call("Bash", line=1, tool_input={"command": "make"},
                   paired_result=None),
        _tool_call("Read", line=2, tool_input={"file_path": "a.py"}),
    ])

    assert _has(report, "Bash"), report
    assert _has(report, "make"), report
    assert report.splitlines()[-1] == "1 orphaned call", report
    # The answered Read must not be listed.
    assert not _has(report, "a.py"), report


def test_tool_orphans_pluralizes_the_count(tmp):
    report = _PS.compute_tool_orphans([
        _tool_call("Bash", line=n, tool_input={"command": "x"},
                   paired_result=None) for n in (1, 2)])

    assert _has(report, "2 orphaned calls"), report


def test_tool_orphans_names_an_unresolved_agent_spawn(tmp):
    report = _PS.compute_tool_orphans(
        [_agent(line=1, agent_name="worker", paired_result=None)])

    assert _has(report, "agent=worker"), report


def test_tool_orphans_ignores_events_that_are_not_calls(tmp):
    """A user message has no result to be missing; it is not an orphan."""
    report = _PS.compute_tool_orphans([
        _user("hi", line=1),
        _assistant("hello", line=2),
    ])

    assert _has(report, "0 orphaned calls"), report


def test_tool_orphans_reports_a_fully_paired_session(tmp):
    report = _PS.compute_tool_orphans([_tool_call("Read", line=1)])

    assert _has(report, "(no orphaned tool calls"), report
    assert _has(report, "0 orphaned calls"), report


# --- compute_timeline -------------------------------------------------------

def test_timeline_labels_every_event_class(tmp):
    report = _PS.compute_timeline([
        _user("hi", line=1),
        _assistant("hello", line=2),
        {"type": "thinking", "ts": _ts(3), "line": 3, "detail": "hmm"},
        _tool_call("Read", line=4, tool_input={"file_path": "a.py"}),
        {"type": "tool_result", "ts": _ts(5), "line": 5, "detail": "ok",
         "is_error": True, "paired_call": _tool_call("Read", line=4)},
        _agent(line=6),
    ])

    assert _has(report, "USER"), report
    assert _has(report, "ASST"), report
    assert _has(report, "THINK"), report
    assert _has(report, "TOOL:Read"), report
    # The error marker rides on the result label, not on a separate row.
    assert _has(report, "RESULT:Read[ERR]"), report
    assert _has(report, "TOOL:Agent"), report
    assert _has(report, "6 events"), report


def test_timeline_renders_a_tool_call_from_its_input(tmp):
    """tool_call events carry no `detail` until something asks for one."""
    report = _PS.compute_timeline(
        [_tool_call("Grep", line=1, tool_input={"pattern": "x"})])

    assert _has(report, "Grep({"), report
    assert _has(report, '"pattern"'), report


def test_timeline_preview_limit_widens_with_verbose(tmp):
    events = [_assistant("q" * 300, line=1)]

    terse = _PS.compute_timeline(events)
    verbose = _PS.compute_timeline(events, verbose=True)

    assert "q" * 80 + "..." in terse, terse
    assert "q" * 200 + "..." in verbose, verbose
    assert "q" * 201 not in verbose


def test_timeline_reports_an_empty_session(tmp):
    report = _PS.compute_timeline([])

    assert _has(report, "(no events)"), report
    assert _has(report, "0 events"), report


# --- compute_script_churn ---------------------------------------------------
# The paths below are string literals inside synthetic events, matched against
# the parser's own `^/tmp/...` regex. Nothing is created on disk, so these stay
# valid on a Windows runner that has no /tmp.

def _write(path, line=1):
    return _tool_call("Write", line=line,
                      tool_input={"file_path": path, "content": ""})


def test_script_churn_flags_a_family_at_the_threshold(tmp):
    report = _PS.compute_script_churn([
        _write("/tmp/probe_a.sh", line=1),
        _write("/tmp/probe_b.sh", line=2),
        _write("/tmp/probe_c.sh", line=3),
    ])

    assert _has(report, "FLAGGED FAMILIES (1)"), report
    assert _has(report, "probe_*.sh  (3 distinct files)"), report
    assert _has(report, "/tmp/probe_c.sh"), report


def test_script_churn_leaves_a_family_one_short_unflagged(tmp):
    """Exactly threshold-1 is the boundary: near, listed, but NOT flagged."""
    report = _PS.compute_script_churn([
        _write("/tmp/probe_a.sh", line=1),
        _write("/tmp/probe_b.sh", line=2),
    ])

    assert _has(report, "(no churn families detected)"), report
    assert _has(report, "NEAR THRESHOLD (2 distinct"), report
    assert _has(report, "probe_*.sh  (2 distinct)"), report


def test_script_churn_does_not_count_rewriting_one_path(tmp):
    """Overwriting a scratch file is iteration, not sibling sprawl."""
    report = _PS.compute_script_churn(
        [_write("/tmp/probe_a.sh", line=n) for n in (1, 2, 3, 4)])

    assert _has(report, "(no churn families detected)"), report
    assert not _has(report, "NEAR THRESHOLD"), report


def test_script_churn_separates_families_by_extension(tmp):
    """probe_*.sh and probe_*.py are different scripts sharing a prefix."""
    report = _PS.compute_script_churn([
        _write("/tmp/probe_a.sh", line=1),
        _write("/tmp/probe_b.sh", line=2),
        _write("/tmp/probe_a.py", line=3),
        _write("/tmp/probe_b.py", line=4),
    ])

    assert _has(report, "(no churn families detected)"), report
    assert _has(report, "probe_*.sh  (2 distinct)"), report
    assert _has(report, "probe_*.py  (2 distinct)"), report


def test_script_churn_ignores_paths_outside_the_tmp_pattern(tmp):
    report = _PS.compute_script_churn([
        _write("/home/u/probe_a.sh", line=1),
        _write("/home/u/probe_b.sh", line=2),
        _write("/home/u/probe_c.sh", line=3),
        _write("/tmp/nested/probe_a.sh", line=4),
        _write("/tmp/probe_a.txt", line=5),
    ])

    assert _has(report, "(no churn families detected)"), report
    assert not _has(report, "NEAR THRESHOLD"), report


def test_script_churn_honours_a_lowered_threshold(tmp):
    events = [_write("/tmp/probe_a.sh", line=1), _write("/tmp/probe_b.sh", line=2)]

    default = _PS.compute_script_churn(events)
    lowered = _PS.compute_script_churn(events, threshold=2)

    assert _has(default, "(no churn families detected)"), default
    assert _has(lowered, "FLAGGED FAMILIES (1)"), lowered
    assert _has(lowered, "(threshold: 2+"), lowered


# --- compute_redirect_read --------------------------------------------------

def _redirect(cmd, line=1, **inp):
    tool_input = {"command": cmd}
    tool_input.update(inp)
    return _tool_call("Bash", line=line, tool_input=tool_input)


def _reader(path, line=2, detail="ok", **inp):
    tool_input = {"file_path": path}
    tool_input.update(inp)
    return _tool_call("Read", line=line, tool_input=tool_input,
                      paired_result={"detail": detail})


def test_redirect_read_flags_a_small_reread(tmp):
    report = _PS.compute_redirect_read([
        _redirect("make > out.txt 2>&1", line=1),
        _reader("out.txt", line=2, detail="12345"),
    ])

    assert _has(report, "redirect target: out.txt"), report
    assert _has(report, "Read (5B returned)"), report
    assert _has(report, "Total: 1 pair(s)"), report


def test_redirect_read_recognizes_every_redirect_form(tmp):
    """POSIX `&>`, PowerShell `*>`, and both PowerShell pipeline forms."""
    for cmd in ("make &> o.txt", "make *> o.txt", "make | Out-File o.txt",
                "make | Tee-Object -FilePath o.txt"):
        report = _PS.compute_redirect_read([
            _redirect(cmd, line=1),
            _reader("o.txt", line=2, detail="x"),
        ])
        assert _has(report, "redirect target: o.txt"), (cmd, report)


def test_redirect_read_flags_a_shell_inspector_downstream(tmp):
    """`cat out.txt` after the redirect is the same antipattern as Read."""
    report = _PS.compute_redirect_read([
        _redirect("make > out.txt 2>&1", line=1),
        _tool_call("Bash", line=2, tool_input={"command": "cat out.txt"},
                   paired_result={"detail": "small"}),
    ])

    assert _has(report, "Bash:cat (5B returned)"), report


def test_redirect_read_ignores_a_background_redirect(tmp):
    """A bg task's output file is a different flow, not this antipattern."""
    report = _PS.compute_redirect_read([
        _redirect("make > out.txt 2>&1", line=1, run_in_background=True),
        _reader("out.txt", line=2, detail="x"),
    ])

    assert _has(report, "(none)"), report


def test_redirect_read_ignores_a_deliberate_slice(tmp):
    """Read with offset/limit is someone paging a file, not dodging inline."""
    windowed = _PS.compute_redirect_read([
        _redirect("make > out.txt 2>&1", line=1),
        _reader("out.txt", line=2, detail="x", offset=10),
    ])
    limited = _PS.compute_redirect_read([
        _redirect("make > out.txt 2>&1", line=1),
        _reader("out.txt", line=2, detail="x", limit=10),
    ])

    assert _has(windowed, "(none)"), windowed
    assert _has(limited, "(none)"), limited


def test_redirect_read_spares_a_result_exactly_at_the_threshold(tmp):
    """The guard is `>= size_threshold`, so 2000B is already "big enough"."""
    big = _PS.compute_redirect_read([
        _redirect("make > out.txt 2>&1", line=1),
        _reader("out.txt", line=2, detail="x" * 2000),
    ])
    just_under = _PS.compute_redirect_read([
        _redirect("make > out.txt 2>&1", line=1),
        _reader("out.txt", line=2, detail="x" * 1999),
    ])

    assert _has(big, "(none)"), big
    assert _has(just_under, "Total: 1 pair(s)"), just_under


def test_redirect_read_ignores_a_read_of_a_different_file(tmp):
    report = _PS.compute_redirect_read([
        _redirect("make > out.txt 2>&1", line=1),
        _reader("other.txt", line=2, detail="x"),
    ])

    assert _has(report, "(none)"), report


def test_redirect_read_ignores_a_downstream_that_does_not_read(tmp):
    """Naming the file is not reading it — a Write or an `rm` is neither."""
    written = _PS.compute_redirect_read([
        _redirect("make > out.txt 2>&1", line=1),
        _tool_call("Write", line=2,
                   tool_input={"file_path": "out.txt", "content": "x"},
                   paired_result={"detail": "ok"}),
    ])
    removed = _PS.compute_redirect_read([
        _redirect("make > out.txt 2>&1", line=1),
        _tool_call("Bash", line=2, tool_input={"command": "rm out.txt"},
                   paired_result={"detail": "ok"}),
    ])

    assert _has(written, "(none)"), written
    assert _has(removed, "(none)"), removed


def test_redirect_read_ignores_a_plain_command(tmp):
    report = _PS.compute_redirect_read([
        _redirect("make", line=1),
        _reader("out.txt", line=2, detail="x"),
    ])

    assert _has(report, "(none)"), report


# --- compute_fast_bg --------------------------------------------------------

def _bg_launch(task_id, line=1, second=0, command="make"):
    return _tool_call(
        "Bash", line=line, second=second,
        tool_input={"command": command, "run_in_background": True},
        paired_result={
            "detail": f"Command running in background with ID: {task_id}"},
    )


def _bg_notify(task_id, line=9, second=1, status="completed",
               operation="enqueue"):
    op = _queue_op(
        f"task-notification <task-id>{task_id}</task-id>"
        f"<status>{status}</status>", line=line, operation=operation)
    op["ts"] = _ts(second)
    return op


def test_fast_bg_flags_a_task_that_finished_immediately(tmp):
    report = _PS.compute_fast_bg(
        [_bg_launch("bg1", line=1, second=0, command="echo hi")],
        [_bg_notify("bg1", line=9, second=2)])

    assert _has(report, "task=bg1"), report
    assert _has(report, "(+2.0s)"), report
    assert _has(report, "cmd: echo hi"), report
    assert _has(report, "Total: 1 fast bg task(s)"), report


def test_fast_bg_spares_a_task_exactly_at_the_threshold(tmp):
    """`gap < threshold_s`, so a 5.0s task is not "fast" at threshold 5.0."""
    at = _PS.compute_fast_bg([_bg_launch("bg1", second=0)],
                             [_bg_notify("bg1", second=5)])
    under = _PS.compute_fast_bg([_bg_launch("bg1", second=0)],
                                [_bg_notify("bg1", second=4)])

    assert _has(at, "(none)"), at
    assert _has(under, "task=bg1"), under


def test_fast_bg_ignores_a_task_that_did_not_complete(tmp):
    """A fast FAILURE is not a backgrounding mistake — it never ran."""
    failed = _PS.compute_fast_bg([_bg_launch("bg1", second=0)],
                                 [_bg_notify("bg1", second=1, status="failed")])
    killed = _PS.compute_fast_bg([_bg_launch("bg1", second=0)],
                                 [_bg_notify("bg1", second=1, status="killed")])

    assert _has(failed, "(none)"), failed
    assert _has(killed, "(none)"), killed


def test_fast_bg_ignores_a_task_that_never_notified(tmp):
    report = _PS.compute_fast_bg([_bg_launch("bg1")], [])

    assert _has(report, "(none)"), report


def test_fast_bg_ignores_a_foreground_call(tmp):
    report = _PS.compute_fast_bg(
        [_tool_call("Bash", line=1, tool_input={"command": "make"},
                    paired_result={
                        "detail": "Command running in background with ID: bg1"})],
        [_bg_notify("bg1", second=1)])

    assert _has(report, "(none)"), report


def test_fast_bg_ignores_a_non_enqueue_queue_operation(tmp):
    report = _PS.compute_fast_bg(
        [_bg_launch("bg1", second=0)],
        [_bg_notify("bg1", second=1, operation="dequeue")])

    assert _has(report, "(none)"), report


def test_fast_bg_ignores_launch_side_noise(tmp):
    """Every launch-side leg that cannot yield a task id must drop out."""
    events = [
        _tool_call("Read", line=1, tool_input={"file_path": "a.py"}),
        # bg, but the call never returned, so there is no task id to key on.
        _tool_call("Bash", line=2,
                   tool_input={"command": "make", "run_in_background": True},
                   paired_result=None),
        # bg and answered, but the result body carries no background ID.
        _tool_call("Bash", line=3,
                   tool_input={"command": "make", "run_in_background": True},
                   paired_result={"detail": "done"}),
    ]

    report = _PS.compute_fast_bg(events, [_bg_notify("bg1", second=1)])

    assert _has(report, "(none)"), report


def test_fast_bg_ignores_notification_side_noise(tmp):
    meta = [
        _queue_op("an ordinary enqueue with no notification", line=8),
        _queue_op("task-notification <status>completed</status>", line=9),
    ]

    report = _PS.compute_fast_bg([_bg_launch("bg1", second=0)], meta)

    assert _has(report, "(none)"), report


def test_fast_bg_ignores_a_task_with_an_unparseable_timestamp(tmp):
    """A malformed ts must drop the row, not raise on the subtraction."""
    launch = _bg_launch("bg1", line=1, second=0)
    launch["ts"] = "not-a-timestamp"

    report = _PS.compute_fast_bg([launch], [_bg_notify("bg1", second=1)])

    assert _has(report, "(none)"), report


def test_fast_bg_honours_a_raised_threshold(tmp):
    events = [_bg_launch("bg1", second=0)]
    meta = [_bg_notify("bg1", second=20)]

    default = _PS.compute_fast_bg(events, meta)
    raised = _PS.compute_fast_bg(events, meta, threshold_s=30.0)

    assert _has(default, "(none)"), default
    assert _has(raised, "task=bg1"), raised
    assert _has(raised, "(+20.0s)"), raised


# --- compute_cache_anomalies ------------------------------------------------

def test_cache_anomalies_reports_a_session_with_no_usage(tmp):
    report = _PS.compute_cache_anomalies([_system("api_error", line=1)])

    assert _has(report, "(no assistant usage records found)"), report


def test_cache_anomalies_flags_a_create_outlier_and_spares_the_rest(tmp):
    report = _PS.compute_cache_anomalies([
        _usage(line=1, request_id="r1", create=1000),
        _usage(line=2, request_id="r2", create=100),
        _usage(line=3, request_id="r3", create=100),
    ])

    section = _section(report, "CACHE_CREATE OUTLIERS")
    # 25% of 1200 is 300: only the 1000-token turn clears it.
    assert len(section) == 1, section
    assert "req=r1" in section[0], section
    assert "cache_create=     1,000" in section[0], section
    assert "83.3% of session" in section[0], section
    assert _has(report, "cache_create: 1,200"), report


def test_cache_anomalies_flags_every_turn_of_an_exact_25_percent_tie(tmp):
    """Four even turns are each 25% — and the bar is 25% of the total."""
    report = _PS.compute_cache_anomalies([
        _usage(line=n, request_id=f"r{n}", create=100) for n in (1, 2, 3, 4)])

    # 25% of 400 = 100 and the compare is `>=`, so all four sit exactly on
    # the boundary and are reported. The pass flags the tie rather than
    # hiding it, which is the behaviour a reader has to be able to rely on.
    assert len(_section(report, "CACHE_CREATE OUTLIERS")) == 4, report


def test_cache_anomalies_flags_a_read_at_the_1m_boundary(tmp):
    at = _PS.compute_cache_anomalies(
        [_usage(line=1, request_id="r1", read=1_000_000)])
    under = _PS.compute_cache_anomalies(
        [_usage(line=1, request_id="r1", read=999_999)])

    assert "req=r1" in "".join(_section(at, "CACHE_READ OUTLIERS")), at
    assert _section(under, "CACHE_READ OUTLIERS") == ["  (none)"], under


def test_cache_anomalies_flags_output_at_the_boundary(tmp):
    at = _PS.compute_cache_anomalies(
        [_usage(line=1, request_id="r1", output=5000)])
    under = _PS.compute_cache_anomalies(
        [_usage(line=1, request_id="r1", output=4999)])

    assert "output= 5,000" in "".join(_section(at, "OUTPUT TOKEN OUTLIERS")), at
    assert _section(under, "OUTPUT TOKEN OUTLIERS") == ["  (none)"], under


def test_cache_anomalies_skips_synthetic_records(tmp):
    """<synthetic> stubs are harness-injected and carry no real API usage.

    Counting them would both inflate the totals and move the 25% bar.
    """
    report = _PS.compute_cache_anomalies([
        _usage(line=1, request_id="r1", create=100),
        _usage(line=2, model="<synthetic>", request_id="r2", create=999_999),
    ])

    assert _has(report, "cache_create: 100"), report
    assert not _has(report, "r2"), report


def test_cache_anomalies_merges_records_sharing_a_request_id(tmp):
    """One request split across records is one turn, and its tokens add up."""
    merged = _PS.compute_cache_anomalies([
        _usage(line=1, request_id="r1", output=3000),
        _usage(line=2, request_id="r1", output=3000),
    ])
    split = _PS.compute_cache_anomalies([
        _usage(line=1, request_id="r1", output=3000),
        _usage(line=2, request_id="r2", output=3000),
    ])

    assert "output= 6,000" in "".join(_section(merged, "OUTPUT TOKEN OUTLIERS"))
    # Same 6,000 tokens, two requests: neither turn alone clears 5,000.
    assert _section(split, "OUTPUT TOKEN OUTLIERS") == ["  (none)"], split


def test_cache_anomalies_keys_a_record_with_no_request_id_by_line(tmp):
    report = _PS.compute_cache_anomalies([_usage(line=7, output=6000)])

    assert _has(report, "req=(no-req-L7)"), report


def main():
    return _util.runner(_util.collect(globals()), tmp_prefix="reports2_")


if __name__ == "__main__":
    raise SystemExit(main())
