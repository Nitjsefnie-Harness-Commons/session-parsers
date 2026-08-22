"""Suite for parse_session.py's outermost layer: filters, rulecheck, branch
analysis and the command line itself.

Every other suite reaches a report by calling its `compute_*` directly. Nothing
covered the path a person actually takes: an argv list, split on `--and`,
parsed into segments, each one narrowing the event stream through
`--range`/`--after`/`--path`/`--tool` before a mode is dispatched, then printed.
Nor the two passes that read raw records rather than events -- `--rulecheck`,
which compares what an assistant CLAIMED about a shell call against the call it
made, and `--branches`, which rebuilds the conversation tree from parentUuid to
find abandoned rewinds.

These fail quietly. A filter that drops one event too many produces a shorter
report, not an error; a rulecheck that silently matches nothing reports a clean
session; a fork detector that mistakes a tool-thread join for a rewind reports
work that was never abandoned. So the assertions below are on the values the
layer produces -- which lines survive a filter, which mismatch is named, how
many forks are found, the exit code and the printed text.

Fixtures are synthetic events and synthetic JSONL written into the temp dir
each test is handed: CI runners have no transcripts of their own, and `main()`
resolves a bare session ID against ~/.claude, so the home directory is
redirected into the temp dir for the duration of every `main()` call.
"""

# The leading-underscore helpers ARE the units here: the argv split, the
# segment label, the rulecheck slot checks and the branch-tree primitives are
# otherwise reachable only through a whole main() run, which is the indirection
# that left this layer untested. Naming them directly is the point of the suite.
# pylint: disable=protected-access
import argparse
import contextlib
import io
import json
import os
import re
import sys
from datetime import datetime, timedelta, timezone

import _util

_PS = _util.load(_util.script("parse_session.py"))

TS = "2026-07-30T12:00:{:02d}Z"


# --- event builders (the shapes extract_events emits) -----------------------

def _call(tool="Bash", line=1, second=1, **inp):
    return {"type": "tool_call", "tool_name": tool, "ts": TS.format(second),
            "line": line, "tool_use_id": f"tu{line}", "tool_input": dict(inp),
            "paired_result": {}, "src": "/corpus/a.jsonl"}


def _text(detail, line=1, second=1):
    return {"type": "assistant_text", "ts": TS.format(second), "line": line,
            "detail": detail}


def _user(detail, line=1, second=1):
    return {"type": "user_message", "ts": TS.format(second), "line": line,
            "detail": detail}


def _res(line=1, second=1, paired_call=None, detail="ok"):
    e = {"type": "tool_result", "ts": TS.format(second), "line": line,
         "detail": detail, "is_error": False}
    if paired_call is not None:
        e["paired_call"] = paired_call
    return e


def _at(ts, line=1):
    """A minimal event carrying only what the time filter looks at."""
    return {"type": "user_message", "ts": ts, "line": line, "detail": "x"}


def _lines(events):
    return [e["line"] for e in events]


# --- namespace builder ------------------------------------------------------

def _ns(**over):
    """An args namespace with every attribute _run_segment reads defaulted off.

    Built from the parser's own flag tuples so a new mode flag cannot leave
    this helper silently short of an attribute.
    """
    fields = dict.fromkeys(_PS._MODE_FLAG_NAMES, False)
    fields.update(dict.fromkeys(_PS._TYPE_FILTER_NAMES, False))
    fields.update(dict.fromkeys(
        ("cheat_sheet", "version", "full", "verbose", "with_source", "timing",
         "find_linegrep"), False))
    fields.update(dict.fromkeys(
        ("jsonl", "tool", "path", "range", "after", "before", "grep",
         "line_grep", "output", "hook_event", "reply_latency_word"), None))
    fields.update(over)
    return argparse.Namespace(**fields)


# --- JSONL record builders --------------------------------------------------

def _write_jsonl(tmp, name, records):
    """Write `records` (dicts, or raw strings taken verbatim) as one JSONL.

    Compact separators, because a real transcript has none and the
    find_linegrep candidate probes match on the exact bytes `"type":"user"` --
    a fixture written with json.dumps' default `": "` spacing would make those
    probes silently find nothing.
    """
    path = os.path.join(tmp, name)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for rec in records:
            fh.write(rec if isinstance(rec, str)
                     else json.dumps(rec, separators=(",", ":")))
            fh.write("\n")
    return path


def _user_rec(uuid, parent, text, second=1, **extra):
    rec = {"type": "user", "uuid": uuid, "parentUuid": parent,
           "timestamp": TS.format(second),
           "message": {"role": "user", "content": text}}
    rec.update(extra)
    return rec


def _asst_rec(uuid, parent, msg_id, blocks, second=2, usage=None, **extra):
    msg = {"role": "assistant", "id": msg_id, "model": "claude-x",
           "content": blocks}
    if usage is not None:
        msg["usage"] = usage
    rec = {"type": "assistant", "uuid": uuid, "parentUuid": parent,
           "timestamp": TS.format(second), "requestId": f"req-{uuid}",
           "message": msg}
    rec.update(extra)
    return rec


def _result_rec(uuid, parent, tool_use_id, second=3):
    return {"type": "user", "uuid": uuid, "parentUuid": parent,
            "timestamp": TS.format(second),
            "message": {"role": "user", "content": [{
                "type": "tool_result", "tool_use_id": tool_use_id,
                "content": "ok"}]}}


def _simple_session(tmp, name="session.jsonl"):
    """One user turn, one assistant turn with a Bash call, one result."""
    return _write_jsonl(tmp, name, [
        _user_rec("u0", None, "list the files", 1),
        _asst_rec("a1", "u0", "m1", [
            {"type": "text", "text": "running it"},
            {"type": "tool_use", "id": "t1", "name": "Bash",
             "input": {"command": "echo alpha"}},
        ], 2, usage={"input_tokens": 10, "output_tokens": 5,
                     "cache_creation_input_tokens": 3,
                     "cache_read_input_tokens": 7}),
        _result_rec("r1", "a1", "t1", 3),
    ])


# --- main() driver ----------------------------------------------------------

def _run_main(tmp, argv):
    """Call main() with an explicit argv; return (exit_code, stdout, stderr).

    exit_code is None when main() returned without calling sys.exit.

    sys.argv, HOME and USERPROFILE are all restored in a finally block: this
    suite runs in one process alongside every other test in the file, so a
    leaked patch would corrupt whatever runs next. HOME/USERPROFILE are
    redirected because main() resolves an unknown path as a session ID against
    ~/.claude/projects, and a test must not depend on -- or read -- the real
    home. (Path.home() consults HOME on POSIX and USERPROFILE on Windows.)
    """
    home_keys = ("HOME", "USERPROFILE")
    saved_argv = sys.argv
    saved_home = {k: os.environ.get(k) for k in home_keys}
    out, err = io.StringIO(), io.StringIO()
    code = None
    try:
        sys.argv = ["parse_session.py"] + list(argv)
        for key in home_keys:
            os.environ[key] = tmp
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                _PS.main()
            except SystemExit as exc:
                code = exc.code
    finally:
        sys.argv = saved_argv
        for key, value in saved_home.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    return code, out.getvalue(), err.getvalue()


# --- filter_by_path ---------------------------------------------------------

def test_filter_by_path_matches_file_path_case_insensitively(tmp):
    events = [_call("Read", line=1, file_path="/repo/Alpha.py"),
              _call("Read", line=2, file_path="/repo/gamma.py")]

    kept = _PS.filter_by_path(events, "alpha")

    # The pattern is lowercased and so is the candidate: a Read of Alpha.py
    # must not be invisible to `--path alpha`.
    assert _lines(kept) == [1], _lines(kept)


def test_filter_by_path_matches_a_bash_command_substring(tmp):
    events = [_call("Bash", line=1, command="grep beta /repo/x"),
              _call("Bash", line=2, command="echo done")]

    kept = _PS.filter_by_path(events, "beta")

    assert _lines(kept) == [1], _lines(kept)


def test_filter_by_path_reads_the_path_field_when_file_path_is_absent(tmp):
    events = [_call("Glob", line=1, path="/repo/delta"),
              _call("Glob", line=2, path="/repo/other")]

    kept = _PS.filter_by_path(events, "delta")

    assert _lines(kept) == [1], _lines(kept)


def test_filter_by_path_keeps_a_result_paired_to_a_matching_call(tmp):
    """A result carries no path of its own -- only its call does.

    Dropping it would show a matching call whose output vanished.
    """
    match = _res(line=2, paired_call={"tool_input": {"file_path": "/r/hit.py"}})
    other = _res(line=3, paired_call={"tool_input": {"file_path": "/r/miss.py"}})
    bare = _res(line=4)

    kept = _PS.filter_by_path([match, other, bare], "hit")

    assert _lines(kept) == [2], _lines(kept)


def test_filter_by_path_passes_conversation_events_through(tmp):
    """User/assistant/thinking events are context, not candidates.

    They survive a path filter that matches nothing, which is what makes a
    filtered report readable rather than a bare list of calls.
    """
    events = [_call("Read", line=1, file_path="/repo/a.py"),
              _user("why", line=2),
              _text("because", line=3),
              {"type": "thinking", "ts": TS.format(4), "line": 4,
               "detail": "hmm"}]

    kept = _PS.filter_by_path(events, "no-such-path")

    assert _lines(kept) == [2, 3, 4], _lines(kept)


# --- filter_line_range ------------------------------------------------------

def test_filter_line_range_is_inclusive_at_both_ends(tmp):
    events = [_at(TS.format(1), line=n) for n in (1, 2, 3, 4, 5)]

    kept = _PS.filter_line_range(events, "2-4")

    assert _lines(kept) == [2, 3, 4], _lines(kept)


def test_filter_line_range_can_select_a_single_line(tmp):
    events = [_at(TS.format(1), line=n) for n in (1, 2, 3)]

    kept = _PS.filter_line_range(events, "2-2")

    assert _lines(kept) == [2], _lines(kept)


def test_filter_line_range_rejects_a_malformed_range(tmp):
    """Fail loudly: a silently-empty result would read as "no such events"."""
    try:
        _PS.filter_line_range([_at(TS.format(1))], "notarange")
    except ValueError:
        return
    raise AssertionError("a malformed --range was accepted")


# --- filter_time ------------------------------------------------------------

def test_filter_time_without_bounds_returns_every_event(tmp):
    events = [_at("2026-07-30T09:30:00Z", 1), _at("nonsense", 2)]

    kept = _PS.filter_time(events, None, None)

    # No filter means no timestamp parsing at all -- even the bad one stays.
    assert _lines(kept) == [1, 2], _lines(kept)


def test_filter_time_intraday_boundary_is_inclusive(tmp):
    events = [_at("2026-07-30T11:59:59Z", 1),
              _at("2026-07-30T12:00:00Z", 2),
              _at("2026-07-30T12:00:01Z", 3)]

    # An event exactly on the boundary passes: the comparator is >=.
    assert _lines(_PS.filter_time(events, "12:00:00", None)) == [2, 3]
    assert _lines(_PS.filter_time(events, None, "12:00:00")) == [1, 2]


def test_filter_time_accepts_bare_hour_and_hour_minute_forms(tmp):
    events = [_at("2026-07-30T09:30:00Z", 1),
              _at("2026-07-30T12:00:00Z", 2)]

    assert _lines(_PS.filter_time(events, "10", None)) == [2]
    assert _lines(_PS.filter_time(events, "09:30", None)) == [1, 2]


def test_filter_time_absolute_date_is_a_midnight_floor(tmp):
    events = [_at("2026-07-30T23:59:59Z", 1), _at("2026-07-31T00:00:00Z", 2)]

    kept = _PS.filter_time(events, "2026-07-31", None)

    # A date crosses days, which the intra-day form cannot: the 30th is out.
    assert _lines(kept) == [2], _lines(kept)


def test_filter_time_absolute_datetime_bounds_a_window(tmp):
    events = [_at("2026-07-30T09:00:00Z", 1),
              _at("2026-07-30T12:00:00Z", 2),
              _at("2026-07-31T15:00:00Z", 3)]

    kept = _PS.filter_time(events, "2026-07-30T12:00:00", "2026-07-31T00:00:00")

    assert _lines(kept) == [2], _lines(kept)


def test_filter_time_relative_lookback_drops_older_events(tmp):
    now = datetime.now(timezone.utc)
    events = [_at((now - timedelta(days=10)).isoformat(), 1),
              _at(now.isoformat(), 2)]

    # `2d`/`3h`/`1w` all measure back from now; only the recent event survives.
    for arg in ("2d", "3h", "1w"):
        assert _lines(_PS.filter_time(events, arg, None)) == [2], arg


def test_filter_time_drops_events_with_an_unparseable_timestamp(tmp):
    """Documented behaviour, and the reason it matters: an event whose ts
    cannot be read has no place in a time-sliced report, so it goes rather
    than being silently treated as in-range.
    """
    events = [_at("2026-07-30T12:00:00Z", 1), _at("not-a-timestamp", 2)]

    kept = _PS.filter_time(events, "00:00:00", None)

    assert _lines(kept) == [1], _lines(kept)


def test_filter_time_treats_a_naive_timestamp_as_utc(tmp):
    """Some records carry no zone suffix. Comparing a naive datetime against
    an aware bound raises, so the filter has to attach UTC -- and the event
    must land where a UTC reading puts it."""
    events = [_at("2026-07-30T09:00:00", 1), _at("2026-07-30T15:00:00", 2)]

    kept = _PS.filter_time(events, "2026-07-30T12:00:00", None)

    assert _lines(kept) == [2], _lines(kept)


def test_filter_time_rejects_an_unparseable_argument(tmp):
    """`30m` is not a supported unit (only d/h/w) and is not a clock time;
    `2026-99-99` looks like a date but is not one.

    Both raise rather than passing everything through -- a silent no-op filter
    would report the whole session as if the window had been applied.
    """
    for bad in ("30m", "2026-99-99"):
        try:
            _PS.filter_time([_at(TS.format(1))], bad, None)
        except ValueError:
            continue
        raise AssertionError(f"an unparseable --after was accepted: {bad}")


# --- rulecheck primitives ---------------------------------------------------

def test_hr0_extract_backticked_reads_a_single_line_literal(tmp):
    lines = ["  `ls -la`  ", "Rulecheck Bash: ..."]

    assert _PS._hr0_extract_backticked(lines, 1) == "ls -la"


def test_hr0_extract_backticked_spans_a_multiline_literal(tmp):
    """A heredoc commit body is one backticked block across several lines.

    Reading only the line above the marker would compare the claim against the
    closing quote instead of the command.
    """
    lines = ["`git commit -m '", "body line", "'`", "Rulecheck Bash: ..."]

    assert _PS._hr0_extract_backticked(lines, 3) == "git commit -m '\nbody line\n'"


def test_hr0_extract_backticked_returns_none_without_a_block(tmp):
    assert _PS._hr0_extract_backticked(["plain text"], 1) is None
    # k == 0 means the marker was the first line: there is nothing above it.
    assert _PS._hr0_extract_backticked(["`ls`"], 0) is None


def test_hr0_extract_backticked_rejects_a_malformed_span(tmp):
    """A stray backtick means the block is not one well-formed literal, and
    guessing which half is the command would invent a mismatch."""
    # Closing line whose head carries a backtick of its own.
    assert _PS._hr0_extract_backticked(["a`b`", "Rulecheck Bash: ..."], 1) is None
    # Candidate opening line with more than one backtick on it.
    assert _PS._hr0_extract_backticked(
        ["`x` `y`", "tail`", "Rulecheck Bash: ..."], 2) is None
    # One backtick, but mid-line: prose, not the start of a literal.
    assert _PS._hr0_extract_backticked(
        ["echo `x", "tail`", "Rulecheck Bash: ..."], 2) is None
    # Nothing above ever opens a block.
    assert _PS._hr0_extract_backticked(
        ["plain prose", "tail`", "Rulecheck Bash: ..."], 2) is None


def test_hr0_strip_quotes_blanks_quoted_spans(tmp):
    stripped = _PS._hr0_strip_quotes("""grep 'a | b' f "c > d" e""")

    assert "|" not in stripped, stripped
    assert ">" not in stripped, stripped
    assert stripped.startswith("grep "), stripped


def test_hr0_count_pipes_ignores_or_and_quoted_pipes(tmp):
    assert _PS._hr0_count_pipes("a | b | c") == 2
    # `||` is boolean OR, not two pipes -- counting it would flag every
    # truthful `cmd || echo` claim as a mismatch.
    assert _PS._hr0_count_pipes("a || b") == 0
    assert _PS._hr0_count_pipes("echo 'a | b'") == 0


def test_hr0_redirect_form_classifies_each_shape(tmp):
    assert _PS._hr0_redirect_form("ls") == "none"
    assert _PS._hr0_redirect_form("ls &> out.txt") == "&> file"
    assert _PS._hr0_redirect_form("ls > out.txt 2>&1") == "> file 2>&1"
    assert _PS._hr0_redirect_form("ls 2>&1 > out.txt") == "> file 2>&1"
    assert _PS._hr0_redirect_form("ls > out.txt") == "other"


def test_hr0_redirect_form_ignores_stdin_redirects(tmp):
    """`<`, `<<` and `<<<` feed input; classing them as output redirects would
    contradict a claim of `redirect=none`."""
    assert _PS._hr0_redirect_form("cat < in.txt") == "none"
    assert _PS._hr0_redirect_form("cat <<< word") == "none"
    assert _PS._hr0_redirect_form("cat <<EOF") == "none"


def test_hr0_redirect_form_treats_powershell_outfile_as_a_redirect(tmp):
    assert _PS._hr0_redirect_form("Get-Item x | Out-File log.txt") == "other"
    assert _PS._hr0_redirect_form("Get-Item x | tee-object log.txt") == "other"


# --- _hr0_check_pair --------------------------------------------------------

_TRUE_CLAIM = "`ls -la`\nRulecheck Bash: |=0, timeout=120000, redirect=none, read-first=N/A"


def test_hr0_check_pair_accepts_a_truthful_claim(tmp):
    assert _PS._hr0_check_pair("ls -la", False, None, _TRUE_CLAIM) is None


def test_hr0_check_pair_reports_no_pair_when_the_marker_is_missing(tmp):
    assert _PS._hr0_check_pair("ls -la", False, None, "just prose") == "no-pair"


def test_hr0_check_pair_reports_no_pair_when_the_literal_does_not_match(tmp):
    """The marker is present but describes a different command.

    Accepting it would let one truthful pair vouch for every later call.
    """
    miss = _PS._hr0_check_pair("rm -rf /", False, None, _TRUE_CLAIM)

    assert miss == "no-pair", miss


def test_hr0_check_pair_accepts_an_unbackticked_command_line(tmp):
    """The literal above the marker is the claim whether or not it is fenced;
    requiring backticks would flag a correct pair written plainly."""
    claim = ("ls -la\nRulecheck Bash: |=0, timeout=120000, redirect=none, "
             "read-first=N/A")

    assert _PS._hr0_check_pair("ls -la", False, None, claim) is None


def test_hr0_check_pair_ignores_a_marker_on_the_first_line(tmp):
    """A marker with nothing above it names no command, so it cannot vouch
    for this call -- and reading pre_lines[-1] would silently pair it with
    the LAST line of the message instead."""
    claim = ("Rulecheck Bash: |=0, timeout=120000, redirect=none, "
             "read-first=N/A\n`ls -la`")

    assert _PS._hr0_check_pair("ls -la", False, None, claim) == "no-pair"


def test_hr0_check_pair_rejects_a_bg_claim_for_a_foreground_call(tmp):
    """`timeout=bg` is not a number: the claim says the call was backgrounded
    when it ran in the foreground."""
    claim = ("`sleep 1`\nRulecheck Bash: |=0, timeout=bg, redirect=none, "
             "read-first=N/A")

    miss = _PS._hr0_check_pair("sleep 1", False, None, claim)

    assert miss == "timeout-mismatch (claimed=bg actual=120000)", miss


def test_hr0_check_pair_flags_a_tool_mismatch(tmp):
    claim = ("`ls`\nRulecheck PowerShell: |=0, timeout=120000, "
             "redirect=none, read-first=N/A")

    miss = _PS._hr0_check_pair("ls", False, None, claim, tool="Bash")

    assert miss is not None and miss.startswith("tool-mismatch"), miss
    assert "claimed=PowerShell" in miss and "actual=Bash" in miss, miss


def test_hr0_check_pair_defaults_the_timeout_to_the_harness_default(tmp):
    """A call with no explicit timeout runs at 120000ms, so that is what a
    truthful claim states -- and 120000 with an explicit timeout set elsewhere
    is a mismatch."""
    assert _PS._hr0_check_pair("ls -la", False, None, _TRUE_CLAIM) is None

    miss = _PS._hr0_check_pair("ls -la", False, 5000, _TRUE_CLAIM)

    assert miss == "timeout-mismatch (claimed=120000 actual=5000)", miss


def test_hr0_check_pair_requires_bg_for_a_backgrounded_call(tmp):
    bg_claim = ("`sleep 1`\nRulecheck Bash: |=0, timeout=bg, "
                "redirect=none, read-first=N/A")

    assert _PS._hr0_check_pair("sleep 1", True, None, bg_claim) is None

    fg_claim = ("`sleep 1`\nRulecheck Bash: |=0, timeout=120000, "
                "redirect=none, read-first=N/A")
    miss = _PS._hr0_check_pair("sleep 1", True, None, fg_claim)
    assert miss == "timeout-mismatch (claimed=120000 actual=bg)", miss


def test_hr0_check_pair_matches_only_the_first_line_of_a_multiline_command(tmp):
    """A shell call whose body is many lines is claimed by its first line OR
    its whole text; both spellings must be accepted."""
    claim = ("`for f in *; do`\nRulecheck Bash: |=0, timeout=120000, "
             "redirect=none, read-first=N/A")

    assert _PS._hr0_check_pair("for f in *; do\n  echo $f\ndone", False, None,
                               claim) is None


# --- compute_rulecheck ------------------------------------------------------

def test_rulecheck_is_silent_for_a_truthful_claim(tmp):
    events = [_text(_TRUE_CLAIM, line=1), _call("Bash", line=2, command="ls -la")]

    assert _PS.compute_rulecheck(events) == ""


def test_rulecheck_flags_a_pipe_count_mismatch(tmp):
    claim = ("`grep x f | wc -l`\nRulecheck Bash: |=0, timeout=120000, "
             "redirect=none, read-first=N/A")
    events = [_text(claim, line=1),
              _call("Bash", line=2, command="grep x f | wc -l")]

    report = _PS.compute_rulecheck(events)

    assert "pipe-mismatch (claimed=|=0 actual=|=1)" in report, report
    assert report.startswith("L2\ttool_use=tu2\t"), report


def test_rulecheck_flags_a_timeout_mismatch(tmp):
    claim = ("`ls`\nRulecheck Bash: |=0, timeout=5000, redirect=none, "
             "read-first=N/A")
    events = [_text(claim, line=1), _call("Bash", line=2, command="ls")]

    report = _PS.compute_rulecheck(events)

    assert "timeout-mismatch (claimed=5000 actual=120000)" in report, report


def test_rulecheck_flags_a_redirect_form_mismatch(tmp):
    claim = ("`ls > out.txt 2>&1`\nRulecheck Bash: |=0, timeout=120000, "
             "redirect=none, read-first=N/A")
    events = [_text(claim, line=1),
              _call("Bash", line=2, command="ls > out.txt 2>&1")]

    report = _PS.compute_rulecheck(events)

    assert "redirect-mismatch" in report, report
    assert "actual='> file 2>&1'" in report, report


def test_rulecheck_flags_a_call_with_no_pair_at_all(tmp):
    report = _PS.compute_rulecheck([_call("Bash", line=2, command="ls")])

    assert report == "L2\ttool_use=tu2\tmiss=no-pair\tcmd=ls", report


def test_rulecheck_reports_exactly_one_finding_per_offending_call(tmp):
    """Two bad calls, two lines -- and each names its own tool_use id."""
    events = [_call("Bash", line=1, command="a"), _call("Bash", line=2, command="b")]

    report = _PS.compute_rulecheck(events)

    assert len(report.splitlines()) == 2, report
    assert "tool_use=tu1" in report and "tool_use=tu2" in report, report


def test_rulecheck_stops_collecting_text_at_the_turn_boundary(tmp):
    """A claim made in a previous turn does not vouch for this turn's call.

    The walk back stops at the user message, so the earlier pair is invisible.
    """
    events = [_text(_TRUE_CLAIM, line=1),
              _user("do it again", line=2),
              _call("Bash", line=3, command="ls -la")]

    report = _PS.compute_rulecheck(events)

    assert "miss=no-pair" in report, report


def test_rulecheck_joins_assistant_text_across_intra_turn_records(tmp):
    """One API response is split across JSONL records and sequenced through
    tool_results, so the pair for a later call can sit in an earlier record."""
    events = [_text("`ls -la`", line=1),
              _res(line=2),
              _text("Rulecheck Bash: |=0, timeout=120000, redirect=none, "
                    "read-first=N/A", line=3),
              _call("Bash", line=4, command="ls -la")]

    assert _PS.compute_rulecheck(events) == ""


def test_rulecheck_ignores_non_shell_tools(tmp):
    """Only Bash and PowerShell carry the HR0 obligation; a Read with no
    Rulecheck line is not a violation."""
    events = [_call("Read", line=1, file_path="/repo/a.py")]

    assert _PS.compute_rulecheck(events) == ""


def test_rulecheck_truncates_a_long_command_in_the_finding(tmp):
    long_cmd = "echo " + "x" * 200
    events = [_call("Bash", line=1, command=long_cmd)]

    report = _PS.compute_rulecheck(events)

    cmd_field = report.split("\tcmd=", 1)[1]
    # 80 chars keeps one finding on one line for a hook that reads them.
    assert len(cmd_field) == 80, len(cmd_field)


# --- branch primitives ------------------------------------------------------

def test_branch_msg_id_tolerates_a_missing_or_non_dict_message(tmp):
    assert _PS._branch_msg_id({"message": {"id": "m1"}}) == "m1"
    assert _PS._branch_msg_id({}) is None
    # Real records carry `"message": null` on meta rows; a crash here would
    # take down the whole --branches pass.
    assert _PS._branch_msg_id({"message": None}) is None


def test_branch_is_tool_result_detects_a_tool_result_block(tmp):
    tool_res = {"message": {"content": [{"type": "tool_result"}]}}

    assert _PS._branch_is_tool_result(tool_res) is True
    assert _PS._branch_is_tool_result({"message": {"content": "plain"}}) is False
    assert _PS._branch_is_tool_result({"message": None}) is False


def test_branch_summary_renders_each_block_kind(tmp):
    rec = {"message": {"content": [
        {"type": "text", "text": "hello   there"},
        {"type": "thinking", "thinking": "..."},
        {"type": "tool_use", "name": "Bash", "input": {"command": "ls"}},
        {"type": "tool_result", "content": "done"},
        "not a dict",
    ]}}

    summary = _PS._branch_summary(rec, 500)

    assert "hello there" in summary, summary  # whitespace collapsed
    assert "thinking…" in summary, summary
    assert 'TOOL Bash: {"command": "ls"}' in summary, summary
    assert "result: done" in summary, summary


def test_branch_summary_truncates_to_the_limit(tmp):
    rec = {"message": {"content": "w " * 100}}

    summary = _PS._branch_summary(rec, 20)

    assert summary.endswith("…"), summary
    assert len(summary) == 21, len(summary)


def test_branch_read_records_skips_junk_and_dedupes_by_uuid(tmp):
    path = _write_jsonl(tmp, "mixed.jsonl", [
        "not json at all", "", "[1, 2]", _user_rec("z0", None, "keep", 1)])

    per_file = _PS._branch_read_records([path, path, os.path.join(tmp, "gone.jsonl")])

    # Same file twice: the second pass is deduped to nothing. The missing file
    # is dropped entirely rather than raising.
    assert [len(recs) for _p, recs in per_file] == [1, 0], per_file
    kept = per_file[0][1][0]
    assert kept["uuid"] == "z0", kept
    # _line is the 1-based physical line, counted past the junk lines.
    assert kept["_line"] == 4, kept


def test_branch_stats_counts_messages_tools_tokens_and_subagents(tmp):
    recs = [
        _user_rec("u0", None, "go", 1),
        _asst_rec("a1", "u0", "m1", [
            {"type": "tool_use", "name": "Bash", "input": {}},
            {"type": "tool_use", "name": "Task", "input": {}},
        ], 2, usage={"output_tokens": 40}),
        {"type": "system", "uuid": "s1", "message": {"usage": {"output_tokens": 999}}},
    ]

    msgs, tools, out_tok, subagents = _PS._branch_stats(recs)

    # The system record contributes nothing: only user/assistant rows count,
    # so its 999 tokens must not reach the total.
    assert (msgs, tools, out_tok, subagents) == (2, 2, 40, 1)


def test_branch_stats_survives_a_record_with_no_usage(tmp):
    msgs, tools, out_tok, subagents = _PS._branch_stats(
        [_user_rec("u0", None, "go", 1)])

    assert (msgs, tools, out_tok, subagents) == (1, 0, 0, 0)


# --- compute_branches -------------------------------------------------------

def _forked_session(tmp, name="fork.jsonl"):
    """One prompt answered twice: the first answer is abandoned by a rewind."""
    return _write_jsonl(tmp, name, [
        _user_rec("u0", None, "do the thing", 1),
        _asst_rec("a1", "u0", "msg_1", [
            {"type": "text", "text": "first attempt"},
            {"type": "tool_use", "name": "Bash", "input": {"command": "ls"}},
        ], 2, usage={"output_tokens": 40}),
        _asst_rec("a2", "u0", "msg_2", [{"type": "text", "text": "second attempt"}],
                  3, usage={"output_tokens": 7}),
    ])


def test_branches_finds_a_genuine_fork_and_splits_live_from_abandoned(tmp):
    report = _PS.compute_branches([_forked_session(tmp)])

    assert "1 genuine rewind ·" in report, report
    # The later child is the file's HEAD, so it is the live branch and the
    # earlier one is the abandoned work only this mode can show.
    assert "○ ABANDONED L2" in report, report
    assert "● LIVE      L3" in report, report
    assert "abandoned: 1 msgs / 1 tools / 40 output tok" in report, report
    assert "live: 2 msgs / 0 tools / 7 output tok" in report, report
    # The abandoned branch is dumped in full, not summarized.
    assert "full trace:" in report, report
    assert "first attempt" in report, report


def test_branches_walks_the_whole_abandoned_subtree(tmp):
    """An abandoned branch is usually several records deep. Counting only the
    fork's immediate child would under-report the work that was thrown away.
    """
    path = _write_jsonl(tmp, "deep.jsonl", [
        _user_rec("u0", None, "do the thing", 1),
        _asst_rec("a1", "u0", "msg_1",
                  [{"type": "tool_use", "id": "t1", "name": "Bash",
                    "input": {"command": "ls"}}], 2, usage={"output_tokens": 10}),
        _result_rec("r1", "a1", "t1", 3),
        _asst_rec("a2", "r1", "msg_2", [{"type": "text", "text": "gave up"}], 4,
                  usage={"output_tokens": 5}),
        {"type": "system", "uuid": "sys1", "parentUuid": "a2",
         "timestamp": TS.format(5), "content": "meta"},
        _asst_rec("b1", "u0", "msg_3", [{"type": "text", "text": "retry"}], 6,
                  usage={"output_tokens": 3}),
    ])

    report = _PS.compute_branches([path])

    assert "1 genuine rewind ·" in report, report
    # 3 abandoned message records (a1, r1, a2) -- the system record is in the
    # subtree but is not a message, so it counts nowhere and prints nowhere.
    assert "abandoned: 3 msgs / 1 tools / 15 output tok" in report, report
    assert "gave up" in report, report
    assert "meta" not in report, report


def test_branches_terminates_on_a_self_parented_record(tmp):
    """A malformed record pointing at itself must not spin the live-branch
    walk forever."""
    path = _write_jsonl(tmp, "cycle.jsonl", [
        _user_rec("u0", None, "go", 1),
        _asst_rec("a1", "a1", "m1", [{"type": "text", "text": "loop"}], 2)])

    report = _PS.compute_branches([path])

    assert "0 genuine rewinds" in report, report


def test_branches_skips_a_file_with_no_usable_records(tmp):
    empty = _write_jsonl(tmp, "empty.jsonl", [])
    real = _forked_session(tmp)

    report = _PS.compute_branches([empty, real])

    # The empty file contributes no section at all, so the surviving section
    # is not tagged as one of several files.
    assert "empty.jsonl" not in report, report
    assert "1 genuine rewind ·" in report, report


def test_branches_reports_none_for_a_linear_file(tmp):
    path = _write_jsonl(tmp, "linear.jsonl", [
        _user_rec("u0", None, "hi", 1),
        _asst_rec("a1", "u0", "m1", [{"type": "text", "text": "yo"}], 2)])

    report = _PS.compute_branches([path])

    assert "0 genuine rewinds" in report, report
    assert "No genuine rewinds" in report, report
    assert "ABANDONED" not in report, report


def test_branches_ignores_a_tool_thread_join(tmp):
    """A tool_use record has two children -- the tool_result and the assistant
    continuation carrying the SAME message id. That is streaming/tool
    threading, not a rewind, and reporting it as one would mark live work
    abandoned."""
    path = _write_jsonl(tmp, "join.jsonl", [
        _user_rec("u0", None, "go", 1),
        _asst_rec("a1", "u0", "m1",
                  [{"type": "tool_use", "name": "Bash", "input": {"command": "ls"}}], 2),
        _result_rec("r1", "a1", "t1", 3),
        _asst_rec("a2", "a1", "m1", [{"type": "text", "text": "done"}], 4),
    ])

    report = _PS.compute_branches([path])

    assert "0 genuine rewinds · 1 tool-thread join ignored" in report, report


def test_branches_handles_no_source_files(tmp):
    report = _PS.compute_branches([])

    assert report.endswith("(no source files)"), report


def test_branches_tags_each_file_when_several_sessions_are_joined(tmp):
    """Folder mode joins distinct sessions. uuids are unique per session, so
    the trees must stay separate -- and each section must say which file it
    describes."""
    one = _forked_session(tmp, "s1.jsonl")
    two = _write_jsonl(tmp, "s2.jsonl", [
        _user_rec("y0", None, "b", 1),
        _asst_rec("y1", "y0", "n1", [{"type": "text", "text": "three"}], 2)])

    report = _PS.compute_branches([one, two])

    assert "[s1.jsonl] 1 genuine rewind" in report, report
    assert "[s2.jsonl] 0 genuine rewinds" in report, report


def test_branches_honours_a_time_window_on_the_branch_point(tmp):
    path = _forked_session(tmp)

    inside = _PS.compute_branches([path], after="2026-07-30")
    outside = _PS.compute_branches([path], after="2026-07-31")

    assert "1 genuine rewind ·" in inside, inside
    # The fork's branch point is on the 30th, so a floor of the 31st hides it.
    assert "0 genuine rewinds" in outside, outside


def test_branches_verbose_stops_truncating_the_summaries(tmp):
    path = _write_jsonl(tmp, "long.jsonl", [
        _user_rec("q0", None, "go", 1),
        _asst_rec("q1", "q0", "m1", [{"type": "text", "text": "word " * 60}], 2),
        _asst_rec("q2", "q0", "m2", [{"type": "text", "text": "live"}], 3)])

    terse = _PS.compute_branches([path])
    verbose = _PS.compute_branches([path], verbose=True)

    assert "…" in terse, terse
    assert "…" not in verbose, verbose
    assert len(verbose) > len(terse)


def test_branches_with_source_tags_a_single_file_on_request(tmp):
    path = _forked_session(tmp)

    assert "[fork.jsonl]" not in _PS.compute_branches([path])
    assert "[fork.jsonl]" in _PS.compute_branches([path], with_source=True)


# --- argv splitting and segment labelling -----------------------------------

def test_split_argv_on_and_returns_one_chunk_without_the_token(tmp):
    assert _PS._split_argv_on_and(["f.jsonl", "--stats"]) == [["f.jsonl", "--stats"]]
    assert _PS._split_argv_on_and([]) == [[]]


def test_split_argv_on_and_preserves_empty_segments(tmp):
    """`--and --and` must survive the split so main() can reject it; silently
    collapsing it would run a chain the user did not write."""
    chunks = _PS._split_argv_on_and(["a", "--and", "b", "--and", "--and", "c"])

    assert chunks == [["a"], ["b"], [], ["c"]], chunks


def test_active_mode_flags_lists_every_active_mode(tmp):
    assert _PS._active_mode_flags(_ns()) == []
    assert _PS._active_mode_flags(_ns(stats=True, latency=True)) == ["stats", "latency"]


def test_active_mode_flags_counts_hook_event_as_the_hooks_mode(tmp):
    """--hook-event routes to compute_hooks on its own, so it is one mode --
    and paired with --hooks it must not count as two."""
    assert _PS._active_mode_flags(_ns(hook_event="Stop")) == ["hooks"]
    assert _PS._active_mode_flags(_ns(hooks=True, hook_event="Stop")) == ["hooks"]


def test_segment_label_prefers_the_short_circuit_flags(tmp):
    assert _PS._segment_label(_ns(version=True, stats=True)) == "--version"
    assert _PS._segment_label(_ns(cheat_sheet=True, version=True)) == "--cheat-sheet"


def test_segment_label_renders_a_mode_in_its_cli_spelling(tmp):
    # The namespace attribute is bg_tasks; the label must read as typed.
    assert _PS._segment_label(_ns(bg_tasks=True)) == "--bg-tasks"
    assert _PS._segment_label(_ns(hook_event="Stop")) == "--hooks"


def test_segment_label_joins_type_filters_and_falls_back(tmp):
    assert _PS._segment_label(_ns(tools=True, user=True)) == "--tools+--user"
    assert _PS._segment_label(_ns()) == "(default)"


# --- _run_segment -----------------------------------------------------------

def _segment_events():
    return [_call("Bash", line=1, second=1, command="echo alpha"),
            _call("Read", line=2, second=2, file_path="/repo/beta.py"),
            _user("hello", line=3, second=3)]


def test_run_segment_dispatches_the_selected_mode(tmp):
    output = _PS._run_segment(_ns(commands=True), _segment_events(), [])

    assert output.startswith("SHELL COMMANDS"), output
    assert "$ echo alpha" in output, output


def test_run_segment_short_circuits_version_without_touching_events(tmp):
    output = _PS._run_segment(_ns(version=True), [], [])

    assert output == f"parse_session.py {_PS.__version__}", output


def test_run_segment_short_circuits_the_cheat_sheet(tmp):
    """Per-segment, so `--cheat-sheet` inside an --and chain prints the table
    instead of running the previous segment's mode again."""
    output = _PS._run_segment(_ns(cheat_sheet=True), _segment_events(), [])

    assert output == _PS._LINE_GREP_CHEAT_SHEET
    assert "--line-grep" in output, output[:200]


def test_run_segment_type_filters_admit_their_own_event_kind(tmp):
    events = [_call("Bash", line=1, second=1, command="echo alpha"),
              _res(line=2, second=2, detail="plain output"),
              _user("hello", line=3, second=3),
              _text("hi back", line=4, second=4),
              {"type": "thinking", "ts": TS.format(5), "line": 5, "detail": "hmm"}]

    def kept(**flags):
        return [ln.split()[0] for ln in
                _PS._run_segment(_ns(**flags), events, []).splitlines() if ln]

    assert kept(tools=True) == ["L1"]
    assert kept(results=True) == ["L2"]
    assert kept(user=True) == ["L3"]
    assert kept(assistant=True) == ["L4"]
    assert kept(thinking=True) == ["L5"]
    assert kept(conversation=True) == ["L3", "L4"]
    # Unioned, not intersected: two filters widen the selection.
    assert kept(tools=True, user=True) == ["L1", "L3"]


def test_run_segment_errors_filter_keeps_only_failing_results(tmp):
    """--errors alone narrows results to the ones that actually failed;
    without that narrowing it would be a synonym for --results."""
    events = [_res(line=1, second=1, detail="all good"),
              _res(line=2, second=2, detail="Traceback (most recent call last)"),
              _call("Bash", line=3, second=3, command="ls")]

    output = _PS._run_segment(_ns(errors=True), events, [])

    assert output.startswith("L2"), output
    assert len(output.splitlines()) == 1, output


def test_run_segment_agents_filter_forces_verbose(tmp):
    """Agent spawns are shown for their prompts, which truncation would hide."""
    args = _ns(agents=True)

    _PS._run_segment(args, _segment_events(), [])

    assert args.verbose is True


def test_run_segment_falls_back_to_a_plain_event_dump(tmp):
    output = _PS._run_segment(_ns(), _segment_events(), [])

    assert len(output.splitlines()) == 3, output
    assert "USER: hello" in output, output


def test_run_segment_applies_the_path_filter(tmp):
    events = _segment_events()

    hit = _PS._run_segment(_ns(commands=True, path="alpha"), events, [])
    miss = _PS._run_segment(_ns(commands=True, path="beta"), events, [])

    # `beta` matches the Read, not the Bash command, so the shell-command
    # report comes back with a header and nothing under it.
    assert "$ echo alpha" in hit, hit
    assert "$ echo alpha" not in miss, miss


def test_run_segment_applies_the_tool_filter(tmp):
    output = _PS._run_segment(_ns(tools=True, tool="Read"), _segment_events(), [])

    assert "Read(" in output, output
    assert "Bash(" not in output, output


def test_run_segment_applies_the_line_range(tmp):
    output = _PS._run_segment(_ns(tools=True, range="2-3"), _segment_events(), [])

    assert output.startswith("L2"), output
    assert "L1" not in output, output


def test_run_segment_grep_keeps_only_matching_blocks(tmp):
    output = _PS._run_segment(_ns(tools=True, grep="alpha"), _segment_events(), [])

    assert "echo alpha" in output, output
    assert "beta.py" not in output, output


def test_run_segment_grep_keeps_a_whole_block_not_just_the_matching_line(tmp):
    """--grep is block-level: a hit on the call line must bring its result
    lines with it, including a wrapped continuation line that matches nothing
    on its own. Line-level filtering would strip the answer off the question.
    """
    events = _segment_events()
    events[0]["paired_result"] = {
        "type": "tool_result", "ts": TS.format(2), "line": 9,
        "detail": "alpha\nsecond line", "is_error": False}

    output = _PS._run_segment(_ns(pairs=True, grep="alpha"), events, [])

    assert "└─ RESULT" in output, output
    assert "second line" in output, output
    assert "beta.py" not in output, output


def test_run_segment_grep_drops_a_report_header_that_does_not_match(tmp):
    """The banner of a compute_* report is its own block. --grep is applied
    after the mode, so a non-matching header goes with everything else."""
    output = _PS._run_segment(_ns(commands=True, grep="alpha"), _segment_events(), [])

    assert "SHELL COMMANDS" not in output, output
    assert "$ echo alpha" in output, output


def test_run_segment_grep_matches_a_block_by_its_result_text(tmp):
    """The match need not be on the L-line: a block is kept when any of its
    lines matches, which is what makes --grep useful on tool output."""
    events = _segment_events()
    events[0]["paired_result"] = {
        "type": "tool_result", "ts": TS.format(2), "line": 9,
        "detail": "needle in the output", "is_error": False}

    output = _PS._run_segment(_ns(pairs=True, grep="needle"), events, [])

    assert output.startswith("L1"), output
    assert "beta.py" not in output, output


def test_run_segment_full_implies_verbose(tmp):
    args = _ns(full=True, timeline=True)

    _PS._run_segment(args, _segment_events(), [])

    assert args.verbose is True


def test_run_segment_does_not_consume_the_shared_event_list(tmp):
    """Every segment of an --and chain runs against the same loaded lists, so
    a filter that mutated them in place would starve the next segment."""
    events = _segment_events()
    before = list(events)

    _PS._run_segment(_ns(commands=True, range="1-1"), events, [])
    second = _PS._run_segment(_ns(tools=True), events, [])

    assert events == before
    assert "beta.py" in second, second


# --- main() -----------------------------------------------------------------

def test_main_version_prints_and_exits_zero(tmp):
    code, out, _err = _run_main(tmp, ["--version"])

    assert code == 0, code
    assert out.strip() == f"parse_session.py {_PS.__version__}", out


def test_main_forces_utf8_on_a_non_utf8_stdout(tmp):
    """Session content carries characters cp1250/cp932 cannot encode, so
    main() reconfigures stdout before printing anything. Without it the first
    such report dies with a UnicodeEncodeError on a Czech or Japanese
    Windows console.
    """
    raw = io.BytesIO()
    stream = io.TextIOWrapper(raw, encoding="cp1252", newline="")
    saved_argv = sys.argv
    try:
        sys.argv = ["parse_session.py", "--version"]
        with contextlib.redirect_stdout(stream):
            try:
                _PS.main()
            except SystemExit:
                pass
        stream.flush()
    finally:
        sys.argv = saved_argv

    assert stream.encoding == "utf-8", stream.encoding
    assert raw.getvalue().decode("utf-8").strip() == (
        f"parse_session.py {_PS.__version__}")


def test_main_cheat_sheet_needs_no_jsonl(tmp):
    code, out, _err = _run_main(tmp, ["--cheat-sheet"])

    assert code == 0, code
    assert "--line-grep" in out, out[:200]


def test_main_missing_file_exits_one_with_a_message_on_stderr(tmp):
    missing = os.path.join(tmp, "nope.jsonl")

    code, out, err = _run_main(tmp, [missing, "--stats"])

    assert code == 1, code
    assert out == "", out
    assert f"File not found: {missing}" in err, err


def test_main_stats_mode_reports_the_event_counts(tmp):
    path = _simple_session(tmp)

    code, out, err = _run_main(tmp, [path, "--stats"])

    assert code is None, code
    assert err == "", err
    assert "SESSION STATISTICS" in out, out
    assert "User msgs:  1" in out, out
    assert "Tool calls: 1" in out, out
    assert "Results:    1 (0 errors)" in out, out


def test_main_commands_mode_prints_the_command(tmp):
    path = _simple_session(tmp)

    _code, out, _err = _run_main(tmp, [path, "--commands"])

    assert "$ echo alpha" in out, out


def test_main_branches_mode_reads_the_raw_records(tmp):
    """--branches re-reads the source file rather than the event stream, so
    this is the only mode whose output depends on the resolved source path
    surviving the filters."""
    path = _forked_session(tmp)

    _code, out, _err = _run_main(tmp, [path, "--branches"])

    assert "1 genuine rewind ·" in out, out
    assert "○ ABANDONED" in out, out


def _projects_dir(tmp, slug):
    """A fake ~/.claude/projects/<slug>/ inside the test's temp dir.

    _run_main points HOME/USERPROFILE here, so main()'s session-ID probe walks
    this tree instead of the machine's real one.
    """
    path = os.path.join(tmp, ".claude", "projects", slug)
    os.makedirs(path, exist_ok=True)
    return path


def test_main_resolves_a_bare_session_id(tmp):
    """A session ID is not a path: main() probes ~/.claude/projects for it,
    which is the form every operator actually types."""
    project = _projects_dir(tmp, "-repo-one")
    _simple_session(project, "abc123.jsonl")

    code, out, err = _run_main(tmp, ["abc123", "--commands"])

    assert code is None, code
    assert err == "", err
    assert "$ echo alpha" in out, out


def test_main_rejects_an_ambiguous_session_id(tmp):
    """The same ID under two projects cannot be resolved silently -- picking
    one would analyse the wrong session and say nothing about it."""
    for slug in ("-repo-one", "-repo-two"):
        _simple_session(_projects_dir(tmp, slug), "dup.jsonl")

    code, out, err = _run_main(tmp, ["dup", "--commands"])

    assert code == 1, code
    assert out == "", out
    assert "Ambiguous session ID 'dup' — matches 2 files:" in err, err
    assert err.count("dup.jsonl") == 2, err


def test_main_find_linegrep_replaces_the_mode_run(tmp):
    path = _simple_session(tmp)

    code, out, err = _run_main(tmp, [path, "--commands", "--find-linegrep"])

    assert code is None, code
    assert err == "", err
    assert out.startswith("FIND --line-grep"), out
    assert "mode: --commands" in out, out
    # The mode's own report is replaced by the search, not printed alongside.
    assert "SHELL COMMANDS" not in out, out


def test_main_accepts_a_directory_and_walks_it(tmp):
    folder = os.path.join(tmp, "corpus")
    os.makedirs(folder, exist_ok=True)
    _simple_session(folder, "one.jsonl")

    _code, out, err = _run_main(tmp, [folder, "--commands"])

    assert "$ echo alpha" in out, out
    assert err == "", err


def test_main_rejects_a_directory_with_no_jsonl(tmp):
    folder = os.path.join(tmp, "empty")
    os.makedirs(folder, exist_ok=True)

    code, _out, err = _run_main(tmp, [folder, "--commands"])

    assert code == 1, code
    assert "No *.jsonl files under" in err, err


def test_main_chained_segments_print_both_labelled_reports(tmp):
    path = _simple_session(tmp)

    code, out, _err = _run_main(tmp, [path, "--commands", "--and", "--stats"])

    assert code is None, code
    assert "===== --commands =====" in out, out
    assert "===== --stats =====" in out, out
    # The load happens once and both segments see it.
    assert out.index("--commands") < out.index("--stats")


def test_main_rejects_two_modes_in_one_chained_segment(tmp):
    path = _simple_session(tmp)

    code, _out, err = _run_main(
        tmp, [path, "--stats", "--commands", "--and", "--latency"])

    assert code == 2, code
    assert "exactly" in err and "ONE mode flag" in err, err


def test_main_rejects_an_empty_chain_segment(tmp):
    path = _simple_session(tmp)

    code, _out, err = _run_main(tmp, [path, "--commands", "--and"])

    assert code == 2, code
    assert "is empty" in err, err


def test_main_rejects_a_jsonl_path_in_a_later_segment(tmp):
    path = _simple_session(tmp)

    code, _out, err = _run_main(tmp, [path, "--commands", "--and", path, "--stats"])

    assert code == 2, code
    assert "may appear only in" in err, err


def test_main_rejects_line_grep_in_a_later_segment(tmp):
    """--line-grep filters at load time, which happens once for the chain, so
    naming it in a later segment could not do what it says."""
    path = _simple_session(tmp)

    code, _out, err = _run_main(
        tmp, [path, "--commands", "--and", "--stats", "--line-grep", "tool_use"])

    assert code == 2, code
    assert "pre-parse filter" in err, err


def test_main_writes_to_the_output_file(tmp):
    path = _simple_session(tmp)
    dest = os.path.join(tmp, "report.txt")

    _code, out, _err = _run_main(tmp, [path, "--commands", "-o", dest])

    assert f"Output written to {dest}" in out, out
    with open(dest, encoding="utf-8") as fh:
        body = fh.read()
    assert "$ echo alpha" in body, body


def test_main_reports_a_failing_segment_and_exits_one(tmp):
    """A segment that raises must not abort the chain -- the load was the
    expensive part -- but the run still ends non-zero."""
    path = _simple_session(tmp)

    code, out, err = _run_main(
        tmp, [path, "--commands", "--after", "30m", "--and", "--stats"])

    assert code == 1, code
    assert "segment 0 (--commands) failed" in err, err
    # The healthy segment still produced its report.
    assert "SESSION STATISTICS" in out, out


def test_main_timing_writes_phase_timings_to_stderr(tmp):
    path = _simple_session(tmp)

    _code, out, err = _run_main(tmp, [path, "--commands", "--timing"])

    assert "$ echo alpha" in out, out
    assert "TOTAL (instrumented phases)" in err, err
    assert "extract_events" in err, err


def test_main_requires_a_jsonl_for_a_mode(tmp):
    code, _out, err = _run_main(tmp, ["--stats"])

    assert code == 2, code
    assert "required: jsonl" in err, err


# --- find_linegrep helpers --------------------------------------------------

def test_gather_jsonl_returns_the_file_or_the_folder_contents(tmp):
    folder = os.path.join(tmp, "corpus")
    os.makedirs(folder, exist_ok=True)
    one = _simple_session(folder, "one.jsonl")
    two = _simple_session(folder, "two.jsonl")
    _write_jsonl(folder, "notes.txt", [{"ignored": True}])

    assert _PS._gather_jsonl(one) == [one]
    # Sorted, and the non-jsonl file is not part of the corpus.
    assert _PS._gather_jsonl(folder) == sorted([one, two])


def test_cache_parse_keeps_the_line_number_and_skips_blanks(tmp):
    path = _write_jsonl(tmp, "gappy.jsonl", [
        _user_rec("u0", None, "hi", 1), "", _user_rec("u1", None, "again", 2)])

    cached = _PS._cache_parse(path)

    # The blank line is dropped but does not shift the numbering: a finding
    # reported as L3 has to point at line 3 of the file on disk.
    assert [line_num for line_num, _raw, _objs in cached] == [1, 3]
    assert cached[0][2][0]["uuid"] == "u0"


def test_cache_parse_salvages_a_torn_line_holding_two_records(tmp):
    """Interleaved writes concatenate two records onto one physical line.
    json.loads fails on it, and dropping the line would lose both records."""
    one = json.dumps(_user_rec("u0", None, "hi", 1), separators=(",", ":"))
    two = json.dumps(_user_rec("u1", None, "again", 2), separators=(",", ":"))
    path = _write_jsonl(tmp, "torn.jsonl", [one + two])

    cached = _PS._cache_parse(path)

    assert len(cached) == 1, cached
    assert [obj["uuid"] for obj in cached[0][2]] == ["u0", "u1"]


def test_build_from_cache_matches_extract_events(tmp):
    """The whole point of the cache path is byte-identical results from a
    reused parse; a divergence here would make find_linegrep recommend a
    filter that changes the report."""
    path = _simple_session(tmp)

    direct_ev, direct_meta = _PS.extract_events(path)
    cached_ev, cached_meta = _PS._build_from_cache(
        _PS._cache_parse(path), path, None, None)

    assert cached_ev == direct_ev
    assert cached_meta == direct_meta
    assert all(e["src"] == path for e in cached_ev)


def test_build_from_cache_applies_a_literal_line_filter(tmp):
    path = _simple_session(tmp)
    cached = _PS._cache_parse(path)

    events, _meta = _PS._build_from_cache(cached, path, "tool_use", None)

    # The user prompt's line carries no `tool_use`, so its event is gone.
    types = [e["type"] for e in events]
    assert "user_message" not in types, types
    assert "tool_call" in types, types


def test_build_from_cache_accepts_a_regex_line_filter(tmp):
    path = _simple_session(tmp)
    cached = _PS._cache_parse(path)

    events, _meta = _PS._build_from_cache(cached, path, "tool_use|no_such", None)

    assert [e["type"] for e in events] == [
        e["type"] for e in _PS._build_from_cache(cached, path, "tool_use", None)[0]]


def test_build_from_cache_accepts_a_precompiled_pattern(tmp):
    """find_linegrep hands the filter down as an object as well as a string;
    both spellings must select the same lines."""
    path = _simple_session(tmp)
    cached = _PS._cache_parse(path)

    compiled, _meta = _PS._build_from_cache(
        cached, path, re.compile("tool_use", re.IGNORECASE), None)
    literal, _meta2 = _PS._build_from_cache(cached, path, "tool_use", None)

    assert compiled == literal


def test_load_from_caches_sorts_and_links_like_the_load_path(tmp):
    path = _simple_session(tmp)
    caches = [(path, _PS._cache_parse(path))]

    events, _meta = _PS._load_from_caches(caches, None)

    assert [e["ts"] for e in events] == sorted(e["ts"] for e in events)
    call = next(e for e in events if e["type"] == "tool_call")
    # link_pairs ran: the call found its result.
    assert call.get("paired_result"), call


def test_count_admitted_counts_raw_lines_case_insensitively(tmp):
    path = _simple_session(tmp)
    caches = [(path, _PS._cache_parse(path))]

    assert _PS._count_admitted(caches, "ASSISTANT") == 1
    assert _PS._count_admitted(caches, "no-such-token") == 0
    assert _PS._count_admitted(caches, "timestamp") == 3


def test_scan_linegrep_candidates_counts_admitting_lines(tmp):
    path = _simple_session(tmp)

    total, counts = _PS._scan_linegrep_candidates([path], "Bash")

    assert total == 3, total
    # Every record carries a timestamp; only the assistant record has a usage.
    assert counts['"timestamp":'] == 3, counts['"timestamp":']
    assert counts['"usage":'] == 1, counts['"usage":']
    assert counts['"name":"Bash"'] == 1, counts['"name":"Bash"']


def test_scan_linegrep_candidates_ignores_blank_lines(tmp):
    """`total` is the denominator of every reported percentage, so a blank
    line counted as a record would understate how selective a filter is."""
    path = _write_jsonl(tmp, "gappy.jsonl", [
        _user_rec("u0", None, "hi", 1), "", _user_rec("u1", None, "again", 2)])

    total, _counts = _PS._scan_linegrep_candidates([path], None)

    assert total == 2, total


def test_scan_linegrep_candidates_ignores_an_unreadable_path(tmp):
    total, counts = _PS._scan_linegrep_candidates(
        [os.path.join(tmp, "does-not-exist.jsonl")], None)

    assert total == 0, total
    assert counts == {}, counts


def test_shared_value_substrings_finds_a_cross_value_discriminator(tmp):
    shared = _PS._shared_value_substrings({"tool_use", "tool_result"})

    # `tool_` spans both record kinds, which no single `"type":"X"` can.
    assert "tool_" in shared, shared
    # Longest first, so the most selective shared substring is tried first.
    assert shared == sorted(shared, key=lambda s: -len(s))
    assert _PS._shared_value_substrings({"abc", "xyz"}) == []


def test_shared_value_substrings_honours_the_share_threshold(tmp):
    values = {"alpha_one", "alpha_two", "beta"}

    assert "alpha_" in _PS._shared_value_substrings(values, min_share=2)
    assert _PS._shared_value_substrings(values, min_share=3) == []


# --- find_linegrep ----------------------------------------------------------

def test_find_linegrep_recommends_a_result_preserving_filter(tmp):
    path = _simple_session(tmp)

    report = _PS.find_linegrep(path, _ns(commands=True, jsonl=path))

    assert "mode: --commands" in report, report
    assert "files: 1" in report and "total lines: 3" in report, report
    # A recommendation is made, and the pattern it names was verified SAFE in
    # the table above it. (Which of several equally cheap patterns wins is not
    # asserted: the candidate pool is built through a set, so ties between
    # equal-admitting patterns are not ordered deterministically.)
    recommend = [ln for ln in report.splitlines() if ln.startswith("RECOMMEND")]
    assert len(recommend) == 1, report
    pattern = recommend[0].split("--line-grep '", 1)[1].rsplit("'", 1)[0]
    assert f"SAFE     {pattern}" in report, report


def test_find_linegrep_refuses_to_recommend_for_a_mode_that_matched_nothing(tmp):
    """The --thinking-with-no-thinking trap: with empty output, every filter
    reproduces the baseline and any recommendation would be meaningless."""
    path = _simple_session(tmp)

    report = _PS.find_linegrep(path, _ns(thinking=True, jsonl=path))

    assert "data-independent" in report, report
    assert "RECOMMEND" not in report, report


def test_find_linegrep_falls_back_to_a_two_way_union(tmp):
    """--conversation needs two disjoint record kinds here (a user prompt with
    no shared discriminator and an assistant reply), so no single substring is
    safe and the bounded OR search is what produces an answer."""
    path = _write_jsonl(tmp, "split.jsonl", [
        _user_rec("u0", None, "list the files", 1, userType="external"),
        _asst_rec("a1", "u0", "m1", [{"type": "text", "text": "ok"}], 2),
        _result_rec("r1", "a1", "t1", 3),
    ])

    report = _PS.find_linegrep(path, _ns(conversation=True, jsonl=path))

    assert "RECOMMEND (OR" in report, report
    assert "admits 2/3" in report, report
    # `"userType":` is the only key unique to the user prompt's line, so any
    # 2-of-3-line union that keeps the prompt has to include it.
    assert '"userType":' in report.rsplit("RECOMMEND", 1)[1], report


def test_find_linegrep_reports_an_empty_corpus(tmp):
    folder = os.path.join(tmp, "empty")
    os.makedirs(folder, exist_ok=True)

    report = _PS.find_linegrep(folder, _ns(commands=True, jsonl=folder))

    assert report.endswith(f"(no *.jsonl under {folder})"), report


def test_try_or_union_returns_none_when_no_pair_preserves_the_result(tmp):
    path = _simple_session(tmp)
    caches = [(path, _PS._cache_parse(path))]
    seg = _ns(conversation=True, jsonl=path)
    baseline = _PS._run_segment(seg, *_PS._load_from_caches(caches, None))

    result = _PS._try_or_union(
        caches, seg, baseline, [('"text":', 1), ('"model":', 1)], 3)

    assert result is None, result


def main():
    return _util.runner(_util.collect(globals()), tmp_prefix="cli_")


if __name__ == "__main__":
    raise SystemExit(main())
