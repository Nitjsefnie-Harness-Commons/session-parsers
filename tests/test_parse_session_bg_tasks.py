"""Suite for parse_session.py --bg-tasks path normalization (Windows drive
spellings). A background task's output read via git-bash `/c/...` or
`/cygdrive/c/...` must count as a read, the same as `C:\\...` / `C:/...`.

Windows-only: the detector skips never-read classification when the output
file doesn't exist on disk, so the tests need a real file at a drive-letter
path.
"""
import os

import _util

_PS = _util.load(_util.script("parse_session.py"))

TS = "2026-07-30T00:00:0{}Z"


def _gitbash(win_path):
    p = win_path.replace("\\", "/")
    return f"/{p[0].lower()}{p[2:]}"  # C:/x/y -> /c/x/y


def _cygdrive(win_path):
    p = win_path.replace("\\", "/")
    return f"/cygdrive/{p[0].lower()}{p[2:]}"


def _events(win_out, read_event=None):
    launch = {
        "type": "tool_call", "tool_name": "Bash", "ts": TS.format(1), "line": 1,
        "tool_use_id": "tu1",
        "tool_input": {"command": "sleep 1", "run_in_background": True},
        "paired_result": {"detail": (
            "Command running in background with ID: abc123\n"
            f"Output is being written to: {win_out}. You will be notified"
        )},
    }
    delivered = {
        "type": "user_message", "ts": TS.format(3), "line": 3,
        "detail": "<task-notification><task-id>abc123</task-id>done</task-notification>",
    }
    events = [launch, delivered]
    if read_event is not None:
        read_event.update({"ts": TS.format(4), "line": 4})
        events.append(read_event)
    meta = [{
        "type": "queue-operation", "ts": TS.format(2), "line": 2,
        "content": (
            "enqueue task-notification <task-id>abc123</task-id>"
            "<status>completed</status>"
            f"<output-file>{win_out}</output-file>"
            "<summary>task finished</summary>"
        ),
    }]
    return events, meta


def _report(tmp, read_event):
    win_out = os.path.join(tmp, "abc123.output")
    with open(win_out, "w", encoding="utf-8") as f:
        f.write("output\n")
    events, meta = _events(win_out, read_event)
    return _PS.compute_bg_tasks(events, meta)


def _read_tool(path):
    return {"type": "tool_call", "tool_name": "Read",
            "tool_input": {"file_path": path}, "paired_result": {}}


def _shell(cmd):
    return {"type": "tool_call", "tool_name": "Bash",
            "tool_input": {"command": cmd}, "paired_result": {}}


def test_windows_spellings_count_as_read(tmp):
    if os.name != "nt":
        return
    win_out = os.path.join(tmp, "abc123.output")
    for spelling in (win_out, win_out.replace("\\", "/"),
                     _gitbash(win_out), _cygdrive(win_out)):
        report = _report(tmp, _read_tool(spelling))
        assert "NOTIFIED BUT NEVER READ:" not in report, spelling
        assert "1 full" in report, spelling


def test_shell_gitbash_read_counts(tmp):
    if os.name != "nt":
        return
    win_out = os.path.join(tmp, "abc123.output")
    for cmd in (f"tail -20 {_gitbash(win_out)}",
                f"grep foo {_cygdrive(win_out)}",
                f"cat '{_gitbash(win_out)}'"):
        report = _report(tmp, _shell(cmd))
        assert "NOTIFIED BUT NEVER READ:" not in report, cmd
        assert "1 full" in report, cmd


def test_indirect_shell_reads_count(tmp):
    # #169: a variable-built path, a loop over task ids, and an interpreter
    # opening the literal path are all reads.
    out = os.path.join(tmp, "abc123.output")
    for cmd in (f"T={tmp}; tail -c 900 $T/abc123.output",
                f"for f in abc123 x; do tail -20 {tmp}/$f.output; done",
                f"python3 - <<'PY'\nprint(open('{out}').read())\nPY"):
        report = _report(tmp, _shell(cmd))
        assert "NOTIFIED BUT NEVER READ:" not in report, cmd
    report = _report(tmp, _shell("ls abc123.output"))
    assert "NOTIFIED BUT NEVER READ:" in report


def test_unread_task_still_flagged(tmp):
    # Control: with no read event at all the task must still be reported as
    # notified-never-read (i.e. the tests above aren't passing vacuously).
    report = _report(tmp, None)
    assert "NOTIFIED BUT NEVER READ:" in report
    assert "1 notified-never-read" in report


def main():
    return _util.runner(_util.collect(globals()), tmp_prefix="bgtaskpaths_")


if __name__ == "__main__":
    raise SystemExit(main())
