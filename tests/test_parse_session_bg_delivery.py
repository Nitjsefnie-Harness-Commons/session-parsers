"""Suite for the harness-delivered background-task reads (issue #383 facet).

A background command's completion can carry its output INLINE, and the session
reads it there:

  1. An explicit model-issued ``GetTask`` call whose result carries
     ``status: completed`` with non-empty ``result.content`` — keyed by its
     ``taskId``, same as ``TaskOutput`` already was.
  2. A harness-delivered completion — ``taskDelivery`` with ``status:
     completed`` and non-empty ``result.content`` — which appears on two
     record shapes: top-level on a task-notification ``user`` record
     (``origin.kind: task-notification``), and nested under ``attachment`` on
     a ``type: attachment`` / ``attachment.type: queued_command`` record
     (mid-turn delivery).

Both count as reads regardless of ``result.isError``: for a background
COMMAND that flag means the command exited nonzero, not that the delivery
failed — and a failed task's output is the one that most needs reading.
Negative cases: ``status: working`` (no ``result`` yet) and empty content are
NOT reads.

Fixture shapes are copied from real transcript records:
  - GetTask result user record: ``toolUseResult`` is the structured
    ``{taskId, status, statusMessage, result{isError, content[]}}`` and the
    tool_result content is its JSON text.
  - taskDelivery user record: ``origin.kind: task-notification``,
    ``turnOrigin: task_notification``, string content holding the
    ``<task-notification>`` text.
  - attachment record: ``{"type": "attachment", "attachment": {"type":
    "queued_command", "taskDelivery": {...}}}``.

The suites run each fixture through the real pipeline — ``extract_events`` →
``annotate_parallel`` → ``link_pairs`` → ``compute_bg_tasks`` — so an ingest
drop (taskDelivery never reaching the events) fails these tests, not just a
Pass-3 miss.
"""
import json
import os

import _util

_PS = _util.load(_util.script("parse_session.py"))

T0 = "2026-09-30T10:00:00Z"


def _ts(n):
    # Deterministic ascending timestamps: 10:00:00, 10:00:01, ...
    minute, sec = divmod(n, 60)
    return f"2026-09-30T10:{minute:02d}:{sec:02d}Z"


# ---------------------------------------------------------------- fixtures


def _launch_pair(tid, out, tuid="tu-launch"):
    """Assistant Bash tool_use + its 'running in background' tool_result."""
    return [
        {"type": "assistant", "timestamp": _ts(0), "message": {"role": "assistant", "content": [
            {"type": "tool_use", "id": tuid, "name": "Bash",
             "input": {"command": "echo bg-output", "run_in_background": True}}]}},
        {"type": "user", "timestamp": _ts(1), "message": {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": tuid,
             "content": (f"Command running in background with ID: {tid}\n"
                         f"Output is being written to: {out}. You will be "
                         f"notified when it finishes.")}]}},
    ]


def _enqueue(tid, out):
    return {"type": "queue-operation", "operation": "enqueue", "timestamp": _ts(2),
            "content": ("<task-notification>\n"
                        f"<task-id>{tid}</task-id>\n"
                        "<status>completed</status>\n"
                        f"<output-file>{out}</output-file>\n"
                        "<summary>task finished</summary>")}


def _dequeue(ts_n=5):
    return {"type": "queue-operation", "operation": "dequeue", "timestamp": _ts(ts_n),
            "content": "<task-notification>"}


def _task_delivery(tid, status="completed", text="success deadbeef",
                   is_error=False, include_result=True):
    td = {"taskId": tid, "status": status,
          "createdAt": T0, "lastUpdatedAt": T0,
          "statusMessage": "Exited with code 1." if is_error else "Exited with code 0."}
    if include_result:
        td["result"] = {"isError": is_error,
                        "content": [{"type": "text", "text": text}]}
    return td


def _gettask_call_pair(tid, status="completed", text="success deadbeef",
                       is_error=False, include_result=True, tuid="tu-gettask"):
    """Explicit model-issued GetTask tool_use + its result record.

    Shaped like the real records: the user record carries the structured
    completion in ``toolUseResult`` and its JSON text as the tool_result
    content.
    """
    tur = _task_delivery(tid, status=status, text=text, is_error=is_error,
                         include_result=include_result)
    return [
        {"type": "assistant", "timestamp": _ts(3), "message": {"role": "assistant", "content": [
            {"type": "tool_use", "id": tuid, "name": "GetTask",
             "input": {"taskId": tid}}]}},
        {"type": "user", "timestamp": _ts(4), "toolUseResult": tur,
         "message": {"role": "user", "content": [
             {"type": "tool_result", "tool_use_id": tuid,
              "content": json.dumps(tur)}]}},
    ]


def _taskdelivery_user_record(td, tid, out):
    return {"type": "user", "timestamp": _ts(3),
            "origin": {"kind": "task-notification", "producer": "session-task"},
            "turnOrigin": "task_notification",
            "taskDelivery": td,
            "message": {"role": "user", "content":
                        ("<task-notification>\n"
                         f"<task-id>{tid}</task-id>\n"
                         "<tool-use-id>tu-launch</tool-use-id>\n"
                         f"<output-file>{out}</output-file>\n"
                         "<status>completed</status>\n"
                         "<summary>task finished</summary>")}}


def _taskdelivery_attachment_record(td):
    return {"type": "attachment", "timestamp": _ts(3),
            "attachment": {"type": "queued_command", "taskDelivery": td}}


# ---------------------------------------------------------------- plumbing


def _write_jsonl(tmp, records):
    path = os.path.join(tmp, "session.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return path


def _report(tmp, records):
    path = _write_jsonl(tmp, records)
    events, meta = _PS.extract_events(path)
    _PS.annotate_parallel(events)
    _PS.link_pairs(events)
    return _PS.compute_bg_tasks(events, meta)


def _scenario(tmp, read_record_builder=None):
    """A full lifecycle: launch -> enqueue -> (optional) read, on real shapes.
    Returns the --bg-tasks report."""
    tid = "bt383aa1"
    out = os.path.join(tmp, f"{tid}.output")
    with open(out, "w", encoding="utf-8") as f:
        f.write("task stdout\n")
    records = _launch_pair(tid, out)
    records.append(_enqueue(tid, out))
    if read_record_builder is not None:
        records.extend(read_record_builder(tid, out))
    records.append(_dequeue())
    return _report(tmp, records)


# ---------------------------------------------------------------- tests


def test_explicit_gettask_completed_counts_as_read(tmp):
    report = _scenario(
        tmp, lambda tid, out: _gettask_call_pair(tid))
    assert "NOTIFIED BUT NEVER READ:" not in report, report
    assert "1 full" in report, report
    # The read is located at the GetTask call's line (L4: the assistant record).
    assert "read=L4 [" in report, report


def test_explicit_gettask_nonzero_exit_still_counts(tmp):
    # result.isError=true means the background command exited nonzero, not
    # that the delivery failed — the content was still delivered and read.
    report = _scenario(
        tmp, lambda tid, out: _gettask_call_pair(tid, is_error=True))
    assert "NOTIFIED BUT NEVER READ:" not in report, report
    assert "1 full" in report, report
    assert "read=L4 [" in report, report


def test_gettask_working_status_is_not_a_read(tmp):
    # status=working results carry no `result` at all (observed shape:
    # "Still running. ... Do not call GetTask again") — nothing read yet.
    report = _scenario(
        tmp, lambda tid, out: _gettask_call_pair(tid, status="working",
                                                 include_result=False))
    assert "1 notified-never-read" in report, report
    assert "NOTIFIED BUT NEVER READ:" in report, report


def test_gettask_empty_content_is_not_a_read(tmp):
    report = _scenario(
        tmp, lambda tid, out: _gettask_call_pair(tid, text=""))
    assert "1 notified-never-read" in report, report


def test_taskdelivery_user_record_counts_as_read(tmp):
    report = _scenario(
        tmp, lambda tid, out: [
            _taskdelivery_user_record(
                _task_delivery(tid), tid, out)])
    assert "NOTIFIED BUT NEVER READ:" not in report, report
    assert "1 full" in report, report
    assert "read=L4 [" in report, report


def test_taskdelivery_user_record_nonzero_exit_still_counts(tmp):
    report = _scenario(
        tmp, lambda tid, out: [
            _taskdelivery_user_record(
                _task_delivery(tid, is_error=True), tid, out)])
    assert "NOTIFIED BUT NEVER READ:" not in report, report
    assert "1 full" in report, report
    assert "read=L4 [" in report, report


def test_taskdelivery_user_record_empty_content_is_not_a_read(tmp):
    report = _scenario(
        tmp, lambda tid, out: [
            _taskdelivery_user_record(
                _task_delivery(tid, text=""), tid, out)])
    assert "1 notified-never-read" in report, report


def test_taskdelivery_attachment_counts_as_read(tmp):
    report = _scenario(
        tmp, lambda tid, out: [
            _taskdelivery_attachment_record(_task_delivery(tid))])
    assert "NOTIFIED BUT NEVER READ:" not in report, report
    assert "1 full" in report, report
    assert "read=L4 [" in report, report


def test_taskdelivery_attachment_nonzero_exit_still_counts(tmp):
    report = _scenario(
        tmp, lambda tid, out: [
            _taskdelivery_attachment_record(
                _task_delivery(tid, is_error=True))])
    assert "NOTIFIED BUT NEVER READ:" not in report, report
    assert "1 full" in report, report
    assert "read=L4 [" in report, report


def test_taskdelivery_attachment_empty_content_is_not_a_read(tmp):
    report = _scenario(
        tmp, lambda tid, out: [
            _taskdelivery_attachment_record(
                _task_delivery(tid, text=""))])
    assert "1 notified-never-read" in report, report


def test_unread_task_still_flagged(tmp):
    # Control: none of the new paths, no read at all -> still a violation.
    report = _scenario(tmp)
    assert "1 notified-never-read" in report, report
    assert "NOTIFIED BUT NEVER READ:" in report, report


def test_delivery_cited_over_later_redundant_file_read(tmp):
    # A taskDelivery followed by a plain file Read of the .output file (the
    # redundant reread a false "never read" block used to force): the
    # delivery is the FIRST consumption, so it keeps the read citation.
    tid = "bt383aa5"
    out = os.path.join(tmp, f"{tid}.output")
    with open(out, "w", encoding="utf-8") as f:
        f.write("task stdout\n")
    records = _launch_pair(tid, out)
    records.append(_enqueue(tid, out))
    records.append(_taskdelivery_user_record(_task_delivery(tid), tid, out))
    records.append(
        {"type": "assistant", "timestamp": _ts(6), "message": {"role": "assistant", "content": [
            {"type": "tool_use", "id": "tu-read", "name": "Read",
             "input": {"file_path": out}}]}},
    )
    records.append(
        {"type": "user", "timestamp": _ts(7), "message": {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "tu-read",
             "content": "task stdout"}]}},
    )
    report = _report(tmp, records)
    assert "1 full" in report, report
    assert "read=L4 [" in report, report
    assert "read=L6" not in report, report


def test_user_record_taskdelivery_survives_ingest(tmp):
    # The ingest layer must carry taskDelivery from the user record onto the
    # emitted user_message event — otherwise Pass 3.5 cannot see it even
    # once it exists.
    tid = "bt383aa2"
    out = os.path.join(tmp, f"{tid}.output")
    with open(out, "w", encoding="utf-8") as f:
        f.write("task stdout\n")
    records = _launch_pair(tid, out)
    records.append(_taskdelivery_user_record(_task_delivery(tid), tid, out))
    path = _write_jsonl(tmp, records)
    events, _meta = _PS.extract_events(path)
    carrying = [e for e in events
                if e.get("type") == "user_message" and e.get("taskDelivery")]
    assert carrying, "no user_message event carries taskDelivery"
    assert carrying[0]["taskDelivery"]["taskId"] == tid
    assert carrying[0]["taskDelivery"]["status"] == "completed"


def test_gettask_branch_ignores_nested_is_error_only(tmp):
    # A hook-BLOCKED GetTask call (is_error on the tool_result itself) must
    # not count even when its detail text embeds a completed payload; but a
    # genuine result whose nested result.isError is true must.
    tid = "bt383aa3"
    out = os.path.join(tmp, f"{tid}.output")
    with open(out, "w", encoding="utf-8") as f:
        f.write("task stdout\n")
    blocked = _gettask_call_pair(tid, is_error=True)
    blocked[1]["message"]["content"][0]["is_error"] = True
    records = _launch_pair(tid, out) + [_enqueue(tid, out)] + blocked + \
        [_dequeue()]
    report = _report(tmp, records)
    assert "1 notified-never-read" in report, report
    # And the genuine nonzero-exit result (block-level is_error absent):
    tid2 = "bt383aa4"
    out2 = os.path.join(tmp, f"{tid2}.output")
    with open(out2, "w", encoding="utf-8") as f:
        f.write("task stdout\n")
    records2 = _launch_pair(tid2, out2) + [_enqueue(tid2, out2)] + \
        _gettask_call_pair(tid2, is_error=True) + [_dequeue()]
    report2 = _report(tmp, records2)
    assert "1 full" in report2, report2


def main():
    return _util.runner(_util.collect(globals()), tmp_prefix="bgdelivery_")


if __name__ == "__main__":
    raise SystemExit(main())
