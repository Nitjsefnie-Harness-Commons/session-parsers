"""Suite for parse_codex.py — the Codex rollout parser.

Everything here exists because of one class of bug: a plausible-looking token
sum that is wrong. Codex's `total_token_usage` is a cumulative counter that a
forked thread inherits from its parent, duplicate `token_count` events repeat
the previous reading verbatim, `cached_input_tokens` is a SUBSET of
`input_tokens`, and `token_count` carries no model of its own. Each of those
produces a number that looks fine and is off by anywhere from 20% to 40x, so
each gets a regression test with hand-computed expected values.

The fixtures are synthetic rollout records in the real schema rather than real
transcripts: a real corpus would drift, and these need exact arithmetic.
"""
import json
import os
import tempfile

import _util

_PC = _util.load(os.path.join(_util.SCRIPTS, "parse_codex.py"))


def _usage(inp, cached, out, write=0, reasoning=0):
    return {
        "input_tokens": inp,
        "cached_input_tokens": cached,
        "cache_write_input_tokens": write,
        "output_tokens": out,
        "reasoning_output_tokens": reasoning,
        "total_tokens": inp + out,
    }


def _token_line(ts, total, last):
    return {"timestamp": ts, "type": "event_msg",
            "payload": {"type": "token_count",
                        "info": {"total_token_usage": total,
                                 "last_token_usage": last,
                                 "model_context_window": 258400},
                        "rate_limits": {"plan_type": "pro",
                                        "primary": {"used_percent": 4.0,
                                                    "window_minutes": 10080}}}}


def _model_line(ts, model):
    return {"timestamp": ts, "type": "turn_context",
            "payload": {"turn_id": "t1", "model": model, "effort": "xhigh"}}


def _meta_line(ts, session_id, forked_from=None):
    payload = {"session_id": session_id, "cwd": "/root", "cli_version": "0.146.0"}
    if forked_from:
        payload["forked_from_id"] = forked_from
    return {"timestamp": ts, "type": "session_meta", "payload": payload}


def _write(tmpdir, name, records):
    path = os.path.join(tmpdir, name)
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")
    return path


def _events(path):
    evs = _PC.extract_events([path])
    _PC.backfill_models(evs)
    _PC.link_pairs(evs)
    return evs


# ---------------------------------------------------------------- pricing
def test_cached_tokens_are_subtracted_not_added():
    """cached_input_tokens is a subset of input_tokens.

    Treating it as an addend is the single largest error available here: at the
    97% cache rates these sessions run, it inflates the fresh-input bill ~40x.
    """
    # 100k prompt, 90k of it a cache hit, 1k output, on Sol.
    cost, fresh, long_ctx = _PC.price_request(_usage(100_000, 90_000, 1_000),
                                              _PC.SOL_RATES)
    assert fresh == 10_000, fresh
    # 10k fresh @ $5/M + 90k cached @ $0.50/M + 1k output @ $30/M
    expected = (10_000 * 5.00 + 90_000 * 0.50 + 1_000 * 30.00) / 1e6
    assert abs(cost - expected) < 1e-12, (cost, expected)
    assert long_ctx is False
    return 0


def test_cache_write_is_its_own_subset_and_rate():
    cost, fresh, _ = _PC.price_request(_usage(100_000, 60_000, 0, write=20_000),
                                       _PC.SOL_RATES)
    assert fresh == 20_000, fresh
    expected = (20_000 * 5.00 + 60_000 * 0.50 + 20_000 * 6.25) / 1e6
    assert abs(cost - expected) < 1e-12, (cost, expected)
    return 0


def test_long_context_multiplier_applies_to_whole_request():
    below = _usage(_PC.LONG_CONTEXT_THRESHOLD, 0, 1_000)
    above = _usage(_PC.LONG_CONTEXT_THRESHOLD + 1, 0, 1_000)
    cost_below, _, flag_below = _PC.price_request(below, _PC.SOL_RATES)
    cost_above, _, flag_above = _PC.price_request(above, _PC.SOL_RATES)
    assert flag_below is False and flag_above is True
    # Input doubles and output takes 1.5x — for the ENTIRE request, not the
    # overflow portion. One extra input token more than doubles the bill.
    assert cost_above > cost_below * 1.9, (cost_below, cost_above)
    return 0


def test_model_variants_map_to_distinct_rates():
    assert _PC.rates_for_model("gpt-5.6-sol") is _PC.SOL_RATES
    assert _PC.rates_for_model("gpt-5.6-terra") is _PC.TERRA_RATES
    assert _PC.rates_for_model("gpt-5.6-luna") is _PC.LUNA_RATES
    # An unknown variant must not silently price as the cheapest tier.
    assert _PC.rates_for_model("gpt-5.7-unknown") is _PC.FALLBACK_RATES
    assert _PC.rates_for_model(None) is _PC.FALLBACK_RATES
    return 0


def test_rates_override_parses_and_rejects_junk():
    got = _PC.parse_rates("input=1,cached=0.1,output=5")
    assert got == {"input": 1.0, "cached": 0.1, "output": 5.0}, got
    for bad in ("nonsense", "banana=3", "", "input=-1", "input=nan",
                "output=inf"):
        try:
            _PC.parse_rates(bad)
        except ValueError:
            continue
        return "parse_rates accepted %r" % bad
    return 0


def test_verbose_is_a_real_alias_for_full():
    args = _PC.build_parser().parse_args(["rollout.jsonl", "--verbose"])
    assert args.full is True
    return 0


# ------------------------------------------------- cumulative counter math
def test_duplicate_token_counts_are_not_double_counted():
    """A repeated token_count must contribute nothing.

    Real rollouts repeat the previous reading verbatim (47 consecutive times in
    one observed file). Summing last_token_usage counts those requests twice.
    """
    with tempfile.TemporaryDirectory() as tmp:
        first = _usage(1000, 0, 10)
        second = _usage(3000, 500, 30)
        path = _write(tmp, "rollout-dupe.jsonl", [
            _meta_line("2026-08-05T10:00:00.000Z", "sess-a"),
            _model_line("2026-08-05T10:00:01.000Z", "gpt-5.6-sol"),
            _token_line("2026-08-05T10:00:02.000Z", first, first),
            _token_line("2026-08-05T10:00:03.000Z", first, first),   # duplicate
            _token_line("2026-08-05T10:00:04.000Z", first, first),   # duplicate
            _token_line("2026-08-05T10:00:05.000Z", second,
                        _usage(2000, 500, 20)),
        ])
        reqs = _PC.request_usage(_events(path))
    assert len(reqs) == 2, [r["usage"] for r in reqs]
    assert sum(r["usage"]["input_tokens"] for r in reqs) == 3000
    assert sum(r["usage"]["output_tokens"] for r in reqs) == 30
    return 0


def test_forked_baseline_is_excluded_from_this_sessions_cost():
    """A fork inherits the parent's cumulative total; it is not our spend.

    Charging the inherited baseline is what makes a naive per-file sum overstate
    a real corpus by ~20%.
    """
    with tempfile.TemporaryDirectory() as tmp:
        # Thread opens already at 5,000,000 input tokens carried from a parent,
        # and its own first request added 2,000 of that.
        first_total = _usage(5_002_000, 4_000_000, 700)
        first_last = _usage(2_000, 1_500, 100)
        second_total = _usage(5_005_000, 4_002_000, 750)
        second_last = _usage(3_000, 2_000, 50)
        path = _write(tmp, "rollout-fork.jsonl", [
            _meta_line("2026-08-05T10:00:00.000Z", "sess-b", forked_from="sess-a"),
            _model_line("2026-08-05T10:00:01.000Z", "gpt-5.6-sol"),
            _token_line("2026-08-05T10:00:02.000Z", first_total, first_last),
            _token_line("2026-08-05T10:00:03.000Z", second_total, second_last),
            # A repeat of the same reading, as real rollouts emit. Without it
            # this fixture would pass under a naive last_token_usage sum too,
            # and pin only half the trap.
            _token_line("2026-08-05T10:00:04.000Z", second_total, second_last),
        ])
        evs = _events(path)
        reqs = _PC.request_usage(evs)
    assert len(reqs) == 2, len(reqs)
    own_input = sum(r["usage"]["input_tokens"] for r in reqs)
    # 2,000 for the first request + 3,000 for the second. NOT 5,005,000 (the
    # cumulative counter) and NOT 8,000 (last_token_usage summed with the dupe).
    assert own_input == 5_000, own_input
    own_output = sum(r["usage"]["output_tokens"] for r in reqs)
    assert own_output == 150, own_output
    # And the inherited baseline really is in the file, unspent by us.
    assert reqs[-1]["cumulative"]["input_tokens"] == 5_005_000
    return 0


def test_delta_sum_telescopes_to_the_cumulative_total():
    """Non-forked session: summed deltas must equal the final counter exactly."""
    with tempfile.TemporaryDirectory() as tmp:
        totals = [_usage(1000, 0, 10), _usage(4000, 800, 25),
                  _usage(9000, 3000, 55)]
        records = [_meta_line("2026-08-05T10:00:00.000Z", "sess-c"),
                   _model_line("2026-08-05T10:00:01.000Z", "gpt-5.6-terra")]
        prev = _usage(0, 0, 0)
        for i, total in enumerate(totals):
            last = {k: total[k] - prev[k] for k in total}
            records.append(_token_line("2026-08-05T10:00:%02d.000Z" % (2 + i),
                                       total, last))
            prev = total
        path = _write(tmp, "rollout-tele.jsonl", records)
        reqs = _PC.request_usage(_events(path))
    assert sum(r["usage"]["input_tokens"] for r in reqs) == 9000
    assert sum(r["usage"]["output_tokens"] for r in reqs) == 55
    return 0


# -------------------------------------------------------- model attribution
def test_model_comes_from_the_preceding_turn_context():
    """token_count carries no model; the most recent turn_context supplies it."""
    with tempfile.TemporaryDirectory() as tmp:
        path = _write(tmp, "rollout-mixed.jsonl", [
            _meta_line("2026-08-05T10:00:00.000Z", "sess-d"),
            _model_line("2026-08-05T10:00:01.000Z", "gpt-5.6-sol"),
            _token_line("2026-08-05T10:00:02.000Z", _usage(1000, 0, 10),
                        _usage(1000, 0, 10)),
            _model_line("2026-08-05T10:00:03.000Z", "gpt-5.6-terra"),
            _token_line("2026-08-05T10:00:04.000Z", _usage(3000, 0, 30),
                        _usage(2000, 0, 20)),
        ])
        reqs = _PC.request_usage(_events(path))
    assert [r["model"] for r in reqs] == ["gpt-5.6-sol", "gpt-5.6-terra"], \
        [r["model"] for r in reqs]
    return 0


def test_backfill_only_when_the_file_declares_one_model():
    """A fork replays parent history before declaring its own turn_context.

    One declared model -> the leading requests can only be that model. Two ->
    guessing which side of the switch they fall on would be fabrication, so
    they stay unattributed.
    """
    with tempfile.TemporaryDirectory() as tmp:
        single = _write(tmp, "rollout-single.jsonl", [
            _meta_line("2026-08-05T10:00:00.000Z", "sess-e", forked_from="sess-a"),
            _token_line("2026-08-05T10:00:01.000Z", _usage(1000, 0, 10),
                        _usage(1000, 0, 10)),
            _model_line("2026-08-05T10:00:02.000Z", "gpt-5.6-sol"),
        ])
        mixed = _write(tmp, "rollout-two.jsonl", [
            _meta_line("2026-08-05T10:00:00.000Z", "sess-f", forked_from="sess-a"),
            _token_line("2026-08-05T10:00:01.000Z", _usage(1000, 0, 10),
                        _usage(1000, 0, 10)),
            _model_line("2026-08-05T10:00:02.000Z", "gpt-5.6-sol"),
            _model_line("2026-08-05T10:00:03.000Z", "gpt-5.6-terra"),
        ])
        single_reqs = _PC.request_usage(_events(single))
        mixed_reqs = _PC.request_usage(_events(mixed))
    assert single_reqs[0]["model"] == "gpt-5.6-sol", single_reqs[0]["model"]
    assert mixed_reqs[0]["model"] is None, mixed_reqs[0]["model"]
    return 0


# ------------------------------------------------------------- extraction
def test_exec_command_string_survives_js_quoting():
    """The command lives inside a JS literal with mixed quoting and escapes."""
    src = ('const r = await tools.exec_command({cmd:"grep -n \\"x\\" a.py '
           '&& echo \\\'done\\\'","workdir":"/root","yield_time_ms":10000}); '
           'text(r.output)')
    assert _PC.js_string_value(src, "cmd") == \
        'grep -n "x" a.py && echo \'done\'', _PC.js_string_value(src, "cmd")
    assert _PC.js_string_value(src, "workdir") == "/root"
    assert _PC.js_string_value(src, "missing") is None
    assert _PC.exec_apis(src) == ["exec_command"]
    return 0


def test_js_string_preserves_windows_paths_and_unknown_escapes():
    src = r'tools.exec_command({cmd:"rg \\d+ C:\\Users\\me\\src", workdir:"C:\\work"})'
    assert _PC.js_string_value(src, "cmd") == r"rg \d+ C:\Users\me\src"
    assert _PC.js_string_value(src, "workdir") == r"C:\work"
    assert _PC.js_string_value(r'{cmd:"\u0041\x42\q"}', "cmd") == "AB\\q"
    return 0


def test_exec_apis_lists_every_tool_in_first_seen_order():
    src = "tools.apply_patch({}); tools.exec_command({}); tools.apply_patch({})"
    assert _PC.exec_apis(src) == ["apply_patch", "exec_command"]
    return 0


def test_failed_script_output_is_classified_as_an_error():
    ok = _PC.classify("response_item",
                      {"type": "custom_tool_call_output", "call_id": "c1",
                       "output": [{"type": "input_text", "text": "Script completed\nfine"}]})
    bad = _PC.classify("response_item",
                       {"type": "custom_tool_call_output", "call_id": "c2",
                        "output": [{"type": "input_text", "text": "Script failed\nboom"}]})
    bg = _PC.classify("response_item",
                      {"type": "custom_tool_call_output", "call_id": "c3",
                       "output": [{"type": "input_text",
                                   "text": "Script running with cell ID 15"}]})
    assert ok[1]["ok"] is True and bad[1]["ok"] is False
    assert bg[1]["bg"] is True
    return 0


def test_tool_calls_pair_with_their_outputs_by_call_id():
    with tempfile.TemporaryDirectory() as tmp:
        path = _write(tmp, "rollout-pairs.jsonl", [
            {"timestamp": "2026-08-05T10:00:00.000Z", "type": "response_item",
             "payload": {"type": "custom_tool_call", "name": "exec",
                         "call_id": "call_1",
                         "input": 'tools.exec_command({cmd:"ls"})'}},
            {"timestamp": "2026-08-05T10:00:01.000Z", "type": "response_item",
             "payload": {"type": "custom_tool_call_output", "call_id": "call_1",
                         "output": [{"type": "input_text",
                                     "text": "Script completed"}]}},
            {"timestamp": "2026-08-05T10:00:02.000Z", "type": "response_item",
             "payload": {"type": "custom_tool_call", "name": "exec",
                         "call_id": "call_2",
                         "input": 'tools.exec_command({cmd:"pwd"})'}},
        ])
        evs = _events(path)
    calls = [e for e in evs if e["kind"] == "tool_call"]
    assert len(calls) == 2, len(calls)
    assert calls[0].get("result") is not None
    assert calls[1].get("result") is None  # orphan: --tool-orphans finds this
    assert calls[0]["cmd"] == "ls"
    return 0


def test_same_basename_files_do_not_merge_usage_or_tool_pairs():
    with tempfile.TemporaryDirectory() as tmp:
        paths = []
        for index, model in enumerate(("gpt-5.6-sol", "gpt-5.6-terra")):
            directory = os.path.join(tmp, str(index))
            os.makedirs(directory)
            path = _write(directory, "rollout-same.jsonl", [
                _meta_line("2026-08-05T1%d:00:00.000Z" % index, "s%d" % index),
                _model_line("2026-08-05T1%d:00:01.000Z" % index, model),
                _token_line("2026-08-05T1%d:00:02.000Z" % index,
                            _usage(1000 + index, 0, 10),
                            _usage(1000 + index, 0, 10)),
                {"timestamp": "2026-08-05T1%d:00:03.000Z" % index,
                 "type": "response_item",
                 "payload": {"type": "custom_tool_call", "name": "exec",
                             "call_id": "reused",
                             "input": 'tools.exec_command({cmd:"cmd%d"})' % index}},
                {"timestamp": "2026-08-05T1%d:00:04.000Z" % index,
                 "type": "response_item",
                 "payload": {"type": "custom_tool_call_output",
                             "call_id": "reused", "output": "result%d" % index}},
            ])
            paths.append(path)
        evs = _PC.extract_events(paths)
        _PC.backfill_models(evs)
        _PC.link_pairs(evs)
        reqs = _PC.request_usage(evs)
    assert len(reqs) == 2, reqs
    assert {r["model"] for r in reqs} == {"gpt-5.6-sol", "gpt-5.6-terra"}
    calls = [e for e in evs if e["kind"] == "tool_call"]
    assert [c["result"]["text"] for c in calls] == ["result0", "result1"]
    return 0


def test_session_id_glob_characters_are_literal():
    with tempfile.TemporaryDirectory() as tmp:
        literal = _write(tmp, "rollout-id[1].jsonl", [_meta_line(
            "2026-08-05T10:00:00.000Z", "literal")])
        _write(tmp, "rollout-id1.jsonl", [_meta_line(
            "2026-08-05T10:00:00.000Z", "wildcard")])
        old_root = _PC.DEFAULT_ROOT
        _PC.DEFAULT_ROOT = tmp
        try:
            assert _PC.resolve_target("id[1]") == [literal]
        finally:
            _PC.DEFAULT_ROOT = old_root
    return 0


def test_malformed_lines_are_skipped_not_fatal():
    """Rollouts are appended to live; a torn final line must not crash a read."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "rollout-torn.jsonl")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(_meta_line("2026-08-05T10:00:00.000Z", "s")) + "\n")
            fh.write("\n")
            fh.write("{not json at all\n")
            fh.write('{"timestamp": "2026-08-05T10:00:01.000Z", "type": "event_msg",'
                     ' "payload": {"type": "user_message", "message": "hi"}}\n')
            fh.write('{"timestamp": "2026-08-05T10:00:02.0')  # torn write
        evs = _events(path)
    kinds = [e["kind"] for e in evs]
    assert "meta" in kinds and "user" in kinds, kinds
    return 0


def test_timestamps_normalize_to_utc():
    got = _PC.parse_ts("2026-08-05T10:00:00.000Z")
    assert got is not None and got.tzinfo is not None
    assert got.hour == 10 and got.utcoffset().total_seconds() == 0
    assert _PC.parse_ts(None) is None
    assert _PC.parse_ts("not a time") is None
    return 0


def test_forks_mode_reports_shared_ids_and_inherited_baselines():
    with tempfile.TemporaryDirectory() as tmp:
        base = _usage(4_000_000, 3_000_000, 500)
        _write(tmp, "rollout-parent.jsonl", [
            _meta_line("2026-08-05T10:00:00.000Z", "shared-id"),
            _model_line("2026-08-05T10:00:01.000Z", "gpt-5.6-sol"),
            _token_line("2026-08-05T10:00:02.000Z", _usage(1000, 0, 10),
                        _usage(1000, 0, 10)),
        ])
        _write(tmp, "rollout-child.jsonl", [
            # A fork keeps the PARENT's session_id — this is why "per session"
            # is ambiguous and why counting files overcounts sessions.
            _meta_line("2026-08-05T11:00:00.000Z", "shared-id",
                       forked_from="shared-id"),
            _model_line("2026-08-05T11:00:01.000Z", "gpt-5.6-sol"),
            _token_line("2026-08-05T11:00:02.000Z", base, _usage(500, 200, 5)),
        ])
        evs = _PC.extract_events(_PC.resolve_target(tmp))
        _PC.backfill_models(evs)
        lines = []
        _PC.mode_forks(evs, None, lines.append)
    text = "\n".join(lines)
    assert "shared-id" in text
    assert "in 2 files" in text, text
    assert "inherited from parent" in text, text
    # The reported baseline is on total_tokens (input + output): the child's
    # first cumulative snapshot is 4,000,500, of which 505 was its own first
    # request, leaving 3,999,995 carried in from the parent.
    assert "3,999,995" in text, text
    return 0


def main():
    ran = failed = 0
    for fn in (test_cached_tokens_are_subtracted_not_added,
               test_cache_write_is_its_own_subset_and_rate,
               test_long_context_multiplier_applies_to_whole_request,
               test_model_variants_map_to_distinct_rates,
               test_rates_override_parses_and_rejects_junk,
               test_verbose_is_a_real_alias_for_full,
               test_duplicate_token_counts_are_not_double_counted,
               test_forked_baseline_is_excluded_from_this_sessions_cost,
               test_delta_sum_telescopes_to_the_cumulative_total,
               test_model_comes_from_the_preceding_turn_context,
               test_backfill_only_when_the_file_declares_one_model,
               test_exec_command_string_survives_js_quoting,
               test_js_string_preserves_windows_paths_and_unknown_escapes,
               test_exec_apis_lists_every_tool_in_first_seen_order,
               test_failed_script_output_is_classified_as_an_error,
               test_tool_calls_pair_with_their_outputs_by_call_id,
               test_same_basename_files_do_not_merge_usage_or_tool_pairs,
               test_session_id_glob_characters_are_literal,
               test_malformed_lines_are_skipped_not_fatal,
               test_timestamps_normalize_to_utc,
               test_forks_mode_reports_shared_ids_and_inherited_baselines):
        rc = fn()
        if rc:
            print("  FAIL  %s: %s" % (fn.__name__, rc))
            failed += 1
        else:
            # Say so on success too: a silent suite and a suite that collected
            # nothing look identical in a CI log, and only one of them is fine.
            print("  PASS  %s" % fn.__name__)
        ran += 1
    print("\n%d/%d passed" % (ran - failed, ran))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
