"""Suite for parse_session.py --cache cost pricing (issue #4).

Pins the Claude 5 family rate keys (Opus 5, Sonnet 5), the Sonnet 5
standard $2/$10 rate regardless of request timestamp, and the visible NOTE
when a model falls back to DEFAULT_RATES.
"""
import _util

_PS = _util.load(_util.script("parse_session.py"))


def _usage(model, ts="2026-07-30T00:00:00Z", **usage):
    return {"type": "assistant_usage", "model": model, "requestId": "r1",
            "line": 1, "ts": ts, "usage": usage}


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


def test_opus5_priced_at_opus_rates(tmp):
    report = _PS.compute_cache(
        [_usage("claude-opus-5-20260101", output_tokens=1_000_000)])
    assert _total(report) == 25.00


def test_sonnet5_standard_rate_does_not_change_in_sep2026(tmp):
    all_categories = {
        "input_tokens": 1_000_000,
        "cache_creation_input_tokens": 2_000_000,
        "cache_read_input_tokens": 1_000_000,
        "output_tokens": 1_000_000,
        "cache_creation": {
            "ephemeral_5m_input_tokens": 1_000_000,
            "ephemeral_1h_input_tokens": 1_000_000,
        },
    }
    before = _PS.compute_cache(
        [_usage("claude-sonnet-5-20260701", ts="2026-07-15T00:00:00Z",
                **all_categories)])
    after = _PS.compute_cache(
        [_usage("claude-sonnet-5-20260701", ts="2026-10-01T00:00:00Z",
                **all_categories)])
    assert _total(before) == 18.70
    assert _total(after) == 18.70
    assert {
        label: _category_cost(after, label)
        for label in ("Fresh input", "Cache create 5m", "Cache create 1h",
                      "Cache read", "Output")
    } == {
        "Fresh input": 2.00,
        "Cache create 5m": 2.50,
        "Cache create 1h": 4.00,
        "Cache read": 0.20,
        "Output": 10.00,
    }
    assert "Sonnet 5 $2/$2.50/$4/$0.20/$10" in after
    assert "intro through" not in after
    assert "then $3/$3.75/$6/$0.30/$15" not in after


def test_sonnet4x_keeps_its_own_rate(tmp):
    report = _PS.compute_cache(
        [_usage("claude-sonnet-4-5", ts="2026-07-15T00:00:00Z",
                output_tokens=1_000_000)])
    assert _total(report) == 15.00


def test_unknown_model_falls_back_with_visible_note(tmp):
    report = _PS.compute_cache([_usage("claude-xyz-9", output_tokens=1_000_000)])
    assert _total(report) == 25.00
    assert "claude-xyz-9" in report and "fallback" in report


def main():
    return _util.runner(_util.collect(globals()), tmp_prefix="costrates_")


if __name__ == "__main__":
    raise SystemExit(main())
