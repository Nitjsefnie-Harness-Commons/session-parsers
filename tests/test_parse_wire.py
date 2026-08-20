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
import _util

_PW = _util.load(_util.script("parse_wire.py"))


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


def main():
    return _util.runner(_util.collect(globals()), tmp_prefix="parsewire_")


if __name__ == "__main__":
    raise SystemExit(main())
