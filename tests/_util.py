"""Shared helpers for the parser suites.

Not a suite itself -- run_tests.py only loads `test_*.py`.

Stdlib only and OS-agnostic: these suites run on Linux, macOS and Windows in
CI, and a POSIX-only assumption here would be a platform failure reported as a
test failure.
"""
import glob
import importlib.util
import os
import tempfile
import traceback
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = str(_PROJECT_ROOT / "session_parsers")
"""The package directory. The suites load a parser by path rather than by
import so that a broken `__init__` cannot mask a parser defect as an
ImportError."""

CLAUDE = os.path.expanduser("~/.claude")


def script(name):
    return str(_PROJECT_ROOT / "session_parsers" / name)


def load(path, name=None):
    """Import a parser by path without running its __main__ block."""
    name = name or ("mod_" + os.path.splitext(os.path.basename(path))[0]
                    .replace("-", "_"))
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def transcripts(limit=None, min_size=0):
    """Real Claude session transcripts on this box, smallest first.

    Empty on a machine that has none -- CI runners included -- so a suite that
    wants one must skip rather than fail when the list comes back empty.
    """
    found = [p for p in glob.glob(os.path.join(CLAUDE, "projects", "**", "*.jsonl"),
                                  recursive=True)
             if os.path.getsize(p) >= min_size]
    found.sort(key=os.path.getsize)
    return found[-limit:] if limit else found


class Skipped(Exception):
    """Raised by a test that cannot hold on this platform."""


def skip(reason):
    """End the running test as skipped, with a reason the log will show."""
    raise Skipped(reason)


def _assertion_site(exc):
    """`file:line: source` of the assert that failed, for bare asserts."""
    frames = traceback.extract_tb(exc.__traceback__)
    if not frames:
        return ""
    last = frames[-1]
    return f"{os.path.basename(last.filename)}:{last.lineno}: {last.line}"


def runner(tests, tmp_prefix="parsertests_"):
    """Shared main(): run every callable, print PASS/FAIL, return exit code.

    Each test takes one argument: an isolated temp dir, handed over fully
    resolved -- macOS resolves /var to /private/var, and a suite comparing a
    path it was given against a path the code produced would otherwise see two
    spellings of the same directory and call them different.
    """
    failed = []
    skipped = []
    with tempfile.TemporaryDirectory(prefix=tmp_prefix) as tmp:
        tmp = os.path.realpath(tmp)
        for t in tests:
            d = os.path.join(tmp, t.__name__)
            os.makedirs(d, exist_ok=True)
            try:
                t(d)
                print(f"  PASS  {t.__name__}")
            except Skipped as e:
                skipped.append(t.__name__)
                print(f"  SKIP  {t.__name__}: {e}")
            except AssertionError as e:
                failed.append(t.__name__)
                # A bare `assert x == y` carries no message, which on a
                # platform you cannot run locally leaves nothing to go on.
                detail = str(e) or _assertion_site(e)
                print(f"  FAIL  {t.__name__}: {detail}")
            except Exception as e:  # noqa: BLE001
                failed.append(t.__name__)
                print(f"  ERROR {t.__name__}: {type(e).__name__}: {e}")
    passed = len(tests) - len(failed) - len(skipped)
    summary = f"\n{passed}/{len(tests)} passed"
    if skipped:
        summary += f", {len(skipped)} skipped"
    print(summary)
    return 1 if failed else 0


def collect(namespace):
    return [v for k, v in sorted(namespace.items())
            if k.startswith("test_") and callable(v)]
