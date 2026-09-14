"""Suite for the wheel's own contract: the three parsers stay reachable.

Nothing else here checks packaging, and packaging is exactly what the move into
a wheel put at risk. A renamed module, a typo in a `[project.scripts]` target,
or a `main` that stopped existing all produce a distribution that installs
cleanly and cannot run -- and the parser suites, which load their target by
path, would not notice any of it.
"""
import os
import re
import subprocess
import sys
import tokenize
import tomllib
from pathlib import Path

import _util

ROOT = Path(_util.SCRIPTS).parent
MODULES = ("parse_session", "parse_wire", "parse_codex", "parse_pi")


def _run(args):
    """Run the package as a subprocess, from a cwd that is not the checkout.

    Run from the repo root, `session_parsers/` is importable because the cwd is
    on sys.path -- which is exactly the accident that would hide a packaging
    error. PYTHONPATH names the root explicitly instead.
    """
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run([sys.executable, "-m", *args], cwd=tempdir_of(),
                          capture_output=True, text=True, env=env, check=False,
                          timeout=120)


_TMP = []


def tempdir_of():
    return _TMP[0]


def test_every_module_is_importable_by_name(tmp):
    _TMP[:] = [tmp]
    for name in MODULES:
        r = _run([f"session_parsers.{name}", "--version"])
        assert r.returncode == 0, f"{name} --version exited {r.returncode}: {r.stderr}"


def test_every_module_reports_its_own_version(tmp):
    """Three tools with independent histories: one version cannot speak for all."""
    _TMP[:] = [tmp]
    seen = {}
    for name in MODULES:
        r = _run([f"session_parsers.{name}", "--version"])
        out = (r.stdout + r.stderr).strip()
        assert name in out, f"{name} --version does not name itself: {out!r}"
        found = re.search(r"(\d+\.\d+\.\d+)", out)
        assert found, f"{name} --version prints no version: {out!r}"
        seen[name] = found.group(1)
    assert len(set(seen.values())) > 1, (
        f"every parser reports the same version {seen}; the per-module "
        "versions have been collapsed into the distribution's")


def test_every_module_has_a_main_the_entry_point_can_call(tmp):
    for name in MODULES:
        mod = _util.load(_util.script(f"{name}.py"), f"contract_{name}")
        main = getattr(mod, "main", None)
        assert callable(main), f"{name} has no callable main()"
        # setuptools console scripts call it with no arguments.
        params = main.__code__.co_argcount - len(main.__defaults__ or ())
        assert params == 0, f"{name}.main requires {params} positional argument(s)"


def test_declared_console_scripts_resolve(tmp):
    """A typo here ships a wheel whose commands do not exist."""
    with open(ROOT / "pyproject.toml", "rb") as fh:
        cfg = tomllib.load(fh)
    scripts = cfg["project"]["scripts"]
    assert scripts, "the wheel declares no commands"
    for command, target in scripts.items():
        module, _, func = target.partition(":")
        path = ROOT / (module.replace(".", os.sep) + ".py")
        assert path.is_file(), f"{command} points at missing module {module}"
        mod = _util.load(str(path), f"entry_{command.replace('-', '_')}")
        assert callable(getattr(mod, func, None)), (
            f"{command} points at {target}, which is not callable")


def test_the_distribution_version_is_the_one_setuptools_reads(tmp):
    with open(ROOT / "pyproject.toml", "rb") as fh:
        cfg = tomllib.load(fh)
    attr = cfg["tool"]["setuptools"]["dynamic"]["version"]["attr"]
    assert attr == "session_parsers.__version__", attr
    init = (ROOT / "session_parsers" / "__init__.py").read_text(encoding="utf-8")
    found = re.search(r'^__version__\s*=\s*"(\d+\.\d+\.\d+)"', init, re.MULTILINE)
    assert found, "no SemVer __version__ in session_parsers/__init__.py"


def _requires_python_floor():
    """`requires-python` as a (major, minor) tuple, from pyproject itself."""
    with open(ROOT / "pyproject.toml", "rb") as fh:
        cfg = tomllib.load(fh)
    spec = cfg["project"]["requires-python"]
    found = re.search(r">=\s*(\d+)\.(\d+)", spec)
    assert found, f"cannot read a floor out of requires-python = {spec!r}"
    return (int(found.group(1)), int(found.group(2)))


def _pep701_uses(path):
    """PEP 701 f-string constructs in `path`, as (line, what) pairs.

    Two of them arrived in 3.12 and are a hard SyntaxError before it: a
    newline inside the replacement field of a singly-quoted f-string, and
    reusing the f-string's own quote character inside it. `ast.parse`'s
    `feature_version` gates NEITHER -- both are decided in the tokenizer,
    which is why this reads tokens rather than parsing.
    """
    found = []
    if not hasattr(tokenize, "FSTRING_START"):
        return found          # 3.11: the file would not have parsed at all
    with open(path, "rb") as fh:
        toks = list(tokenize.tokenize(fh.readline))
    depth = 0
    opened_at = quote = None
    for tok in toks:
        if tok.type == tokenize.FSTRING_START:
            if depth == 0:
                opened_at, quote = tok.start[0], tok.string.lstrip("fFrRbB")
            depth += 1
            continue
        if tok.type == tokenize.FSTRING_END:
            depth -= 1
            if depth == 0:
                opened_at = quote = None
            continue
        # A triple-quoted delimiter is 3 characters; a singly-quoted one is
        # 1, and only the latter is forbidden from spanning lines pre-3.12.
        if depth and quote and len(quote) < 3:
            if tok.start[0] != opened_at and tok.type != tokenize.NL:
                found.append((opened_at, "newline inside the replacement field"))
            if tok.type == tokenize.STRING and tok.string.startswith(quote[0]):
                found.append((opened_at, "the f-string quote reused inside it"))
    return sorted(set(found))


def test_no_module_uses_syntax_the_oldest_supported_python_cannot_parse(tmp):
    """`requires-python` is a promise the SYNTAX has to keep, not just the API.

    A module carrying a PEP 701 f-string is not degraded on the floor
    version, it cannot be imported there at all. parse_wire.py carried one:
    every suite that did not import it passed everywhere, and the 3.11 lane
    went red only once something finally did.
    """
    floor = _requires_python_floor()
    if floor >= (3, 12):
        _util.skip(f"requires-python is already {floor[0]}.{floor[1]}")
    problems = []
    for path in sorted(Path(_util.SCRIPTS).glob("*.py")):
        for line, what in _pep701_uses(path):
            problems.append(f"{path.name}:{line}: {what}")
    assert not problems, (
        f"PEP 701 syntax, a SyntaxError on {floor[0]}.{floor[1]}: "
        + "; ".join(problems))


def test_the_package_ships_every_parser(tmp):
    with open(ROOT / "pyproject.toml", "rb") as fh:
        cfg = tomllib.load(fh)
    assert cfg["tool"]["setuptools"]["packages"] == ["session_parsers"]
    for name in MODULES:
        assert (ROOT / "session_parsers" / f"{name}.py").is_file(), name


def main():
    return _util.runner(_util.collect(globals()), tmp_prefix="pkgcontract_")


if __name__ == "__main__":
    raise SystemExit(main())
