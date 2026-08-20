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
import tomllib
from pathlib import Path

import _util

ROOT = Path(_util.SCRIPTS).parent
MODULES = ("parse_session", "parse_wire", "parse_codex")


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
