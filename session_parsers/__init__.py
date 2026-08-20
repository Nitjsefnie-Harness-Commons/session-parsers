"""Transcript parsers for the agent harnesses the bundle installs.

Three formats, three modules, one wheel: `parse_session` reads Claude Code
session JSONL, `parse_wire` reads Kimi `wire.jsonl`, and `parse_codex` reads
Codex rollout JSONL. They share no code -- the formats have nothing in common
-- but they share a release, a CI matrix and a lint gate, which is the whole
reason they live together.

Each module keeps its own `__version__`, and that is what its `--version`
prints: the three tools have independent histories and a caller pinning one of
them is not pinning the others. The `__version__` here is the DISTRIBUTION's,
and the only one setuptools reads.
"""

__version__ = "1.0.1"
