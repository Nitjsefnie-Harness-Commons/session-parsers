# session-parsers

Transcript parsers for Claude Code, Kimi and Codex agent sessions.

## Install

Not on PyPI. Every release publishes the wheel with a `SHA256SUMS` file beside
it, and checking against it is the point: fetching "the newest release" is
otherwise a promise about a URL, not about the artifact CI built.

```sh
gh release download v1.0.1 --repo Nitjsefnie-Harness-Commons/session-parsers
sha256sum -c SHA256SUMS
pip install ./session_parsers-1.0.1-py3-none-any.whl
```

| Tool | Reads |
|---|---|
| `parse-session` | Claude Code session JSONL (`~/.claude/projects/<slug>/<uuid>.jsonl`) |
| `parse-wire` | Kimi `wire.jsonl` |
| `parse-codex` | Codex rollout JSONL (`~/.codex/sessions/`) |

Each takes `--help`, and each reports its own version with `--version`.
