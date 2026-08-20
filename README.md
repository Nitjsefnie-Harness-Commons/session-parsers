# session-parsers

Transcript parsers for Claude Code, Kimi and Codex agent sessions.

Install: `pip install session-parsers`

| Tool | Reads |
|---|---|
| `parse-session` | Claude Code session JSONL (`~/.claude/projects/<slug>/<uuid>.jsonl`) |
| `parse-wire` | Kimi `wire.jsonl` |
| `parse-codex` | Codex rollout JSONL (`~/.codex/sessions/`) |

Each takes `--help`, and each reports its own version with `--version`.
