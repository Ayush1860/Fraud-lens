# Journal

A dated log of bugs, surprises and decisions.

## 2026-11-16 — Phase 0 scaffold

- **Decision:** secrets (`FRAUDLENS_API_KEY`, `FRAUDLENS_CARD_HMAC_KEY`) and both store URLs
  have no defaults in `Settings`. A missing value fails at start-up instead of the service
  running with a guessable key. Tests build settings from the environment with `.env` disabled.
- **Decision:** Makefile pipeline targets are stubs that exit non-zero and name their phase,
  so `make all` cannot look like a successful run before the pipeline exists.
- **Surprise:** the Postgres 18 image keeps its data under `/var/lib/postgresql/<major>/`,
  so the Compose volume mounts `/var/lib/postgresql`, not the older `/var/lib/postgresql/data`.
- **Decision:** the dev machine has no Docker, so the "both containers report healthy"
  acceptance check runs as its own CI job (`compose`) rather than being taken on trust.
- **Decision:** `astral-sh/setup-uv` publishes no moving major tag, so CI pins the exact
  release; `actions/checkout` is pinned to its major tag.
