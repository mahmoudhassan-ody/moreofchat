# More Of Chat

Multi-tenant AI agent platform. Egyptian Arabic + English, across WhatsApp,
Instagram, Messenger, Telegram, Email. Verticals: universities, real estate.

## Read before working
- `docs/moreofchat-design.md` — architecture and locked decisions
- `docs/eval-harness-spec.md` — quality gates
- `docs/plans/2026-08-16-p0-week1-plan.md` — current plan; work task by task

## Non-negotiable rules
- Script-first: the LLM never composes fees, prices or payment figures.
  Facts come from retrieval or a calculator tool.
- No provider SDK (anthropic, openai, twilio, qdrant_client, meilisearch)
  imported outside its adapter package. Enforced by import-linter.
- Every tenant-scoped table gets RLS with FORCE ROW LEVEL SECURITY.
- All Qdrant access goes through the single tenant-filtered repository.
- Redact PII before the embedding call, not just before the LLM call.
- Compose ports bind to 127.0.0.1 only. This is a public VPS.
- TDD: write the failing test, run it, then implement.

## Environment
Ubuntu 26.04, Python 3.14, uv, Docker 29. 3.3 GB RAM — mind memory in ingestion.
Internal prefix: `moc`.
