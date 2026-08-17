# More Of Chat — Platform Design Document

**Product:** More Of Chat
**Internal code prefix:** `moc` (repo, packages, DB, container names)
**Owner:** Mahmoud Hassan / Odyssey Technology
**Status:** Awaiting approval before build
**Date:** 2026-08-16

**Goal:** A commercial, multi-tenant SaaS that answers customer questions from a tenant-authored script in Egyptian Arabic and English across WhatsApp, Instagram, Messenger, Telegram and Email — packaged for two verticals: Universities and Real Estate (brokers + developers).

---

## 1. Locked decisions

| # | Decision | Choice | Consequence |
|---|---|---|---|
| D1 | Codebase | Greenfield | No inheritance from the existing contact center; patterns only |
| D2 | Vector store | Qdrant | Two-store consistency problem → outbox pattern required |
| D3 | LLM | Claude **and** OpenAI, both via **first-party direct APIs** | Router selects per task; either fails over to the other; two DPAs; processing occurs in the US (see §2) |
| D4 | Embeddings | OpenAI `text-embedding-3-large`, direct API | Second US vendor in the data path; reindex needed if the model changes |
| D5 | Lexical search | Meilisearch retained | Fusion happens in the app layer (RRF), not in Qdrant |
| D6 | WhatsApp access | Twilio (BSP) | Per-message markup; adapter must be swappable to direct Meta later |
| D7 | Hosting | EU (Frankfurt or Amsterdam) for the app; **US for model inference** | Two cross-border hops: Egypt → EU (app) → US (models). Both disclosed in tenant consent |
| D8 | OS | Ubuntu 26.04 LTS | PostgreSQL 18, Python 3.14, Valkey 9, Docker 29; cgroup v1 removed |

### 1.1 Open items requiring decision during P0

- Twilio channel coverage for Instagram and Messenger — verify current support. If not covered, those two go direct to Meta Graph API (one integration, two surfaces).
- Egypt on-prem edition: build as a deployment target now, or defer to an enterprise upsell tier.
- Whether to create the OpenAI project as an EU-region project even while calling the US endpoint — the only irreversible choice in the stack (see §2.3).
- Dual-provider scope: failover only, or task-level routing, or per-tenant provider choice (see §2.6).

---

## 2. LLM access paths (D3 + D7)

**v1 decision: both providers are called through their first-party direct APIs.** `api.anthropic.com` and `api.openai.com`. No AWS Bedrock, no Vertex AI, no `eu.api.openai.com`.

This is a deliberate trade of compliance posture for speed and simplicity, taken with the exit route documented in §2.4 and §2.5.

### 2.1 What this buys

- **No approval gates.** Bedrock model access is a per-model request; OpenAI EU residency is granted by OpenAI rather than self-served. Both are lead-time items on a critical path. Direct APIs are self-serve today.
- **Full feature surface immediately.** Prompt caching and the Batch API are first-class on the first-party APIs — and prompt caching is the primary cost lever in §10, so having it verified rather than assumed matters.
- **Newest models first.** Frontier releases reach the first-party API before the cloud resellers.
- **One less vendor.** No AWS account, IAM policy, or cross-account billing to administer.

### 2.2 What this costs

**Latency.** Cairo → Frankfurt is ~70 ms; Frankfurt → US East adds roughly 100–150 ms round trip. Both are small against the model call, which §2.5 measured at a 4551 ms median and which dominates the turn. The network hop is not where this budget goes — measure before optimising it.

**Compliance posture — the real cost.** The data path is now:

```
Egypt (customer)  →  EU (app, Frankfurt)  →  US (model inference)
```

Two cross-border transfers, not one. Consequences:

- **Egypt PDPL (Law 151/2018)** requires consent for cross-border transfer and, depending on data category, authorization. This was already true of EU hosting; the US hop adds a second jurisdiction with no adequacy arrangement with Egypt.
- **GDPR** obligations for European-standard tenants now rest on the vendors' standard contractual clauses rather than in-region processing.
- **Public universities are where this surfaces.** Procurement asks where processing happens, and "the United States" is a harder answer than "Frankfurt." Private brokers will not ask. Plan for the objection on the education side specifically.

**Required now, not later:** the tenant consent copy and privacy notice must state that processing occurs outside Egypt and the EU. Retrofitting consent after 200 live conversations means re-consenting 200 people. Write it before the first tenant connects a channel.

### 2.3 The one irreversible choice

Everything in §2 is reversible except this. Embeddings are only comparable within the same model and dimension, so switching later means re-embedding every chunk for every tenant.

**If OpenAI EU project eligibility is easy to obtain, create the project as an EU-region project even while calling the US endpoint.** An OpenAI project cannot be converted to an EU project after creation. Creating it EU-region now costs nothing and removes a full corpus reindex from the future migration. If eligibility is slow or unavailable, proceed with a standard project and accept the reindex as the price of a later move.

### 2.4 Migration route, if a deal requires it

Kept documented because the education vertical is likely to ask.

| Requirement | Path | Cost |
|---|---|---|
| EU processing for Claude | AWS Bedrock `eu-central-1`, or Vertex AI EU regional endpoints (~10% premium) | New adapter file behind the existing `LLMProvider` port. Verify prompt caching and Batch support on the target before committing. |
| EU processing for OpenAI | EU-region project on `eu.api.openai.com`, or Azure OpenAI in France Central / Sweden Central | New adapter, plus a full re-embed unless §2.3 was taken |
| In-country Egyptian processing | On-prem edition with self-hosted models | A deployment target, not a config change. Enterprise tier. |

**What keeps this cheap:** both providers sit behind one internal `LLMProvider` port, and no endpoint or `base_url` may appear outside `src/moc/llm/`. The import-linter contract already enforces that no other module imports the provider SDKs. A region change must remain a new adapter, never a refactor — if a `base_url` assumption leaks into the router or a call site, this table stops being true.

### 2.5 Latency budget

Cairo → Frankfurt ~70 ms, Frankfurt → US ~100–150 ms.

**Targets, measured 2026-08-17:**

| Segment | Target | Basis |
|---|---|---|
| Answer-composition model call | p95 under **5000 ms** | Six samples, below |
| End to end, inbound webhook → outbound message | p95 **7000 ms** | Model call plus retrieval, guards, channel hop |
| Model call breakage ceiling | **8000 ms** | Not a target — above this something is wrong |

The earlier figure was a 4 s end-to-end p95, and it was an **unmeasured estimate** carried from the first draft of this document. It was never true. Six samples from the Frankfurt VPS against `claude-sonnet-5`, thinking disabled, effort medium, producing a realistic ~310-token grounded Arabic fee answer:

```
2746  4386  4551  4716  4934  5052   ms
min 2746 · median 4551 · max 5052
```

The **model call alone consumed 109 % of the old 4 s budget at the median**, before retrieval, guard evaluation, or the channel round trip. The numbers above replace the estimate rather than the estimate constraining the numbers; `config/llm/routing.yaml` carries them as data and `tests/llm/test_live_smoke.py` re-measures them on every live run.

**Perceived latency is mitigated by the typing indicator, not by a faster model.** WhatsApp, Instagram and Messenger all expose one; sending it on inbound receipt is what makes a 5 s reply read as considered rather than broken. This is the intended mitigation for the gap between these targets and an ideal sub-2 s reply — a chat user tolerates a visible wait far better than a silent one.

**Two levers are measured and deliberately not pulled.** `effort: low` roughly halved output tokens in probing (171 against 331), and instructing a shorter reply would too. Both trade against grounding: the tokens at risk are the `as_of` disclosure and the second grounded fact. Whether that trade is acceptable is an eval-suite question (§8.3), and the 22 worked examples are explicitly too few to settle it. They stay on the shelf, documented, until the mined corpus can price the quality cost.

Track the model-call segment separately in tracing, so a regression is attributable to the network hop rather than to retrieval or generation.

### 2.6 Dual-provider routing policy

Recommended shape: **task-level routing with cross-provider failover**, not per-tenant provider choice. Per-tenant choice doubles the eval matrix and every prompt becomes two prompts to maintain.

| Task | Primary | Failover |
|---|---|---|
| Customer-facing answer composition (Masri/MSA) | Claude Sonnet (direct API) | OpenAI flagship chat model (direct API) |
| Intent + slot extraction, language/register detection | Claude Haiku | OpenAI small/mini model |
| Retrieval query rewriting | Claude Haiku | OpenAI small/mini model |
| Embeddings | OpenAI `text-embedding-3-large` (direct API) | none — see §7.3 |
| Offline eval grading, script compilation | Claude Opus | OpenAI flagship |

Rules:

- **Answer composition is the one task that must stay pinned by default.** Egyptian-dialect register quality differs between providers, so the primary is whichever wins the Arabic eval suite (§8.3) — re-run that comparison quarterly rather than assuming.
- Failover is **automatic on breaker-open, rate-limit or 5xx**, and logged as a degraded turn in `usage_ledger` so quality regressions are attributable to the fallback path.
- Both providers sit behind one internal `LLMProvider` port. Prompts are stored per provider family because system-prompt conventions and tool-call formats differ; the eval suite runs against both, and a prompt that has not passed eval on a provider is not eligible as that task's failover.
- Prompt caching semantics differ between the two — do not assume cache-hit economics carry across. Measure per provider.
- **Never fail over mid-conversation for a single turn** if the two providers produce visibly different register; prefer scripted fallback for that turn and switch provider at conversation boundaries.

**Latency budget:** Cairo → Frankfurt round trip is roughly 60–80 ms. Acceptable for chat. The turn budget is §2.5's measured 7 s end-to-end p95 — the model call is the dominant segment, not the network hop.

---

## 3. Architecture

```
  WhatsApp (Twilio) ─┐
  Instagram ─────────┤
  Messenger ─────────┼──► Channel Gateway ──► Valkey Stream ──► Agent Worker
  Telegram ──────────┤    (verify, normalize,                      │
  Email (SendGrid) ──┘     ACK < 3s)                               │
                                                    ┌─────────────┴─────────────┐
                                                    ▼                           ▼
                                            Script Engine              Retrieval Service
                                          (deterministic FSM)      (Meilisearch + Qdrant, RRF)
                                                    │                           │
                                                    └──────────┬────────────────┘
                                                               ▼
                                                     Claude / OpenAI (direct)
                                                               │
                                          ┌────────────────────┼────────────────────┐
                                          ▼                    ▼                    ▼
                                    Outbound Sender      Agent Inbox         Usage Ledger
```

### 3.1 Design principle: script-first, LLM-in-the-loop

The LLM never freely decides what to say about fees, prices, payment plans, or admission rules. The script engine owns the flow and the facts; Claude handles four bounded jobs:

1. **Intent + slot extraction** from the inbound message (Haiku)
2. **Retrieval query rewriting** (Haiku)
3. **Answer composition** constrained to retrieved passages and script-approved copy (Sonnet)
4. **Register and dialect rendering** — Masri vs MSA (same Sonnet call)

Any turn where retrieval confidence is below threshold and the script has no matching node routes to human handoff, not to an improvised answer. This is the single most important constraint in the system: a hallucinated tuition figure or unit price is a commercial incident for the tenant.

### 3.2 Request path rules

- Webhook handler does: signature verify → tenant resolve → enqueue → 200 OK. **No LLM calls, no DB writes beyond the raw event row.**
- All model work happens in workers. Twilio and Meta both retry on slow ACK; a synchronous LLM call in the webhook path will cause duplicate message delivery.
- Idempotency: dedupe on `(channel, provider_message_id)` with a unique index.

---

## 4. Module layout

```
app/
  api/              # FastAPI routers: webhooks, admin, inbox, auth
  channels/
    base.py         # MessagingProvider protocol
    twilio_wa.py    # WhatsApp via Twilio
    meta_ig.py      # Instagram
    meta_fb.py      # Messenger
    telegram.py
    email_sg.py     # SendGrid inbound parse + outbound
  agent/
    orchestrator.py # per-turn state machine
    script_engine.py
    prompts/        # versioned, per-vertical
    guards.py       # PII redaction, confidence gate, handoff trigger
  retrieval/
    ingest.py       # chunking, outbox emit
    embedder.py     # OpenAI text-embedding-3-large
    lexical.py      # Meilisearch
    vectors.py      # Qdrant, tenant-filtered repository
    fusion.py       # reciprocal rank fusion + rerank
  arabic/
    normalize.py    # Franco-Arab, diacritics, alef/ya/hamza folding
    register.py     # Masri vs MSA policy
  llm/
    base.py         # LLMProvider port: complete(), classify(), stream()
    anthropic_direct.py   # api.anthropic.com
    openai_direct.py      # api.openai.com
    router.py       # task → provider/model, failover policy, breaker per provider
    cache.py        # prompt cache block assembly (per-provider semantics)
  tenancy/
    context.py      # tenant resolution + RLS session var
    metering.py     # usage_ledger writes
  verticals/
    education/
    realestate/
  workers/
    inbound.py  outbound.py  ingest_sync.py  reconcile.py  nightly.py
```

**Rule:** nothing outside `channels/` may import a provider SDK. Nothing outside `llm/` may import the Anthropic or OpenAI SDK. Enforce with an import-linter rule in CI.

---

## 5. Data model (PostgreSQL 18)

Every tenant-scoped table carries `tenant_id uuid not null` with row-level security enabled and a policy bound to `current_setting('app.tenant_id')`. The tenancy middleware sets it per request; workers set it per job.

| Table | Purpose | Notes |
|---|---|---|
| `tenants` | Account, vertical, plan, locale defaults | `vertical` enum drives which pack loads |
| `channel_accounts` | One row per connected channel per tenant | Provider credentials encrypted at rest |
| `contacts` | End customer identity per channel | Merge key: normalized phone or platform id |
| `conversations` | Thread state, assigned agent, status | `state` jsonb holds script cursor + slots |
| `messages` | Inbound/outbound, provider ids, cost | Unique `(channel, provider_message_id)` |
| `scripts` / `script_versions` | Versioned YAML, immutable once published | Conversations pin a version |
| `kb_documents` / `kb_chunks` | Source of truth for retrieval | `chunks.embedding_version` for reindex tracking |
| `kb_outbox` | Pending Qdrant/Meilisearch sync ops | Drained by `ingest_sync` worker |
| `handoffs` | Reason, timestamps, resolution | Feeds the containment-rate KPI |
| `agents` | Human agents, roles, availability | |
| `usage_ledger` | Per-tenant metered events | Messages, LLM tokens, provider cost |
| `eval_cases` / `eval_runs` | Arabic quality harness | See §8 |
| `audit_log` | Admin actions, consent events | PDPL/GDPR evidence |

`usage_ledger` must exist in P0. Retrofitting metering into a live billing product is far harder than it looks.

---

## 6. Channel layer

### 6.1 Normalized inbound schema

```python
class InboundMessage(BaseModel):
    tenant_id: UUID
    channel: Literal["whatsapp","instagram","messenger","telegram","email"]
    channel_account_id: UUID
    provider_message_id: str
    sender_ref: str            # phone, IG id, chat id, email address
    thread_ref: str | None     # email Message-ID / References chain
    text: str | None
    media: list[MediaRef]
    received_at: datetime
    raw: dict                  # stored, never parsed downstream
```

### 6.2 Per-channel notes

**WhatsApp (Twilio):** Write against the internal `MessagingProvider` protocol with Twilio as one implementation. Twilio adds its own per-message fee on top of Meta's; migration to direct Meta once volume justifies it should be a new adapter file, not a refactor. Do not build a template management UI — proxy Twilio's. Respect the 24-hour customer service window: outside it, only approved templates send.

**Instagram / Messenger:** If Twilio does not currently cover these, one Meta Graph webhook integration serves both. Instagram enforces stricter human-agent handoff windows.

**Telegram:** Bot API in webhook mode. Cheapest and least restricted — use it as the demo and pilot channel.

**Email:** SendGrid inbound parse for receiving (Twilio owns SendGrid, so one vendor relationship). Thread using `Message-ID` / `In-Reply-To` / `References`. SPF, DKIM and DMARC must be correct per sending domain or tenant replies land in spam — this is a common silent failure in Egypt-hosted setups.

**Outbound:** single sender worker per channel with token-bucket rate limiting in Valkey, exponential backoff, and dead-letter after N attempts with an alert to the tenant's admin.

---

## 7. Retrieval design

### 7.1 Ingestion

1. Tenant uploads or syncs source (PDF, DOCX, sheet, URL, CRM records).
2. Chunk: 400–600 tokens, 15% overlap, **respecting Arabic sentence boundaries** — naive character splitting mangles RTL text and breaks numbers away from their labels.
3. Attach payload: `tenant_id`, `doc_id`, `chunk_id`, `vertical`, `lang`, `entity_ref` (project id, program id), `effective_from`, `effective_to`.
4. Write chunks + outbox rows in one transaction.
5. `ingest_sync` worker embeds and upserts to Qdrant and Meilisearch. Point IDs are UUIDv5 from `chunk_id` — replays are idempotent.

### 7.2 Qdrant configuration

- Collections: `kb_education`, `kb_realestate`. **Not** collection-per-tenant — per-collection segment overhead becomes the binding constraint around a few dozen tenants.
- Tenant partitioning via an indexed `tenant_id` payload field with tenant-aware partitioning enabled.
- On-disk vectors + scalar quantization from day one.
- All access through a single repository class that injects the tenant filter. No raw client calls elsewhere in the codebase — a missing filter is a cross-tenant data leak.
- Snapshots to object storage nightly; nightly reconciliation job compares per-tenant counts and checksums against Postgres.

### 7.3 Embeddings

**No failover for embeddings.** Vectors are only comparable within the same model and dimension count, so a "fallback embedding provider" would silently return vectors from a different space and retrieval quality would collapse without erroring. If the embedding endpoint is down, ingest queues and search degrades to Meilisearch-only — that is the correct behaviour, and the fusion layer must handle a missing dense arm gracefully.

**Redact before embedding, not just before completion.** The embedding call sits *earlier* in the pipeline than the LLM call, and it receives the customer's raw message at query time. National IDs, payment details and student IDs must be stripped in `agent/guards.py` before either call.

`text-embedding-3-large`, truncated via Matryoshka to **1024 dimensions**. The model supports dimension reduction with minimal quality loss, and 1024 cuts Qdrant storage roughly threefold versus 3072. Store `embedding_version` on every chunk so a future model change is a tracked backfill rather than a guess.

### 7.4 Meilisearch configuration

- One index per tenant per vertical, or a shared index with a tenant filter plus tenant-scoped API keys. Prefer tenant-scoped keys — Meilisearch supports them natively and it is a stronger isolation boundary.
- Arabic setup: custom stop words, synonym map per tenant (project names, program abbreviations, brand variants), typo tolerance on — it does real work on Franco-Arab and misspelled Arabic.
- Index both the original and the normalized text (see §8).

### 7.5 Fusion

Reciprocal rank fusion over Meilisearch top-20 and Qdrant top-20, then a cross-encoder rerank to top-5. Hard rule: if the top fused score is below the tenant's threshold, the turn does **not** reach answer composition — it routes to the script's fallback node or to handoff.

---

## 8. Egyptian Arabic handling

Arabic support is the product differentiator and also where quality quietly fails. Enterprise support tooling still performs materially worse in Arabic than English, and dialect handling requires deliberate engineering rather than model choice alone.

### 8.1 Normalizer (`arabic/normalize.py`)

Applied at index time and query time, with the original text preserved:

- Franco-Arab transliteration: `3` → ع, `7` → ح, `2` → ء/ق, `5`/`kh` → خ, `sh` → ش, and so on. "3ayez a3raf el masareef" must retrieve the fees document.
- Alef folding (أ إ آ → ا), ya folding (ى → ي), ta marbuta (ة → ه as a search variant), diacritic stripping, tatweel removal.
- Arabic-Indic digit normalization (٠١٢٣ → 0123) — critical for prices, dates and phone numbers.
- Emoji and repeated-character collapse.

### 8.2 Register policy (`arabic/register.py`)

Per-script-node setting:

- **Masri** — conversational turns, greetings, clarifications, small talk.
- **MSA** — official statements: fees, admission regulations, contract terms, payment plan legal text.
- **English** — mirrors the customer's language, or forced by tenant policy.

Code-switching is the norm in Egyptian messaging, not an edge case. Never route on detected language alone; route on the tenant's configured reply policy plus the detected language as a hint.

### 8.3 Evaluation harness — build this first

Before the retrieval layer, before the inbox, before the verticals:

- 150 graded cases per vertical in real Egyptian dialect, drawn from actual conversations where possible (the Sinai University corpus is a starting point for education).
- Each case: question, expected facts, forbidden claims, acceptable register.
- Graded by Opus 5 offline against a rubric, spot-checked by a human weekly.
- Runs in CI on every prompt, script, chunking or retrieval change. A regression blocks merge.

Metrics: answer accuracy, hallucinated-fact rate (target zero on figures), containment rate, register correctness, retrieval recall@5.

---

## 9. Human handoff and agent inbox

Triggers: low retrieval confidence, explicit customer request, negative sentiment, script node marked `handoff`, three consecutive clarification loops, or any high-value real-estate lead above a tenant-set threshold.

Inbox is React + SSE. Requirements: full conversation history across channels for one contact, canned replies in both languages, the agent's reply going out through the same provider adapter, and a "return to bot" action that restores the script cursor.

---

## 10. Multi-tenancy, metering, commercial

**Isolation:** shared schema + RLS. Schema-per-tenant is tempting for enterprise sales but multiplies migration pain across hundreds of tenants; revisit only for the on-prem edition.

**Metering:** every inbound message, outbound message, LLM call (input/output tokens, model, cached tokens) and provider cost writes to `usage_ledger` with the tenant id. Nightly rollup into `usage_daily`.

**Pricing shape:** platform subscription per tenant + bundled conversation allowance + metered overage. WhatsApp costs must be passed through, because two things move independently of your margin: Meta's per-template-message billing (since July 2025), Meta's charge to AI Providers for non-template messages in certain jurisdictions (from February 2026), and Twilio's markup on top of both. A flat all-inclusive price will erode without warning.

**Cost controls:** prompt caching on the tenant script + persona + policy preamble; the cheapest adequate model for all classification work; batch endpoints for nightly summarization and eval runs; per-tenant monthly token cap with soft-warning and hard-stop thresholds.

Dual providers also give you a price lever: the router's task table can be re-pointed per task when either vendor changes pricing, provided the target has passed the Arabic eval suite for that task. Track cost per conversation per provider in `usage_ledger` so that decision is evidence-based rather than a guess.

---

## 11. Vertical packs

### 11.1 Education (Universities)

Scripts: admission requirements by faculty, credit-hour fees and payment deadlines, application status lookup, program comparison, scholarships and discounts, transportation and hostel, exam and calendar queries, transfer/credit-hour equivalency, contact routing to the right department.

Integrations: SIS lookup for application status (read-only API or nightly export), payment gateway status, admissions CRM for lead capture.

Engineering constraint: the **Tansik season spike**. Design for 20× burst on the education tier — pre-cache the top 200 questions per tenant, autoscale workers, and load-test before August each year. This is the single most likely cause of a public failure.

KPIs: containment rate, applications initiated, response p95, cost per conversation.

### 11.2 Real Estate (Brokers + Developers)

Brokers need multi-project inventory search and comparison. Developers need single-project depth plus routing to the right sales team. Model this as a tenant flag, not two products.

Scripts: unit availability by area/bedrooms/budget, price and payment plan calculation (down payment %, installment years, delivery date, maintenance deposit), project location and amenities, site-visit booking, resale/rental distinction, lead qualification and scoring.

Integrations: inventory source (sheet, CRM or API), calendar for viewings, CRM push. Zoho CRM first, given the existing partner practice.

Hard rule: prices and availability are **never** LLM-composed from memory — always retrieved from a structured inventory table with an `as_of` timestamp, and the reply states that timestamp.

KPIs: qualified leads per 100 conversations, viewings booked, lead response time, cost per qualified lead.

---

## 12. Compliance

- **Egypt PDPL 151/2018:** lawful basis and explicit consent captured at first contact per channel; consent events written to `audit_log`; **both** cross-border transfers (EU app, US inference) disclosed in the tenant's privacy notice; data subject access and erasure endpoints in the admin console.
- **GDPR:** the app runs in an EU region but model inference is in the US (§2). DPAs with Anthropic and with OpenAI, SCCs where applicable, documented sub-processor list per tenant. Note the EU AI Act became broadly applicable on 2 August 2026 — transparency obligations apply: end users must be told they are talking to an AI system.
- **AI disclosure:** the first bot message on every new conversation identifies the sender as an automated assistant, in the customer's language. Non-negotiable, and also a Meta policy requirement.
- **PII minimization:** redact national ID numbers, full payment details and student IDs before any LLM call. Log redacted forms only.
- **Retention:** per-tenant configurable message retention with a default of 24 months and a hard delete job.

---

## 13. Observability and SLOs

OpenTelemetry traces through the whole turn, Prometheus + Grafana, Loki for logs, self-hosted Langfuse for LLM traces with per-tenant cost attribution (add at P1).

| SLO | Target |
|---|---|
| Webhook ACK | p99 < 3 s |
| Inbound → outbound | p95 < 7 s (§2.5, measured) |
| Availability | 99.5% monthly |
| Hallucinated figures | 0 |
| Failed outbound after retries | < 0.1% |

Alerts: webhook queue depth, provider error rate per channel, LLM breaker open, Qdrant/Postgres reconciliation drift, per-tenant cost anomaly.

---

## 14. Deployment

Single Ubuntu 26.04 LTS host in Frankfurt or Amsterdam to start. Docker Compose; K3s only when a second node is genuinely needed.

Services: `caddy`, `api`, `worker-inbound`, `worker-outbound`, `worker-ingest`, `postgres:18`, `valkey:9`, `qdrant`, `meilisearch`, `prometheus`, `grafana`, `loki`.

**Sizing:** 8 vCPU / 32 GB / NVMe for two verticals with real tenants. 16 GB works for pilots only — Qdrant, Meilisearch and Postgres sharing a box is the constraint.

**Watch out:** cgroup v1 is fully removed in 26.04. Any container config or compose file carried over from a 24.04 host may fail to start.

Backups: pgBackRest to object storage, Qdrant snapshots nightly, Meilisearch dumps nightly, restore drill before the first paying tenant.

CI/CD: GitHub Actions → build, test, Arabic eval suite, import-linter, deploy via SSH with migration gate. Secrets in SOPS or Vault, never in compose files.

---

## 15. Phase plan

Both verticals launch together, so the vertical packs move forward out of P3 and the retrieval layer must support **two grounding modes** — document chunks (education) and structured inventory with an `as_of` (real estate) — from P1 rather than P3.

**P0 — Foundations (3 weeks)**
Tenant model + RLS + `usage_ledger` skeleton · `MessagingProvider` protocol + Twilio WhatsApp adapter + Telegram adapter · webhook → verify → enqueue → ACK · script engine v1 (YAML flows, slot filling, register switch) · Arabic normalizer including area-name and unit-word parsing · **evaluation harness, both verticals, both fixtures frozen**
*Acceptance:* a scripted WhatsApp conversation completes end to end in Masri with zero LLM-composed figures, the eval suite runs green in CI **against both providers**, and forced failover produces an acceptable reply rather than an error.

**P1 — Knowledge, inventory and humans (4 weeks — one week longer than the single-vertical plan)**
Document ingestion + outbox + Qdrant/Meilisearch sync · **structured inventory connector + `payment_plan_calculator` tool** · fusion + rerank + confidence gate · agent inbox with handoff and return-to-bot · Langfuse · reconciliation job
*Acceptance:* retrieval recall@5 above 0.85 on the education set; `arithmetic_in_model_rate` and `sold_unit_offered_rate` both zero on the real-estate set; handoff round trip works from the inbox.

**P2 — Channel breadth (2 weeks)**
Instagram, Messenger, Email · per-channel window and template rules · outbound rate limiting
*Acceptance:* the same contact is threaded correctly across two channels.

**P3 — Vertical depth and integrations (3 weeks)**
Education: SIS lookup for application status · Real estate: inventory sync from sheet/CRM, viewing calendar, Zoho lead push · full script libraries, capped at ~15 intents per vertical for first release
*Acceptance:* a real-estate payment plan reply is fully traceable to a calculator output over a timestamped inventory row; an education application-status lookup returns live SIS data.

**P4 — Commercial (3 weeks)**
Billing and metered invoicing · self-serve tenant onboarding · analytics dashboards · load test at 20× education burst
*Acceptance:* a new tenant onboards, connects WhatsApp and publishes a script without engineering involvement.

### 15.1 Sequencing note

Get a **pilot broker** connected during P1, not P3. Real estate has no conversation corpus, so its eval suite starts mostly synthetic and stays guesswork until real traffic arrives. Education has the Sinai corpus and can wait; real estate cannot. This also hedges the revenue timing: brokers close fast, universities buy through procurement committees, and the 2026 Tansik season is not reachable on this timeline anyway.

---

## 16. Top risks

| Risk | Mitigation |
|---|---|
| Hallucinated fees or prices | Script-first architecture; confidence gate; zero-tolerance eval metric |
| Tansik-season overload | 20× load test; pre-cached top questions; queue-based backpressure |
| Cross-tenant data leak via Qdrant | Single filtered repository; import-linter; per-tenant integration test in CI |
| Margin erosion from Meta/Twilio pricing changes | Pass-through pricing; `usage_ledger` from P0; monthly cost review |
| PDPL blocks public university deals | On-prem Egypt edition as an enterprise tier |
| LLM vendor outage or a model withdrawn without notice | Dual provider with automatic task-level failover; scripted fallback if both are down |
| Failover silently degrades Arabic register | Every provider/task pair must pass the eval suite before becoming eligible; degraded turns flagged in `usage_ledger` |
| Prompt drift between the two providers | Prompts versioned per provider family; eval suite runs both on every change in CI |
| Embedding model change forces reindex | `embedding_version` on every chunk; backfill job written in P1 |
| Real-estate eval suite stays synthetic and unrepresentative | Pilot broker connected in P1; weekly rule to mine new phrasings from live traffic and append cases |
| Dual-vertical launch dilutes focus | Shared core is genuinely shared; cap first release at ~15 intents per vertical rather than exhaustive scripts |
| Bot appears to negotiate price | Negotiation intents route to handoff unconditionally; `adversarial_figures` cases gate every release |

---

## 17. Brand and channel identity

**Palette:** purple primary, lime accent, white type. Define legal pairings in the console theme tokens: white-on-purple and lime-on-purple are approved; **lime-on-white fails contrast** and must be blocked at the token level, not left to judgement.

**Critical distinction — platform brand vs tenant brand.** End customers never see More Of Chat branding in the conversation. They see the tenant's WhatsApp display name, avatar and sender identity. More Of Chat appears in:

- the admin console and agent inbox
- the marketing site and sales material
- tenant-facing email (onboarding, invoices, alerts)
- the tenant's optional "powered by" footer, if the plan includes it

This means chat-avatar assets are a **tenant onboarding requirement**, not a branding deliverable — the onboarding flow must collect each tenant's display name and avatar, and warn them about the circle crop.

**Asset gaps to close before the console is built:**

| Asset | Why |
|---|---|
| Square/circular mark (glyph or `MOC` monogram) | Favicon, app icon, console header, integrations. The horizontal lockup is unusable in a circle. |
| Small-size variant tested at 32px | The zigzag-M loses legibility below ~64px; verify before it lands in a favicon |
| Arabic descriptor lockup | Local marketing and the RTL console. The logo itself never mirrors in RTL — only surrounding layout does. |
| Monochrome and single-colour versions | Invoices, print, WhatsApp template headers |
| Console theme tokens | Purple/lime/white with the contrast rule encoded |

**Before anything is printed or registered:** trademark search in Egypt, and domain availability for `moreofchat.com` plus the `.com.eg` variant.

**Positioning note:** the name reads as *more* conversation, while the value sold to tenants is containment and deflection — fewer human touches per resolution. The tagline has to carry the positioning the name does not.

---

## 18. Naming conventions

| Thing | Convention |
|---|---|
| Repo | `moreofchat` |
| Python package | `moc` |
| Database | `moc_prod`, `moc_staging` |
| Containers | `moc-api`, `moc-worker-inbound`, `moc-qdrant` |
| Qdrant collections | `moc_kb_education`, `moc_kb_realestate` |
| Meilisearch indexes | `moc_{tenant_slug}_{vertical}` |
| Env prefix | `MOC_` |

---

## 19. Configuration surface

**Principle: nothing that varies is a literal in code.** Anything that differs between tenants, verticals, languages, or over time is data — loaded from config, and exposed through an admin UI control. Code holds algorithms; config holds values.

This is a build-order commitment, not a P4 aspiration. A value embedded in a function today is a migration and a refactor when it needs a UI. The path is: **YAML file (P0–P1) → database table (P3) → admin UI control (P4)**, with the loader written so each step is a one-file change.

### 19.1 Three tiers

| Tier | Who edits | Where it lives | Examples |
|---|---|---|---|
| **Tenant** | Tenant admin, via console | DB, `tenant_id`-scoped | Script flows, register policy per node, area/faculty synonyms, business hours, handoff triggers, confidence threshold, payment-plan terms, retention period, escalation contacts, greeting and AI-disclosure copy |
| **Platform** | Odyssey staff, via admin console | DB, global | Arabic lexicons, approximation markers, model routing table, prompt versions, rate limits, chunking parameters, fusion weights, rerank cutoff, per-plan quotas |
| **Fixed** | Nobody — code only | Source, under test | See §19.3 |

### 19.2 What must be data from day one

The lexicons are the immediate case, because they change constantly and differ per tenant:

```
config/
  arabic/
    lexicon.yaml        # unit words, digit maps, ordinal and floor markers,
                        # approximation markers, franco-arab transliteration
    stopwords.yaml
  retrieval/
    defaults.yaml       # chunk size, overlap, fusion weights, rerank cutoff
  llm/
    routing.yaml        # task -> provider/model, failover policy
  verticals/
    education.yaml      # category sets, KPI definitions
    realestate.yaml
```

`src/moc/arabic/numerals.py` contains **zero** Arabic literals. It reads `config/arabic/lexicon.yaml`. Same for every module below it: the algorithm is code, the vocabulary is data.

Per-tenant overrides layer on top of platform defaults — a broker's area synonyms extend the platform Arabic lexicon rather than replacing it. Resolution order: tenant override → platform default → schema default.

### 19.3 What stays fixed, and why

Four rules are **not** configuration. They are the product guarantee, and a UI toggle that disables them is a toggle that lets a tenant break the protection they are paying for.

| Rule | Consequence if configurable |
|---|---|
| The LLM never composes fees, prices or payment figures | A wrong tuition figure reaches a student on WhatsApp |
| Payment-plan arithmetic comes from the calculator tool, never the model | A plausible-looking instalment the developer must honour or deny |
| Every tenant-scoped table enforces the RLS predicate | Cross-tenant data leak between a university and a developer |
| Unavailable inventory is never presented as available | A broker's sold unit offered to a second buyer |

**Fixed enforcement, configurable parameters.** The distinction matters: the confidence *threshold* that triggers handoff is tenant-configurable; that a below-threshold turn cannot compose a figure is not. The approximation *markers* are platform-configurable; that an approximation trips the grounding check is not.

Nothing here is hardcoded in the sense of magic numbers. These are enforced invariants, each pinned by a test that fails if the enforcement is removed.

### 19.4 The cost, and how it is paid

Every knob is a support surface, a way for a tenant to break their own bot, and a new dimension in the eval matrix. Three mitigations:

1. **Every config key has a schema** with a validated type, range, and default. A tenant cannot set a confidence threshold of 0, or a retention period of 0 days.
2. **Config changes are audited** — `audit_log` records who changed what, when, and the previous value. A tenant reporting "the bot got worse yesterday" needs an answer.
3. **Config is versioned, and the eval suite pins the version** (see harness spec §2.3). A regression measured against a different lexicon is not a measurement.

### 19.5 Admin UI scope (P4)

Two consoles, because the audiences differ:

- **Tenant console** — script editor, synonym manager, threshold sliders with recommended ranges, business hours, handoff rules, preview-and-test against their own KB before publishing.
- **Platform console** (Odyssey only) — lexicon editor, routing table, prompt versions, plan quotas, per-tenant override inspector.

Both write through the same validated config service, so a value set in the UI and a value set in YAML take the identical code path. No second implementation.
