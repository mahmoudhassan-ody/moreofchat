# More Of Chat — Demo-Complete Plan

> **For Claude:** implement task-by-task. Write the failing test, run it, confirm it fails for the right reason, then implement. Commit after every task. Stop at each boundary for review.

**Target:** a full working demo to Sinai University, a real-estate broker, and a developer. They see their own data, their own name in the console, conversations on WhatsApp and Telegram, and a human taking over a handoff. No date pressure — the demo moves rather than the scope.

**Prerequisite state:** `ec47a91`. 1299 tests, 8 import contracts, both verticals measured, CI green with the red-path alert proven. 8 GB / 40 GB VPS.

**Read first:** `CLAUDE.md`, design doc §11 (verticals), §17 (brand), §19 (config surface), and the two console mocks in `docs/design/`.

**Standing constraints, all test-enforced.** Everything from the prior plans still holds, plus three new ones:

- **Bilingual from the first component.** Every string through i18n, `dir` at the root, CSS logical properties only. There is no "English first, Arabic later" — RTL retrofitted is a rewrite.
- **Console language and reply language are independent settings.** An officer works in English while the bot answers students in Masri. Conflating them repeats the register/language collapse fixed in composition.
- **`#7D39EB` is the only colour.** It appears where something is live, actionable, or verified. Everything else is greyscale. Identity is carried by the logo and typography, not by colour.

---

## Track A — The tenant console (3 weeks)

The long pole. Nothing built so far has an interface.

### Task 28: Agent authentication

Carried since P1 Task 22, where `AgentAuthenticator` was left as a seam with no default — because a header-derived tenant id is an authorization bypass wearing the shape of a feature.

**Tests first:**

```python
async def test_a_session_resolves_exactly_one_tenant()
async def test_a_token_for_tenant_a_cannot_read_tenant_bs_inbox(app_engine, two_tenants)
async def test_an_expired_session_is_refused_not_renewed()
async def test_the_tenant_id_never_comes_from_a_request_header():
    """Structural. The bypass arrives as a convenience — a header the frontend
    already sends, trusted because it is usually right."""
async def test_logout_invalidates_server_side_not_only_the_cookie()
async def test_password_hashing_uses_a_slow_kdf():
    """Assert the algorithm, not that hashing happened."""
```

**Notes:** sessions in Postgres, httpOnly + SameSite cookie, tenant resolved server-side from the session row and never from the client. This is the fourth security-critical file — flagged for line-by-line human review alongside `guards.py`, `webhooks.py`, and the Qdrant repository.

**Acceptance:** the cross-tenant test proven to fail when tenant resolution is moved to a header.

---

### Task 29: Console shell — i18n, RTL, theme

Everything after this inherits it, so it comes before any screen.

**Tests first:**

```python
def test_every_visible_string_comes_from_the_catalogue():
    """AST/scan over components. A literal that ships is a string that never
    translates, and it is found by a customer, not a test."""

def test_both_catalogues_have_the_same_keys():
    """A missing Arabic key renders English inside an Arabic sentence."""

def test_no_component_uses_left_or_right_css():
    """margin-left in an RTL layout is the bug that looks like a design
    choice. Logical properties only."""

def test_the_theme_exposes_exactly_one_accent()
def test_console_language_and_reply_language_are_separate_settings()
def test_language_preference_is_per_user_not_per_tenant()
```

**Notes:** React + Vite + TypeScript, i18next, IBM Plex Sans Arabic. Tokens from a single theme file — `#7D39EB`, greys, and the semantic names (`live`, `verified`, `actionable`) that map to it. A second accent added later should require editing one file and failing one test.

---

### Task 30: Tenant identity and onboarding

**Tests first:**

```python
async def test_the_header_shows_the_tenants_name_and_logo()
async def test_a_tenant_without_a_logo_falls_back_to_its_initials():
    """Not to the More Of Chat mark — a tenant seeing our logo where theirs
    should be reads as a product that does not know who they are."""
async def test_logo_upload_rejects_a_non_image_by_content_not_extension()
async def test_powered_by_appears_on_every_page()
async def test_tenant_branding_is_tenant_scoped(app_engine, two_tenants)
```

**Notes:** name, logo, default language, timezone on `tenants`. White-label — their colours, their domain — is **out of scope** and stays out: your mark in the corner of a working product is how the university's IT director tells the broker about you.

---

### Task 31: Knowledge screen — upload and ingestion feedback

Where "they feed their own data" becomes true.

**Tests first:**

```python
async def test_an_uploaded_document_becomes_searchable_chunks()
async def test_the_screen_shows_chunk_count_and_a_sample_before_confirming():
    """Chunking is where a corpus quietly breaks. A sentence severed from its
    number surfaces months later as an orphan figure in a correct reply."""
async def test_a_failed_ingest_names_the_row_and_the_reason()
async def test_re_uploading_an_unchanged_document_costs_no_embedding():
    """The content-addressed cache, visible in the UI as 'unchanged'."""
async def test_ingest_writes_an_embedding_call_row_to_the_ledger()
async def test_upload_is_tenant_scoped(app_engine, two_tenants)
async def test_a_document_can_be_removed_and_its_chunks_go_with_it()
```

**Notes:** the real-estate side is a catalogue sync rather than documents — same screen, different source shape. Show `as_of` prominently: a broker looking at stale inventory should see the date without asking.

---

### Task 32: The inbox

The mock is the spec: `docs/design/console-inbox-purple.html`.

**Tests first:**

```python
async def test_the_thread_shows_every_channel_for_one_contact()
async def test_a_grounded_figure_links_to_the_chunk_that_grounded_it():
    """The differentiator. The data exists — grounding results, passages,
    as_of — and has been discarded at the UI boundary until now."""
async def test_the_source_pane_shows_the_gates_that_passed()
async def test_taking_over_stops_the_bot_replying()
async def test_returning_to_the_bot_resumes_at_the_right_node()
async def test_an_agent_reply_goes_out_through_the_same_provider_adapter()
async def test_sse_only_streams_the_agents_own_tenant()
```

**Notes:** the provenance pane is the reason a dean believes the product. It costs nothing to build — the data is already computed and thrown away.

---

### Task 33: Scripts and settings

**Tests first:**

```python
async def test_a_script_can_be_edited_and_previewed_before_publishing()
async def test_publishing_pins_a_version_and_in_flight_conversations_keep_theirs()
async def test_a_script_cannot_lower_the_confidence_gate():
    """§19.3 through the UI. The console must not offer what the engine
    already refuses — a disabled slider is a worse lie than no slider."""
async def test_synonyms_are_editable_per_tenant():
    """Every broker names areas differently. This is the screen that stops
    that being an engineering ticket."""
async def test_settings_changes_are_written_to_the_audit_log()
```

---

### Task 34: Analytics

**Tests first:**

```python
async def test_containment_rate_is_shown_and_never_gated():
    """Gating it creates pressure to answer rather than hand off."""
async def test_cost_per_conversation_comes_from_the_ledger_not_an_estimate()
async def test_an_unpriced_model_shows_unknown_not_zero()
async def test_analytics_are_tenant_scoped(app_engine, two_tenants)
```

**Notes:** what a buyer asks — how many answered, how many handed off, what it cost, what people asked about. The ledger now supports the cost answer; nothing else does.

---

## Track B — Channels (1.5 weeks)

### Task 35: Telegram end to end

The adapter exists and has never carried a real message. Cheapest channel, no approval, and the one to demo alongside WhatsApp.

### Task 36: Instagram and Messenger

One Meta Graph webhook integration serves both. Stricter human-agent windows on Instagram.

### Task 37: Email

SendGrid inbound parse; thread on `Message-ID` / `References`. SPF, DKIM, DMARC per sending domain or the tenant's replies land in spam.

**Every adapter goes through the existing `MessagingProvider` port. The one-send-path contract already forbids the alternative.**

---

## Track C — The developer variant (0.5 weeks)

### Task 38: Single-project depth and sales routing

A tenant flag, not a product (design §11.2). Brokers search across projects; developers answer deeply about one and route to the right sales team.

**Tests first:**

```python
async def test_a_developer_tenant_scopes_every_search_to_its_own_project()
async def test_routing_sends_a_qualified_lead_to_the_configured_team()
async def test_the_no_substitution_rule_still_holds_within_one_project()
```

**Note:** you have no developer data. That's content work and it's the long pole of this track, not the code.

---

## Track D — Demo hardening (1 week)

### Task 39: The real-phone path

Never been driven end to end outside tests. Everything else is theatre if this doesn't work.

### Task 40: Typing indicator

Twilio's endpoint shipped 2026-06-14 and needs `messageId`, which we already carry. Two known caveats: it marks the message read (not separable), and the vendor's own pages disagree on GA versus beta. A second HTTP client — different host, JSON not form, auth scheme untested.

### Task 41: Ingest the real corpora

Sinai's KB and the broker's catalogue, the day before — not during. Their column names will not match the fixtures. This is where surprises live.

### Task 42: Rehearsal

Run the demo script end to end on the real tenants. Every question you plan to ask, plus three you don't.

---

## Exit criteria

Each line needs evidence, not a tick.

- [ ] A message from a real phone produces a grounded reply on WhatsApp and on Telegram
- [ ] Three tenants, each seeing their own name, logo and data
- [ ] Console usable in Arabic and English, toggled per user, RTL correct in both
- [ ] A document uploaded through the console becomes answers
- [ ] A handoff appears in the inbox, an agent replies, the customer receives it
- [ ] A grounded figure in the inbox links to the chunk that grounded it
- [ ] Cost per conversation shown from the ledger, no unpriced rows
- [ ] Cross-tenant tests proven to fail when isolation is removed — auth, inbox, upload, analytics
- [ ] The rehearsal run completed with no manual intervention

## Not in this plan

- White-label: tenant colours, custom domains
- SIS and CRM integrations (P3)
- Billing and self-serve onboarding (P4)
- Tansik load testing (P4)
- The corpus. Both suites remain ~95% ours; that closes with pilot traffic, not with building.

## Two things carried, and neither is code here

**The TypeScript repo.** `locations.ts`'s corrupt Alamein aliases and the cross-type substitution in the reply path. Both live, both costing leads, and that repo is not on this VPS.

**`edu-0012`.** Fails on every model measured — both sonnet and haiku relabel a nearby figure as tuition under adversarial pressure, and only the judge's `figure_labelling` check catches it. A system defect rather than a model comparison, and it deserves its own diagnosis after the demo.
