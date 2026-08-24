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

### Task 41b: Inventory provenance

The source pane is blank for two of the three tenants, and it is the thing the
demo is built around.

A chunk-grounded figure links to the chunk that grounded it (Task 32). An
inventory figure has the same promise and a different shape: a price traces to
a **row** — a unit id, a compound, and the `as_of` that row was snapshotted at
— and an instalment traces to a **calculator output** with the inputs it ran
with. Both are computed today and both are discarded at the worker boundary:
`InventoryTurn` carries `presented_unit_ids` and `computation`, and the inbound
worker writes `provenance=None` because the pane renders chunks.

**Tests first:**

```python
async def test_a_price_traces_to_the_row_it_was_read_from()
async def test_an_instalment_traces_to_the_calculator_inputs_that_produced_it():
    """Not to the row alone. §19.3: the arithmetic is the tool's, and the
    evidence for a number the model never composed is the computation."""
async def test_the_as_of_travels_with_the_figure_rather_than_beside_it():
    """A price separated from its date is a price the tenant cannot stand
    behind, and `asof_disclosure_rate` grades the reply rather than the pane."""
async def test_a_figure_with_no_row_and_no_computation_behind_it_is_not_sent():
    """The same gate the chunk path already has, on the other grounding mode."""
def test_the_source_pane_renders_a_row_and_a_computation()
```

**Notes:** one `figures` list, one renderer, `source: "inventory" | "calculator"`
beside the existing `"chunk" | "script"`. Not a second provenance shape — a
second shape is a second thing the pane can fail to render, and the promise
being made to all three tenants is the same one.

**Acceptance:** a broker's price in the inbox opens the unit row and its
snapshot date; a payment plan opens the calculator's inputs and output. Proven
by removing the trace and watching the figure fail to send.

---

### Task 42: Rehearsal

Run the demo script end to end on the real tenants. Every question you plan to ask, plus three you don't.

**The runner exists:** `evals/demo/rehearsal.yaml` is the script and
`scripts/rehearse.py` runs it — three tenants, their own numbers and data,
through the real programs, with every figure's provenance checked on every
turn. `--only <slug>` re-runs one tenant. Replace the questions with the real
ones before the run that counts.

**What the first runs found**, all fixed and pinned by tests:

- An out-of-vocabulary slot value **killed the turn and the customer got
  silence** — a student asking about a faculty this university does not run, and
  a customer asking a project-scoped developer about the project next door.
  §2.6 is not "no wrong answer reaches the customer", it is "no error does",
  and nothing at all is the worst version of one. Both verticals.
- `quoted_unit_id` outranked the unit named in the current message, so browsing
  Madinaty and then asking about a Noor City unit was answered about Madinaty —
  fluent, grounded, every figure traced, wrong property. Nothing in the eval
  suite caught it because every payment-plan case is a first turn.
- A dead-letter row carried `repr(exc)` and no traceback, which is the one
  place "why did this customer get nothing" has to be answerable.
- Orphaned workers from a killed run kept consuming the same consumer group,
  so two generations answered alternate messages and the older one won some.
  That is what a deploy which forgets to stop the old containers does.

---

### Task 42b: A customer cannot change their mind — DONE 2026-08-24

`extraction_v1.md` shows the model the slots already held. On the second turn
of "I want an apartment in Madinaty" → "the unit in Noor City at six and a
half million, what's the instalment?", the extractor returns `near_price` and
**no compound**: it has been told Madinaty is held, and it does not contradict
it. Everything downstream is then correct about the wrong unit, and no fix
below extraction can recover a value the model did not return.

The prompt already carries the rule — *"Repeat a held slot only if this message
changes it"* — with a worked example of an explicit correction (`مش التجمع،
الشيخ زايد`). This phrasing is not explicit: naming a unit in another compound
reads as continuing the same conversation.

**Tests first:**

```python
async def test_naming_a_unit_in_another_compound_changes_the_compound()
async def test_an_explicit_correction_still_works():
    """The existing case, so the fix does not trade one for the other."""
async def test_a_message_that_names_no_place_still_keeps_the_held_one():
    """The reason held slots exist. "And at 40% down?" names nothing."""
```

**Notes:** an eval case in `evals/cases/realestate.yaml` as a *second* turn —
every payment-plan case today is a first turn, which is exactly why the suite
was silent. The prompt version is pinned by the harness (§19.4), so this is a
measured change rather than an edit.

**Acceptance:** the rehearsal's broker section passes without the expectation
being relaxed.

**Done.** The broker section passes: the second turn now extracts
`{compound: Noor City, property_type: apartment, near_price: 6500000}` and the
reply is the Noor City instalment schedule from the calculator. re-0025 is the
case, as a second turn, and it passes 3 of 3. Fresh baselines for both suites
are in the harness spec (§2.4, 2026-08-24): real estate 100.0% on 21 cases,
education 86.3% (82.4–88.2), measurable.

**The prompt edit took three attempts and both failed ones were worked
examples, not rules.** The first illustrated the naming rule with
`الوحدة في نور سيتي بـ ٦ مليون`, and re-0016's franco ceiling `b 6 melion`
flipped from `budget_max` to `near_price`. The second replaced it with a
sentence disclaiming any price-slot meaning, and `property_type` on re-0024's
sentence fell from 10/10 to 1/10. The final example names no figure at all.
Every step was measured against both prompts rather than reasoned about, which
is the only reason either was caught —

**— because the test that should have caught them was dead.**
`test_live_the_two_price_slots_are_not_confused_in_either_direction` read
`live_runner._agent` after the fixture started yielding `(runner, session)`,
so it raised `AttributeError` before its first assertion. `live` tests do not
run in CI, so nothing reported it. Fixed and passing.

---

### Task 42c: project scope is not enforced, it was an accident

**Introduced-by-exposure in 42b, and the underlying gap is older.** The
rehearsal's developer turn — `وعندكم إيه في نور سيتي؟` asked of a tenant whose
`project` is Madinaty — now answers *"عندنا apartment في Madinaty بسعر
5,800,000"*. A cross-project substitution, which is the one thing Task 38's
vertical exists to prevent.

The mechanism, measured 5 of 5 under each prompt:

- A project-scoped tenant's catalogue holds one compound, so the extraction
  vocabulary offers one compound. Asked about another, the model must return a
  listed value, and the only listed value is Madinaty.
- Before 42b it returned `Noor City` anyway — breaking the prompt's own "use
  the exact value from the list" rule — which raised `ExtractionFailed`, which
  Task 42 turned into a scripted handoff. The right behaviour, produced by the
  model disobeying an instruction.
- 42b's prompt is followed more closely, the disobedience stopped, and with it
  the only thing standing between a buyer and another developer's compound.

So this is not a regression in the sense of correct-then-broken: the guarantee
was never implemented. It was a safety net under an accident, and the accident
was load-bearing.

**Recommended fix — vocabulary scope is not search scope.** Give a
project-scoped tenant the *full* compound vocabulary for extraction while the
search stays filtered to the project. The model then names Noor City
correctly, and the runner compares the extracted compound against
`tenants.project` and hands off with the scripted "we are the developer of
Madinaty" reply. Deterministic, and it does not depend on any model behaviour.

**Tests first:**

```python
async def test_a_compound_outside_the_project_hands_off()
async def test_the_project_s_own_compound_still_answers()
async def test_the_search_is_still_filtered_to_the_project()
    # Widening the vocabulary must not widen what can be quoted.
```

**Acceptance:** the rehearsal's developer turn passes `expect: handoff`
without the expectation being relaxed.

---

### Task 42d: a held compound survives a message naming another region

**Pre-existing, unchanged by 42b, and it fails in front of a buyer.** The
broker's third turn asks `في استوديو في الساحل الشمالي؟` and is answered
*"مفيش studio في Noor City دلوقتي. عندنا studio في ZED East"*. The customer
named the North Coast; the reply names neither the North Coast nor anything on
it — ZED East is New Cairo.

**Extraction is not at fault.** It returns `{city: North Coast,
property_type: studio}`, 5 of 5, under both prompts. The held `compound` from
two turns earlier is never cleared, `with_slots` merges it forward, and the
search filters on compound *and* city — an empty intersection, reported as
"no studio in <the compound they stopped talking about>".

This is 42b's family and a different member of it: not a value replaced by
another value of the same slot, but a value made **inconsistent** by a value of
a different slot. `clear_slots` is the mechanism and nothing invokes it here,
because the customer did not say "somewhere else" — they said somewhere.

**Recommended fix — below extraction, not in the prompt.** A compound belongs
to exactly one city in the catalogue. When a turn sets `city` and the held
`compound` is not in it, the compound is stale and must be dropped in the
merge. Deterministic, reads from the catalogue, and needs no model behaviour.
Deciding it in the prompt would ask the model to know which compounds are on
the North Coast, which is the one thing the vocabulary design forbids.

**Tests first:**

```python
async def test_naming_a_city_drops_a_held_compound_elsewhere()
async def test_naming_a_city_keeps_a_held_compound_inside_it()
async def test_a_turn_that_names_no_city_keeps_the_compound()
```

**Acceptance:** the rehearsal's broker third turn names the North Coast, and
still names no chalet.

---

### The rehearsal's own checks were not asserting

Both findings above passed a 10/10 rehearsal before the checks were tightened,
and one of them had been passing since Task 42.

**`expect:` was parsed and never read.** Every turn in `rehearsal.yaml`
declares `answer`, `handoff` or `any`, and `judge()` looked only at
`must_not_contain`, `every_figure_traced`, `traces_to_calculator` and
`must_state_asof`. "10/10 turns held their expectations" was reporting on four
content flags and calling it the script. It now reads `handoffs` for the
thread — not the reply text, since "a colleague will follow up" is a sentence
a composed answer can also contain.

**`must_contain` did not exist.** The forbidden-claim check can only say what a
reply must not be. The broker's coast turn forbade `شاليه` and the reply
contained no chalet, so it passed while never mentioning the region the
customer asked about. A check that only forbids cannot catch an answer to a
different question.

With both in place the rehearsal reports **8/10**, and the two failures are
42c and 42d. The number went down because the checks went up; nothing about
the system got worse between the two runs.

---

## Exit criteria

Each line needs evidence, not a tick.

- [ ] A message from a real phone produces a grounded reply on WhatsApp and on Telegram
- [ ] Three tenants, each seeing their own name, logo and data
- [ ] Console usable in Arabic and English, toggled per user, RTL correct in both
- [ ] A document uploaded through the console becomes answers
- [ ] A handoff appears in the inbox, an agent replies, the customer receives it
- [ ] A grounded figure in the inbox links to what grounded it — the chunk
      for a document answer, the unit row and its `as_of` for a price, the
      calculator's inputs for an instalment
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
