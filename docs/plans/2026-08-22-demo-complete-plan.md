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

### Task 42c: project scope is not enforced, it was an accident — DONE 2026-08-24

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

**Acceptance:** the rehearsal's developer turn passes without the expectation
being relaxed.

**Done.** `وعندكم إيه في نور سيتي؟` now answers *"إحنا مطوّري Madinaty وده اللي
عندنا"* — the project named, no unit, no price, no alternative, because the
alternative *is* the leak.

`vocabulary()` is the one read not narrowed to the project, and it binds a
named `_EVERY_PROJECT` rather than a bare `None` at the call site, where an
omission would read as a bug and be "fixed" by the next person through.
`search`, `get` and `compounds` stay scoped, each asserted directly. The AST
guard that every statement shares one predicate is unchanged — what the
predicate is *bound to* is asserted in behaviour instead.

`refuse`, not `handoff`: a colleague cannot sell Noor City either, and
escalating a question that has a complete answer makes the answer look like an
escalation. Task 42's handoff was the shape of an out-of-vocabulary error, not
a decision about what the customer should hear.

**Task 38's opposite decision was reversed, not deleted.** Its test said an
unscoped vocabulary lets the extractor resolve a compound the sales team does
not sell. Right about the consequence, wrong about the cause: offered one
compound, the extractor does not fail to name the other — it names *this* one.
Both that test and the docstring arguing for it now say so.

Sabotage-verified in both halves: disabling the guard fails two tests,
re-scoping the vocabulary fails three.

Nothing stores a turn's action, so the rehearsal asserts `refuse` by what it
does not do — no handoff row, and no figure in the provenance. Both matter:
the failure being replaced was a real price for a real unit nobody asked
about.

---

### Task 42d: a held compound survives a message naming another region — DONE 2026-08-24

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

**Done.** `في استوديو في الساحل الشمالي؟` now answers *"مفيش studio في North
Coast دلوقتي. عندنا studio في Noor City بسعر 2,930,000 EGP"* — the region the
customer named in the half that says what is missing, and the alternative
still the same type, still no chalet.

**The relation is declared, not coded.** `compound:` in the search script gains
`narrows: city`, and the engine clears any held slot that declares `narrows: X`
when a turn names `X` and not the slot itself. The engine never names a
vertical's slots — asserted over the module's string constants, because the
word "compound" appears in an unrelated comment and a substring scan reads that
as the slot.

Distinct from `clear_slots`, and the difference is the whole bug: clearing is
the customer saying *this no longer applies* with no replacement — "somewhere
else". Here they named a replacement, one slot up, and every mechanism in the
system was watching for the other sentence.

Only when this message does not also name the narrower slot: `شقة في مدينتي في
التجمع الخامس` names both and means both.

Sabotage-verified twice — not clearing the stale slots fails three tests,
removing `narrows: city` from the script fails four.

**The rehearsal is 10/10 with the tightened checks**, and both suites were
re-baselined because `config_hash` moved (§2.4, 2026-08-24 later). No eval case
moved, which is the finding rather than the reassurance: no multi-turn case in
either suite names a city while holding a compound. That absence is why the
rehearsal found this and the suites did not, and it is the same shape as the
gap 42b closed — worth a case per vertical rather than a note here.

---

### The displacement hole, closed as cases — DONE 2026-08-24

**re-0026** and **edu-0018**, one per vertical. Both rehearsal-only findings
(42b's compound precedence, 42d's stale compound) were multi-turn
*displacement* — a later turn moving the customer off a held slot — and every
multi-turn case in both suites accumulated instead. re-0018 gathers three
slots, edu-0007 gathers three, and re-0001 turn 3 clears two only because the
customer said "in any other location" out loud. Nothing tested a value being
replaced or invalidated without the customer announcing it, which is how
people actually talk.

**They are not the same mechanism, and cannot be.** 42d is a *wider* slot
invalidating a narrower held one, which needs a containment hierarchy among
the slots — a compound sits inside a city. Education has none: `branch` and
`faculty` are orthogonal, a faculty is taught at both campuses, and edu-0007
turn 2 correctly *keeps* the faculty when a branch arrives. Declaring a
`narrows:` there to make the pair symmetric would invent a relation the domain
does not have. So education covers the same class by the mechanism it actually
has — a new value replacing a held one on the same slot.

re-0026 passes 3 of 3. edu-0018 fails 3 of 3 and found Task 42e below on its
first run, which is what a case written to cover a hole is for.

---

### Task 42e: a bare slot value cannot change a held slot — DONE 2026-08-25

**Found by edu-0018 immediately, and it is the same hole one layer down.**

`الحد الأدنى للقبول في طب الأسنان في العريش بالمعادلة العربية؟` answers 75%.
`وفي القنطرة؟` — a branch and nothing else — returns the fallback
disambiguation list: *"تقصد أنهي واحدة فيهم؟"* with three unrelated Qantara
titles. The customer asked a threshold question one turn ago, named the other
campus, and was asked to pick from a menu.

**Mechanism, read from `ScriptEngine._resumed`.** A turn carrying slots and no
intent resumes the node it came from — that is how edu-0007's bare `العريش`
works. Resumption is defined as filling a slot the node is *still waiting for*:

```python
pending = (requires_slots | requires_any_slot) - set(state.slots)
return state.node if pending & set(turn.slots) else None
```

`branch` is already held, so it is not pending, so nothing intersects, so the
turn falls to the fallback node and clarifies. The docstring states the
assumption outright — *"repeating a slot the node already holds answers
nothing"* — and that is true of a repeat and false of a replacement. The two
are the same shape at the type level and opposite in meaning, which is exactly
the conflation Task 42b fixed in the extraction prompt and 42d fixed in the
merge. This is its third appearance.

`slot_retention_accuracy` is 100.0% on the run that fails this case. The state
after turn 2 is correct — `{branch: qantara, faculty: dentistry, certificate:
arab_equivalent}`. Extraction is right, the merge is right, and the routing
decision made from them is wrong.

**Recommended fix.** Resume when the turn's slots intersect the node's declared
slots at all, not only the pending ones — a value for a slot this node reads is
this node's business whether or not it already had one. The docstring's real
concern is a *topic change*, and that is already handled by the intent being
non-null; a bare value for a slot the node does not read still falls back.

**Tests first:**

```python
def test_a_bare_value_replacing_a_held_slot_resumes_the_node()
def test_a_bare_value_filling_a_missing_slot_still_resumes()   # edu-0007
def test_a_bare_value_for_a_slot_this_node_does_not_read_still_falls_back()
def test_a_turn_carrying_an_intent_still_routes_by_intent()
```

**Acceptance:** edu-0018 passes without its expectations being relaxed, and
edu-0007 still passes.


**Done.** `_resumed` now resumes on a slot the node *declares*, held or not,
instead of only one it was still waiting for. `expected_action_accuracy` went
from 90.5% (85.7–95.2) to **95.2% with zero spread**, which is the one metric
that moved outside its previous spread and the one the change was aimed at.
The rehearsal is 10/10 and real estate holds 100.0% on 22 cases.

**One behaviour changed beyond the case, and it is a trade.** An *exact*
repeat — the same faculty named twice — used to fall to the fallback and now
re-answers. Routing could tell a repeat from a replacement by comparing
values, and deliberately does not: routing would then read slot values and not
just their names, for an edge case with no evidence either way, and the
behaviour it preserves is the worse half — an exact repeat used to be answered
with "ممكن توضّحلي أكتر", a request to clarify a message the system had
understood perfectly.

**What that costs, recorded rather than solved.** The fallback's clarification
counted toward `max_consecutive_clarifications`, so a customer repeating
themselves three times escalated to a human. They now get the same answer three
times and never escalate. Repetition is a frustration signal and this drops it.
The escalation belongs to whatever reads frustration, not to a routing rule
that happened to be catching it — see §16.

**edu-0018 does not pass, so 42e's acceptance is not met.** It fails on a
different thing now, one layer down: Task 42f.

---

### Task 42f: a resumed turn retrieves on the fragment

**Uncovered by 42e, and it is the same conflation one layer further down.**

`وفي القنطرة؟` now routes to `admission_thresholds` correctly, holds
`{branch: qantara, certificate: arab_equivalent, faculty: dentistry}`
correctly, and answers: *"المعلومات المتاحة لدي لا تحدد مدة دراسة كلية طب
الأسنان في فرع القنطرة"* — about study **duration**, which nobody asked about.
The judge scored helpfulness 0.

**Mechanism, structural rather than probabilistic.** `Orchestrator.handle`
runs extraction and retrieval concurrently:

```python
turn, retrieval = await asyncio.gather(extract(), search())
...
return await search_with.search(query=redaction.text)
```

The query is the raw redacted message and nothing else, *by construction* —
retrieval starts before extraction finishes, so it cannot see this turn's
slots. On a first turn the message carries the topic and this is invisible.
On a resumed turn the message is two words, and the topic lives in the state
the query cannot reach.

Directly evidenced from the run before 42e, where the same message reached the
fallback: its five retrieved titles were the Qantara branch address, the
Qantara programme list, the internship page, transport and dorms. The
thresholds chunk was not among them, on either side of the routing fix.

**Why this is not a small change.** The `gather` is deliberate — retrieval and
extraction are the two slowest things in the intake phase and §2.5's budget is
already the tightest gate in the suite. Enriching the query with *this turn's*
slots serialises them. Three shapes are worth costing before choosing:

- **Held slots and the held node's topic**, both available *before* extraction,
  so the `gather` survives. Cheapest. Misses the value the customer just named
  — the query would say "dentistry thresholds Arish" while they asked about
  Qantara — which may still retrieve the right chunk, since the chunk holds
  both branches. Fragile in exactly the way that sentence suggests.
- **Re-retrieve after extraction when the turn resumed a node**, keeping the
  concurrent search for the common case and paying a second round trip only on
  resumed turns. Correct, and it costs one retrieval on the turns that need it.
- **Serialise always.** Simplest to reason about, and it puts the whole
  extraction latency in front of every retrieval.

The second looks right and none of them should be picked without measuring
against §2.5.

**Tests first:**

```python
async def test_a_resumed_turn_retrieves_on_more_than_the_message()
async def test_a_first_turn_still_retrieves_concurrently()   # the latency path
async def test_the_query_carries_the_slot_this_turn_named()
```

**Acceptance:** edu-0018 passes without its expectations being relaxed, and
`p95_latency_ms` is measured on the same run rather than assumed unchanged.

**And a metric that cannot see this.** `retrieval_recall_at_5` reads 100.0% on
the run where turn 2 retrieved nothing relevant, because recall is scored per
*case* against `gold_chunks` and turn 1 retrieved the gold chunk. Third metric
in three days found structurally unable to see the failure beside it — after
`hallucinated_figure_rate` and `slot_retention_accuracy`. Recorded in the
harness spec; a per-turn recall would say it, and changing a denominator
mid-plan makes every recorded run incomparable under §2.3.


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
