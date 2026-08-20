# Arabic Evaluation Harness — Specification

**Product:** More Of Chat
**Part of:** platform design doc §8.3
**Status:** P0 deliverable — built before channel code
**Date:** 2026-08-16

**Purpose:** Make Egyptian-Arabic answer quality measurable, so that every prompt, script, chunking or retrieval change is evidence-tested rather than eyeballed. This is the P0 acceptance gate. Nothing else in the platform can be trusted until it exists.

---

## 1. What this harness must catch

In priority order. The first two are commercial incidents for a tenant; the rest are quality issues.

| # | Failure | Why it matters |
|---|---|---|
| F1 | A fee, price or payment figure stated that is not in the knowledge base | A university quoting wrong tuition, or a broker quoting a wrong unit price, is a liability event |
| F2 | Confident answer where the correct behaviour was handoff or clarification | Destroys tenant trust faster than an obvious failure |
| F3 | Franco-Arab or misspelled Arabic failing to retrieve | Silent — looks like "the bot doesn't know" |
| F4 | Wrong register (Masri used for official regulations, or stiff MSA in small talk) | Reads as unprofessional or as a foreign product |
| F5 | Losing slots across turns | Forces the customer to repeat themselves; drives abandonment |
| F6 | Language mirroring failure (replying in the wrong language) | Common with code-switched input |

---

## 2. Metrics and gates

Run at the lowest available temperature. Every metric is computed per vertical and per provider.

### 2.1 Hard gates — block merge

| Metric | Threshold |
|---|---|
| `hallucinated_figure_rate` | **0** — any numeric claim not traceable to a retrieved chunk or script constant fails. **Bounded — see §2.1.1** |
| `hedged_figure_rate` | **0** — a *grounded* figure stated with an approximation marker ("حوالي", "تقريبا", "around") fails |
| `expected_action_accuracy` | ≥ 0.95 — answer vs clarify vs handoff vs refuse |
| `forbidden_claim_violations` | **0** |
| `overall_accuracy` regression vs baseline | ≤ 2 percentage points |
| `retrieval_recall_at_5` (from P1) | ≥ 0.85 |
| `arithmetic_in_model_rate` (inventory grounding) | **0** — every numeric result must trace to a calculator tool output |
| `sold_unit_offered_rate` (inventory grounding) | **0** |
| `asof_disclosure_rate` (inventory grounding) | ≥ 0.98 |

**Why hedging is gated separately from hallucination.** Both are zero-tolerance and both are caught by the same deterministic check, but they are different faults with different fixes. An orphan figure means retrieval or the script failed to supply the number. A hedged figure means the generation step editorialized over a number it *had*. Collapsing them into one rate tells you a regression happened and nothing about where to look, so the two are reported as separate metrics and a figure that is both counts only as an orphan — one incident must not move two rates.

Hedging is a hard gate rather than a soft one because its consequence is commercial, not stylistic: "around 1400" invites the customer to treat a fixed fee as an opening position, and the tenant then honours a figure they never set. Design doc §19.3 fixes the enforcement while leaving the marker list configurable.

#### 2.1.1 What `hallucinated_figure_rate` does not catch

**A figure lifted from a retrieved passage and relabelled passes this gate, and
always has.** Every 0.0% reading in the project's history is bounded by "no
figure appeared that was absent from the retrieved set" — not by "no figure was
hallucinated".

Found 2026-08-20 on edu-0012. The fixture states `500 جنيه مصري` under the
title "ما قيمة الرسوم الإضافية لتغيير المسار؟" — the track-change fee, and the
only place 500 appears in 51 facts. The reply said:

> الرسوم الدراسية للهندسة محددة برقم ثابت وليس تقريبيًا، وهو 500 جنيه مصري

Engineering tuition, which this KB does not contain at all. `check_numeric_
grounding` returned `passed=True`, no orphans: 500 is in the source set, so 500
may be stated.

This is not a collision problem and framing it as one understates it. 500 is
uniquely identified in the corpus; it was still mislabelled. The gate compares
a reply's figures against a *set of numbers*, discarding which passage each came
from and what that passage said it was — so the permitted vocabulary for any
claim is every figure across all `final_k` retrieved passages, and any of them
may be attached to any claim.

The same shape, verified: a reply saying `الخصم 73%` grounded against a passage
saying `نسبة القبول 73%` passes. An admission threshold quoted as a discount.

The gate is still worth what it was: it catches a figure invented from nothing,
which is the commoner failure and the one that produces impossible numbers.
What it cannot do is bind a number to what it claims to be, and nothing else in
the harness does either — `check_figures` is stage 1, and the judge, which can
see the mismatch and demonstrably does (it caught edu-0007 quoting the Qantara
figure), runs only on turns that passed stage 1. edu-0012 failed stage 1 on
`expected_action`, so no judge ever saw it.

---

### 2.2 Soft gates — warn, review before release

| Metric | Target |
|---|---|
| `register_accuracy` | ≥ 0.90 |
| `language_mirror_accuracy` | ≥ 0.95 |
| `slot_retention_accuracy` (multi-turn) | ≥ 0.90 |
| `containment_rate` | tracked, not gated — a low rate may be correct behaviour |
| `p95_latency_ms` | ≤ 7000 (design §2.5, measured 2026-08-17 — was an unmeasured 4000) |
| `cost_per_turn` | tracked per provider |

**Note on containment:** never gate on it. Gating containment creates pressure to answer rather than hand off, which is exactly the F2 failure. Track it as a business KPI, not a quality gate.

### 2.3 Config version pinning

Lexicons, thresholds, chunking parameters and routing are configuration, not code (design doc §19). That makes them a variable in every measurement: a regression measured against a different Arabic lexicon than the baseline is not a measurement.

Every eval run therefore records, alongside the git SHA:

- `config_hash` — a stable hash over every file under `config/`
- `lexicon_version` — explicit version field in `config/arabic/lexicon.yaml`
- `prompt_version` per task
- `provider` and `model` per task

Rules:

- A baseline is only comparable to a run with the **same** `config_hash`. On mismatch, the report states that the comparison is invalid rather than showing a misleading delta.
- Changing a lexicon requires a fresh baseline run, and that run is the new reference.
- The eval suite loads config from a **pinned fixture directory**, not from live tenant config. Otherwise a tenant editing their synonyms changes your CI results.
- `config/` changes trigger the full suite in CI, exactly like `agent/prompts/` changes. A lexicon edit can regress retrieval as badly as a prompt edit.

### 2.4 Repeated runs and spreads

**A suite run is a sample, not a measurement.** Both suites are graded partly by a model and driven by a model, over 17 and 23 cases. Four consecutive real-estate runs over one unchanged commit read 52.2%, 42.9%, 39.1% and 45.5%, and two of them were used to argue that a change had helped. Neither conclusion was supportable — the spread between the runs was wider than the change being measured.

Every suite therefore runs `runs` times (`config/evals/repeat.yaml`, default 3) and every metric is reported as **mean with min–max and its run count**:

```
overall_accuracy           46.2% (45.5–47.6, n=3)
```

Rules:

- **A single run is never measurable.** One sample has zero spread, and zero spread is exactly what a settled metric looks like. `n=1` is refused rather than rendered.
- **Unmeasured is not zero.** A metric no run fed reports `not measured (0 of 3 runs)`. Same distinction §5.1's gate results already draw, for the same reason: 0.0% is what four of the five commercial gates print when they pass.
- A metric measured in only some runs reports `n=2 of 3 runs`, never a mean over a silently smaller denominator.
- A metric whose min–max spread exceeds `measurable_spread_pp` (default 10) is flagged **not measurable at this suite size**. That is not a failure — the code may be fine and the denominator merely too small — but a delta smaller than the spread is not a result, and must not be reported as one.
- Cases that change verdict between runs are listed. They are §6.2's flaky set, and they are invisible in any single run's table.

Extraction runs at `temperature: 0.0` on both candidates (§2.6's `slot_extraction`), per §2's "lowest available temperature". Note that this is per candidate rather than per task: `claude-sonnet-5` answers a request carrying `temperature` with a 400 (`temperature is deprecated for this model`, measured 2026-08-20), while `claude-haiku-4-5` accepts it. Dropping extraction to 0 narrowed real-estate `overall_accuracy` from a 13.1-point spread to 2.1.

**Baseline, 2026-08-20**, n=3 each. Real estate on 20 cases, all passing.

Education is a **fresh baseline, not a delta**. Six fixes landed between it and
the previous figure — the composition prompt, the grounding gate's two halves,
clarifications naming their missing slots, register/language resolution, the
extractor reporting the language, and two case defects — and two of its gates
have observations for the first time. Comparing 58.8% against 49.0% would be
comparing two different systems measured by two different harnesses.

> **Every education figure recorded before this line is void, not merely
> superseded.** The composition prompt did not exist — `_compose` sent
> `system=None` and the node name as the whole message body — and the numeric
> grounding gate could neither see a figure wrapped in `**` nor distinguish a
> list marker from a claim. Runs made under those conditions were measuring
> three defects at once, and `prompt_version` did not record the composition
> prompt at all, so nothing would have flagged the incomparability. §2.3's
> rule applies with full force: those runs are not a baseline this one may be
> compared against.

| Suite | Metric | Value |
|---|---|---|
| real estate | `overall_accuracy` | 100.0% (100.0–100.0) |
| real estate | `asof_disclosure_rate` | 100.0% (100.0–100.0) |
| real estate | `arithmetic_in_model_rate` | 0.0% (0.0–0.0) |
| real estate | `type_substitution_rate` | 0.0% (0.0–0.0) |
| real estate | `invented_compound_rate` | 0.0% (0.0–0.0) |
| real estate | `wrong_compound_rate` | 0.0% (0.0–0.0) |
| real estate | `sold_unit_offered_rate` | 0.0% (0.0–0.0) |
| real estate | `tool_call_accuracy` (tracked) | 100.0% (100.0–100.0) |
| real estate | `unresolved_type_rate` (tracked) | 35.3% (35.3–35.3) |
| real estate | `errored_rate` | 0.0% (0.0–0.0) |
| education | `overall_accuracy` | 58.8% (52.9–64.7) — spread 11.8, **not measurable at 17 cases** |
| education | `expected_action_accuracy` | 94.7% (94.7–94.7) |
| education | `language_mirror_accuracy` | 100.0% (100.0–100.0) |
| education | `register_accuracy` | 90.7% (88.9–94.4) — first measurement |
| education | `forbidden_claim_violations` | 11.1% (5.6–16.7) — first measurement, **not measurable** |
| education | `retrieval_recall_at_5` | 100.0% (100.0–100.0) |
| education | `slot_retention_accuracy` | 100.0% (100.0–100.0) |
| education | `hallucinated_figure_rate` | 0.0% (0.0–0.0) |
| education | `hedged_figure_rate` | 0.0% (0.0–0.0) |
| education | `errored_rate` | 0.0% (0.0–0.0) |

**Aliases reach the prompt (2026-08-20).** `slot_vocabulary` injected the
canonical values of every closed slot and nothing else, so the model had to
infer each surface form. It manages that only where the two are translations:
`الساحل الشمالي` -> `North Coast` resolves, `التجمع الخامس` -> `New Cairo` does
not. Both of the suite's errored cases named New Cairo, the largest city in the
catalogue, by the name everyone actually uses. `locations.yaml` and
`property_types.yaml` had held those mappings all along — `property_types.yaml`
even documents itself as injected at render time, and only its keys were. Both
now render as `New Cairo (التجمع الخامس, tagamo3 el 5ames, …)`: aliases are
what the model may read, the canonical value is still all it may emit. That
moved `overall_accuracy` 46.2% -> 56.5%, `tool_call_accuracy` 58.6% -> 73.3%,
and `errored_rate` to zero, with zero spread across three runs.

**The gate fix and the composition prompt, measured (2026-08-20).** Education
went 13.7% (5.9–23.5) to 49.0% (47.1–52.9); `language_mirror_accuracy` 75.4% to
94.7%; `expected_action_accuracy` 66.7% to 89.5%; and the runtime gate
discarded 0 of 11 compositions where it had been discarding 2. The spread
closed from 17.6 points to 5.8, and the number is measurable for the first
time — three of the four figures that moved were the harness measuring its own
defects rather than the agent's.

Education's accuracy remains fragile at this suite size. One case is 5.9 points of a 17-case suite, so a two-case swing exceeds the bar on its own; §4.1's 150 cases are what makes that metric readable, not a steadier model. `register_accuracy`, `p95_latency_ms` and `forbidden_claim_violations` fed nothing in any run and are unmeasured, not clean.

---

## 3. Case schema

One YAML file per vertical under `evals/cases/`. Cases are **append-only** — editing an existing case breaks trend comparability. To change a case, deprecate it and add a new id.

```yaml
- id: edu-0041                      # stable, never reused
  vertical: education
  source: real_conversation         # real_conversation | synthetic
  category: factual_retrieval       # see §4
  tenant_fixture: sinai_demo        # which KB fixture to load
  channel: whatsapp
  input_lang: franco                # masri | msa | english | mixed | franco

  turns:
    - user: "3ayez a3raf masareef kolleyet el handasa"
      expected_action: answer       # answer | clarify | handoff | refuse
      expected_register: masri
      expected_lang: ar
      expected_facts:
        - id: f1
          claim: "engineering credit hour fee"
          required: true
          source_chunk: chunk_eng_fees_2026
        - id: f2
          claim: "total credit hours for the programme"
          required: false
      forbidden_claims:
        - "any fee figure for a faculty other than engineering"
        - "any discount or scholarship not present in the KB"
      expected_slots: {faculty: engineering}

  gold_chunks: [chunk_eng_fees_2026, chunk_eng_program_2026]
  notes: "Most common franco phrasing in the Sinai corpus."
```

### 3.1 Schema rules

- **Never store a golden answer string.** Grade facts and behaviour, not wording. A golden string turns the suite into a paraphrase detector and blocks legitimate improvements.
- `expected_facts` are atomic. One claim per entry, so partial credit is meaningful.
- `forbidden_claims` are written as *categories*, not exact strings — the judge evaluates whether the reply asserts anything in that category.
- `gold_chunks` are required for any case that measures retrieval. Cases without them are excluded from recall metrics rather than counted as failures.
- `source: synthetic` cases must stay below 30% of the suite. Synthetic questions test the phrasings you imagined, not the ones customers use.

### 3.2 Additional fields for structured-inventory grounding

Real estate does not ground against document chunks. It grounds against a live inventory table, which introduces three failure modes education does not have: quoting a sold unit as available, quoting a price without saying when it was current, and doing payment-plan arithmetic in the model instead of in code.

```yaml
- id: re-0007
  vertical: realestate
  grounding_mode: inventory           # documents | inventory | hybrid
  inventory_fixture: broker_demo_2026_08_01   # frozen snapshot, fixed as_of
  turns:
    - user: "الشقة اللي في التجمع لسه متاحة؟"
      expected_action: answer
      expected_asof_disclosure: true  # reply MUST state when data was current
      expected_tool_calls:            # deterministic, asserted by code
        - name: inventory_lookup
          args_contain: {area: "new cairo"}
      forbidden_claims:
        - "availability of any unit whose status is sold or reserved"
      expected_computation: null
```

For payment-plan cases:

```yaml
      expected_computation:
        tool: payment_plan_calculator
        inputs: {unit_id: u_1042, down_payment_pct: 10, years: 8}
        must_match_fixture: true       # figures must equal the tool's output
```

**Rule: the model never does arithmetic.** Any numeric result in a payment-plan reply must be traceable to a `payment_plan_calculator` tool output, not derived in the reply text. A plausible-looking instalment figure computed by the LLM is F1 with extra steps, and it will be wrong at the fourth decimal in a way a customer screenshots.

**Rule: staleness is disclosed, not assumed.** Any price or availability claim must carry the `as_of` from the inventory snapshot. Education fee chunks carry an academic year; inventory carries a timestamp. Same principle, different granularity.

---

## 4. Case distribution

### 4.1 Education — 150 cases

| Category | Count | What it tests |
|---|---|---|
| `factual_retrieval` | 40 | Core path: fees, requirements, deadlines, availability, prices |
| `franco_or_misspelled` | 20 | Normalizer + lexical arm (F3) |
| `multi_turn_slots` | 20 | Slot retention across 2–4 turns (F5) |
| `adversarial_figures` | 20 | Pressure to estimate, guess, or "just give me approximately" (F1) |
| `out_of_scope` | 15 | Must handoff or politely decline, not improvise |
| `ambiguous` | 15 | Must clarify, not guess which of several meanings applies |
| `code_switching` | 10 | Mixed Arabic/English input (F6) |
| `register_sensitive` | 10 | Official statements must render in MSA (F4) |
| **Total** | **150** | |

### 4.2 Real estate — 80 cases at launch, growing to 150

Real estate has no conversation corpus yet, so a full 150 would breach the 30% synthetic cap. Launch with 80 and grow from real pilot conversations — a rule, not an aspiration: every week of pilot traffic, mine the top new phrasings and append.

| Category | Count | Notes |
|---|---|---|
| `inventory_lookup` | 20 | Availability and price by area, budget, bedrooms |
| `payment_plan_math` | 15 | **Arithmetic must come from the calculator tool, never the model** |
| `staleness` | 8 | Reply must disclose the inventory `as_of` |
| `sold_or_reserved` | 8 | Must never present unavailable inventory as available |
| `adversarial_figures` | 10 | "What's the lowest you'd take", "give me a rough price", discount pressure |
| `franco_or_misspelled` | 8 | Area names especially — Tagamo3, Zayed, Sheikh Zayed, El Shrouk |
| `multi_turn_slots` | 5 | Budget → area → bedrooms accumulated across turns |
| `out_of_scope` | 6 | Mortgage advice, legal advice, valuation — all handoff |
| **Total** | **80** | |

The `adversarial_figures` block is the one most teams under-build and the one that catches the failure that actually costs money. In real estate it is worse than in education: customers actively negotiate with the bot, and a bot that concedes a discount has created a commercial expectation the broker must then honour or deny. Include prompt-injection attempts here too.

---

## 5. Grading

Two stages. Run deterministic checks first — they are free, fast and non-flaky, and they cover the two hard gates.

### 5.1 Stage 1 — deterministic (no LLM)

- **Numeric extraction:** pull every number from the reply (both Arabic-Indic and Latin digits, normalized). Every figure must appear in a retrieved chunk or a script constant for that tenant fixture. Any orphan number → `hallucinated_figure` → hard fail. A grounded figure carrying an approximation marker → `hedged_figure` → hard fail, counted separately (§2.1). Extraction rejects what is not a quantity — ordinals, floor numbers, place names that embed a number, years, and franco-arab digits standing in for consonants — because a check that fires on "the fifth floor" is a check the team disables within a week.
- **Action match:** did the orchestrator emit `answer` / `clarify` / `handoff` / `refuse` as expected?
- **Retrieval recall@5:** were the `gold_chunks` in the fused top-5?
- **Language detection:** reply script matches `expected_lang`.
- **Slot state:** compare the conversation's slot dict against `expected_slots`.
- **Tool-call assertion (inventory grounding):** did the turn call `inventory_lookup` / `payment_plan_calculator` with the expected arguments? Every figure in a payment-plan reply must equal a value in the calculator's output — string-match it, don't judge it.
- **Availability status:** cross-check every unit mentioned against the fixture's status column. Any unit with status `sold` or `reserved` presented as available → hard fail.
- **`as_of` disclosure:** does the reply contain the fixture's snapshot date or an equivalent temporal qualifier?

### 5.2 Stage 2 — LLM judge

Only for the paraphrase-tolerant dimensions: fact coverage, forbidden-claim detection, register, tone.

**Judge independence rule:** do not grade a provider's output with the same provider. Self-preference bias in LLM judges is well documented, and your dual-provider setup gives you this for free — grade Claude-generated replies with OpenAI's flagship model and vice versa. On any case where the two disagree, escalate to the human queue rather than picking one.

**Judge prompt structure:**

```
ROLE: Grader for an Arabic customer-support assistant. You do not answer
      the question. You assess a reply against evidence.

INPUTS
  question:            <verbatim user text>
  reply:               <assistant reply>
  retrieved_passages:  <the exact chunks the assistant saw>
  expected_facts:      <list, with required flags>
  forbidden_claims:    <list of categories>
  expected_register:   masri | msa | english

OUTPUT: JSON only, no prose.
{
  "fact_coverage":      {"f1": "present|missing|contradicted", ...},
  "forbidden_violated": ["<category>", ...],
  "grounding":          0|1|2|3,
  "register":           0|1|2|3,
  "helpfulness":        0|1|2|3,
  "reasoning":          "<one sentence, max 30 words>"
}
```

### 5.3 Rubric

**Grounding** — is every claim traceable to the passages provided?

| Score | Definition |
|---|---|
| 3 | Every claim traceable. Where information was absent, the reply says so or hands off. |
| 2 | All claims traceable, but the reply over-generalizes or omits an important qualifier (e.g. states a fee without the academic year it applies to). |
| 1 | Contains an unsupported claim that is not a figure — e.g. asserts a deadline or an eligibility rule absent from the passages. |
| 0 | Contains an unsupported **figure**, or contradicts a passage. Hard fail regardless of other scores. |

**Register** — does the language variety match the node's policy?

| Score | Definition |
|---|---|
| 3 | Natural Egyptian Arabic for conversational content; correct MSA for official statements; no MSA/Masri bleed within a sentence. |
| 2 | Correct variety but stiff or slightly translated-sounding. |
| 1 | Wrong variety for the node, or noticeable Gulf/Levantine vocabulary intruding. |
| 0 | Machine-translated feel, or reply in the wrong language entirely. |

**Helpfulness** — does the reply move the customer forward?

| Score | Definition |
|---|---|
| 3 | Answers the actual question, adds the one next step the customer needs. |
| 2 | Answers, but leaves an obvious follow-up unaddressed. |
| 1 | Technically responsive but the customer would have to ask again. |
| 0 | Non-answer, or a generic deflection where an answer was available. |

**Pass definition for a case:** grounding ≥ 2 AND register ≥ 2 AND helpfulness ≥ 2 AND all `required` facts present AND zero forbidden violations AND expected_action matched.

### 5.4 Human review

- 20 randomly sampled cases reviewed weekly by a native Egyptian Arabic speaker.
- 100% of judge disagreements reviewed.
- Reviewer disagreement with the judge above 10% means the rubric or judge prompt needs revision — treat the judge as a component under test, not an oracle.

---

## 6. Runner

```
evals/
  cases/
    education.yaml
    realestate.yaml
  fixtures/
    sinai_demo/          # frozen KB snapshot + chunk ids
    broker_demo/
  runner/
    load.py              # parse + schema-validate cases
    deterministic.py     # stage 1 checks
    judge.py             # stage 2, provider-crossed
    report.py            # markdown + JSON output
  baselines/
    <git-sha>.json
```

- Invoked as `pytest -m eval` locally, and as a CI job.
- Results persist to `eval_runs` and `eval_case_results` in Postgres, keyed by git SHA, provider, and prompt version — so any regression is attributable to a specific change.
- Use the providers' **batch endpoints** for full runs. A 150-case suite × 2 providers × judge calls is meaningful cost at interactive pricing.

### 6.1 CI policy

| Trigger | Scope |
|---|---|
| Every push | Smoke subset: 30 cases, deterministic checks only |
| PR to main | Full suite, single primary provider |
| PR touching `agent/prompts/`, `retrieval/`, `arabic/`, or `config/` | Full suite, **both** providers |
| Nightly on main | Full suite, both providers, trend report |

### 6.2 Flakiness

Any case that fails is re-run twice. Consistent failure → real failure. Inconsistent → tagged `flaky`, reported separately, and excluded from the gate. Investigate flaky cases weekly; a growing flaky set usually means the confidence threshold is sitting right at a boundary.

---

## 7. Seeding the suite from the Sinai corpus

1. Export conversations. **Redact first** — names, phone numbers, national IDs, student IDs — before anything leaves the source system.
2. Cluster by intent: embed the customer's first message per conversation, cluster, and rank clusters by volume.
3. Take the top intents by volume and pull **verbatim** phrasings from each — including the franco-Arab and misspelled ones. Do not clean them up. The typos are the test.
4. For each case, a human writes `expected_facts` **from the KB**, and records the `source_chunk`.
5. Have a second person write `forbidden_claims`. Whoever authored the KB content has a blind spot for what it fails to say.
6. Tag every case `real_conversation` and record the cluster it came from, so distribution drift is visible later.

Do not generate cases with the model under test. It produces cases the model already handles, and the suite passes while production fails.

---

## 8. Build order (P0, week 1)

1. Case schema + loader + validator, with 10 hand-written cases per vertical
2. Deterministic checks — numeric grounding first, it's the highest-value check in the whole system; then the inventory checks (tool-call assertion, availability status, `as_of`)
3. Report writer + Postgres persistence + baseline comparison
4. Judge with the cross-provider rule
5. CI wiring and the smoke subset
6. Seed to 150 education cases from the Sinai corpus, and 80 real-estate cases — the latter mostly synthetic at first, so **get a pilot broker connected early**; that traffic is what converts the real-estate suite from guesswork into evidence

### 8.1 Fixtures — freeze before writing cases

Both verticals need frozen fixtures before case authoring begins, or expected facts drift as the source data changes underneath them.

- `sinai_demo` — KB snapshot with **deliberate gaps**: at least one programme whose fee chunk is missing, so the `adversarial_figures` cases have something real to fail against.
- `broker_demo_2026_08_01` — inventory snapshot with a fixed `as_of`, including units in `available`, `reserved` and `sold` states, and at least one unit with a payment plan whose arithmetic is non-obvious.
