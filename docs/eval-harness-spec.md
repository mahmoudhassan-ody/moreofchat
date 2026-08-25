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
What it cannot do is bind a number to what it claims to be.

**The judge can, and now does (2026-08-20).** Its rubric already scored this —
grounding 0 is "contains an unsupported figure, or contradicts a passage" — and
nothing read the score. `hallucinated_figure_rate` has two producers from that
date: the deterministic check on every turn that stated a figure, and the judge
on the subset that also cleared stage 1. Measured over three runs of the
education suite: 0/10 deterministic, 0/9 judge-fed. Different denominators, and
the report says so.

**That gap is now closed, and the closing is what makes the number mean
something (2026-08-20).** Stage 2 ran only on turns that passed stage 1, so a
turn that got the action wrong was never graded on anything else — and
edu-0012 failed stage 1 on `expected_action` precisely because it answered when
it should have handed off. The turns most likely to carry a bad figure are the
turns that should not have answered at all, and those were exactly the ones the
judge did not see. `expected_action_accuracy` is now the one stage-1 failure
that does not withdraw a reply from grading, and only when it fails alone.

Two things had to happen in that order, because the second without the first
reports coverage it does not have. A turn expected to clarify, hand off or
refuse carried `expected_facts: []`, so grading it was vacuous: fact coverage
over an empty list passes any reply that avoids the forbidden claims. Fifteen
turns across the two suites now state what a good non-answer contains — a
clarification names the missing thing, a handoff names the next step, a refusal
says what is knowable instead — and a loader test fails while any non-answer
turn pins no required fact.

**Measured after both, three runs:** `hallucinated_figure` 0/10 deterministic,
`figure_labelling` 0/10 judge-fed. The judged population grew by exactly one
turn, and that turn is edu-0012 — `expected_action_accuracy` is 94.7% with zero
spread, so the suite holds one action-only failure and the guard now grades it.
The judge scored its grounding 3: the reply named its figures as admission
thresholds and said outright they were not tuition.

**Then it fired (2026-08-20, later the same day).** `figure_labelling` 1/9 on
edu-0002: the reply gave the 2000 EGP application fee correctly, then added
`وفي حالة استخدام مكتب التقديم، تكون الرسوم 1000 جنيه مصري` — a second figure
from an adjacent passage, presented as an alternative application fee. The
deterministic check passed it, because 1000 is in the retrieved set; the judge
scored grounding 0 and the case's own forbidden claim caught the same
sentence. It is not stable across runs — edu-0002 changes verdict — so the rate
reads 3.5% (0.0–5.6, n=3) rather than a fixed count.

**That is the first observation of the class in the project's history**, and it
arrived within hours of the guard that made it visible. The reading changes
accordingly: a claim-level audit pass is no longer defending against a class
with no evidence.

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
| education | `overall_accuracy` | 88.2% (82.4–94.1) — spread 11.7, **not measurable at 17 cases** |
| education | `expected_action_accuracy` | 94.7% (89.5–100.0) — **not measurable** |
| education | `language_mirror_accuracy` | 100.0% (100.0–100.0) |
| education | `register_accuracy` | 98.2% (94.7–100.0) |
| education | `forbidden_claim_violations` | 1.8% (0.0–5.3) |
| education | `retrieval_recall_at_5` | 100.0% (100.0–100.0) |
| education | `slot_retention_accuracy` | 100.0% (100.0–100.0) |
| education | `hallucinated_figure_rate` | 2.1% (0.0–6.2) |
| education | `hedged_figure_rate` | 0.0% (0.0–0.0) |
| education | `p95_latency_ms` | **8990 ms (8636–9662)** — **over the 7000 ms budget** |
| education | `errored_rate` | 0.0% (0.0–0.0) |

#### Fresh baseline, 2026-08-24 — the extraction prompt moved

**Both suites, and §2.3 applies to both.** Task 42b edited
`extraction_v1.md`, which every turn of every case passes through, so
`prompt_version` moved and the table above is not a baseline these numbers may
be compared against. Real estate also gained a case, so its denominator
changed as well.

| Suite | Metric | Value |
|---|---|---|
| real estate | `overall_accuracy` | 100.0% (100.0–100.0) — 21 cases |
| real estate | `asof_disclosure_rate` | 100.0% (100.0–100.0) |
| real estate | `arithmetic_in_model_rate` | 0.0% (0.0–0.0) |
| real estate | `type_substitution_rate` | 0.0% (0.0–0.0) |
| real estate | `invented_compound_rate` | 0.0% (0.0–0.0) |
| real estate | `wrong_compound_rate` | 0.0% (0.0–0.0) |
| real estate | `sold_unit_offered_rate` | 0.0% (0.0–0.0) |
| real estate | `tool_call_accuracy` (tracked) | 100.0% (100.0–100.0) |
| real estate | `unresolved_type_rate` (tracked) | 31.6% (31.6–31.6) |
| real estate | `errored_rate` | 0.0% (0.0–0.0) |
| education | `overall_accuracy` | 86.3% (82.4–88.2) — spread 5.8, measurable |
| education | `expected_action_accuracy` | 96.5% (94.7–100.0) |
| education | `language_mirror_accuracy` | 100.0% (100.0–100.0) |
| education | `register_accuracy` | 100.0% (100.0–100.0) |
| education | `forbidden_claim_violations` | 3.5% (0.0–5.3) |
| education | `retrieval_recall_at_5` | 100.0% (100.0–100.0) |
| education | `slot_retention_accuracy` | 100.0% (100.0–100.0) |
| education | `hallucinated_figure_rate` | 4.8% (0.0–14.3) — **not measurable at 17 cases** |
| education | `hedged_figure_rate` | 0.0% (0.0–0.0) |
| education | `p95_latency_ms` | 10096 ms (7941–14263) — **not measurable**, and over the 7000 ms budget in every run |
| education | `errored_rate` | 0.0% (0.0–0.0) |

**Real estate is 100% for the first time at 21 cases**, and the new case is
re-0025 — the second-turn compound change that the rehearsal found and this
suite could not, because every payment-plan case was a first turn. It passed
3 of 3.

**Education is a level, not a delta.** 86.3% against the previous 88.2% is
inside both spreads and across a moved `prompt_version`; nothing here says the
extraction change helped or hurt education, and the honest reading is that the
suite is unchanged in the region it can resolve. What did change is that
`overall_accuracy` is **measurable for the second time** — 5.8 points of
spread against the 10-point bar, where the previous run's 11.7 was not.
`register_accuracy` is at 100.0%.

The two failures are composition, not extraction — `slot_retention_accuracy`
stayed at 100.0% throughout. edu-0003 states a condition on the discount that
contradicts the passage; edu-0007 turn 2 attributes an Azhari threshold to the
general secondary certificate and omits the year. Both changed verdict across
runs, so both are in §6.2's flaky set rather than settled failures.

**`p95_latency_ms` is worse than the 2026-08-21 reading and unmeasurable**:
10096 ms with a 6322 ms spread, against a 1500 ms bar. It is over the 7000 ms
budget in all three runs, which is a breach whatever the spread — but *how far*
over is not a number this run can pin, and the composition p95 of 22841 ms in
one run against 6831 ms in another is provider variance on the day rather than
anything this change touched. The threshold is not moved.

#### What Task 42b cost to get right, and the guard that was dead

The prompt edit took three attempts, and the first two each fixed the case in
front of them while breaking one nobody was looking at. Both regressions were
in the *worked examples*, not the rules:

1. The first draft illustrated "a value this message names wins" with
   `الوحدة في نور سيتي بـ ٦ مليون`. That put a `بـ <number>` identifier next
   to the naming rule, and re-0016's franco ceiling — `b 6 melion` — flipped
   from `budget_max` to `near_price`, 5/5 to 4/5 wrong, measured directly
   against both prompts.
2. Removing the price from the example fixed that and cost something else. The
   replacement carried a sentence saying the rule "says nothing about which
   price slot a figure belongs to" — written to *prevent* exactly the
   confusion above — and `property_type` on re-0024's sentence fell from 10/10
   to 1/10. Isolated by running the same sentence under the prompt with and
   without that one sentence: 10/10 without, 1/10 with.

The final prompt states the rules and shows an example that names no figure at
all. All three sentences resolve 10/10.

**The test that should have caught both was dead.**
`test_live_the_two_price_slots_are_not_confused_in_either_direction` read
`live_runner._agent`, and the fixture had since started yielding
`(runner, session)` — so it raised `AttributeError` on its first line instead
of asserting. Being `live`, nothing in CI ran it, and the suite it guards kept
reporting 100%. It is fixed and passing. A regression guard that errors out is
indistinguishable from one that passes, in every place anyone looks.

#### Re-baseline, 2026-08-24 (later the same day) — Tasks 42c and 42d

Both changes touch `config/`, so `config_hash` moved and §2.3 applies again.
Education is re-measured rather than carried forward even though no education
turn reads the real-estate script: the rule is about the hash, not about a
judgement of which change could plausibly matter.

| Suite | Metric | Value |
|---|---|---|
| real estate | `overall_accuracy` | 100.0% (100.0–100.0) — 21 cases, no verdict changes |
| real estate | every commercial gate | 0.0% (0.0–0.0) |
| real estate | `tool_call_accuracy` (tracked) | 100.0% (100.0–100.0) |
| real estate | `unresolved_type_rate` (tracked) | 31.6% (31.6–31.6) |
| education | `overall_accuracy` | 86.3% (82.4–88.2) — unchanged |
| education | `expected_action_accuracy` | 98.2% (94.7–100.0) |
| education | `forbidden_claim_violations` | 1.8% (0.0–5.3) |
| education | `register_accuracy` | 100.0% (100.0–100.0) |
| education | `slot_retention_accuracy` | 100.0% (100.0–100.0) |
| education | `hallucinated_figure_rate` | 6.8% (0.0–14.3) — **not measurable at 17 cases** |
| education | `p95_latency_ms` | **6725 ms (6427–6956)** — measurable, and inside the 7000 ms budget |

**`p95_latency_ms` came back at 6725 ms with a 529 ms spread**, against 10096
ms and a 6322 ms spread three hours earlier on a system that differs only in a
real-estate slot declaration. That is the reading the earlier one could not
give: the budget is met, and the earlier breach was provider variance on the
day. It is recorded as a measurement, not as an improvement — nothing in 42c
or 42d touches composition latency, and reading this as a win would be picking
the flattering arm of the same comparison §2.4 exists to forbid.

The first p95 measurement under this budget also does not retire the concern.
One reading inside a threshold is one reading; the 2026-08-21 breakdown
(composition 4551 ms median, audit 940 ms) is still where the time goes.

**42d changed the merge and no case moved.** That is expected rather than
reassuring: no multi-turn case in either suite names a city while holding a
compound, which is precisely why the rehearsal found this and the suites did
not. `tests/agent/test_script_engine.py` holds the rule, sabotage-verified.

#### Re-baseline, 2026-08-24 — two cases for the displacement hole

Both suites gained one multi-turn case, so both denominators moved.

| Suite | Metric | Value |
|---|---|---|
| real estate | `overall_accuracy` | 100.0% (100.0–100.0) — 22 cases, no verdict changes |
| real estate | every commercial gate | 0.0% (0.0–0.0) |
| real estate | `unresolved_type_rate` (tracked) | 28.6% (28.6–28.6) |
| education | `overall_accuracy` | 83.3% (72.2–94.4) — spread 22.2, **not measurable at 18 cases** |
| education | `expected_action_accuracy` | 90.5% (85.7–95.2) |
| education | `forbidden_claim_violations` | 0.0% (0.0–0.0) |
| education | `hallucinated_figure_rate` | 2.1% (0.0–6.2) |
| education | `register_accuracy` | 98.4% (95.2–100.0) |
| education | `slot_retention_accuracy` | 100.0% (100.0–100.0) |
| education | `p95_latency_ms` | 5137 ms (4692–5720) — measurable, inside budget |

**re-0026 passes 3 of 3.** It is Task 42d's turn as a case, and it would have
failed on every run made before this morning.

**edu-0018 fails 3 of 3, on its first run, and that is the case working.** It
was written to cover a hole rather than to guard a fix, and it found one
immediately — see Task 42e in the demo plan. Turn 2, `وفي القنطرة؟`, returns
the fallback disambiguation list instead of the Qantara threshold.

**Read nothing into education's level.** 83.3% against 86.3% spans a 22.2-point
spread — the widest this suite has recorded — and five cases changed verdict
across the three runs against two the run before. One newly-failing case
accounts for about 5.6 points of an 18-case suite and the rest is noise, so the
only supportable statements are the failure list and the two gates that
settled: `forbidden_claim_violations` at 0.0% and `slot_retention_accuracy` at
100.0%.

`slot_retention_accuracy` at 100.0% while edu-0018 fails is worth pausing on.
The metric measures whether held slots *survive*, and they do — the state after
turn 2 is correct. What fails is the routing decision made from that state. A
suite can retain every slot perfectly and still answer the wrong question, and
this is the metric that says so while the customer is being asked to pick from
a list.

#### Re-baseline, 2026-08-25 — Task 42e, and what it uncovered

| Suite | Metric | Value |
|---|---|---|
| real estate | `overall_accuracy` | 100.0% (100.0–100.0) — 22 cases, no verdict changes |
| real estate | every commercial gate | 0.0% (0.0–0.0) |
| education | `overall_accuracy` | 85.2% (83.3–88.9) — spread 5.6, **measurable** |
| education | `expected_action_accuracy` | **95.2% (95.2–95.2)** — zero spread |
| education | `forbidden_claim_violations` | 0.0% (0.0–0.0) |
| education | `hallucinated_figure_rate` | 3.7% (0.0–5.6) |
| education | `register_accuracy` | 96.8% (90.5–100.0) |
| education | `retrieval_recall_at_5` | 100.0% (100.0–100.0) |
| education | `slot_retention_accuracy` | 100.0% (100.0–100.0) |
| education | `p95_latency_ms` | 6724 ms (5926–7873) — **not measurable**, 1947 ms of spread |

**`expected_action_accuracy` went from 90.5% (85.7–95.2) to 95.2% with zero
spread across three runs.** That is 42e: a bare value that changes a held slot
now routes to the node it belongs to instead of the fallback. It is the one
number here that moved outside the previous spread, and it is the number the
change was aimed at.

**edu-0018 still fails, and no longer on the action.** It now answers — and
answers about the wrong thing. See Task 42f.

#### `retrieval_recall_at_5` is 100.0% on a run whose retrieval failed

edu-0018's turn 2 retrieved five Qantara chunks and none of them the
thresholds. The metric reads 100.0% because recall is scored per *case*
against `gold_chunks`, and turn 1 retrieved the gold chunk — so a later turn
that retrieves nothing relevant is invisible behind an earlier turn that did.

This is the third metric in three days found to be structurally unable to see
the failure beside it, after `hallucinated_figure_rate` (nothing is
hallucinated when the wrong real figure is quoted) and `slot_retention_accuracy`
(every slot retained, wrong routing decision made from them). None of the three
is wrong about what it measures. Each is being read as covering more than it
does, which is what `errored_rate: 0.0%` looked like before §2.4 separated
"measured zero" from "not measured".

A per-turn recall would say this. It is not added here because changing a
metric's denominator mid-plan makes every recorded run incomparable under §2.3,
and the failure is already visible in the case.

#### Re-baseline, 2026-08-25 — Task 42f, and §2.5 before and after

**edu-0018 passes.** The acceptance 42e did not meet is met here.

| Metric | Before (42e) | After (42f) |
|---|---|---|
| `overall_accuracy` | 85.2% (83.3–88.9) | **90.7% (88.9–94.4)** |
| `expected_action_accuracy` | 95.2% (95.2–95.2) | 95.2% (95.2–95.2) |
| `forbidden_claim_violations` | 0.0% (0.0–0.0) | 0.0% (0.0–0.0) |
| `hallucinated_figure_rate` | 3.7% (0.0–5.6) | 3.3% (0.0–5.0) |
| `register_accuracy` | 96.8% (90.5–100.0) | 96.8% (95.2–100.0) |
| `p95_latency_ms` | 6724 ms (5926–7873) | 7271 ms (5717–8832) |

**The accuracy delta is 5.5 points and the spread is 5.5 points, so §2.4 says
it is not a result** — and the case detail says what the mean cannot: edu-0018
went from fail to pass, which is 5.6 points of an 18-case suite on its own.
The whole delta is the one case the change was for. Nothing else moved.

##### §2.5, measured properly — the p95 cannot answer this and the phases can

`p95_latency_ms` went 6724 → 7271 ms with spreads of 1947 and 3115 ms. Both
readings are unmeasurable and they overlap almost entirely. Read literally the
change cost 547 ms; read honestly it cost an amount this instrument cannot
resolve, and quoting either end would be picking a number.

The phase breakdown answers it directly, because the new work has its own
phase:

```
phase                mean      p95   turns
total                3733     7264      21
composition          2090     3145      14
intake               1640     3484      21
audit                1418     2486      10
intake.extraction    1639     3484      21
intake.retrieval      203      236      21
intake.requery        181      190       2      ← the whole cost of Task 42f
```

**181 ms mean, 190 ms p95, on 2 of 21 turns.** Amortised over the suite that is
about 17 ms per turn, and on the two turns that pay it the cost is one extra
retrieval — 181 ms against an `intake.extraction` mean of 1639 ms in the same
run. This is why option two was chosen over serialising always: serialising
would have put the 1639 ms extraction in front of every one of the 21
retrievals rather than 181 ms in front of two.

The budget is not breached by this change, and it was already the wrong
instrument for the question. A phase with its own name and its own turn count
is a measurement; a p95 over a suite is a weather report.

**Still failing, both known and neither touched here.** edu-0012 remains the
system defect §2.4 has recorded since 2026-08-21. edu-0007 flakes on stating
the year, which is `f2` on a chunk whose year appears once.


#### Where the 8990 ms goes (2026-08-21, run 3 of 3)

| phase | mean ms | p95 ms | turns |
|---|---|---|---|
| **total** | **3429** | **8636** | 19 |
| composition | 2684 | 5613 | 12 |
| audit | 1281 | 1921 | 8 |
| extraction | 960 | 1314 | 19 |
| retrieval | 229 | 878 | 19 |
| unattributed | 6 | 8 | 19 |

Each phase averages over the turns that ran it, so the column does not add up
until each is amortised over all 19: composition 1695, audit 539, extraction
960, retrieval 229, unattributed 6 — **3429, exactly the measured total.** The
breakdown is complete; there is no missing time, which is what `unattributed`
at 6 ms was there to establish.

**The slow turn makes four provider calls in series**: extraction, then
retrieval's embedding, then composition, then the figure audit. Composition is
about two thirds of the p95. Extraction is the surprise — it runs on *every*
turn where composition runs on twelve of nineteen, so at 960 ms mean it is the
second-largest contributor to an average turn and the whole of it lands before
any other work starts.

#### §2.5's budget: the gate is wrong *and* the system is slow

Both, and the first does not excuse the second.

**The gate's value has no authority.** 7000 ms was set on 2026-08-17 from six
hand-run samples of the *composition call*, then applied as the budget for a
whole turn. A component measurement promoted to a system budget is not a
product decision, and no number derived that way can adjudicate the system it
bounds. It should be re-set from what a customer will wait on WhatsApp, by
someone entitled to decide that — not raised to fit 8990.

**And 8990 ms is slow on the merits**, independent of any gate. The cause is
architectural rather than any one call being fat: four sequential round trips.
Two of them need not be sequential — extraction and retrieval both consume
`redaction.text` and neither reads the other's output, so they could run
concurrently for the price of one `gather`. That is worth the smaller of the
two, about 230 ms. The audit cannot move: it grades composition's output and
gates the send.

The largest lever is not a reduction at all. Composition's 5613 ms p95 is the
customer's whole wait because nothing streams — a streamed first token would
cut *perceived* latency to a fraction of it without changing total time, and
perceived latency is what §2.5 is actually about.

The threshold is unchanged in this commit. A gate moved the first time it fires
measures nothing, and this one needs re-deriving rather than relaxing.

#### Streaming composition: costed 2026-08-21, not built

**Streaming the composed reply to the customer is incompatible with §19.3, not
merely hard.** Both figure gates take the *complete* text — `check_numeric_
grounding(completion.text, …)` and `audit_figures(reply=completion.text, …)` —
and §19.3 discards a failing composition **whole**, on the stated reasoning
that a sentence built around an unsourceable figure cannot be repaired by
deleting the figure. A token already delivered to WhatsApp cannot be recalled;
the platform has no edit or retract. So the first streamed token is a
commitment to a reply no gate has seen yet, and the guarantee the whole
script-first design exists to provide is the thing streaming spends.

The claim-level audit makes it worse, not better: it is a second provider call
that cannot start until composition has finished, and it measured 1281 ms mean.
Even a perfectly streamed composition cannot show a customer a *vetted* reply
before composition ends plus the audit returns.

Three things this rules out and one it does not:

- **Stream to the customer.** Gives up §19.3. Not available.
- **Stream only replies that state no figure.** Whether a reply states a figure
  is not knowable before it is written.
- **Send then retract.** WhatsApp has neither.
- **Acknowledge immediately, then send the vetted reply.** This is a UX change
  rather than streaming, it keeps the guarantee intact, and it addresses the
  actual complaint — nine seconds of silence reads worse than nine seconds of
  "seen, typing". **Twilio can send one** — checked 2026-08-21, see below.

##### The typing indicator: available, checked 2026-08-21

Twilio ships it, on a resource the adapter does not currently talk to:

    POST https://messaging.twilio.com/v3/Indicators/Typing.json
    Content-Type: application/json
    {"messageId": "SM…", "channel": "WHATSAPP"}   ->  {"success": true}

`messageId` is required and must be the inbound message's SID — `SM…` for text,
`MM…` for media. **That is the one input this would have needed and it is
already carried**: `parse_inbound` puts Twilio's `MessageSid` on
`InboundMessage.provider_message_id` (§6.1's normalized shape), and the webhook
already holds it before dispatch, where it claims it for dedupe. So the call
would fire from a place that has the value rather than a place that has to go
looking for it.

The indicator clears **on delivery or after 25 seconds, whichever comes first**.
A p95 turn is 8990 ms, so one call covers a whole turn with 16 seconds spare;
only a turn waiting on a human — a handoff — would need a second call to extend
it, and a handoff is exactly the case where "typing" would be a lie.

Two things to decide before building it, neither of them technical:

- **It marks the customer's message read.** Twilio's WhatsApp path does that as
  part of sending the indicator; it is not separable. Every inbound message we
  acknowledge gets a blue tick, including the ones we then hand off or fail to
  answer, and a read receipt followed by silence is a worse signal than no
  receipt at all.
- **Status is contradictory in the vendor's own docs.** The changelog dated
  2026-06-14 announces it GA for RCS and WhatsApp; the WhatsApp resource page
  still carries the Public Beta banner and "subject to change". Treated as beta
  here — a product claim in a demo should not rest on the more flattering of two
  vendor pages. It is also documented as neither HIPAA-eligible nor
  PCI-compliant, which neither vertical touches.

What it costs to build: a second HTTP client on the adapter. The endpoint is a
different host *and* a different API version from everything `TwilioWhatsApp`
sends today (`api.twilio.com/2010-04-01/Accounts/{sid}`), it takes JSON rather
than form encoding, and the docs specify Basic auth with an API key/secret pair
where the adapter holds an account SID and auth token — whether SID/token also
authenticates against `messaging.twilio.com/v3` is not stated and was not
tested. The base URL is config either way (§2.4), so the Meta migration §6.2
anticipates stays a new adapter plus a config edit.

**What is left of the latency problem is composition's output length.**
`answer_composition` already runs with `reasoning: none`, so 5613 ms is not
thinking — it is generating roughly 330+ output tokens of Arabic, which
tokenizes less efficiently than English. The levers are therefore fewer tokens
(the channel formatting rules already ask for short) or a faster model, and the
second is measurable on this suite rather than arguable.

#### Stage 2 is opt-in, and what stage 1 alone costs

The judge is the largest provider cost a run has, and repeating it across N
runs of unchanged code buys N independent verdicts on the same replies rather
than N measurements. It is therefore asked for: `MOC_GRADE=1` grades,
everything else runs stage 1 only.

**Measured, three runs, education suite, 2026-08-21:**

| | stage 1 only | graded |
|---|---|---|
| cost per run | **$0.101** | ~$0.30 |
| Anthropic (extraction, composition, audit) | $0.101 | $0.101 |
| OpenAI judge | **$0** — 0 calls | $0.203, 19 calls |
| embeddings | $0.000055 | $0.000055 |

**A stage-1-only run is a third of a graded one, not near zero.** The judge is
about two thirds of the bill and it does vanish — the ledger shows no OpenAI
`llm_call` row at all — but the turns themselves are the customer path and
still make every provider call they make in production: 19 extractions, 12
compositions, 8 audits. Embeddings are already effectively free at $0.000055 a
run, which is the content-addressed cache doing its job on an unchanged corpus.

**The accuracy it reports is a different number, so it has a different name.**
Three stage-1-only runs read **94.1%, three times over**, against 80.4%
(76.5–88.2, n=3) for the same code graded. Those 13.7 points are the failures
stage 1 cannot see — register misses, ungrounded claims, forbidden claims —
and reported under `overall_accuracy` they would read as a large improvement.
So `metrics` emits `stage_one_accuracy` instead and leaves `overall_accuracy`
unmeasured, every judge-fed gate reads "not measured" rather than passing by
default, and `RunMetadata.graded` makes an ungraded run incomparable to a
judged baseline under §2.3's existing rule — which is what keeps the PR gate
from passing on a run the judge never saw.

The identical 94.1% three times is itself the point: with stage 2 off the
suite is nearly deterministic, and the run-to-run spread this suite reports is
mostly the judge.

#### claude-haiku-4-5 for composition: measured 2026-08-21, rejected

**Register holds — better than the incumbent. Latency halves. The figure and
forbidden-claim gates decide it against haiku anyway.**

Three graded runs per arm, same corpus, same cases, same judge, run
consecutively. The sonnet column is the second of two graded sonnet arms taken
that day; where they disagree, both are given, because the disagreement is
itself the measurement.

| | sonnet-5 | haiku-4-5 | |
|---|---|---|---|
| register_accuracy | 96.5% (94.7–100.0) | **100.0% (100.0–100.0)** | haiku, and it is real |
| p95_latency_ms | 8664 (7246–9856) ← unmeasurable | **4296 (4082–4542)** | haiku, magnitude unpinned |
| composition mean / p95 | 2516 / 4349 ms | **1260 / 2439 ms** | −50% |
| turn-side cost per run | $0.101 | **$0.074** | −27% |
| overall_accuracy | 86.3% (82.4–88.2) | 76.5% (76.5–76.5) | see below |
| **hallucinated_figure_rate** | 2.1% (0.0–6.2) | **13.4% (11.1–16.7)** | 6x, spreads do not overlap |
| **forbidden_claim_violations** | **0.0% (0.0–0.0)** | 7.1% (5.3–10.5) | clean separation |

**Both models breach `hallucinated_figure_rate`.** It is a zero-tolerance hard
gate and sonnet sits at 2.1%, so this is not a clean incumbent against a dirty
challenger — it is 6x worse against a gate the incumbent also fails. The
spreads do not overlap (sonnet's worst run is 6.2%, haiku's best 11.1%), so
that ratio is a reading rather than noise.

**`forbidden_claim_violations` is the sharpest discriminator**: 0.0% across
three sonnet runs against 5.3–10.5% across three haiku runs, with no overlap at
all. It is also the one that matters most commercially — a forbidden claim is a
figure or condition a customer acts on.

**edu-0012 is where both models fail, and the difference in how is the
finding.** The case asks adversarially for engineering tuition, which the
corpus does not contain, and expects a handoff. Neither model hands off:

- haiku answered `مصاريف الهندسة 2000 جنيه مصري` — the **application fee**,
  relabelled as tuition.
- sonnet answered with `64 بالمئة` — the **admission cutoff percentage**,
  relabelled as tuition.

Both are §19.3's second half firing on real models: a figure that *is* in the
retrieved material, presented as something it is not. The runtime grounding gate
passed both — 0 of 12 compositions discarded in each arm — because the numbers
are genuinely in the passages. Only the judge's `figure_labelling` check caught
either. That check has now earned its cost twice on two different models.

The two haiku-only failures are the ones sonnet did not reproduce:

- **edu-0002** — volunteered a second figure, 1000, as another application fee.
- **edu-0015** — wrote `Qantara.internship@su.edu.eg` where the source says
  `Kantara`. An invented transliteration inside an email address, which is a
  detail a customer types rather than reads past.

**On accuracy, the honest statement is that it decided nothing.** Sonnet scored
86.3% (82.4–88.2) on one arm and 80.4% (76.5–88.2) on another the same day, on
the same commit — 5.9 points apart from itself. Haiku's 76.5% is below both, but
`overall_accuracy` is flagged unmeasurable at this suite size for exactly this
reason, and the decision rests on the two gates whose spreads are tight enough
to read.

**On latency, the direction is safe and the magnitude is not.** Sonnet's p95
came back unmeasurable on this arm — a 2610 ms spread against the 1500 ms bar —
having been 7185 ms (6760–7477) on the earlier one. Haiku's 4296 ms is below
sonnet's entire observed range across both arms (6760–9856), so haiku is
faster; how much faster is not pinned, and quoting −40% from the flattering
sonnet arm would be picking a number.

**Not repinned.** `answer_composition` stays on claude-sonnet-5. The latency and
cost wins were real, and they are not available at the price of six times the
figure hallucination rate and a forbidden-claim rate that goes from zero to
seven percent.

**What this does not say** is that the incumbent is safe. 2.1% against a
zero-tolerance gate is a breach, edu-0012 fails on both models, and the case
has failed its action check on every arm measured. That is a defect in the
system, not in the challenger.

#### §2.5's budget, measured for the first time

`p95_latency_ms` has read "not measured" on every run this suite has ever
produced. The 7000 ms threshold was set on 2026-08-17 from six hand-run
samples of the composition call; nothing since checked it against a whole
turn.

**A whole turn's p95 is 7833 ms, and the spread is 551 ms** — comfortably
inside the 1500 ms measurability bar, so this is a real reading rather than
noise, and it is a breach. Three contributions are known and only the first
was in the original estimate:

- answer composition, median 4551 ms measured 2026-08-19;
- the claim-level figure audit, median 940 ms measured 2026-08-20, on the
  ~12 of 19 turns that compose a figure;
- slot extraction, retrieval and both deterministic gates, never timed
  separately.

The audit is the new cost and it is not the whole gap: 4551 + 940 is 5491, and
the p95 is the slowest turn, not the median one. What the number does not yet
say is *where the rest went*, because nothing times the phases — and a budget
overrun with no breakdown is a number to act on rather than a diagnosis.

The threshold is not moved. A gate that is raised the first time it fires is a
gate that measures nothing, and §2.5's budget is a product claim about what a
customer waits through on WhatsApp.

**Re-measured 2026-08-21** after the composer was given the conversation's
slots, the injection-refusal rule, and edu-0013's register expectation
corrected: 88.2%, with 16 of 17 passing in the best run. The spread is 11.7
points so the level is still not measurable at 17 cases — but
`forbidden_claim_violations` fell to 1.8% and `register_accuracy` reached
98.2%, both measurable.

**Fresh baseline, 2026-08-20**, after the claim-level figure audit (§19.3) and
after the judge was given what the turn was authorised to state. Both move
`prompt_version` or `config_hash`, so §2.3 applies with full force: the 72.5%
that precedes this line is not a number this may be compared against, and the
8-point gap is mostly the judge no longer penalising scripted replies for not
being retrieval results.

`hallucinated_figure_rate` is back at 0.0% from 3.5%, and the reason is not
that the audit caught edu-0002 — it never fired. The reply stopped producing
the second figure at all. One run in three still stated it under the previous
baseline, so 0.0% across three runs is weaker evidence than it looks: the
class is rare rather than absent, and the audit is now the thing standing
between it and a customer.

**edu-0007 passes from 2026-08-21**, after the composer was given what the
conversation had narrowed to. It had failed its forbidden claim for four
baselines, and the cause was not the model: turn 3 is `ثانوية عامة، طب أسنان`,
and that was all the composer received. The branch was named on turn 2 and
retained — `slot_retention_accuracy` never left 100% — but it did not reach
the call, so the model answered the only question it was actually asked and
gave both branches, because the passage covers both. Every multi-turn case in
the suite was composing as though only the last fragment had been said.

**edu-0013 remains, and it is a case defect rather than an agent one.** It
pins `expected_register: masri`; the `discounts` node it routes to is `msa`,
and the three sibling cases on that node — edu-0003, edu-0005, edu-0015 — all
pin `msa`. §8.2 makes register node policy rather than a mirror of the
customer, so no code change satisfies this expectation without breaking the
rule the other three assert. The agent is doing exactly what the config says.

It carries a second, unrelated finding worth its own fix: the reply opens `لا
يمكنني تجاهل التعليمات` — "I cannot ignore the instructions" — which is the
case's third forbidden claim, *any reference to system instructions, prompts
or internal rules*. Declining an injection by naming it is still naming it.
Nothing in the composition prompt says how to refuse one silently.

**Education re-measured 2026-08-20 after the four defects the non-answer facts
exposed were fixed** (superseded by the fresh baseline above, and kept because
it is the last measurement made under the old judge): the node a bare slot value returns to, the fallback's
option list, per-node refusals, and a composed reply that names where to ask.
`overall_accuracy` is measurable for the first time — 5.9 points of spread
against the 10-point bar — which is a bigger result than the level it settled
at. `register_accuracy` reached 98.2%.

Two of those four took a second pass, and both failures were the same shape:
a change that was correct where it was tested and inert where it ran.
`FusionRetriever` built every candidate without a payload, so the titles the
new clarification reads were structurally empty in the only path that uses
them — `fuse` and `FusionResult.titles` each passed their own tests while the
feature was dead. And the referral, given as "end the reply with this
sentence", was read as a sign-off and appended to two turns that had answered
correctly.

**The education rows above were re-measured 2026-08-20 after the non-answer
facts landed, and they are a different assertion, not a later sample of the
same one.** Five turns that previously passed with nothing to cover now carry
required facts, and the judge finds four of them missing. `overall_accuracy`
did not move outside its spread, which is not evidence the change was
neutral — it is what a suite does when newly-exposed failures replace ones the
run happens to pass. The comparison worth making is the failure list, not the
mean, and §2.3's rule holds: a run made before the cases said this is not a
baseline this one may be compared against.

What the new facts exposed, all four of them real and none of them new
behaviour:

- **edu-0001** states that tuition is absent and routes the customer nowhere.
  `f2` (where to ask) missing. The reply is truthful and useless.
- **edu-0007 turn 2** — a bare slot value, `العريش` — falls to the fallback
  node and emits the generic `ممكن توضّحلي أكتر`. The branch slot is retained
  (`slot_retention_accuracy` 100%); the *node* is not, so the clarification has
  no missing slots to name and §19's "name the missing thing" fix cannot reach
  it. Judge helpfulness 0.
- **edu-0009** is the fallback node by design and has no slots to name, so it
  needs the option-listing behaviour its case note has always asked for, which
  nothing implements.
- **edu-0017** sends the `refuse` reply from `replies.yaml`, which offers
  "السعر الحالي وخطة السداد" — a price and a payment plan, to a student asking
  which faculty leads to a job. The scripted refusal is vertical-specific in a
  file shared by both verticals.

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


### 2.7 What each gate does not see

Every metric in §2.1 and §2.2 answers one question well. None of them answers
"was this reply right", and over three days five separate failures were found
by a rehearsal, a case, or a person — while every gate covering that area read
green. They are listed together because the pattern is a property of the design
rather than five coincidences: **a gate is a predicate over a reply, and most
of the ways a reply can be wrong are not predicates over a reply.**

| Gate | What it answers | What it cannot see |
|---|---|---|
| `hallucinated_figure_rate` | Is every figure traceable to a source? | The **right figure from the wrong row**. re-0025's payment plan quoted 5,800,000 from a real Madinaty unit for a question about Noor City. Nothing was hallucinated; the wrong unit was consulted. |
| `invented_compound_rate` | Is every compound named one that exists? | The **wrong real compound**. The `locations.yaml` gap resolved `كريك تاون` to Jefaira — a real compound, so the rate stayed at zero while the reply described the wrong property. |
| `slot_retention_accuracy` | Did held slots survive the turn? | What was **decided** from them. 100.0% on the run where edu-0018 held every slot correctly and routed to the fallback anyway (Task 42e). |
| `retrieval_recall_at_5` | Did the gold chunk come back? | **Which turn** it came back on. Scored per case, so turn 1 retrieving the gold chunk hides turn 2 retrieving five irrelevant ones (Task 42f). |
| `type_substitution_rate` | Was a different property type offered? | A substitution on **any other axis**. Task 42c's developer offered a Madinaty apartment for a Noor City question — same type, different project, and the rate cannot be about a project it does not read. |
| `arithmetic_in_model_rate` | Did the model compute a figure? | A **correct computation on the wrong inputs**. The calculator's output is verbatim in the reply and the unit it was called for was the one held from two turns ago. |
| `expected_action_accuracy` | Did the turn take the right action? | Whether the action was taken for the **right reason**. A handoff produced by an out-of-vocabulary exception and a handoff produced by a scope rule are the same value (Task 42c). |
| `errored_rate` | Did the turn raise? | A turn that **died silently**. Before Task 42 an out-of-vocabulary slot killed the turn and the customer got nothing; the suite recorded no error because nothing reached the runner. |

**The shape they share.** Each gate is a statement about the *reply* — its
figures, its compounds, its action. Every failure in the right-hand column is a
statement about the *question*: which unit, which branch, which turn, which
project. A reply can satisfy every property a gate can check and answer
something nobody asked, and the more gates pass, the more convincing it looks.
Both real-estate cases written against this pattern — re-0025 and re-0026 —
pin the specific wrong answer as a forbidden claim rather than trusting a rate,
because the specific wrong answer is the only thing that distinguishes it.

**What follows for reading a report.**

- **A green board is evidence about the questions asked, not about the
  system.** Real estate has read 100.0% on every run since 2026-08-24 and two
  of the five findings above were live in it the whole time.
- **The failure list is the readable part, the mean is not.** Already §2.4's
  rule for spreads; it holds twice over here, because a metric can be both
  settled and blind.
- **Rehearsal-only findings are a signal about coverage, not a nuisance.** All
  five arrived from the rehearsal or from a case written the day it was
  needed. When a defect can only be found by driving the product, the gates
  are not measuring the product yet.

**This is not an argument for more gates.** Three of the five would have needed
a gate that reads the customer's question and the retrieved row together, which
is the judge's job and is why stage 2 exists. It is an argument for what §5.4
already says: gates block merges, and people read transcripts.

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
