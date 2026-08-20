"""Slot and intent extraction — design §3.1, §19, and §2.6's routing table.

Replaces the keyword stubs that stood in for this on both verticals. Those
stubs bounded every number the suites produced: `overall_accuracy` and
`tool_call_accuracy` were substantially measuring them, and `asof_disclosure_rate`
failed on turns that never reached the inventory path because no intent was
recognised.

Three things this file exists to pin:

**The vocabulary is not in the prompt.** Intents come from the script's nodes,
slot values from the script's `slots:` block and the shared config files it
points at. A value named only in the prompt text is a value the resolver
cannot parse back, and the failure is silent — the filter matches nothing and
reads as "no inventory" rather than as an extraction error.

**A malformed extraction is a failed turn.** Not an empty slot dict. An
extractor that swallows a bad response returns "the customer said nothing",
which routes to a clarification and looks like a customer who was vague.

**`prompt_version` moves with the file.** `config_hash` covers `config/` and
this prompt lives under `src/`, so without a digest an edit would change every
extracted slot while every run still claimed comparability (§2.3).
"""

import json
from pathlib import Path

import pytest

from moc.agent.extraction import (
    ExtractionFailed,
    LlmSlotExtractor,
    render_prompt,
    slot_vocabulary,
)
from moc.agent.state import ConversationState
from moc.config_store import load

EDUCATION = "scripts/education/fees"
REALESTATE = "scripts/realestate/search"
PROMPT = Path("src/moc/agent/prompts/extraction_v1.md")


class FakeRouter:
    """Records what it was asked and returns a canned completion.

    Its signature mirrors `Router.complete` exactly rather than accepting
    `**kwargs`. A permissive double hid a real mismatch once: this module
    passed `max_tokens=`, which `Router.complete` does not take, and every
    unit test passed while all 17 live cases errored on the first call.
    """

    def __init__(self, text: str = '{"intent": null, "slots": {}}') -> None:
        self.text = text
        self.calls: list[dict] = []

    async def complete(
        self, *, task, messages, system=None, cache_blocks=(), exclude_provider=None
    ):
        kwargs = {
            "task": task,
            "messages": messages,
            "system": system,
            "cache_blocks": cache_blocks,
            "exclude_provider": exclude_provider,
        }
        self.calls.append(kwargs)

        class Completion:
            def __init__(self, text): self.text, self.provider, self.model = text, "anthropic", "m"
            input_tokens = output_tokens = cached_tokens = 0
            degraded = False

        return Completion(self.text)


def state(**slots) -> ConversationState:
    return ConversationState(script_id="s", script_version=1, slots=dict(slots))


# ─────────────────── the vocabulary comes from config ───────────────────


def test_no_compound_or_city_is_named_in_the_prompt_text():
    """The constraint this whole design turns on.

    A compound named in the prompt is a compound the resolver cannot parse
    back — the model has no way to know which spelling the catalogue uses, so
    it emits a plausible one and every filter downstream returns nothing. That
    reads as "we have no stock in Mivida", not as a bug.
    """
    text = PROMPT.read_text(encoding="utf-8")
    named = {
        name
        for values in CATALOGUE.values()
        for name in values
        if name.lower() in text.lower()
    }
    named |= {n for n in load("arabic/locations")["aliases"] if n.lower() in text.lower()}
    assert named == set(), f"the prompt names locations: {named}"

    types = set(load("agent/property_types")["types"])
    assert {t for t in types if t in text} == set(), "the prompt names property types"


def test_the_property_type_vocabulary_matches_the_catalogue_column():
    """A type the extractor can emit and the connector cannot filter on
    returns nothing and reads as absent stock."""
    rows = [
        json.loads(line)
        for line in Path(
            "evals/fixtures/broker_demo_2026_08_01/units.jsonl"
        ).read_text(encoding="utf-8").splitlines()
    ]
    assert set(load("agent/property_types")["types"]) == {r["property_type"] for r in rows}


def test_slot_vocabulary_is_resolved_from_the_script():
    """Each vertical's slots come from its own script, so a flow and the
    values its slots may hold cannot disagree."""
    education = slot_vocabulary(EDUCATION)
    assert "faculty" in education
    assert "dentistry" in education["faculty"]
    assert "property_type" not in education, "a vertical sees only its own slots"

    realestate = slot_vocabulary(REALESTATE, catalogue=CATALOGUE)
    assert "studio" in realestate["property_type"]
    assert "North Coast" in realestate["city"]
    assert "Creek Town" in realestate["compound"]
    assert realestate["bedrooms"] == "integer"
    assert realestate["unit_id"] == "free"


CATALOGUE = {
    "city": ("New Cairo", "North Coast", "Sheikh Zayed"),
    "compound": ("Creek Town", "Jefaira", "Noor City", "SouthMED", "Stei8ht"),
}


def test_the_location_vocabulary_is_the_catalogue_not_a_config_list():
    """The values the model may emit are the values the connector filters on,
    read from the same place. They were two lists and one held ten of
    ninety-four compounds — see tests/arabic/test_location_coverage.py."""
    vocabulary = slot_vocabulary(REALESTATE, catalogue=CATALOGUE)
    assert vocabulary["compound"] == (
        "Creek Town",
        "Jefaira",
        "Noor City",
        "SouthMED",
        "Stei8ht",
    )
    assert vocabulary["city"] == ("New Cairo", "North Coast", "Sheikh Zayed")


def test_a_catalogue_value_keeps_the_catalogue_s_own_spelling():
    """`SouthMED`, not `Southmed`. The old design title-cased a lower-case
    config key, which cannot produce `SouthMED`, `Stei8ht` or `L'Avenir` —
    and a resolved slot is used as a filter directly, so a near-miss spelling
    matches nothing and reads as absent stock."""
    vocabulary = slot_vocabulary(REALESTATE, catalogue=CATALOGUE)
    assert "SouthMED" in vocabulary["compound"]
    assert "Southmed" not in vocabulary["compound"]


def test_a_catalogue_slot_without_a_catalogue_refuses_rather_than_empties():
    """An empty vocabulary would reject every extraction as out-of-vocabulary,
    which reads as a bad model rather than as unwired plumbing."""
    with pytest.raises(ValueError, match="catalogue"):
        slot_vocabulary(REALESTATE, catalogue=None)


def test_a_catalogue_value_with_no_alias_is_offered_bare():
    """Ninety-four compounds, forty-five aliased. `Creek Town` carries an
    Arabic name; most compounds are Latin in the catalogue and typed that
    way, and empty brackets would read as a list to choose from."""
    prompt = render_prompt(
        script=REALESTATE, message="x", held_slots={}, catalogue=CATALOGUE
    )
    line = next(ln for ln in prompt.splitlines() if ln.startswith("- compound:"))
    assert "Creek Town (كريك تاون)" in line, "the compound the model substituted for"
    assert "Jefaira (جفيرة, jefaira)" in line
    assert "Stei8ht" in line and "Stei8ht (" not in line, "no empty brackets"


async def test_a_slot_may_hold_two_values_when_the_customer_named_two():
    """re-0018 turn 3: `الشيخ زايد أو أكتوبر`.

    The prompt now tells the model that `أو` means both, so it returns a list
    — and validation rejected the list *as a value*, which turned a correct
    extraction into an errored case. Every element is checked; the list is
    kept.
    """
    extractor = LlmSlotExtractor(
        router=FakeRouter(
            '{"intent": "inventory_lookup", '
            '"slots": {"city": ["Sheikh Zayed", "New Cairo"]}}'
        ),
        script=REALESTATE,
        catalogue=CATALOGUE,
    )
    turn = await extractor.extract(text="x", state=state())
    assert turn.slots == {"city": ["Sheikh Zayed", "New Cairo"]}


async def test_one_bad_value_in_a_list_still_fails_the_turn():
    """A list is not a way past the vocabulary."""
    extractor = LlmSlotExtractor(
        router=FakeRouter(
            '{"intent": "inventory_lookup", "slots": {"city": ["Sheikh Zayed", "Zamalek"]}}'
        ),
        script=REALESTATE,
        catalogue=CATALOGUE,
    )
    with pytest.raises(ExtractionFailed, match="Zamalek"):
        await extractor.extract(text="x", state=state())


async def test_an_empty_list_is_not_said_rather_than_malformed():
    extractor = LlmSlotExtractor(
        router=FakeRouter('{"intent": "inventory_lookup", "slots": {"city": []}}'),
        script=REALESTATE,
        catalogue=CATALOGUE,
    )
    turn = await extractor.extract(text="x", state=state())
    assert turn.slots == {}


async def test_a_stated_unit_price_is_a_different_slot_from_a_budget():
    """`near_price` identifies WHICH unit; `budget_max` filters.

    `الوحدة في نور سيتي بـ ٦ مليون و نص` read as a ceiling picks the first
    Noor City unit under it — a 2,930,000 studio — and quotes a correct
    instalment for a unit nobody asked about. The connector has resolved by
    nearest stated price since the calculator landed; the slot simply was not
    in the vocabulary, so nothing could ever fill it.
    """
    extractor = LlmSlotExtractor(
        router=FakeRouter(
            '{"intent": "payment_plan", '
            '"slots": {"compound": "Noor City", "near_price": 6500000}}'
        ),
        script=REALESTATE,
        catalogue=CATALOGUE,
    )
    turn = await extractor.extract(text="x", state=state())
    assert turn.slots == {"compound": "Noor City", "near_price": 6500000}


def test_both_price_slots_are_described_and_contrasted():
    """They are one preposition apart in Masri, so the prompt has to draw the
    line rather than leave it to inference — and it has to draw it in both
    directions, because reading a ceiling as an identifier is just as wrong."""
    prompt = render_prompt(
        script=REALESTATE, message="x", held_slots={}, catalogue=CATALOGUE
    )
    near = next(ln for ln in prompt.splitlines() if ln.startswith("- near_price:"))
    ceiling = next(ln for ln in prompt.splitlines() if ln.startswith("- budget_max:"))

    assert "في حدود" in ceiling, "the phrase that marks a ceiling"
    assert "الوحدة" in near, "the phrase that marks a specific unit"
    assert "budget" in near or "ceiling" in near, "each names the other to contrast"


def test_a_free_slot_says_what_it_is_for():
    """`unit_id` rendered as "exactly as the customer wrote it" and nothing
    else, so the model filled it with the first number in the sentence:
    `شقة مدينتي 95 متر` gave `unit_id: '95'` — 95 square metres. The lookup
    then missed and the turn handed off a question the catalogue could answer.

    The description is config, next to the slot, because what a free slot
    holds is a vertical's business.
    """
    prompt = render_prompt(
        script=REALESTATE, message="x", held_slots={}, catalogue=CATALOGUE
    )
    line = next(ln for ln in prompt.splitlines() if ln.startswith("- unit_id:"))
    assert "catalogue reference" in line
    assert "area" in line or "size" in line


def test_a_city_is_never_offered_as_a_compound():
    """`locations.yaml`'s `kind` map exists for this. Filtering `city` with a
    compound name returns nothing and reads as "no inventory"."""
    vocabulary = slot_vocabulary(REALESTATE, catalogue=CATALOGUE)
    assert "Mivida" not in vocabulary["city"]
    assert "North Coast" not in vocabulary["compound"]


def test_the_rendered_prompt_carries_the_vocabulary_and_the_held_slots():
    prompt = render_prompt(
        script=REALESTATE,
        message="عايز شقة في ميفيدا",
        held_slots={"city": "New Cairo"},
        catalogue=CATALOGUE,
    )
    assert "Creek Town" in prompt
    assert "studio" in prompt
    assert "New Cairo" in prompt, "held slots travel, or multi-turn cannot accumulate"
    assert "inventory_lookup" in prompt, "intents come from the script's nodes"
    assert "{message}" not in prompt and "{slots}" not in prompt


def test_a_location_is_offered_with_the_names_customers_use_for_it():
    """Canonical values alone are not enough, and this cost two live cases.

    The model maps a surface form to a canonical value only when the two are
    translations of each other: `الساحل الشمالي` -> `North Coast` resolves,
    `الشيخ زايد` -> `Sheikh Zayed` resolves. `التجمع الخامس` -> `New Cairo` is
    not a translation, it is local knowledge, and Haiku does not have it — it
    put the raw Arabic into `compound`, which is a hard rejection and an
    errored case (measured 2026-08-20).

    `locations.yaml` already holds that mapping. It simply never reached the
    prompt. The aliases come from config like the canonical values do, so the
    §3.1 rule is intact: what the model may EMIT is still only what the
    resolver can parse back.
    """
    prompt = render_prompt(
        script=REALESTATE, message="x", held_slots={}, catalogue=CATALOGUE
    )

    assert "التجمع الخامس" in prompt, "the Arabic name that errored re-0001"
    assert "tagamo3 el 5ames" in prompt.lower(), "the franco name that errored re-0016"

    line = next(ln for ln in prompt.splitlines() if ln.startswith("- city:"))
    assert "New Cairo" in line and "التجمع الخامس" in line, (
        "an alias must sit beside the canonical value it maps to, or the "
        "model has a list of names and no mapping"
    )


def test_the_emittable_values_are_still_only_the_canonical_ones():
    """Aliases are input, never output. A prompt that reads as if
    `التجمع الخامس` were an acceptable slot VALUE re-creates the failure it
    was added to fix."""
    vocabulary = slot_vocabulary(REALESTATE, catalogue=CATALOGUE)
    assert "التجمع الخامس" not in vocabulary["city"]
    assert set(vocabulary["city"]) == set(CATALOGUE["city"])


def test_a_property_type_is_offered_with_the_names_customers_use_for_it():
    """The same gap as locations, in a file that already documents itself as
    "injected into the extraction prompt at render time". Only the canonical
    keys ever were — `property_types.yaml` has carried `شقة` and `sha22a`
    since it was written, and neither reached the model.

    It is what re-0001 and re-0016 failed on once the location half was
    fixed: both name an apartment, neither writes `apartment`.
    """
    prompt = render_prompt(
        script=REALESTATE, message="x", held_slots={}, catalogue=CATALOGUE
    )
    line = next(ln for ln in prompt.splitlines() if ln.startswith("- property_type:"))
    assert "شقة" in line and "sha22a" in line
    assert "apartment (" in line, "the alias must sit beside the value it maps to"


def test_a_slot_with_no_alias_file_renders_as_a_plain_list():
    """`listing_kind` declares its values inline and has no surface forms.
    It must not grow empty brackets that read as a list to choose from."""
    prompt = render_prompt(
        script=REALESTATE, message="x", held_slots={}, catalogue=CATALOGUE
    )
    line = next(ln for ln in prompt.splitlines() if ln.startswith("- listing_kind:"))
    assert "(" not in line, line


def test_the_prompt_distinguishes_a_correction_from_a_choice():
    """re-0018 turn 3: `الشيخ زايد أو أكتوبر` — Sheikh Zayed *or* October.

    The correction rule was written for `مش التجمع، الشيخ زايد` and is right
    there. Applied to `أو` it drops half the question: the turn returned
    `Sheikh Zayed` alone, and the customer's second city never reached the
    filter. Both readings are one sentence apart, so the exception has to be
    stated rather than left to inference.
    """
    text = PROMPT.read_text(encoding="utf-8")
    assert "أو" in text, "the Arabic conjunction is what the turn actually contains"
    assert '"or"' in text or "` or `" in text
    assert "list" in text.lower(), "a choice returns both values, which is a list"


def test_only_intents_the_engine_can_route_are_offered():
    """An intent the script has no node for routes to the fallback, so
    offering it produces a clarification the customer cannot resolve."""
    prompt = render_prompt(script=EDUCATION, message="كام؟", held_slots={})
    routable = {
        intent
        for node in load(EDUCATION)["nodes"].values()
        for intent in node.get("intents", [])
    }
    assert routable
    # One name per node is offered, not every alias — see
    # `test_exactly_one_intent_name_is_offered_per_node`. Every offered name
    # must be routable; not every routable name is offered.
    offered = {
        line[2:].split(" — ")[0].strip()
        for line in prompt.splitlines()
        if line.startswith("- ") and "—" in line
    }
    assert offered <= routable
    assert len(offered) >= 8, "a node with intents is missing from the menu"
    assert "inventory_lookup" not in prompt, "a vertical is not offered another's intents"


# ─────────────────── malformed is a failed turn ───────────────────


async def test_malformed_json_is_a_failed_turn_not_an_empty_slot_dict():
    """An extractor that swallows a bad response reports "the customer said
    nothing", which routes to a clarification and looks like a vague customer.

    The distinction decides where an engineer looks: at the prompt, or at a
    case they will conclude is badly written.
    """
    extractor = LlmSlotExtractor(router=FakeRouter("not json at all"), script=EDUCATION)
    with pytest.raises(ExtractionFailed, match="malformed"):
        await extractor.extract(text="كام المصاريف؟", state=state())


async def test_an_unknown_intent_is_a_failed_turn():
    """A model inventing an intent has misread the task, and routing on it
    sends the customer down a flow built for a different question."""
    extractor = LlmSlotExtractor(
        router=FakeRouter('{"intent": "book_a_viewing", "slots": {}}'), script=EDUCATION
    )
    with pytest.raises(ExtractionFailed, match="book_a_viewing"):
        await extractor.extract(text="عايز أزور", state=state())


async def test_a_slot_value_outside_the_vocabulary_is_a_failed_turn():
    """The silent failure this guards: `Mivida Heights` filters nothing and
    reads as absent stock rather than as an invented value."""
    extractor = LlmSlotExtractor(
        router=FakeRouter(
            '{"intent": "inventory_lookup", "slots": {"compound": "Mivida Heights"}}'
        ),
        script=REALESTATE, catalogue=CATALOGUE,
    )
    with pytest.raises(ExtractionFailed, match="Mivida Heights"):
        await extractor.extract(text="عايز شقة", state=state())


async def test_an_unknown_slot_key_is_a_failed_turn():
    extractor = LlmSlotExtractor(
        router=FakeRouter('{"intent": "inventory_lookup", "slots": {"colour": "blue"}}'),
        script=REALESTATE, catalogue=CATALOGUE,
    )
    with pytest.raises(ExtractionFailed, match="colour"):
        await extractor.extract(text="عايز شقة", state=state())


async def test_a_fenced_object_is_accepted():
    """Models fence JSON despite being told not to, and rejecting a correct
    object over its wrapper would fail turns that extracted perfectly."""
    extractor = LlmSlotExtractor(
        router=FakeRouter('```json\n{"intent": "fees", "slots": {"faculty": "pharmacy"}}\n```'),
        script=EDUCATION,
    )
    turn = await extractor.extract(text="مصاريف الصيدلة", state=state())
    assert turn.intent == "fees"
    assert turn.slots == {"faculty": "pharmacy"}


async def test_numbers_are_parsed_as_integers():
    extractor = LlmSlotExtractor(
        router=FakeRouter(
            '{"intent": "inventory_lookup", "slots": {"bedrooms": "3", "budget_max": 17000000}}'
        ),
        script=REALESTATE, catalogue=CATALOGUE,
    )
    turn = await extractor.extract(text="٣ غرف في حدود ١٧ مليون", state=state())
    assert turn.slots == {"bedrooms": 3, "budget_max": 17000000}


async def test_an_explicit_null_means_the_slot_was_not_said():
    """Found live: re-0014 is "I don't want an exact number, just a rough
    range", and the model returned `budget_max: null` — a correct reading,
    reported in the JSON way rather than by omission.

    Failing the turn on it confused two different things. A null value is a
    well-formed answer meaning "not said"; a malformed response is one that
    cannot be read at all. Only the second is a failed turn. This is not the
    silent-empty-dict failure either: nothing is being swallowed, one
    well-formed key is being normalised to the absence it denotes.
    """
    extractor = LlmSlotExtractor(
        router=FakeRouter(
            '{"intent": "inventory_lookup", "slots": {"budget_max": null, '
            '"property_type": "apartment"}}'
        ),
        script=REALESTATE, catalogue=CATALOGUE,
    )
    turn = await extractor.extract(text="مش عايز رقم بالظبط", state=state())
    assert turn.slots == {"property_type": "apartment"}


async def test_a_non_numeric_number_is_a_failed_turn():
    """A budget that is not a number would be compared against a price."""
    extractor = LlmSlotExtractor(
        router=FakeRouter('{"intent": "inventory_lookup", "slots": {"budget_max": "a lot"}}'),
        script=REALESTATE, catalogue=CATALOGUE,
    )
    with pytest.raises(ExtractionFailed, match="budget_max"):
        await extractor.extract(text="عايز حاجة رخيصة", state=state())


# ─────────────────── routing and versioning ───────────────────


async def test_extraction_uses_the_slot_extraction_task():
    """§2.6's table decides the model, not this module. Haiku primary with the
    OpenAI small model as failover, both configured in llm/routing.yaml."""
    router = FakeRouter('{"intent": "fees", "slots": {}}')
    await LlmSlotExtractor(router=router, script=EDUCATION).extract(
        text="كام؟", state=state()
    )
    assert router.calls[0]["task"] == "slot_extraction"

    routing = load("llm/routing")["tasks"]["slot_extraction"]
    assert routing["primary"]["model"].startswith("claude-haiku")
    assert routing["failover"]["provider"] == "openai"


def test_prompt_version_moves_with_the_prompt_file():
    """`config_hash` covers config/ and this prompt lives under src/. Without
    a digest, editing the instructions would change every extracted slot while
    every run still claimed comparability with the old baseline (§2.3)."""
    extractor = LlmSlotExtractor(router=FakeRouter(), script=EDUCATION)
    version = extractor.prompt_version
    assert version.startswith("extraction_v1+")

    original = PROMPT.read_text(encoding="utf-8")
    try:
        PROMPT.write_text(original + "\nAlso be brief.\n", encoding="utf-8")
        assert LlmSlotExtractor(router=FakeRouter(), script=EDUCATION).prompt_version != version
    finally:
        PROMPT.write_text(original, encoding="utf-8")


async def test_held_slots_survive_a_turn_that_names_none():
    """Multi-turn accumulation. re-0018 gathers over three turns, and a turn
    that reports nothing must not clear what the customer already said."""
    extractor = LlmSlotExtractor(
        router=FakeRouter('{"intent": "inventory_lookup", "slots": {}}'),
        script=REALESTATE,
        catalogue=CATALOGUE,
    )
    turn = await extractor.extract(text="أيوه", state=state(city="New Cairo"))
    assert turn.slots == {}, "the extractor reports the turn, not the state"

    prompt = render_prompt(
        script=REALESTATE,
        message="أيوه",
        held_slots={"city": "New Cairo"},
        catalogue=CATALOGUE,
    )
    assert "New Cairo" in prompt, "but the model is told what is already held"


def test_the_fake_router_matches_the_real_signature():
    """The double that hid a live-only failure.

    `FakeRouter.complete` used to take `**kwargs`, so it accepted a
    `max_tokens=` this module was wrongly passing. Every unit test passed and
    all 17 live cases errored on the first call. A test double looser than the
    thing it stands for tests the double.
    """
    import inspect

    from moc.llm.router import Router

    real = inspect.signature(Router.complete).parameters
    fake = inspect.signature(FakeRouter.complete).parameters
    assert set(real) == set(fake), (
        f"the double and the router disagree: only-real={set(real) - set(fake)}, "
        f"only-fake={set(fake) - set(real)}"
    )


def test_exactly_one_intent_name_is_offered_per_node():
    """A node's aliases stay routable and are not offered as choices.

    Listing them made the model return the whole comma-joined string —
    `"discounts, scholarships, financial_aid"` — as the intent, and every such
    turn failed as unroutable. The list must read as a menu of values, not as
    a description of a group.
    """
    prompt = render_prompt(script=EDUCATION, message="x", held_slots={})
    offered = [
        line[2:].split(" — ")[0].strip()
        for line in prompt.splitlines()
        if line.startswith("- ") and "—" in line
    ]
    assert offered, "no intents rendered"
    for name in offered:
        assert "," not in name, f"{name!r} is a list, not a value"

    routable = {
        intent
        for node in load(EDUCATION)["nodes"].values()
        for intent in node.get("intents", [])
    }
    assert set(offered) <= routable
