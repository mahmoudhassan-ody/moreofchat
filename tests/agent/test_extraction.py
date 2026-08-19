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
    catalogue = set(load("arabic/locations")["kind"])
    named = {name for name in catalogue if name.lower() in text.lower()}
    assert named == set(), f"the prompt names locations from config: {named}"

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

    realestate = slot_vocabulary(REALESTATE)
    assert "studio" in realestate["property_type"]
    assert "North Coast" in realestate["city"]
    assert "Mivida" in realestate["compound"]
    assert realestate["bedrooms"] == "integer"
    assert realestate["unit_id"] == "free"


def test_a_city_is_never_offered_as_a_compound():
    """`locations.yaml`'s `kind` map exists for this. Filtering `city` with a
    compound name returns nothing and reads as "no inventory"."""
    vocabulary = slot_vocabulary(REALESTATE)
    assert "Mivida" not in vocabulary["city"]
    assert "North Coast" not in vocabulary["compound"]


def test_the_rendered_prompt_carries_the_vocabulary_and_the_held_slots():
    prompt = render_prompt(
        script=REALESTATE, message="عايز شقة في ميفيدا", held_slots={"city": "New Cairo"}
    )
    assert "Mivida" in prompt
    assert "studio" in prompt
    assert "New Cairo" in prompt, "held slots travel, or multi-turn cannot accumulate"
    assert "inventory_lookup" in prompt, "intents come from the script's nodes"
    assert "{message}" not in prompt and "{slots}" not in prompt


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
    for intent in routable:
        assert intent in prompt
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
        script=REALESTATE,
    )
    with pytest.raises(ExtractionFailed, match="Mivida Heights"):
        await extractor.extract(text="عايز شقة", state=state())


async def test_an_unknown_slot_key_is_a_failed_turn():
    extractor = LlmSlotExtractor(
        router=FakeRouter('{"intent": "inventory_lookup", "slots": {"colour": "blue"}}'),
        script=REALESTATE,
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
        script=REALESTATE,
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
        script=REALESTATE,
    )
    turn = await extractor.extract(text="مش عايز رقم بالظبط", state=state())
    assert turn.slots == {"property_type": "apartment"}


async def test_a_non_numeric_number_is_a_failed_turn():
    """A budget that is not a number would be compared against a price."""
    extractor = LlmSlotExtractor(
        router=FakeRouter('{"intent": "inventory_lookup", "slots": {"budget_max": "a lot"}}'),
        script=REALESTATE,
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
        router=FakeRouter('{"intent": "inventory_lookup", "slots": {}}'), script=REALESTATE
    )
    turn = await extractor.extract(text="أيوه", state=state(city="New Cairo"))
    assert turn.slots == {}, "the extractor reports the turn, not the state"

    prompt = render_prompt(
        script=REALESTATE, message="أيوه", held_slots={"city": "New Cairo"}
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
