"""Slot and intent extraction — design §3.1, §19, and §2.6's routing table.

The model reads one customer message and reports what it asked for. It does not
answer, and it does not decide what is true — §3.1's split, applied at the
front of the turn.

**The vocabulary is not in the prompt.** Intents come from the script's own
nodes; slot values come from the script's `slots:` block and the shared config
files it points at. Both are injected at render time.

That is not tidiness. A value named only in the prompt text is a value the
resolver cannot parse back, and the model cannot tell the difference — it
emits a plausible spelling, the filter matches nothing, and the reply reads as
"we have no stock in that compound" rather than as an extraction error. The
same argument makes the intent list come from the script: an intent with no
node routes to the fallback, producing a clarification the customer has no way
to resolve.

**A malformed extraction is a failed turn.** Not an empty slot dict. An
extractor that swallows a bad response reports "the customer said nothing",
which routes to a clarification and looks like a vague customer — and that
decides where an engineer looks: at the prompt, or at a case they will
conclude is badly written. Every rejection here names what was wrong with it.

**`prompt_version` moves with the file.** `config_hash` covers `config/` and
this prompt lives under `src/`, so the digest is what keeps a run comparable
to the baseline it claims to be comparable to (§2.3).
"""

import hashlib
import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from moc.agent.state import ConversationState, TurnInput
from moc.config_store import load
from moc.llm.base import Message, Task

_CONFIG = "agent/extraction"
_PROMPTS = Path(__file__).parent / "prompts"
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)

#: Slot descriptors that carry no closed vocabulary. `integer` is validated as
#: a number; `free` is passed through for a downstream resolver to judge.
INTEGER, FREE = "integer", "free"


class ExtractionFailed(Exception):
    """The model's output could not be trusted as an extraction.

    Raised rather than degraded to an empty result. The two are opposite
    claims — "I could not read this" and "the customer said nothing" — and
    only one of them is true when a response is malformed.
    """


@lru_cache(maxsize=8)
def slot_vocabulary(script: str) -> dict[str, Any]:
    """What each slot of `script` may hold.

    Resolved from the script's own `slots:` block, following `from:` into the
    shared config files rather than restating their values. A second copy of
    the property types or the location aliases would drift into a value the
    connector cannot filter on.
    """
    document = load(script).get("slots", {})
    vocabulary: dict[str, Any] = {}
    for slot, spec in document.items():
        if spec.get("free"):
            vocabulary[slot] = FREE
        elif spec.get("type") == "integer":
            vocabulary[slot] = INTEGER
        elif "values" in spec:
            vocabulary[slot] = tuple(spec["values"])
        elif spec.get("from") == "agent/property_types":
            vocabulary[slot] = tuple(load("agent/property_types")["types"])
        elif spec.get("from") == "arabic/locations":
            kinds = load("arabic/locations")["kind"]
            wanted = spec["kind"]
            vocabulary[slot] = tuple(
                sorted(
                    catalogue_name(name)
                    for name, kind in kinds.items()
                    if kind == wanted
                )
            )
        else:
            raise ValueError(f"{script}: slot {slot!r} declares no vocabulary")
    return vocabulary


@lru_cache(maxsize=8)
def slot_aliases(script: str) -> dict[str, dict[str, tuple[str, ...]]]:
    """Per slot, the surface forms that map to each canonical value.

    **The canonical values alone are not a mapping.** The model resolves a
    surface form only when the two are translations of each other:
    `الساحل الشمالي` -> `North Coast` and `الشيخ زايد` -> `Sheikh Zayed` both
    land. `التجمع الخامس` -> `New Cairo` does not — it is local knowledge, not
    translation — and Haiku put the raw Arabic into `compound`, which is a
    hard rejection and an errored case. Two of twenty-three real-estate cases
    died that way, both naming the largest city in the catalogue.

    `locations.yaml` has held that mapping since it was transcribed. It simply
    never reached the prompt. Reading it here rather than writing the names
    into the template keeps §3.1 intact: aliases are what the model may READ,
    the vocabulary is still all it may EMIT.
    """
    document = load(script).get("slots", {})
    resolved: dict[str, dict[str, tuple[str, ...]]] = {}
    for slot, spec in document.items():
        source = spec.get("from")
        if source == "arabic/locations":
            locations = load("arabic/locations")
            wanted = spec["kind"]
            resolved[slot] = {
                catalogue_name(name): _forms(forms)
                for name, forms in locations["aliases"].items()
                if locations["kind"].get(name) == wanted
            }
        elif source == "agent/property_types":
            # The same omission as locations, in a file whose own header says
            # it is "injected into the extraction prompt at render time".
            # Only the canonical keys ever were. `شقة` and `sha22a` have been
            # sitting here since it was written, and the model never saw
            # either — which is what re-0001 and re-0016 failed on once their
            # city resolved.
            resolved[slot] = {
                name: _forms(forms)
                for name, forms in load("agent/property_types")["types"].items()
            }
    return resolved


def _forms(spec: dict[str, Any]) -> tuple[str, ...]:
    """Arabic first, then Latin. Order is the file's, not sorted: the list is
    maintained commonest-first and that is the order worth showing."""
    return tuple(spec.get("arabic", []) + spec.get("latin", []))


def catalogue_name(canonical: str) -> str:
    """`new cairo` -> `New Cairo`. The catalogue's own columns are title case
    and a resolved slot is used as a filter directly."""
    return " ".join(word.capitalize() for word in canonical.split())


@lru_cache(maxsize=8)
def _intent_names(script: str) -> frozenset[str]:
    """Every routable intent name, for validating what came back."""
    return frozenset(
        intent
        for node in load(script)["nodes"].values()
        for intent in node.get("intents", [])
    )


def _intents(script: str) -> tuple[str, ...]:
    """Only intents the engine can route, each with the node's own one-line
    description where it has one.

    The gloss comes from the script rather than the prompt for the same reason
    the slot values do: a description written into the prompt describes a node
    that may not exist. It also carries the franco and Arabic surface forms a
    topic actually arrives as — `khasm`, `manh` — which the intent *name*
    cannot, and without which the model returns null on exactly the cases the
    suite was built around."""
    lines = []
    for node in load(script)["nodes"].values():
        names = node.get("intents", [])
        if not names:
            continue
        gloss = node.get("describe")
        # ONE name per line. Listing a node's aliases made the model return
        # the whole comma-joined string as the intent — "discounts,
        # scholarships, financial_aid" — and every one of those turns failed
        # as unroutable. The aliases stay routable for other callers; the
        # model is offered exactly one value per node.
        lines.append(f"{names[0]}" + (f" — {gloss.strip()}" if gloss else ""))
    return tuple(sorted(lines))


def _describe(
    vocabulary: dict[str, Any], aliases: dict[str, dict[str, tuple[str, ...]]]
) -> str:
    lines = []
    for slot, allowed in sorted(vocabulary.items()):
        if allowed is INTEGER:
            lines.append(f"- {slot}: an integer, no separators or units")
        elif allowed is FREE:
            lines.append(f"- {slot}: exactly as the customer wrote it")
        else:
            known = aliases.get(slot, {})
            lines.append(
                f"- {slot}: one of "
                + ", ".join(_offer(value, known.get(value, ())) for value in allowed)
            )
    return "\n".join(lines)


def _offer(value: str, forms: tuple[str, ...]) -> str:
    """`New Cairo (التجمع الخامس, tagamo3 el 5ames)`.

    The parenthesis is what the customer may write; the value outside it is
    the only thing that may come back. A slot with no alias file — property
    types — renders as the bare value, with no empty brackets to read as a
    list the model must choose from.
    """
    return f"{value} ({', '.join(forms)})" if forms else value


def render_prompt(*, script: str, message: str, held_slots: dict[str, Any]) -> str:
    """Fill the template with this script's vocabulary and the held state.

    `held_slots` travels because multi-turn cases accumulate — re-0018 gathers
    over three turns — and a model that cannot see what is already held will
    either repeat it or contradict it.
    """
    template = _template()
    return (
        template.replace("{message}", message)
        .replace("{held_slots}", json.dumps(held_slots, ensure_ascii=False) or "{}")
        .replace("{intents}", "\n".join(f"- {line}" for line in _intents(script)))
        .replace(
            "{slots}", _describe(slot_vocabulary(script), slot_aliases(script))
        )
    )


@lru_cache(maxsize=1)
def _template() -> str:
    return (_PROMPTS / f"{load(_CONFIG)['prompt']}.md").read_text(encoding="utf-8")


class LlmSlotExtractor:
    """§2.6's `slot_extraction` task, behind the orchestrator's extractor port."""

    def __init__(self, *, router: Any, script: str, config: dict[str, Any] | None = None) -> None:
        self._router = router
        self._script = script
        self._config = config or load(_CONFIG)

    @property
    def prompt_version(self) -> str:
        """Digest read from the file, not from the render cache.

        `_template` is cached because it is on the turn path. The version must
        not be: a cached digest reports the prompt this process started with,
        which is the §2.3 failure arriving through the mechanism meant to
        prevent it. Construction happens once per worker, so reading the file
        here costs nothing that matters.
        """
        path = _PROMPTS / f"{self._config['prompt']}.md"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        return f"{self._config['prompt']}+{digest[:12]}"

    async def extract(self, *, text: str, state: ConversationState) -> TurnInput:
        prompt = render_prompt(
            script=self._script, message=text, held_slots=dict(state.slots)
        )
        # No `max_tokens` here: `llm/routing.yaml` sets it per task, and a
        # second value in this module would be a second thing to keep in step
        # with §2.6's table.
        completion = await self._router.complete(
            task=Task.slot_extraction,
            messages=[Message(role="user", content=prompt)],
            system=None,
        )
        return self._parse(completion.text)

    # ─────────────────────────── strict parsing ───────────────────────────

    def _parse(self, text: str) -> TurnInput:
        fenced = _FENCE.search(text or "")
        try:
            document = json.loads(fenced.group(1) if fenced else text)
        except (json.JSONDecodeError, TypeError) as exc:
            raise ExtractionFailed(
                f"malformed extraction: response was not JSON ({exc})"
            ) from exc
        if not isinstance(document, dict):
            raise ExtractionFailed("malformed extraction: response was not an object")

        intent = document.get("intent")
        if intent is not None and intent not in _intent_names(self._script):
            raise ExtractionFailed(
                f"unroutable intent {intent!r}: the script has no node for it, so the "
                f"turn would clarify in a way the customer cannot resolve"
            )

        raw = document.get("slots") or {}
        if not isinstance(raw, dict):
            raise ExtractionFailed("malformed extraction: `slots` was not an object")

        return TurnInput(
            intent=intent,
            slots=self._validate(raw),
            grounded=True,
            explicit_handoff_request=bool(document.get("explicit_handoff_request")),
        )

    def _validate(self, raw: dict[str, Any]) -> dict[str, Any]:
        """Every key known, every value in vocabulary, every number a number.

        Rejected rather than dropped. Silently discarding an unknown slot
        turns a model that misread the task into a customer who said less than
        they did, and the reply that follows asks them to repeat themselves.
        """
        vocabulary = slot_vocabulary(self._script)
        clean: dict[str, Any] = {}
        for slot, value in raw.items():
            if value is None or (isinstance(value, str) and not value.strip()):
                # An explicit null or an empty string is "not said", reported
                # the JSON way rather than by omission. Well-formed, and not
                # the same thing as a response that cannot be read — only the
                # second is a failed turn. Nothing is swallowed here: one key
                # is normalised to the absence it already denotes.
                continue
            if slot not in vocabulary:
                raise ExtractionFailed(
                    f"unknown slot {slot!r}: not declared in {self._script}"
                )
            allowed = vocabulary[slot]
            if allowed is INTEGER:
                clean[slot] = _as_int(slot, value)
            elif allowed is FREE:
                clean[slot] = str(value)
            elif value in allowed:
                clean[slot] = value
            else:
                raise ExtractionFailed(
                    f"{slot}={value!r} is outside the configured vocabulary; a value "
                    f"the resolver cannot parse back filters nothing and reads as "
                    f"absent stock rather than as an error"
                )
        return clean


def _as_int(slot: str, value: Any) -> int:
    try:
        return int(str(value).replace(",", "").strip())
    except (TypeError, ValueError) as exc:
        raise ExtractionFailed(
            f"{slot}={value!r} is not a number, and it would be compared against one"
        ) from exc


__all__ = [
    "ExtractionFailed",
    "LlmSlotExtractor",
    "catalogue_name",
    "render_prompt",
    "slot_aliases",
    "slot_vocabulary",
]
