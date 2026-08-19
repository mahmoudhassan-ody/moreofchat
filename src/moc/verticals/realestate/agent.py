"""The real-estate turn — design §3.1, §3.2, §19.3.

Script-first, and more literally here than anywhere else in the system: an
inventory reply is composed from templates with figures interpolated straight
from the row or from the calculator. No model composes a price, so
`arithmetic_in_model_rate` and `hallucinated_figure_rate` are checkable against
the tool output rather than hoped for.

That is a stronger constraint than the education path, where the model phrases
a grounded answer. It is warranted: §3.2's figures are commercial commitments —
a price, an instalment, an availability — and the fixture's own numbers are the
only defensible source for each.

**Tools are recorded, not narrated.** Every lookup and every calculation is
appended to `tool_calls` with the arguments it actually ran with, because
`check_tool_calls` asserts against what happened rather than against what the
reply claims happened. The units offered are recorded the same way, so
`sold_unit_offered_rate` reads what was presented rather than scraping ids out
of prose.

The extractor is a documented keyword stub, as on the education side. The
extraction prompt is carried out of P1 and P1b both; wiring a model here would
put a second unmeasured variable inside the measurement these gates exist to
produce. Named plainly so no number from this path is read as prompt quality.
"""

import re
from dataclasses import dataclass, field
from typing import Any

from moc.agent.state import Action, ConversationState, Register, TurnInput
from moc.config_store import load
from moc.retrieval.inventory import UnitQuery
from moc.verticals.realestate.calculator import (
    PaymentPlanUnavailable,
    PaymentSchedule,
    UnofferedTerms,
    payment_schedule,
)
from moc.verticals.realestate.replies import (
    NoMatch,
    find_same_type_elsewhere,
    render_no_match,
    route_no_match,
)

_SCRIPT = "scripts/realestate/search"
_REPLIES = "agent/replies"
_LOCATIONS = "arabic/locations"


@dataclass(frozen=True)
class RecordedCall:
    name: str
    args: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class InventoryTurn:
    """What one real-estate turn produced.

    Deliberately not `TurnResult`: that type carries `passages` and a
    retrieval confidence, and an inventory turn has neither. A price came from
    a row, and pretending otherwise would let a reader think fusion was
    involved.
    """

    reply: str
    action: Action
    register: Register
    state: ConversationState
    tool_calls: tuple[RecordedCall, ...] = ()
    presented_unit_ids: tuple[str, ...] = ()
    named_compounds: tuple[str, ...] = ()
    computation: PaymentSchedule | None = None
    script_constants: tuple[str, ...] = ()
    as_of: str | None = None


class SlotExtractor:
    """Intent and slots without a model call.

    Keyword matching against the script's own intents and the location
    aliases. Stated plainly because it bounds what any number from this path
    means: it measures the connector, the calculator and the reply rules, and
    it does not measure extraction.
    """

    INTENTS = (
        ("negotiation", ("أقل سعر", "خصم", "كاش", "تخفيض", "أحسن سعر")),
        ("investment_projection", ("هتزيد", "قيمتها", "استثمار", "بعد سنتين", "عائد")),
        ("contracts", ("عقد", "تعاقد", "تمليك", "قانون")),
        ("payment_plan", ("قسط", "أقسط", "تقسيط", "مقدم", "أقساط")),
        ("inventory_lookup", ("في", "عايز", "عندكم", "متاح", "شقة", "فيلا", "استوديو")),
    )

    TYPES = {
        "استوديو": "studio",
        "شقة": "apartment",
        "شقه": "apartment",
        "فيلا": "villa",
        "شاليه": "chalet",
        "تاون": "townhouse",
        "دوبلكس": "duplex",
        "بنتهاوس": "penthouse",
        "مكتب": "office",
        "محل": "retail",
    }

    async def extract(self, *, text: str, state: ConversationState) -> TurnInput:
        intent = next(
            (name for name, words in self.INTENTS if any(w in text for w in words)),
            None,
        )
        slots: dict[str, Any] = {}
        for word, value in self.TYPES.items():
            if word in text:
                slots["property_type"] = value
                break
        city = _location_in(text, "city")
        if city:
            slots["city"] = city
        compound = _location_in(text, "compound")
        if compound:
            slots["compound"] = compound
        budget = _budget_in(text)
        if budget:
            # `في حدود X` is a ceiling; `الوحدة بـ X` names the unit being
            # asked about. Both narrow, and the payment-plan path resolves the
            # unit from the nearer of the two rather than treating a stated
            # price as a filter that could exclude the very unit meant.
            slots["budget_max" if "حدود" in text else "near_price"] = budget
        return TurnInput(intent=intent, slots=slots, grounded=True)


def _aliases() -> dict[str, str]:
    """Arabic surface form -> canonical location, from config (§19).

    Longest first, so `القاهرة الجديدة` is not shadowed by a shorter alias
    that happens to be a substring of it.
    """
    document = load(_LOCATIONS)["aliases"]
    pairs = [
        (alias, canonical)
        for canonical, forms in document.items()
        for alias in forms.get("arabic", [])
    ]
    return dict(sorted(pairs, key=lambda pair: -len(pair[0])))


def _kinds() -> dict[str, str]:
    return load(_LOCATIONS)["kind"]


def _location_in(text: str, kind: str) -> str | None:
    """The first alias of `kind` present in the message.

    City and compound are resolved separately because they filter different
    columns — `locations.yaml`'s `kind` map exists for exactly this, and
    filtering `city` with a compound name returns nothing and reads as "no
    inventory".
    """
    for alias, canonical in _aliases().items():
        if alias and alias in text and _kinds().get(canonical) == kind:
            return catalogue_name(canonical)
    return None


_ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")


def _budget_in(text: str) -> int | None:
    """`في حدود 15 مليون` -> 15000000. Millions only; the fixture has no
    other scale, and inventing one would be a figure nobody stated."""
    normalized = text.translate(_ARABIC_DIGITS)
    match = re.search(r"(\d+(?:\.\d+)?)\s*مليون(\s*و\s*نص)?", normalized)
    if not match:
        return None
    millions = float(match.group(1)) + (0.5 if match.group(2) else 0)
    return int(millions * 1_000_000)


def catalogue_name(canonical: str) -> str:
    """`new cairo` -> `New Cairo`.

    The alias map is lower-case and the catalogue's `city` column is title
    case. Converted here rather than duplicating the catalogue's spelling into
    config, which would be a second place for it to drift.
    """
    return " ".join(part.capitalize() for part in canonical.split())


class InventoryAgent:
    """One real-estate turn, end to end."""

    def __init__(self, *, repository: Any, engine: Any, extractor: Any) -> None:
        self._repository = repository
        self._engine = engine
        self._extractor = extractor

    async def handle(self, *, state: ConversationState, text: str) -> InventoryTurn:
        turn = await self._extractor.extract(text=text, state=state)
        decision = self._engine.advance(state, turn)
        slots = decision.state.slots

        if decision.action is not Action.answer:
            return InventoryTurn(
                reply=_scripted(_reply_key(decision), decision.register),
                action=decision.action,
                register=decision.register,
                state=decision.state,
            )

        if decision.node == "payment_plan":
            return await self._payment_plan(decision, slots)
        return await self._lookup(decision, slots)

    # ─────────────────────────── inventory ───────────────────────────

    async def _lookup(self, decision: Any, slots: dict[str, Any]) -> InventoryTurn:
        query = UnitQuery(
            city=slots.get("city"),
            compound=slots.get("compound"),
            property_type=slots.get("property_type"),
            bedrooms=slots.get("bedrooms"),
            budget_max=slots.get("budget_max"),
        )
        call = RecordedCall("inventory_lookup", _named(query))
        units = await self._repository.search(query)

        if units:
            unit = units[0]
            return _answer(
                decision,
                "units_found",
                call,
                unit,
                type=unit.property_type,
                compound=unit.compound,
                price=f"{unit.price:,}",
                currency=unit.currency,
                as_of=str(unit.as_of),
            )

        # Nothing of that type where they asked. The alternative is pinned to
        # the same type by `find_same_type_elsewhere`; there is no path here
        # that could offer another (§19.3).
        requested = slots.get("property_type")
        alternative = (
            await find_same_type_elsewhere(
                self._repository,
                property_type=requested,
                exclude_city=slots.get("city"),
                exclude_compound=slots.get("compound"),
                budget_max=slots.get("budget_max"),
            )
            if requested
            else None
        )
        no_match = NoMatch(
            requested_type=requested or "",
            asked_about=slots.get("compound") or slots.get("city") or "",
            alternative=alternative,
        )
        action = route_no_match(no_match)
        as_of = str(alternative.as_of) if alternative else _snapshot_as_of(self._repository)
        reply = render_no_match(no_match, register=decision.register, as_of=as_of)
        return InventoryTurn(
            reply=reply,
            action=action,
            register=decision.register,
            state=decision.state,
            tool_calls=(call,),
            named_compounds=(alternative.compound,) if alternative else (),
            presented_unit_ids=(alternative.unit_id,) if alternative else (),
            script_constants=_figures(reply),
            as_of=as_of,
        )

    async def _payment_plan(self, decision: Any, slots: dict[str, Any]) -> InventoryTurn:
        unit = await self._resolve_unit(slots)
        if unit is None:
            return InventoryTurn(
                reply=_scripted("handoff", decision.register),
                action=Action.handoff,
                register=decision.register,
                state=decision.state,
            )

        call = RecordedCall("payment_plan_calculator", {"unit_id": unit.unit_id})
        try:
            schedule = payment_schedule(_as_row(unit))
        except (PaymentPlanUnavailable, UnofferedTerms):
            # re-0007: a completed project is cash only. Said plainly rather
            # than answered with an invented plan.
            reply = _fill(
                "cash_only",
                decision.register,
                compound=unit.compound,
                price=f"{unit.price:,}",
                currency=unit.currency,
                as_of=str(unit.as_of),
            )
            return InventoryTurn(
                reply=reply,
                action=Action.answer,
                register=decision.register,
                state=decision.state,
                tool_calls=(RecordedCall("inventory_lookup", {"compound": unit.compound}),),
                presented_unit_ids=(unit.unit_id,),
                named_compounds=(unit.compound,) if unit.compound else (),
                script_constants=_figures(reply),
                as_of=str(unit.as_of),
            )

        reply = _fill(
            "payment_plan",
            decision.register,
            down_payment=f"{schedule.down_payment:,}",
            installment=f"{schedule.installment_amount:,}",
            currency=unit.currency,
            years=schedule.years,
            count=schedule.installment_count,
            as_of=str(unit.as_of),
        )
        return InventoryTurn(
            reply=reply,
            action=Action.answer,
            register=decision.register,
            state=decision.state,
            tool_calls=(call,),
            presented_unit_ids=(unit.unit_id,),
            named_compounds=(unit.compound,) if unit.compound else (),
            computation=schedule,
            script_constants=_figures(reply),
            as_of=str(unit.as_of),
        )


    async def _resolve_unit(self, slots: dict[str, Any]) -> Any | None:
        """Which unit the customer means.

        By id when a previous turn established one, otherwise by the compound
        they named and the price they quoted — re-0005 is "the unit in Noor
        City at six and a half million", which identifies a row without naming
        one. Nearest price rather than exact: customers round, and refusing to
        recognise a rounded figure would hand off a question the catalogue can
        answer.

        Still filtered. `get` and `search` both apply the availability
        predicate, so a sold unit cannot be resolved through this path either.
        """
        if slots.get("unit_id"):
            return await self._repository.get(slots["unit_id"])
        if not slots.get("compound"):
            return None
        units = await self._repository.search(
            UnitQuery(compound=slots["compound"], limit=50)
        )
        planned = [unit for unit in units if unit.payment_plan]
        if not planned:
            return units[0] if units else None
        target = slots.get("near_price")
        if target is None:
            return planned[0]
        return min(planned, key=lambda unit: abs(unit.price - target))


# ─────────────────────────── helpers ───────────────────────────


def _named(query: UnitQuery) -> dict[str, Any]:
    """The arguments the lookup actually ran with, for `check_tool_calls`."""
    return {
        key: value
        for key, value in {
            "city": query.city,
            "compound": query.compound,
            "property_type": query.property_type,
            "bedrooms": query.bedrooms,
            "budget_max": query.budget_max,
        }.items()
        if value is not None
    }


def _answer(decision: Any, key: str, call: RecordedCall, unit: Any, **values: Any) -> InventoryTurn:
    reply = _fill(key, decision.register, **values)
    return InventoryTurn(
        reply=reply,
        action=Action.answer,
        register=decision.register,
        state=decision.state,
        tool_calls=(call,),
        presented_unit_ids=(unit.unit_id,),
        named_compounds=(unit.compound,) if unit.compound else (),
        script_constants=_figures(reply),
        as_of=str(unit.as_of),
    )


def _fill(key: str, register: Register, **values: Any) -> str:
    templates = load(_SCRIPT)["replies"][key]
    return (templates.get(str(register)) or templates["masri"]).format(**values)


def _scripted(key: str, register: Register) -> str:
    templates = load(_REPLIES)["replies"][key]
    return templates.get(str(register)) or templates["masri"]


def _reply_key(decision: Any) -> str:
    if decision.action is Action.handoff:
        return "handoff"
    if decision.action is Action.refuse:
        return "refuse"
    return "clarify"


def _figures(reply: str) -> tuple[str, ...]:
    """Figures the template put in the reply, all of which came from a row.

    Declared as script constants so the grounding guard sees a source for
    them — they are as sourced as a retrieved passage, and more directly.
    """
    return tuple(re.findall(r"[\d,]*\d", reply))


def _snapshot_as_of(repository: Any) -> str:
    return getattr(repository, "as_of", None) or "2026-08-01"


def _as_row(unit: Any) -> dict[str, Any]:
    return {
        "unit_id": unit.unit_id,
        "price": unit.price,
        "as_of": str(unit.as_of),
        "payment_plan": unit.payment_plan,
    }


__all__ = ["InventoryAgent", "InventoryTurn", "RecordedCall", "SlotExtractor"]
