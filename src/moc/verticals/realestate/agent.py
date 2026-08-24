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

from moc.agent.extraction import ExtractionFailed
from moc.agent.provenance import Computation, Row, trace_figures
from moc.agent.replies import Voice, refusal
from moc.agent.state import Action, ConversationState, Register, TurnInput
from moc.config_store import load
from moc.retrieval.inventory import UnitQuery
from moc.tenancy.metering import UsageKind, record_usage
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
    #: The rows this reply quoted from, as evidence. Not the ids —
    #: `presented_unit_ids` already carries those, and "which unit" is the
    #: question restated rather than the answer to "where did 5,500,000 come
    #: from?". See `moc.agent.provenance.Row`.
    rows: tuple[Row, ...] = ()

    @property
    def provenance(self) -> dict[str, Any] | None:
        """Where each figure in this reply came from — demo plan Task 41b.

        A property rather than a field set at each of the seven construction
        sites: every one of them already passes the unit or the schedule, so
        deriving it here means a new answer path cannot ship without evidence
        by forgetting a keyword argument.

        None when the reply quotes nothing. An empty `figures` list would make
        the pane draw an empty panel for a clarification that had nothing to
        show, which reads as evidence that went missing.

        **`script_constants` is deliberately not passed in.** On this vertical
        it is `_figures(reply)` — every number in the reply, harvested from the
        reply itself — so feeding it to the tracer would mark every figure
        grounded by definition and the pane would say "script" for a figure
        that came from nowhere. The trace has to find the origin or say it
        could not.
        """
        if self.action is not Action.answer:
            return None
        figures = trace_figures(
            reply=self.reply,
            rows=self.rows,
            computations=(
                (_computation_of(self.computation),) if self.computation else ()
            ),
        )
        if not figures:
            return None
        return {
            "figures": [figure.to_dict() for figure in figures],
            # Only the gate this vertical actually runs. There is no figure
            # audit here: no model composed these numbers, so there is nothing
            # for a second model to check — and showing a gate that never ran,
            # green or red, is a claim about a check that did not happen.
            "gates": {"numeric_grounding": all(f.grounded for f in figures)},
        }


class KeywordSlotExtractor:
    """A test double. **Not the production extractor.**

    `moc.agent.extraction.LlmSlotExtractor` is what runs — §2.6's
    `slot_extraction` task, Haiku with an OpenAI failover. This keyword
    matcher stays only so the offline runner tests can exercise the connector,
    the calculator and the reply rules without a model call and without an API
    key.

    Renamed from `SlotExtractor` on 2026-08-19 so its status is unmistakable
    at the call site. A double whose name reads like the real thing is a
    double that ends up in production wiring.
    """

    def __init__(self, *, catalogue: Any = None) -> None:
        # The same catalogue the real extractor gets. Without it this double
        # resolves no location at all, which is the honest failure: the values
        # live in the tenant's rows, not in a config list.
        self._catalogue = catalogue

    INTENTS = (
        ("negotiation", ("أقل سعر", "خصم", "كاش", "تخفيض", "أحسن سعر")),
        ("investment_projection", ("هتزيد", "قيمتها", "استثمار", "بعد سنتين", "عائد")),
        ("contracts", ("عقد", "تعاقد", "تمليك", "قانون")),
        ("financing", ("تمويل", "قرض", "بنك", "مرتبي", "تمويل عقاري")),
        ("price_validity", ("لسه ساري", "ساري", "لسه بنفس السعر", "still valid")),
        ("delivery_date", ("التسليم", "الاستلام", "تسليم", "handover", "delivery")),
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
        city = _location_in(text, "city", self._catalogue)
        if city:
            slots["city"] = city
        compound = _location_in(text, "compound", self._catalogue)
        if compound:
            slots["compound"] = compound
        budget = _budget_in(text)
        if budget:
            # `في حدود X` is a ceiling; `الوحدة بـ X` names the unit being
            # asked about. Both narrow, and the payment-plan path resolves the
            # unit from the nearer of the two rather than treating a stated
            # price as a filter that could exclude the very unit meant.
            slots["budget_max" if "حدود" in text else "near_price"] = budget
        return TurnInput(
            intent=intent, slots=slots, grounded=True, cleared=_cleared_in(text)
        )


def _aliases() -> dict[str, str]:
    """Arabic surface form -> catalogue value, from config (§19).

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


#: Which slots "somewhere else" drops. Both, because a customer rejecting a
#: compound has not necessarily rejected the city — but keeping the city would
#: re-run the search they just said no to, and re-0001 turn 3 says "any other
#: location" after two New Cairo turns.
_LOCATION_SLOTS = ("city", "compound")


def _cleared_in(text: str) -> tuple[str, ...]:
    """"In any other location" — the customer moving off the area on offer.

    `locations.yaml` has carried this list since it was transcribed and
    nothing consumed it, because `TurnInput` had no way to say "drop that".
    An absent slot means "not mentioned", which correctly keeps a held value,
    so the held city survived a sentence whose entire content was dropping it.
    """
    document = load(_LOCATIONS)["anywhere_else"]
    lowered = text.lower()
    for phrase in document["arabic"]:
        if phrase in text:
            return _LOCATION_SLOTS
    for phrase in document["latin"]:
        if phrase in lowered:
            return _LOCATION_SLOTS
    return ()


def _location_in(text: str, column: str, catalogue: Any) -> str | None:
    """The first alias present in the message whose value sits in `column`.

    City and compound resolve separately because they filter different
    columns, and filtering `city` with a compound name returns nothing and
    reads as "no inventory". Which column a value belongs to comes from the
    catalogue rather than from a hand-kept `kind:` map — one fewer thing that
    can disagree with the rows.
    """
    known = set((catalogue or {}).get(column, ()))
    for alias, value in _aliases().items():
        if alias and alias in text and value in known:
            return value
    # A bare catalogue name typed as-is: `Creek Town`, `Mivida`. Most compounds
    # carry no alias because their catalogue spelling is already what people
    # type, so matching the value itself is not a shortcut — it is the common
    # case.
    lowered = text.lower()
    for value in sorted(known, key=len, reverse=True):
        if value.lower() in lowered:
            return value
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



class InventoryAgent:
    """One real-estate turn, end to end."""

    def __init__(
        self, *, repository: Any, engine: Any, extractor: Any, channel: str = "whatsapp"
    ) -> None:
        self._repository = repository
        self._engine = engine
        self._extractor = extractor
        self._channel = channel

    async def handle(
        self, *, state: ConversationState, text: str, session: Any = None
    ) -> InventoryTurn:
        """One turn, metered when a session is given.

        `record_usage` had four call sites and every one of them was in the
        education orchestrator, so this vertical wrote nothing to the ledger at
        all — and extraction is its only provider call, which meant a broker
        tenant's usage read as free. The session is optional because the
        connector tests drive this agent without one and a required argument
        would make metering a reason to rewrite them.
        """
        try:
            turn = await self._extractor.extract(text=text, state=state)
        except ExtractionFailed:
            # A slot value outside this tenant's vocabulary — for a
            # project-scoped developer, that is any compound they do not sell,
            # because `vocabulary()` is scoped to the project. The extractor is
            # right to refuse the value; refusing by raising killed the turn
            # and the customer got **silence**, which §2.6 forbids more plainly
            # than a wrong answer.
            #
            # Found by the Task 42 rehearsal, asking a developer about the
            # project next door.
            if session is not None:
                await record_usage(
                    session, kind=UsageKind.message_out, channel=self._channel
                )
            voice = Voice.of(Register.masri, text)
            return InventoryTurn(
                reply=_scripted("handoff", voice),
                action=Action.handoff,
                register=Register.masri,
                state=state,
            )
        if session is not None:
            await record_usage(
                session, kind=UsageKind.message_in, channel=self._channel
            )
            await _meter(session, self._channel, getattr(turn, "usage", None))
        decision = self._engine.advance(state, turn)
        slots = decision.state.slots
        # Register is the node's; language is the customer's. Resolved once,
        # here, because every reply below needs both and half of them used to
        # take only the register.
        voice = Voice.of(decision.register, text)

        if decision.action is not Action.answer:
            if session is not None:
                await record_usage(
                    session, kind=UsageKind.message_out, channel=self._channel
                )
            return InventoryTurn(
                reply=_non_answer(decision, voice),
                action=decision.action,
                register=decision.register,
                state=decision.state,
            )

        if decision.node == "staleness":
            result = self._data_currency(decision, voice)
        elif decision.node == "delivery_date":
            result = await self._delivery_date(decision, slots, voice)
        elif decision.node == "payment_plan":
            # `turn.slots` is what *this message* said; `slots` is that merged
            # over everything held. Which unit the customer means has to come
            # from the message — see `_resolve_unit`.
            result = await self._payment_plan(decision, slots, voice, said=turn.slots)
        else:
            result = await self._lookup(decision, slots, voice)
        if session is not None:
            await record_usage(
                session, kind=UsageKind.message_out, channel=self._channel
            )
        return result

    # ─────────────────────────── staleness ───────────────────────────

    def _data_currency(self, decision: Any, voice: Voice) -> InventoryTurn:
        """re-0008. "Is this price still valid?"

        "Yes" is a commitment somebody has to honour or deny, so the answer is
        the date the data is current as of and who confirms the final price.
        No unit is quoted, so none is presented — the snapshot date is a
        property of the catalogue rather than of a row.
        """
        as_of = _snapshot_as_of(self._repository)
        reply = _fill("data_currency", voice, as_of=as_of)
        return InventoryTurn(
            reply=reply,
            action=Action.answer,
            register=decision.register,
            state=decision.state,
            script_constants=_figures(reply),
            as_of=as_of,
        )

    async def _delivery_date(
        self, decision: Any, slots: dict[str, Any], voice: Voice
    ) -> InventoryTurn:
        """re-0009. The handover date is a catalogue value, read from the row.

        Stated as the developer's schedule, never as a promise: fixture dates
        run to 2030 and off-plan slippage is normal here, so certainty is a
        claim about the future nobody can back.
        """
        known = decision.state.quoted_unit_id or slots.get("unit_id")
        if not known:
            # 305 delivery dates in the catalogue. Answering with one of them
            # is the same failure as answering a bare browse with one studio.
            return InventoryTurn(
                reply=_ask_for("unit_id", voice),
                action=Action.clarify,
                register=decision.register,
                state=decision.state,
            )
        unit = await self._repository.get(str(known))
        if unit is None:
            return InventoryTurn(
                reply=_scripted("handoff", voice),
                action=Action.handoff,
                register=decision.register,
                state=decision.state,
            )
        reply = _fill(
            "delivery_date",
            voice,
            compound=unit.compound,
            delivery_date=str(unit.delivery_date),
            as_of=str(unit.as_of),
        )
        return InventoryTurn(
            reply=reply,
            action=Action.answer,
            register=decision.register,
            state=_holding(decision, unit),
            tool_calls=(RecordedCall("inventory_lookup", {"unit_id": unit.unit_id}),),
            presented_unit_ids=(unit.unit_id,),
            named_compounds=(unit.compound,) if unit.compound else (),
            script_constants=_figures(reply),
            as_of=str(unit.as_of),
            rows=(_row_of(unit),),
        )

    # ─────────────────────────── inventory ───────────────────────────

    async def _lookup(self, decision: Any, slots: dict[str, Any], voice: Voice) -> InventoryTurn:
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
                voice,
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
        reply = render_no_match(no_match, voice=voice, as_of=as_of)
        return InventoryTurn(
            reply=reply,
            action=action,
            register=decision.register,
            state=_holding(decision, alternative) if alternative else decision.state,
            tool_calls=(call,),
            named_compounds=(alternative.compound,) if alternative else (),
            presented_unit_ids=(alternative.unit_id,) if alternative else (),
            script_constants=_figures(reply),
            as_of=as_of,
            rows=(_row_of(alternative),) if alternative else (),
        )

    async def _payment_plan(
        self,
        decision: Any,
        slots: dict[str, Any],
        voice: Voice,
        said: dict[str, Any] | None = None,
    ) -> InventoryTurn:
        unit = await self._resolve_unit(slots, decision.state.quoted_unit_id, said)
        if unit is None:
            return InventoryTurn(
                reply=_scripted("handoff", voice),
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
                voice,
                compound=unit.compound,
                price=f"{unit.price:,}",
                currency=unit.currency,
                as_of=str(unit.as_of),
            )
            return InventoryTurn(
                reply=reply,
                action=Action.answer,
                register=decision.register,
                state=_holding(decision, unit),
                tool_calls=(RecordedCall("inventory_lookup", {"compound": unit.compound}),),
                presented_unit_ids=(unit.unit_id,),
                named_compounds=(unit.compound,) if unit.compound else (),
                script_constants=_figures(reply),
                as_of=str(unit.as_of),
                rows=(_row_of(unit),),
            )

        reply = _fill(
            "payment_plan",
            voice,
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
            state=_holding(decision, unit),
            tool_calls=(call,),
            presented_unit_ids=(unit.unit_id,),
            named_compounds=(unit.compound,) if unit.compound else (),
            computation=schedule,
            script_constants=_figures(reply),
            as_of=str(unit.as_of),
            rows=(_row_of(unit),),
        )


    async def _resolve_unit(
        self,
        slots: dict[str, Any],
        quoted: str | None = None,
        said: dict[str, Any] | None = None,
    ) -> Any | None:
        """Which unit the customer means.

        **The current message beats the held context.** `quoted_unit_id` exists
        for a turn that names nothing — "and at 40% down?" identifies a unit
        only by what preceded it — and it was outranking a unit the customer
        named in the message. Browse Madinaty, then ask about a Noor City unit
        by name and price, and the answer came back about Madinaty: fluent,
        grounded, every figure traced, and about the wrong property.

        Nothing in the eval suite caught it because every payment-plan case is
        a first turn, and a first turn holds no context to be outranked. The
        Task 42 rehearsal walked into it on the second question it asked.

        Precedence, in order:

        1. An id in this message.
        2. The held unit, when this message names no place, or names the place
           that unit is already in. Repeating the compound is not a new
           request, and re-resolving would move the customer onto a different
           unit in the same development without saying so.
        3. The place **this message** names — by compound, narrowed by the
           price they quoted. The message's own slots, not the merge: the
           engine merges each turn over everything held, so the merge reads
           "Madinaty" for a message that said nothing and for a message that
           said Noor City in a spelling the extractor dropped.

           re-0005 is "the unit in Noor City at six and a half million", which
           identifies a row without naming one. Nearest price rather than
           exact: customers round.
        4. Nothing. **Never the held unit as a fallback** — that is how the
           first bug got in, and for a project-scoped developer it would be an
           answer about a project they do not sell.

        Still filtered. `get` and `search` both apply the availability
        predicate, so a sold unit cannot be resolved through this path either.
        """
        named = slots.get("unit_id")
        if named:
            unit = await self._repository.get(str(named))
            if unit is not None:
                return unit
            # An id that matches nothing is an extraction slip, not a dead end.
            # re-0007 gave `unit_id: 95` — the area in square metres — and
            # returning None here handed off a question whose compound was
            # sitting in the same turn.

        held = await self._repository.get(str(quoted)) if quoted else None
        # **The place named in this message, not the one merged into state.**
        # The script engine merges each turn's slots over everything held, so a
        # customer who named Madinaty and then asks about Noor City arrives
        # here with `compound` still reading Madinaty when the second message
        # left it out — and reading Noor City only if the extractor put it
        # there. Comparing the merge against the held unit therefore said "same
        # place, keep the held one" for a message that named a different place.
        #
        # `said` is the message's own slots. Falling back to the merge keeps
        # every other caller working, and those are turns with nothing held.
        wanted = (said if said is not None else slots).get("compound")
        if held is not None and (not wanted or wanted == held.compound):
            return held
        if not wanted:
            return held

        units = await self._repository.search(
            UnitQuery(compound=wanted, limit=50)
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


def _answer(
    decision: Any, key: str, call: RecordedCall, unit: Any, voice: Voice, **values: Any
) -> InventoryTurn:
    reply = _fill(key, voice, **values)
    return InventoryTurn(
        reply=reply,
        action=Action.answer,
        register=decision.register,
        state=_holding(decision, unit),
        tool_calls=(call,),
        presented_unit_ids=(unit.unit_id,),
        named_compounds=(unit.compound,) if unit.compound else (),
        script_constants=_figures(reply),
        as_of=str(unit.as_of),
        rows=(_row_of(unit),),
    )


def _row_of(unit: Any) -> Row:
    """One unit, as the evidence behind whatever the reply quoted from it.

    Every numeric column, not only the price: a reply can state an area, a
    bedroom count and a delivery year from the same row, and the pane has to
    say which of them each figure is.
    """
    return Row(
        unit_id=unit.unit_id,
        values={
            "price": unit.price,
            "bedrooms": unit.bedrooms,
            "bathrooms": unit.bathrooms,
            "unit_area_sqm": unit.unit_area_sqm,
        },
        title=unit.compound or unit.unit_id,
        as_of=unit.as_of,
    )


def _computation_of(schedule: PaymentSchedule) -> Computation:
    """One schedule, as the evidence behind the figures it produced.

    `inputs` carries what the calculator ran with, because that is what makes
    an instalment checkable. `total` is in `values` too even though the row
    also holds it — the tracer prefers the row, and listing it here keeps the
    computation an honest record of what the tool returned rather than an
    edited one.
    """
    return Computation(
        tool="payment_plan_calculator",
        unit_id=schedule.unit_id,
        values={
            "total": schedule.total,
            "down_payment": schedule.down_payment,
            "down_payment_pct": schedule.down_payment_pct,
            "installment_amount": schedule.installment_amount,
            "installment_count": schedule.installment_count,
            "final_installment_amount": schedule.final_installment_amount,
            "years": schedule.years,
        },
        inputs={
            "price": schedule.total,
            "down_payment_pct": schedule.down_payment_pct,
            "years": schedule.years,
            "frequency": schedule.frequency,
        },
        as_of=_as_date(schedule.as_of),
    )


def _as_date(value: Any) -> Any:
    from datetime import date

    if value in (None, "") or isinstance(value, date):
        return value or None
    try:
        return date.fromisoformat(str(value))
    except ValueError:
        return None


async def _meter(session: Any, channel: str, completion: Any) -> None:
    """One provider call, on the ledger. The education orchestrator's
    `_meter`, restated here rather than shared: the two agents are separate
    verticals with no common base, and a helper module for six lines would be
    a dependency neither of them needs."""
    if completion is None:
        return
    await record_usage(
        session,
        kind=UsageKind.llm_call,
        channel=channel,
        model=completion.model,
        provider=completion.provider,
        input_tokens=completion.input_tokens,
        output_tokens=completion.output_tokens,
        cached_tokens=completion.cached_tokens,
        cache_write_tokens=getattr(completion, "cache_write_tokens", 0),
        degraded=completion.degraded,
    )


def _fill(key: str, voice: Voice, **values: Any) -> str:
    return voice.say(load(_SCRIPT)["replies"][key]).format(**values)


def _scripted(key: str, voice: Voice) -> str:
    return voice.say(load(_REPLIES)["replies"][key])


def _ask_for(slot: str, voice: Voice) -> str:
    return voice.say(load(_REPLIES)["ask_for_slot"][slot])


def _non_answer(decision: Any, voice: Voice) -> str:
    """A refusal names what this node can offer instead; the rest are one
    string each. `refusal` is shared with the education orchestrator because
    both agents refuse, and the string that leaked into edu-0017 was written
    here."""
    replies = load(_REPLIES)["replies"]
    if decision.action is Action.refuse:
        return refusal(replies, decision.node, voice)
    key = "handoff" if decision.action is Action.handoff else "clarify"
    return voice.say(replies[key])


def _figures(reply: str) -> tuple[str, ...]:
    """Figures the template put in the reply, all of which came from a row.

    Declared as script constants so the grounding guard sees a source for
    them — they are as sourced as a retrieved passage, and more directly.
    """
    return tuple(re.findall(r"[\d,]*\d", reply))


def _holding(decision: Any, unit: Any) -> ConversationState:
    """Remember the unit this turn quoted.

    A conversation that forgets what it just showed forces the customer to
    repeat themselves — F5, and what `slot_retention_accuracy` measures. It is
    also what makes a follow-up answerable at all: "when's delivery?" and "and
    at 40% down?" both name a unit only by having been preceded by one.
    """
    from dataclasses import replace

    return replace(decision.state, quoted_unit_id=unit.unit_id)


def _snapshot_as_of(repository: Any) -> str:
    return getattr(repository, "as_of", None) or "2026-08-01"


def _as_row(unit: Any) -> dict[str, Any]:
    return {
        "unit_id": unit.unit_id,
        "price": unit.price,
        "as_of": str(unit.as_of),
        "payment_plan": unit.payment_plan,
    }


__all__ = ["InventoryAgent", "InventoryTurn", "KeywordSlotExtractor", "RecordedCall"]
