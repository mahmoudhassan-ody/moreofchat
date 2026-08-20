"""The no-substitution rule — design §19.3, owner decision of 2026-08-17.

**A request for one property type is never answered with another.**

A chalet is not a studio, an office is not retail, and the closest-priced unit
of the wrong kind is the wrong answer wearing the right price. The customer
discovers it at the viewing, which is the worst possible moment and the one
that costs the tenant the relationship.

The rule is in code and the wording is in config. That split is deliberate:
`config/scripts/realestate/search.yaml` holds strings a tenant's own people can
read and edit, and this module holds the part that must not be editable by
accident. A template is free to be reworded and not free to introduce a second
type — which is why both halves interpolate the same `{type}`, and why a test
asserts it appears twice.

**re-0022 is where the rule is under real pressure.** The North Coast holds 33
units — 19 chalets, 11 townhouses, 3 villas — and no studio. Every naive
ranking answers a studio request there with a chalet: it matches the city, the
budget and the language, and it is the most common thing on that coast. The
rule has to survive the case where breaking it looks helpful, so the type is
pinned into the query rather than preferred by a ranker. There is no step at
which the wrong type could win, because there is no step at which it is a
candidate.

Two outcomes, and the difference matters (§9):

- The type exists elsewhere -> answer, naming both the place asked about and
  the place that has one. That is re-0002 and re-0022.
- It exists nowhere in range -> hand off, naming no alternative. That is
  re-0021: all five villas are 23.9M and up, so a 15M villa budget has nothing
  to offer, and naming one above budget would be answering a different
  question rather than this one.
"""

from dataclasses import dataclass
from typing import Any

from moc.agent.replies import Voice
from moc.agent.state import Action
from moc.config_store import load

_SCRIPT = "scripts/realestate/search"


class TypeSubstitution(Exception):
    """An alternative of a different type than the one requested.

    Raised rather than corrected. A renderer that quietly dropped the mismatch
    would produce a reply naming no alternative, which reads as "we have
    nothing" — a different and also wrong answer. The caller built something
    incoherent and needs to know.
    """


@dataclass(frozen=True)
class NoMatch:
    """Nothing of the requested type where the customer asked.

    `asked_about` is what *they* named — a compound for re-0002, a city for
    re-0022 — and is echoed back rather than generalised. "No villa in New
    Cairo" when they asked about Mivida answers a question they did not ask,
    and sounds like the catalogue is thinner than it is.
    """

    requested_type: str
    asked_about: str
    alternative: Any | None = None


async def find_same_type_elsewhere(
    repository: Any,
    *,
    property_type: str,
    exclude_city: str | None = None,
    exclude_compound: str | None = None,
    budget_max: int | None = None,
) -> Any | None:
    """The nearest unit **of the same type**, or None.

    `property_type` is pinned into the query, so a unit of another kind cannot
    come back. There is deliberately no parameter through which the type could
    be relaxed, widened or replaced — the same shape the tenant filter and the
    availability filter use, for the same reason: a rule that depends on every
    caller remembering it is a rule with an expiry date.

    Returns None rather than the nearest anything. re-0021 is that case, and
    the correct reply there names no alternative at all.
    """
    from moc.retrieval.inventory import UnitQuery

    units = await repository.search(
        UnitQuery(property_type=property_type, budget_max=budget_max, limit=50)
    )
    for unit in units:
        if exclude_city and unit.city == exclude_city:
            continue
        if exclude_compound and unit.compound == exclude_compound:
            continue
        if unit.compound:
            return unit
    return None


def route_no_match(no_match: NoMatch) -> Action:
    """Answer when there is a same-type alternative, hand off when there is not.

    Handing off for every no-match would be a bot that never answers the
    question re-0002 exists to test; answering every one would mean inventing
    an alternative for re-0021, where none exists.
    """
    return Action.answer if no_match.alternative is not None else Action.handoff


def render_no_match(no_match: NoMatch, *, voice: Voice, as_of: str) -> str:
    """The customer-facing reply, from the tenant's own wording.

    Both halves receive the same type. It is read once, from the request, and
    formatted once — a second type value in scope here is how the halves come
    apart in a later edit.

    `as_of` is not optional (§3.2). Inventory moves faster than a snapshot, and
    a price without a date is one the tenant cannot stand behind.
    """
    requested_type = no_match.requested_type
    templates = load(_SCRIPT)["replies"]

    if no_match.alternative is None:
        return _fill(
            templates["no_match_anywhere"],
            voice,
            type=requested_type,
            asked_about=no_match.asked_about,
            as_of=as_of,
        )

    alternative = no_match.alternative
    if alternative.property_type != requested_type:
        raise TypeSubstitution(
            f"asked for a {requested_type} and the alternative is a "
            f"{alternative.property_type}; a different compound is a legitimate "
            f"alternative and a different type never is"
        )

    return _fill(
        templates["no_match_same_type"],
        voice,
        type=requested_type,
        asked_about=no_match.asked_about,
        compound=alternative.compound,
        price=f"{alternative.price:,}",
        currency=alternative.currency,
        as_of=as_of,
    )


def _fill(templates: dict[str, str], voice: Voice, **values: Any) -> str:
    """Pick this turn's wording and fill it — see `moc.agent.replies.Voice`."""
    return voice.say(templates).format(**values)


__all__ = [
    "NoMatch",
    "TypeSubstitution",
    "find_same_type_elsewhere",
    "render_no_match",
    "route_no_match",
]
