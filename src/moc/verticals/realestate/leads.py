"""Lead qualification and sales routing — design §11.2, demo plan Task 38.

A developer answers deeply about one project and routes the lead to the right
sales team. This is the second half of that sentence; the first half is the
project scope in `moc.retrieval.inventory`.

**Qualification reads slots and never the customer's wording.** A lead score is
a number a sales manager acts on — who gets called first, who waits — and
inferring enthusiasm from language is exactly the guess this platform refuses
everywhere else. So `qualify` takes the slots the script filled and counts what
is known. A customer who wrote three exclamation marks and named no budget is
not a warmer lead than one who named a budget; they are a lead we know less
about, and the honest output is `missing`.

**Qualified means actionable, not interested.** The two slots a salesperson
needs before they can do anything are the type and the budget. Everything else
narrows; those two decide whether there is a call to make. `missing` names what
is absent so the agent picking the lead up knows the first question to ask.

**A routing rule is a column on the team it routes to.** There is no rules
table beside a teams table, because then a rule can name a team that does not
exist — and the failure is silent: the lead is routed nowhere, the row looks
fine, and nobody calls. `sales_teams.property_type` *is* the rule, and NULL is
the fallback. Two partial unique indexes make "two teams claim villas" and "two
fallbacks" refusals from the database rather than from whichever router read
them first; neither has a behavioural signature, since a router returns *a*
team either way.

**An unroutable lead is not a dropped lead.** A tenant who has configured no
fallback has a configuration problem. The customer who asked has a question,
and it must still be in front of somebody — so `route` returns None, the
handoff opens with no team, and the inbox shows it unassigned. Choosing the
nearest specialist instead would be the substitution rule's mistake wearing a
different hat: the closest team is not the right team, and the buyer discovers
that on the phone.
"""

import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from moc.agent.handoff import Handoff, HandoffStore

#: What a salesperson needs before there is a call to make. Both, not either:
#: a budget with no type is a number, and a type with no budget is a wish.
_REQUIRED = ("property_type", "budget")

#: Everything else that narrows the call. Counted, never required — a lead is
#: not less real for arriving without a bedroom count.
_SIGNALS = ("city", "compound", "bedrooms", "delivery_date")

_BUDGET_SLOTS = ("budget_max", "budget_min")


@dataclass(frozen=True)
class Lead:
    """What is known about somebody who wants to buy.

    Frozen: a lead a later step can upgrade to qualified is a lead that becomes
    qualified because somebody needed it to be.
    """

    qualified: bool
    score: int
    property_type: str | None
    budget_max: int | None
    missing: tuple[str, ...]


@dataclass(frozen=True)
class SalesTeam:
    team_key: str
    name: str
    contact: str
    #: The routing rule. None is the fallback — see the module docstring for
    #: why the rule lives on the team rather than in a table of its own.
    property_type: str | None = None


def qualify(slots: Mapping[str, Any]) -> Lead:
    """Score a lead from the slots the script filled.

    No parameter carries the customer's words, and there must never be one:
    the moment wording is an input, the score is a judgement about a person
    rather than a count of what we know about them.
    """
    property_type = slots.get("property_type")
    budget_max = slots.get("budget_max")
    has_budget = any(slots.get(name) is not None for name in _BUDGET_SLOTS)

    missing = tuple(
        name
        for name, present in (
            ("property_type", property_type is not None),
            ("budget", has_budget),
        )
        if not present
    )

    known = sum(
        1
        for name in (*_REQUIRED, *_SIGNALS)
        if (slots.get(name) is not None if name != "budget" else has_budget)
    )

    return Lead(
        qualified=not missing,
        score=known,
        property_type=property_type,
        budget_max=budget_max if isinstance(budget_max, int) else None,
        missing=missing,
    )


def route(lead: Lead, teams: Sequence[SalesTeam]) -> SalesTeam | None:
    """The team this lead belongs to, or None.

    Matched on the type the lead actually named. A lead that named none is not
    a villa lead because villas happens to be the only specialist configured —
    that is the naive router, and with one team on the list it routes
    everything there.

    None rather than the nearest team when nothing matches and no fallback
    exists. The caller opens the handoff anyway.
    """
    fallback: SalesTeam | None = None
    for team in teams:
        if team.property_type is None:
            fallback = team
        elif lead.property_type is not None and team.property_type == lead.property_type:
            return team
    return fallback


class SalesTeamStore:
    """`sales_teams` for one tenant. RLS does the scoping; no query names a
    tenant, exactly as everywhere else."""

    def __init__(self, *, session: AsyncSession) -> None:
        self._session = session

    async def all(self) -> tuple[SalesTeam, ...]:
        rows = (
            await self._session.execute(
                text(
                    "SELECT team_key, name, contact, property_type FROM sales_teams "
                    "ORDER BY property_type NULLS LAST, team_key"
                )
            )
        ).all()
        return tuple(
            SalesTeam(
                team_key=row.team_key,
                name=row.name,
                contact=row.contact,
                property_type=row.property_type,
            )
            for row in rows
        )

    async def add(
        self, *, team_key: str, name: str, contact: str, property_type: str | None = None
    ) -> SalesTeam:
        """Add one team.

        No upsert. Two teams claiming the same type is a unique-index violation
        here rather than a silently replaced rule — a tenant who meant to move
        villas from one team to another should say so, because the version
        where it happens quietly is the version where nobody knows which team
        stopped getting leads.
        """
        await self._session.execute(
            text(
                "INSERT INTO sales_teams (id, tenant_id, team_key, name, contact, property_type) "
                "VALUES (:id, nullif(current_setting('moc.tenant_id', true), '')::uuid, "
                ":team_key, :name, :contact, :property_type)"
            ),
            {
                "id": uuid.uuid4(),
                "team_key": team_key,
                "name": name,
                "contact": contact,
                "property_type": property_type,
            },
        )
        return SalesTeam(
            team_key=team_key, name=name, contact=contact, property_type=property_type
        )


async def open_lead(
    session: AsyncSession,
    *,
    conversation_id: uuid.UUID,
    reason: str,
    slots: Mapping[str, Any],
    resume_state: dict[str, Any],
) -> tuple[Handoff, Lead]:
    """Qualify, route, and put the lead in front of somebody.

    One function rather than three call sites, because the order matters and
    the last step is the one that must not be skipped: a lead that is qualified
    and routed and never written is a lead nobody sees.

    The handoff opens whether or not a team was found. See the module
    docstring: unroutable is not dropped.
    """
    lead = qualify(slots)
    team = route(lead, await SalesTeamStore(session=session).all())
    handoff = await HandoffStore(session=session).open(
        conversation_id=conversation_id,
        reason=reason,
        resume_state=resume_state,
        team=team.team_key if team is not None else None,
        lead_qualified=lead.qualified,
        lead_score=lead.score,
    )
    return handoff, lead


__all__ = [
    "Lead",
    "SalesTeam",
    "SalesTeamStore",
    "open_lead",
    "qualify",
    "route",
]
