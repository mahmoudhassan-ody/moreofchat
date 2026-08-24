"""A real-estate turn behind the worker's port — demo plan Task 39b.

The inbound worker was written around one turn shape: the education
orchestrator's. Real estate is a different agent producing a different result
type — `InventoryTurn`, which has no passages and no retrieval confidence,
because a price came from a row rather than a passage. Until this file existed
there was no path from the worker to that agent at all, which meant a broker's
number could not be connected: the worker either refused the tenant or, worse
before the refusal existed, answered their customer with the education script.

**Everything here is built per turn, and that is not laziness.** Three of the
collaborators are properties of the tenant rather than of the process:

- the repository is bound to the session, and the session is bound to the
  tenant whose turn this is;
- the *catalogue* the extractor resolves slot values against comes from that
  tenant's own inventory rows — `LlmSlotExtractor` reads it once at
  construction, so one extractor per process resolves every broker's compounds
  against whichever broker started first, and the failure is a confident
  answer about the wrong compound;
- the project scope, which the repository reads from the tenant row.

The catalogue costs two `SELECT DISTINCT`s per turn. That is the right trade
against caching it: inventory is re-ingested whenever a broker updates a sheet,
and a cached catalogue is a bot that cannot see the stock it is being asked
about — the same staleness the `as_of` disclosure exists to make visible.
"""

from collections.abc import Callable
from typing import Any

from moc.agent.script_engine import ScriptEngine
from moc.retrieval.inventory import InventoryRepository
from moc.verticals.realestate.agent import InventoryAgent, InventoryTurn


class InventoryRunner:
    """`TurnRunner` for a real-estate tenant.

    `extractor` is a factory taking the tenant's catalogue rather than an
    extractor, because there is no such thing as a tenant-independent one here.
    """

    def __init__(
        self,
        *,
        script: str,
        extractor: Callable[[Any], Any],
    ) -> None:
        self._script = script
        self._engine = ScriptEngine.from_config(script)
        self._extractor = extractor

    async def handle(
        self,
        *,
        session: Any,
        state: Any,
        text: str,
        channel: str,
        engine: ScriptEngine | None = None,
        retriever: Any = None,
    ) -> InventoryTurn:
        """One turn.

        `retriever` is accepted and unused. Inventory answers come from a
        filtered row with an `as_of`, never from fusion — §3.2's whole point —
        and taking the argument keeps one worker path rather than making the
        worker ask which vertical it is carrying before it builds a call.
        """
        repository = InventoryRepository(session=session)
        agent = InventoryAgent(
            repository=repository,
            engine=engine or self._engine,
            extractor=self._extractor(await repository.vocabulary()),
            channel=channel,
        )
        return await agent.handle(state=state, text=text, session=session)


__all__ = ["InventoryRunner"]
