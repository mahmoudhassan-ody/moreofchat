"""The inventory connector — design §3.2.

Structured inventory, not document chunks. A unit's price is answered from a
*row*: filtered on availability, carrying its own `as_of`, with the developer's
payment terms attached. That is why the broker fixture is deliberately absent
from `kb_chunks` — a passage holding the same price would bypass both the
filter and the disclosure, and nothing would raise. `sold_unit_offered_rate`
would read zero while a sold unit's price sat in a retrieved chunk.

**The availability filter is unrepresentable-absent, not merely applied.**

The same shape `vectors.py` uses for the tenant filter, for the same reason: a
filter whose absence has no behavioural signature is a filter that will
eventually be absent. Removing it does not raise, does not fail a type check,
and does not return fewer results — it returns *more*, which reads as better
coverage right up until a customer is shown a flat somebody else has bought.

So the guarantee is structural rather than remembered:

- `UnitQuery` — the object a caller fills in — has no field that speaks about
  availability, so there is no `include_sold=True` to pass.
- No public method takes a status, a `where`, or raw SQL.
- The predicate is written once, in `_where`, which has a single return.
- Every statement in this module that names `inventory_units` names
  `availability` too, including the one that reads a single unit by id — the
  id path is exactly what a caller reaches for when search says no.

There is no `count_all` or equivalent. An unfiltered method on the answering
path would make the filter a convention again, and counting rows is an
ingestion concern that belongs to whoever loaded them.

Tenant scoping is RLS, as everywhere else: no query here names `tenant_id`,
and the caller supplies a session whose tenant is already set.
"""

import json
import uuid
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from functools import lru_cache
from pathlib import Path
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from moc.config_store import load

_INVENTORY = "retrieval/inventory"

#: The one place the availability comparison is written. A second occurrence
#: would be a second thing to keep correct, and the test that counts them is
#: what makes "written once" checkable rather than aspirational.
_AVAILABLE = "availability = :available"

_COLUMNS = (
    "unit_id, as_of, title, listing_kind, property_type, compound, area, city, "
    "price, currency, unit_area_sqm, bedrooms, bathrooms, finish, furnished, "
    "availability, delivery_date, project_status, payment_plan, address"
)


@dataclass(frozen=True)
class UnitQuery:
    """What a caller may ask for.

    Every field narrows. None of them widens, and in particular none of them
    reaches availability — that is the whole point of the type. A customer's
    question becomes one of these, so a slot the extractor never filled stays
    `None` and simply does not constrain.
    """

    city: str | None = None
    compound: str | None = None
    property_type: str | None = None
    bedrooms: int | None = None
    budget_max: int | None = None
    budget_min: int | None = None
    limit: int | None = None


@dataclass(frozen=True)
class Unit:
    """One available unit, as the agent may present it.

    `as_of` travels on the unit rather than being fetched alongside it: a
    price separated from its date is a price the tenant cannot stand behind,
    and `asof_disclosure_rate` has nothing to disclose if the row does not
    carry it.
    """

    unit_id: str
    as_of: date
    property_type: str
    price: int
    currency: str
    availability: str
    title: str | None = None
    listing_kind: str | None = None
    compound: str | None = None
    area: str | None = None
    city: str | None = None
    unit_area_sqm: Decimal | None = None
    bedrooms: int | None = None
    bathrooms: int | None = None
    finish: str | None = None
    furnished: bool | None = None
    delivery_date: date | None = None
    project_status: str | None = None
    payment_plan: dict[str, Any] | None = None
    address: str | None = None


@lru_cache(maxsize=1)
def _settings() -> dict[str, Any]:
    return load(_INVENTORY)


def available_status() -> str:
    """The word this tenant's CRM uses for a unit that may be offered (§19)."""
    return _settings()["available_status"]


def _where(query: UnitQuery) -> tuple[str, dict[str, Any]]:
    """The one predicate builder, and the only place availability is compared.

    A single return on purpose. A conditional availability clause is the
    version of this bug that survives review — correct on every path anyone
    tested, absent on the one they did not — so there is no path here that can
    omit it.

    Optional filters append to it and can never replace it. There is no
    argument in `UnitQuery` that could.
    """
    clauses = [_AVAILABLE]
    params: dict[str, Any] = {"available": available_status()}

    for column, value in (
        ("city", query.city),
        ("compound", query.compound),
        ("property_type", query.property_type),
        ("bedrooms", query.bedrooms),
    ):
        if value is None or value == []:
            # An empty list is the absence of a filter, not a filter on
            # nothing. A slot the extractor cleared must not turn into zero
            # rows, which reads as "we have no stock".
            continue
        if isinstance(value, list | tuple):
            # `أو` means either — re-0018 asks for Sheikh Zayed or October.
            # `= ANY` rather than an expanded IN list so the parameter stays
            # one bind and nothing here builds SQL out of values.
            clauses.append(f"{column} = ANY(:{column})")
            params[column] = list(value)
        else:
            clauses.append(f"{column} = :{column}")
            params[column] = value

    # Inclusive on both ends. re-0011 asserts nothing above 17M, and an
    # off-by-one here quotes a price the customer already said they cannot pay.
    if query.budget_max is not None:
        clauses.append("price <= :budget_max")
        params["budget_max"] = query.budget_max
    if query.budget_min is not None:
        clauses.append("price >= :budget_min")
        params["budget_min"] = query.budget_min

    return " AND ".join(clauses), params


def _limit(query: UnitQuery) -> int:
    settings = _settings()["query"]
    requested = query.limit or settings["default_limit"]
    return min(requested, settings["max_limit"])


def _as_date(value: Any) -> date | None:
    """Parse a snapshot date before binding, not in SQL.

    asyncpg infers a parameter's type from the column it lands in, so it tries
    to encode the string as a date before any `cast(... as date)` in the
    statement can apply. The cast reads as if it handles this; it does not.
    """
    if value is None or isinstance(value, date):
        return value
    return date.fromisoformat(value)


def _unit(row: Any) -> Unit:
    values = dict(row._mapping)
    plan = values.get("payment_plan")
    if isinstance(plan, str):
        values["payment_plan"] = json.loads(plan)
    return Unit(**values)


class InventoryRepository:
    """The sole read path onto `inventory_units`.

    Holds a session whose tenant is already set; RLS does the tenant half and
    `_where` does the availability half. Neither is a parameter.
    """

    def __init__(self, *, session: AsyncSession) -> None:
        self._session = session

    async def search(self, query: UnitQuery) -> list[Unit]:
        """Available units matching `query`, cheapest first.

        Returns an empty list when nothing matches, never the nearest thing.
        re-0021 is the case: no villa exists under 15M, and returning the
        closest is where cross-type substitution begins — at the data layer,
        before any model has a chance to be tempted.
        """
        where, params = _where(query)
        order = _settings()["query"]["order_by"]
        # One f-string, not implicit concatenation: the statement is read as a
        # whole by the guard test, and a split literal reads as unfiltered.
        sql = (
            f"SELECT {_COLUMNS} FROM inventory_units WHERE {where} "  # noqa: S608
            f"ORDER BY {order}, unit_id LIMIT :limit"
        )
        rows = (
            await self._session.execute(text(sql), {**params, "limit": _limit(query)})
        ).all()
        return [_unit(row) for row in rows]

    async def get(self, unit_id: str) -> Unit | None:
        """One unit by id, subject to the same filter.

        The id path is what a caller reaches for when search says no, so it
        applies the availability predicate too — otherwise the filter is
        advisory and the way around it is a single method call.
        """
        where, params = _where(UnitQuery())
        sql = (
            f"SELECT {_COLUMNS} FROM inventory_units "  # noqa: S608
            f"WHERE unit_id = :unit_id AND {where}"
        )
        row = (
            await self._session.execute(text(sql), {**params, "unit_id": unit_id})
        ).one_or_none()
        return _unit(row) if row is not None else None

    async def vocabulary(self) -> dict[str, frozenset[str]]:
        """The values a slot may hold, read from the catalogue itself.

        **The extractor's vocabulary and the connector's filter are one list.**
        They used to be two — `locations.yaml` declared ten of ninety-four
        compounds — and the gap did not present as a missing alias. It
        presented as a confident answer about the wrong compound: asked about
        `كريك تاون`, the model picked the nearest allowed value and the reply
        described a Jefaira townhouse. Nothing raised, and
        `invented_compound_rate` stayed at zero because Jefaira is real.

        Exact spelling, not a normalised form. `SouthMED`, `Stei8ht`,
        `HAPTown` and `L'Avenir` are what the columns hold, and a resolved
        slot is used as a filter directly.

        Which column a value came from is its kind: a value in `city` filters
        `city`. That replaces a hand-maintained kind map, which is one more
        thing that could disagree with the rows.
        """
        where, params = _where(UnitQuery())
        columns = {}
        for column in ("city", "compound"):
            sql = (
                f"SELECT DISTINCT {column} FROM inventory_units "  # noqa: S608
                f"WHERE {where} AND {column} IS NOT NULL"
            )
            rows = (await self._session.execute(text(sql), params)).scalars().all()
            columns[column] = frozenset(rows)
        return columns

    async def compounds(self) -> frozenset[str]:
        """The catalogue a named compound is checked against (§19.3).

        Read from the same rows an answer comes from, so
        `invented_compound_rate` grades against the inventory rather than a
        second list that would drift from it.
        """
        where, params = _where(UnitQuery())
        sql = (
            f"SELECT DISTINCT compound FROM inventory_units "  # noqa: S608
            f"WHERE {where} AND compound IS NOT NULL"
        )
        rows = (await self._session.execute(text(sql), params)).scalars().all()
        return frozenset(rows)


_UPSERT = text("""
INSERT INTO inventory_units (
  id, tenant_id, unit_id, fixture, as_of, title, listing_kind, property_type,
  compound, area, city, price, currency, unit_area_sqm, bedrooms, bathrooms,
  finish, furnished, availability, delivery_date, project_status, payment_plan,
  address, source_row
) VALUES (
  :id, nullif(current_setting('moc.tenant_id', true), '')::uuid, :unit_id,
  :fixture, :as_of, :title, :listing_kind, :property_type,
  :compound, :area, :city, :price, :currency, :unit_area_sqm, :bedrooms,
  :bathrooms, :finish, :furnished, :availability, :delivery_date,
  :project_status, cast(:payment_plan as jsonb), :address, :source_row
)
ON CONFLICT (tenant_id, unit_id) DO UPDATE SET
  as_of = EXCLUDED.as_of,
  title = EXCLUDED.title,
  property_type = EXCLUDED.property_type,
  compound = EXCLUDED.compound,
  area = EXCLUDED.area,
  city = EXCLUDED.city,
  price = EXCLUDED.price,
  currency = EXCLUDED.currency,
  unit_area_sqm = EXCLUDED.unit_area_sqm,
  bedrooms = EXCLUDED.bedrooms,
  bathrooms = EXCLUDED.bathrooms,
  finish = EXCLUDED.finish,
  furnished = EXCLUDED.furnished,
  availability = EXCLUDED.availability,
  delivery_date = EXCLUDED.delivery_date,
  project_status = EXCLUDED.project_status,
  payment_plan = EXCLUDED.payment_plan,
  address = EXCLUDED.address,
  source_row = EXCLUDED.source_row
""")

_FIELDS = (
    "unit_id", "fixture", "as_of", "title", "listing_kind", "property_type",
    "compound", "area", "city", "price", "currency", "unit_area_sqm",
    "bedrooms", "bathrooms", "finish", "furnished", "availability",
    "delivery_date", "project_status", "address", "source_row",
)


async def load_units(*, session: AsyncSession, path: Path) -> int:
    """Ingest a snapshot into `inventory_units`, returning the row count.

    An upsert on `(tenant_id, unit_id)`, so re-ingesting a snapshot updates
    rather than accumulating a second copy. Two rows for one unit is one unit
    offered twice, which is the failure the whole table is filtered to prevent.

    Sold and reserved rows are stored, not dropped. They are what a later
    snapshot updates *from*, and dropping them would make a unit reappear as
    available the moment it left the export.
    """
    # Read off the event loop: a snapshot is a few hundred KB today and a
    # broker's real export is not, and blocking the loop on a file read stalls
    # every other turn the worker is carrying.
    import asyncio

    text_content = await asyncio.to_thread(path.read_text, encoding="utf-8")
    rows = [json.loads(line) for line in text_content.splitlines() if line.strip()]
    for row in rows:
        plan = row.get("payment_plan")
        await session.execute(
            _UPSERT,
            {
                "id": uuid.uuid4(),
                "payment_plan": json.dumps(plan, ensure_ascii=False) if plan else None,
                **{field: row.get(field) for field in _FIELDS},
                "as_of": _as_date(row.get("as_of")),
                "delivery_date": _as_date(row.get("delivery_date")),
            },
        )
    return len(rows)


__all__ = [
    "InventoryRepository",
    "Unit",
    "UnitQuery",
    "available_status",
    "load_units",
]
