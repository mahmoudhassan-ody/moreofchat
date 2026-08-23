"""Guards — what must be true of a turn regardless of what the model produced.

**Flagged for line-by-line human review.** Two of the platform's three
irreversible failures pass through this file: a customer identifier leaving
Egypt (§11.2) and a fee figure the knowledge base never stated (§19.3, F1).
Neither announces itself at runtime. A redactor that silently stops matching
produces perfectly normal-looking turns, and so does a grounding gate that
always returns True.

Two guards live here:

**Redaction (§7.3).** Runs on the inbound message *before either provider call*.
The one people miss is the embedding call: it sits earlier in the pipeline than
completion, it receives the customer's raw text at query time, and it is easy to
forget precisely because it does not look like "sending the message to an LLM".
`orchestrator.py` therefore redacts once at the top of the turn and never holds
the raw text afterwards — a rule about ordering is only as good as the last
person who read it, while having nothing to leak needs no vigilance.

**Numeric grounding (§19.3).** Every figure in a reply must trace to a retrieved
passage or a script constant. This lives here rather than in `moc.evals` because
it is a *runtime gate first* and a measurement second: the eval harness reports
how often it would fire, the orchestrator refuses to send when it does. One
implementation serves both — `moc.evals.deterministic` re-exports it — because
two would drift, and the drift would show up as an eval suite passing while
production emitted an orphan figure. The dependency runs evals -> agent; an
import-linter contract forbids the reverse.

The confidence gate named in the §3 module map is **not** here: it is in
`script_engine.advance`, because routing a below-threshold turn to the script's
fallback node is a flow decision and the engine owns flow. Splitting it out
would mean two modules deciding what happens next.
"""

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache

from moc.arabic.numerals import extract_numbers, extract_quantities, normalize_digits
from moc.config_store import load

_REDACTION = "agent/redaction"

#: The redaction vocabulary. A taxonomy, not a lexical value — the same status
#: as `UsageKind`. Patterns are config; what categories exist is a schema
#: question, and a typo'd label in YAML must be a loud error rather than a
#: placeholder that merely looks like a redaction.
PLACEHOLDER_LABELS = frozenset({"national_id", "payment_card", "phone", "student_id"})


@dataclass(frozen=True)
class Redaction:
    """Redacted text plus what was found, in order of appearance.

    `found` exists so the caller can log *that* an identifier was present
    without logging the identifier — §11.2 requires logging redacted forms
    only, and "we redacted a national ID from this turn" is the audit trail.
    """

    text: str
    found: tuple[str, ...] = ()

    @property
    def clean(self) -> bool:
        return not self.found


@dataclass(frozen=True)
class _Pattern:
    label: str
    regex: re.Pattern[str]


@lru_cache(maxsize=1)
def _patterns() -> tuple[tuple[_Pattern, ...], str]:
    document = load(_REDACTION)
    compiled = []
    for entry in document["patterns"]:
        label = entry["label"]
        if label not in PLACEHOLDER_LABELS:
            raise ValueError(
                f"redaction config declares unknown label {label!r}; "
                f"known labels are {sorted(PLACEHOLDER_LABELS)}"
            )
        compiled.append(_Pattern(label=label, regex=re.compile(entry["regex"])))
    return tuple(compiled), document["placeholder"]


def redact(text: str) -> Redaction:
    """Strip identifiers from `text` before it reaches any provider.

    Matching runs against a digit-normalized copy so Arabic-Indic input is
    covered, and the spans are then cut out of the *original* — normalization
    is length-preserving, so the offsets hold, and the customer's own digit
    script survives into the prompt. Rewriting their numerals would change the
    register of the message we are about to answer.

    Patterns apply in configured order and a match consumes its span, so an
    identifier cannot be claimed twice by two patterns that both describe it.
    """
    patterns, placeholder = _patterns()
    normalized = normalize_digits(text)

    spans: list[tuple[int, int, str]] = []
    for pattern in patterns:
        for match in pattern.regex.finditer(normalized):
            start, end = match.span()
            if any(start < taken_end and taken_start < end for taken_start, taken_end, _ in spans):
                continue
            spans.append((start, end, pattern.label))
    spans.sort()

    pieces: list[str] = []
    found: list[str] = []
    cursor = 0
    for start, end, label in spans:
        pieces.append(text[cursor:start])
        pieces.append(placeholder.format(label=label))
        found.append(label)
        cursor = end
    pieces.append(text[cursor:])

    return Redaction(text="".join(pieces), found=tuple(found))


# ─────────────────────────── numeric grounding ───────────────────────────


@dataclass(frozen=True)
class GroundingResult:
    """Two independent zero-tolerance gates, never conflated.

    `orphan_numbers` feeds `hallucinated_figure_rate` (harness §2.1) — the
    figure has no source. `hedged_numbers` feeds `hedged_figure_rate` — the
    figure has a source but the reply hedged it.

    They are separate because the fixes are different. An orphan means
    retrieval or the script failed to supply the figure; a hedge means the
    generation step editorialized over a figure it had. A single number
    combining both tells you a regression happened and nothing about where.

    The lists are disjoint. A figure that is both orphan and hedged counts
    only as an orphan, so one incident cannot move both rates.
    """

    passed: bool
    orphan_numbers: list[int | float] = field(default_factory=list)
    hedged_numbers: list[int | float] = field(default_factory=list)
    reply_numbers: list[int | float] = field(default_factory=list)
    source_numbers: list[int | float] = field(default_factory=list)


def check_numeric_grounding(
    reply: str,
    retrieved_passages: Sequence[str],
    script_constants: Iterable[float | str],
) -> GroundingResult:
    """Every figure in `reply` must appear in a passage or a script constant.

    Two ways to fail, reported separately and gated separately (§2.1):

    - **Orphan figure** -> `hallucinated_figure_rate`. A number with no source.
      This is F1 — a fee the knowledge base never stated, reaching a student on
      WhatsApp.
    - **Hedged figure** -> `hedged_figure_rate`. A *grounded* number the reply
      hedged anyway. Design doc §19.3 makes this non-negotiable: the markers
      are configurable, the fact that an approximation trips the check is not.
      "Roughly 1400" invites the customer to treat a fixed fee as an opening
      position, which is how a tenant ends up honouring a figure they never set.

    Comparison happens after digit normalization, so a reply in Arabic-Indic
    digits matches a source in Latin ones — otherwise the check would fire on
    every correctly-grounded Arabic reply and be switched off within a week.
    """
    quantities = extract_quantities(reply)

    # Years qualify a figure rather than asserting one, and the extractor already
    # drops them. Percentages and counts still need a source: a down-payment
    # share and a bedroom count are both claims a tenant can be held to.
    source_numbers = _collect_sources(retrieved_passages, script_constants)
    reply_numbers = [q.value for q in quantities]

    grounded = [q for q in quantities if is_grounded(q.value, source_numbers)]
    orphans = [q.value for q in quantities if not is_grounded(q.value, source_numbers)]
    # Only grounded figures can be hedged. An orphan that is also hedged is one
    # incident, and counting it twice would move both rates for a single fault.
    hedged = [q.value for q in grounded if q.approximate]

    return GroundingResult(
        passed=not orphans and not hedged,
        orphan_numbers=orphans,
        hedged_numbers=hedged,
        reply_numbers=reply_numbers,
        source_numbers=sorted(source_numbers),
    )


def _collect_sources(
    retrieved_passages: Sequence[str], script_constants: Iterable[float | str]
) -> set[float]:
    """Numbers the reply is allowed to state, from both grounding surfaces."""
    numbers: set[float] = set()
    for passage in retrieved_passages:
        numbers.update(float(n) for n in extract_numbers(passage))
    for constant in script_constants:
        # Constants arrive as numbers from a calculator tool or as strings from
        # a script node; parse strings through the same extractor so "١٤٠٠"
        # and 1400 are the same permission.
        if isinstance(constant, str):
            numbers.update(float(n) for n in extract_numbers(constant))
        else:
            numbers.add(float(constant))
    return numbers


def is_grounded(value: float, sources: set[float]) -> bool:
    """Exact match only.

    No tolerance window on purpose. A figure that is merely close is the
    failure this check exists to catch — spec §3.2 says the same about
    payment-plan arithmetic, and a tolerance here would let a rounded fee pass.

    Public because `moc.agent.provenance` traces figures to the chunks that
    grounded them and must decide "grounded" *identically* — a source pane
    that disagreed with this gate would be the version people believe. One
    function, exported deliberately, rather than a second implementation that
    starts out matching.
    """
    return float(value) in sources


# ─────────────────────────── inventory grounding ───────────────────────────


@dataclass(frozen=True)
class SubstitutionResult:
    """Whether the reply offered the type the customer asked for.

    `substituted` lists the (unit, type) pairs that differ from the request.
    Kept as pairs rather than a count because the fix depends on which type
    was swapped in: retail-for-office is a filter that is too loose, and
    townhouse-for-chalet is usually a ranker reaching for the nearest price.
    """

    passed: bool
    requested_type: str | None = None
    substituted: tuple[tuple[str, str], ...] = ()


def check_type_substitution(
    requested_type: str | None, presented: Mapping[str, str]
) -> SubstitutionResult:
    """No unit of a different property type may be presented.

    `presented` maps unit id to that unit's `property_type` from the catalogue,
    read off what the turn actually offered rather than scraped from the reply.

    **A different compound is a legitimate alternative; a different type never
    is.** A customer who asked for a chalet and is shown a townhouse has been
    answered with the wrong thing at the right price, and they find out at the
    viewing — after the tenant's agent has spent the trip. The closest-priced
    unit of another type is the single most tempting substitution and the one
    that costs the most trust, which is why this is a gate and not a ranking
    preference.

    An unknown requested type does not silently pass: with nothing to compare
    against there is no way to say the offer was right, so the caller gets a
    result with `requested_type=None` and must treat it as unchecked rather
    than as clean.
    """
    if requested_type is None:
        return SubstitutionResult(passed=False, requested_type=None)
    wrong = tuple(
        (unit_id, unit_type)
        for unit_id, unit_type in presented.items()
        if unit_type != requested_type
    )
    return SubstitutionResult(
        passed=not wrong, requested_type=requested_type, substituted=wrong
    )


@dataclass(frozen=True)
class EntityGroundingResult:
    """Names a reply used that the catalogue does not contain."""

    passed: bool
    invented: tuple[str, ...] = ()
    matched: tuple[str, ...] = ()


def check_named_entities(named: Sequence[str], catalogue: Iterable[str]) -> EntityGroundingResult:
    """Every compound a reply names must exist in the catalogue.

    An invented compound is the same failure as an invented price: a customer
    is told something specific and checkable that nobody can honour, and it is
    *more* convincing than a wrong number because a plausible Egyptian compound
    name reads as local knowledge. "Madinaty East" does not exist; a customer
    who drives there has been sent somewhere by a bot.

    Comparison is exact against the catalogue's own spelling. No fuzzy match:
    a near-miss is precisely the case worth catching, and tolerance here would
    let "Mivida Heights" pass because "Mivida" is real.
    """
    known = set(catalogue)
    invented = tuple(name for name in named if name not in known)
    return EntityGroundingResult(
        passed=not invented,
        invented=invented,
        matched=tuple(name for name in named if name in known),
    )


__all__ = [
    "PLACEHOLDER_LABELS",
    "EntityGroundingResult",
    "GroundingResult",
    "Redaction",
    "SubstitutionResult",
    "check_named_entities",
    "check_numeric_grounding",
    "check_type_substitution",
    "is_grounded",
    "redact",
]
