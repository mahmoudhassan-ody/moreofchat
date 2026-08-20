"""Claim-level figure grounding — §19.3, and the class the numeric gate cannot see.

**What `check_numeric_grounding` cannot do.** It compares a reply's figures
against a *set of numbers* collected from every retrieved passage, discarding
which passage each came from and what that passage said it was. The permitted
vocabulary for any claim is therefore every figure in the retrieved set, and any
of them may be attached to any claim. A figure lifted from a neighbouring
passage and relabelled passes, and always has — every 0.0% reading in the
project's history is bounded by "no figure appeared that was absent from the
retrieved set", not by "no figure was hallucinated" (harness §2.1.1).

edu-0002 is that failure, measured 2026-08-20 and the first observation of the
class the project has ever had: the reply gave the 2000 EGP application fee
correctly, then added `وفي حالة استخدام مكتب التقديم، تكون الرسوم 1000 جنيه
مصري` — a figure from an adjacent passage, presented as a second application
fee. The deterministic check passed it, because 1000 is in the retrieved set.

**The model proposes and code disposes.** The auditor returns, per figure, the
verbatim span of the material that states this number is that thing; this module
checks the span really appears in the material and really contains the figure.
Nothing here trusts the auditor's opinion — an auditor that paraphrases,
summarises or invents its evidence fails the same check as a composer that
invents a fee. That is what makes this a guard rather than a second opinion.

**It fails open, and says so.** An auditor outage costs the relabelling check
and nothing else: the deterministic gate still ran and still caught every
orphan, so the turn degrades to the protection that shipped the day before.
Failing closed would turn one provider's outage into the loss of every composed
answer that states a figure — a larger incident than the one being prevented.
`degraded` exists so a run where the auditor was down and a run where it passed
do not report the same number.

Separate module rather than a third guard in `guards.py`, because this one
calls a provider and that file is imported by `moc.evals.deterministic`. A
runtime guard that needs a router must not drag one into the measurement layer.
"""

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol

from moc.arabic.numerals import extract_quantities, normalize_digits
from moc.config_store import load
from moc.llm.base import Completion, Message, Role, Task

_CONFIG = "agent/figure_audit"
_PROMPTS = Path(__file__).parent / "prompts"
_FENCE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)
#: Trailing marks on a figure as the auditor echoed it. `%` in particular: the
#: material writes `73%` and the reply may write `73 %` or `٧٣٪`.
_TRIM = "%٪ \t\n"


class Auditor(Protocol):
    """The router, structurally. A seam so the guard is testable without a
    network — a guard whose tests need a provider is a guard nobody runs."""

    async def complete(self, **kwargs: Any) -> Any: ...


@dataclass(frozen=True)
class UnsupportedClaim:
    """One figure the material does not state is what the reply says it is."""

    figure: str
    claim: str
    #: What the auditor offered, kept whether or not it verified. A span that
    #: is not in the material and a span that is but lacks the figure are
    #: different faults, and only the text distinguishes them.
    span: str | None = None


@dataclass(frozen=True)
class FigureAudit:
    passed: bool
    unsupported: list[UnsupportedClaim] = field(default_factory=list)
    #: How many figures were actually verified. Zero with `passed` True means
    #: the reply stated none, or the auditor was unavailable — `degraded`
    #: tells those apart, and without it a skipped guard reads as a clean one.
    checked: int = 0
    degraded: bool = False
    #: The call that produced this verdict, for the ledger. None when no call
    #: was made — a reply with no figure, or a provider that never answered.
    #: Carried rather than metered here because this module has no session and
    #: no tenant, and a ledger write outside the caller's transaction would
    #: survive a turn that rolled back.
    completion: Completion | None = None


async def audit_figures(
    *, router: Auditor, reply: str, material: Sequence[str]
) -> FigureAudit:
    """Check every figure in `reply` is labelled the way the material labels it.

    `material` is everything the turn was entitled to state from — retrieved
    passages and script constants alike. The customer's question is
    deliberately not passed: given it, the auditor starts deciding whether the
    reply was *responsive*, which is the judge's job, and a guard that drifts
    into quality scoring is a guard that begins refusing good replies.
    """
    if not extract_quantities(reply):
        # One extra call on the customer-facing path. A reply with nothing to
        # check does not pay for it.
        return FigureAudit(passed=True)

    settings = load(_CONFIG)
    prompt = (
        _template(settings["prompt"])
        .replace("{material}", "\n\n".join(material))
        .replace("{reply}", reply)
    )
    try:
        completion = await router.complete(
            task=Task.figure_audit,
            messages=[Message(role=Role.user, content=prompt)],
        )
    except Exception:  # noqa: BLE001 — any provider fault is the same outcome here
        return FigureAudit(passed=True, degraded=True)

    claims = _parse(completion.text)
    if claims is None:
        # An auditor that returned prose has not audited anything. Same class
        # as an outage: a harness fault, not a finding.
        return FigureAudit(passed=True, degraded=True, completion=completion)

    haystack = normalize_digits("\n\n".join(material))
    unsupported = [claim for claim in claims if not _supported(claim, haystack)]
    return FigureAudit(
        passed=not unsupported,
        unsupported=unsupported,
        checked=len(claims),
        completion=completion,
    )


def _supported(claim: UnsupportedClaim, haystack: str) -> bool:
    """Two conditions, and the second is the one that catches relabelling.

    The span must be in the material — otherwise the auditor wrote its own
    evidence — *and* it must contain the figure, because a span about the right
    subject that never states the number is the relabelling failure wearing a
    citation.
    """
    if not claim.span:
        return False
    span = normalize_digits(claim.span)
    figure = normalize_digits(claim.figure).strip(_TRIM)
    return bool(figure) and span in haystack and figure in span


def _parse(text: str) -> list[UnsupportedClaim] | None:
    """None for anything that is not a well-formed audit.

    Missing keys and wrong types are the auditor failing to answer, not the
    reply failing, so they must not be distinguishable from prose here — both
    land as degraded upstream rather than as a refusal.
    """
    fenced = _FENCE.match(text)
    try:
        document = json.loads(fenced.group(1) if fenced else text)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(document, dict):
        return None
    figures = document.get("figures")
    if not isinstance(figures, list):
        return None

    claims = []
    for item in figures:
        if not isinstance(item, dict) or "figure" not in item or "claim" not in item:
            return None
        span = item.get("span")
        claims.append(
            UnsupportedClaim(
                figure=str(item["figure"]),
                claim=str(item["claim"]),
                span=str(span) if span else None,
            )
        )
    return claims


@lru_cache(maxsize=1)
def _template(name: str) -> str:
    return (_PROMPTS / f"{name}.md").read_text(encoding="utf-8")


__all__ = ["FigureAudit", "UnsupportedClaim", "audit_figures"]
