"""Claim-level figure grounding — §19.3, and the class the numeric gate cannot see.

`check_numeric_grounding` compares a reply's figures against a *set of numbers*
drawn from every retrieved passage, discarding which passage each came from and
what that passage said it was. So the permitted vocabulary for any claim is
every figure in the retrieved set, and any of them may be attached to any claim.

edu-0002 is that failure, measured 2026-08-20: the reply gave the 2000 EGP
application fee correctly and then added `وفي حالة استخدام مكتب التقديم، تكون
الرسوم 1000 جنيه مصري` — a figure from an adjacent passage, presented as an
alternative application fee. The deterministic check passed it, because 1000 is
in the retrieved set.

The model proposes and code disposes: the auditor returns, per figure, the
verbatim span of the material that states this number is that thing, and this
module checks the span is really in the material and really contains the
figure. A model that invents a supporting span fails the same check as a model
that invents a fee.
"""

import json

import pytest

from moc.agent.figure_audit import audit_figures
from moc.llm.base import Completion

MATERIAL = (
    "ما هي رسوم التقديم؟\n2000 جنيه مصري",
    "ما قيمة الرسوم الإضافية لتغيير المسار؟\n500 جنيه مصري",
)


class FakeAuditor:
    """A router stand-in that returns one canned audit.

    Records its calls, because "the audit did not run" and "the audit passed"
    are the same observable outcome from the reply alone — and the first is
    what a skipped guard looks like.
    """

    def __init__(self, figures=None, *, fail=None, text=None):
        self.calls: list[dict] = []
        self._fail = fail
        self._text = (
            text if text is not None else json.dumps({"figures": figures or []})
        )

    async def complete(self, **kwargs):
        self.calls.append(kwargs)
        if self._fail is not None:
            raise self._fail
        return Completion(text=self._text, provider="anthropic", model="haiku")


async def test_a_figure_whose_span_states_what_it_is_passes():
    auditor = FakeAuditor(
        [{"figure": "2000", "claim": "رسوم التقديم", "span": "ما هي رسوم التقديم؟\n2000 جنيه مصري"}]
    )
    result = await audit_figures(
        router=auditor, reply="رسوم التقديم 2000 جنيه مصري.", material=MATERIAL
    )
    assert result.passed
    assert result.unsupported == []
    assert result.checked == 1


async def test_a_figure_the_material_never_labels_that_way_fails():
    """edu-0012's shape. 500 is in the material, uniquely, and it is the
    track-change fee — so an auditor asked whether the material says 500 is
    tuition has nothing verbatim to return."""
    auditor = FakeAuditor(
        [{"figure": "500", "claim": "المصاريف الدراسية للهندسة", "span": None}]
    )
    result = await audit_figures(
        router=auditor,
        reply="الرسوم الدراسية للهندسة هي 500 جنيه مصري.",
        material=MATERIAL,
    )
    assert not result.passed
    assert [claim.figure for claim in result.unsupported] == ["500"]


async def test_a_span_that_is_not_in_the_material_fails():
    """The half that makes this a check rather than a second opinion. An
    auditor that paraphrases, summarises, or invents its evidence fails the
    same way a composer that invents a fee does — the span is matched against
    the material, not trusted."""
    auditor = FakeAuditor(
        [{"figure": "2000", "claim": "رسوم التقديم", "span": "رسوم التقديم هي 2000 جنيه"}]
    )
    result = await audit_figures(
        router=auditor, reply="رسوم التقديم 2000 جنيه.", material=MATERIAL
    )
    assert not result.passed


async def test_a_span_that_does_not_contain_the_figure_fails():
    """A span about the right subject that never states the number is the
    relabelling failure wearing a citation."""
    auditor = FakeAuditor(
        [{"figure": "1000", "claim": "رسوم التقديم", "span": "ما هي رسوم التقديم؟"}]
    )
    result = await audit_figures(
        router=auditor, reply="رسوم التقديم 1000 جنيه.", material=MATERIAL
    )
    assert not result.passed


async def test_a_reply_stating_no_figure_is_never_audited():
    """Cost. The audit is one extra call on the customer-facing path, and a
    reply with nothing to check must not pay for it."""
    auditor = FakeAuditor([])
    result = await audit_figures(
        router=auditor, reply="مفيش مصاريف مذكورة في البيانات المتاحة.", material=MATERIAL
    )
    assert result.passed
    assert result.checked == 0
    assert auditor.calls == [], "the auditor was called for a reply with no figure"


async def test_digits_are_normalised_before_the_span_is_checked():
    """A reply in Arabic-Indic digits against material in Latin ones is the
    ordinary case, not an edge case. Comparing them raw would fail every
    correctly-grounded Arabic reply, and the guard would be switched off in a
    week — the same reasoning `check_numeric_grounding` already carries."""
    auditor = FakeAuditor(
        [{"figure": "٢٠٠٠", "claim": "رسوم التقديم", "span": "ما هي رسوم التقديم؟\n2000 جنيه مصري"}]
    )
    result = await audit_figures(
        router=auditor, reply="رسوم التقديم ٢٠٠٠ جنيه.", material=MATERIAL
    )
    assert result.passed


async def test_an_auditor_outage_degrades_rather_than_blocking():
    """Fail open, flagged.

    The deterministic gate still ran and still caught every orphan, so an
    auditor outage costs the relabelling check and nothing else — it degrades
    to the protection that shipped yesterday. Failing closed would convert one
    provider's outage into the loss of every composed answer that states a
    figure, which is a larger incident than the one being prevented.

    `degraded` is what keeps that honest: a run where the auditor was down and
    a run where it passed must not report the same number.
    """
    auditor = FakeAuditor(fail=RuntimeError("all providers unavailable"))
    result = await audit_figures(
        router=auditor, reply="رسوم التقديم 2000 جنيه.", material=MATERIAL
    )
    assert result.passed
    assert result.degraded
    assert result.checked == 0


async def test_an_unparseable_audit_degrades_rather_than_blocking():
    """An auditor that returned prose has not audited anything. That is the
    same class as an outage — a harness fault, not a finding — and scoring it
    as a failure would charge the model under test for the grader's mistake."""
    auditor = FakeAuditor(text="I think this reply looks fine.")
    result = await audit_figures(
        router=auditor, reply="رسوم التقديم 2000 جنيه.", material=MATERIAL
    )
    assert result.passed
    assert result.degraded


async def test_a_fenced_object_still_parses():
    auditor = FakeAuditor(
        text="```json\n"
        + json.dumps(
            {
                "figures": [
                    {
                        "figure": "2000",
                        "claim": "رسوم التقديم",
                        "span": "ما هي رسوم التقديم؟\n2000 جنيه مصري",
                    }
                ]
            }
        )
        + "\n```"
    )
    result = await audit_figures(
        router=auditor, reply="رسوم التقديم 2000 جنيه.", material=MATERIAL
    )
    assert result.passed
    assert not result.degraded


async def test_the_auditor_never_receives_the_customer_s_question():
    """It grades a reply against material. Given the question it starts
    deciding whether the reply was *responsive*, which is the judge's job and
    a different one — and a guard that drifts into quality scoring is a guard
    that starts refusing good replies."""
    auditor = FakeAuditor(
        [{"figure": "2000", "claim": "رسوم التقديم", "span": "ما هي رسوم التقديم؟\n2000 جنيه مصري"}]
    )
    await audit_figures(
        router=auditor, reply="رسوم التقديم 2000 جنيه.", material=MATERIAL
    )
    sent = auditor.calls[0]["messages"][0].content
    assert "رسوم التقديم 2000 جنيه." in sent
    assert all(passage in sent for passage in MATERIAL)


@pytest.mark.parametrize("figures", [None, "not a list", [{"figure": "2000"}]])
async def test_a_malformed_figures_list_degrades(figures):
    """Missing keys and wrong types are the auditor failing to answer, not the
    reply failing. Both must land as degraded rather than as a refusal."""
    auditor = FakeAuditor(text=json.dumps({"figures": figures}))
    result = await audit_figures(
        router=auditor, reply="رسوم التقديم 2000 جنيه.", material=MATERIAL
    )
    assert result.passed
    assert result.degraded
