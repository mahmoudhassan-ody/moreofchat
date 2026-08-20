"""The answer-composition prompt — design §3.1, §8.2, harness F4 and F6.

**There was no prompt.** `_compose` sent `system=None` and a message body of
`decision.reason or decision.node`, so the model received the retrieved
passages and the string `"fees"` — not the customer's question, not the
register the node had chosen, and no instruction of any kind.

Everything the education suite was failing on follows from that. Four Arabic
questions came back as English markdown, because the passages were English and
nothing said to mirror the customer. The markdown then broke the numeric gate
in both directions at once (see tests/agent/test_guards.py), so two correct,
fully-grounded answers were discarded and replaced with an apology.
"""

import pytest

from moc.agent.composition import prompt_version, render_composition
from moc.agent.state import Register

PASSAGES = ("رسوم التقديم 2000 جنيه مصري",)


def render(**overrides) -> str:
    base = {
        "message": "رسوم التقديم كام؟",
        "register": Register.msa,
        "channel": "whatsapp",
        "passages": PASSAGES,
    }
    return render_composition(**{**base, **overrides})


def test_the_prompt_carries_the_customer_s_own_words():
    """It received the node name. A model answering a question it was never
    shown is answering the retrieval, which is what it did."""
    assert "رسوم التقديم كام؟" in render()


def test_the_reply_language_is_stated_and_mirrors_the_customer():
    """§8.2 and F6. An Arabic question about an English chunk gets an Arabic
    answer — the passages' language is an accident of the corpus."""
    arabic = render(message="السكن الجامعي متاح؟")
    assert "Arabic" in arabic
    assert "English" not in arabic.split("LANGUAGE")[1].split("\n\n")[0]

    english = render(message="Is housing available?")
    assert "English" in english


def test_franco_is_answered_in_arabic():
    """Franco is Latin script and Arabic language — the same distinction the
    scripted replies needed."""
    assert "Arabic" in render(message="fe sakan gam3y wala la2?")


def test_the_register_the_node_chose_is_stated():
    """Register does not mirror (§8.2): a fee is MSA because fees are
    official, whatever variety the customer wrote in."""
    assert "MSA" in render(register=Register.msa)
    assert "Masri" in render(register=Register.masri)


def test_the_channel_s_formatting_rules_are_stated():
    """WhatsApp renders none of it, and the markdown was not cosmetic — it
    broke the hallucination gate."""
    whatsapp = render(channel="whatsapp")
    for banned in ("heading", "table", "emoji"):
        assert banned in whatsapp.lower(), banned


def test_a_richer_channel_may_allow_more():
    """Email is not WhatsApp. The rules are per channel and in config, so a
    tenant adding a channel does not edit a prompt."""
    from moc.config_store import load

    channels = load("agent/composition")["formatting"]
    assert set(channels) >= {"whatsapp", "email", "default"}
    assert channels["whatsapp"] != channels["email"]


def test_an_unknown_channel_falls_back_to_the_strictest_rules():
    """A channel nobody configured must not silently permit everything."""
    assert render(channel="carrier_pigeon") == render(channel="default")


def test_the_passages_are_the_only_source_of_fact():
    text = render()
    assert "رسوم التقديم 2000 جنيه مصري" not in text, (
        "passages travel as cache blocks, not in the prompt body — they are "
        "the stable prefix and the question is the volatile part"
    )
    assert "invent" in text.lower() or "not in the" in text.lower()


def test_prompt_version_moves_with_the_file():
    """Same §2.3 rule as extraction: `config_hash` covers `config/` and this
    prompt lives under `src/`, so the digest is what keeps a run comparable to
    the baseline it claims to be comparable to."""
    from pathlib import Path

    version = prompt_version()
    assert version.startswith("composition_v1+")

    path = Path(__file__).parents[2] / "src" / "moc" / "agent" / "prompts"
    assert (path / "composition_v1.md").is_file()


@pytest.mark.parametrize("marker", ["#", "|", "```"])
def test_the_prompt_does_not_itself_demonstrate_what_it_bans(marker):
    """A prompt showing a markdown table while forbidding tables is a prompt
    that will produce one."""
    from pathlib import Path

    path = (
        Path(__file__).parents[2]
        / "src"
        / "moc"
        / "agent"
        / "prompts"
        / "composition_v1.md"
    )
    body = "\n".join(
        line for line in path.read_text(encoding="utf-8").splitlines()
        if not line.strip().startswith("- ") and "never" not in line.lower()
    )
    assert marker not in body
