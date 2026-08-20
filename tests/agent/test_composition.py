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


# ─────────────── language and register resolve together ───────────────


def test_an_english_reply_is_never_in_modern_standard_arabic():
    """edu-0015's rendered prompt said "Write the whole reply in English" and
    two paragraphs later "Modern Standard Arabic (MSA)". The model resolved
    the contradiction by ignoring the register, which is the best of the
    available bad outcomes.

    Register is the node's policy (§8.2) and language is the customer's, but
    they are not independent: MSA is a variety OF Arabic. An English turn on
    an MSA node gets the English register — the formality survives, the
    language does not."""
    prompt = render_composition(
        message="How much are the fees?", register=Register.msa, channel="whatsapp"
    )
    assert "English" in prompt
    assert "Arabic" not in prompt.split("REGISTER")[1].split("FORMATTING")[0]


def test_an_english_turn_on_a_masri_node_is_also_english():
    prompt = render_composition(
        message="Is housing available?", register=Register.masri, channel="whatsapp"
    )
    assert "Masri" not in prompt.split("REGISTER")[1].split("FORMATTING")[0]


def test_an_arabic_turn_keeps_the_node_s_arabic_register():
    """The resolution runs one way only. An Arabic customer on an MSA node
    gets MSA, and one on a Masri node gets Masri — that is §8.2 and it must
    not be softened by this."""
    assert "MSA" in render(message="المصاريف كام؟", register=Register.msa)
    assert "Masri" in render(message="عامل إيه؟", register=Register.masri)


def test_an_english_register_on_an_arabic_turn_becomes_arabic():
    """The other direction, which is just as wrong: a node pinned to English
    must not answer an Arabic customer in English.

    Masri, not MSA. A node whose author chose plain English chose the
    conversational end of the scale, and Masri is where that lands in Arabic.
    """
    prompt = render_composition(
        message="السكن متاح؟", register=Register.english, channel="whatsapp"
    )
    assert "Arabic" in prompt
    assert "Masri" in prompt.split("REGISTER")[1].split("FORMATTING")[0]


def test_a_figure_must_be_named_for_what_it_is():
    """Reply quality, and explicitly NOT protection.

    edu-0012 stated the 500 EGP track-change fee as engineering tuition. A
    reply that says what a figure is, in the retrieved material's own words,
    is a better reply and makes the mismatch visible to a reader — but the
    model is being asked, not constrained, and this session has twice shown
    what happens when a prompt is treated as a guarantee. The guarantee, if we
    build one, is a verified citation pass.
    """
    text = render()
    assert "Say what each figure is" in text
    assert "500" not in text, "the rule is stated, not illustrated with a figure"


# ───────── edu-0001: a reply that cannot answer still names a next step ─────────


def test_the_prompt_carries_the_script_s_referral():
    """edu-0001. The reply correctly said the material holds admission
    thresholds and not tuition — grounded, truthful, and a dead end. The case
    asks for "where to ask" and the customer was given nowhere.

    The prompt said "say what is missing and stop", which is the reply that
    was produced. What it may say instead is not the model's to invent: a
    contact route is tenant data, so the sentence is configured per script and
    quoted verbatim.
    """
    from moc.agent.script_engine import ScriptEngine

    referral = ScriptEngine.from_config("scripts/education/fees").referral("ar")
    assert referral, "the education script configures no referral"

    system = render_composition(
        message="المصاريف كام لكلية الهندسة؟",
        register=Register.msa,
        channel="whatsapp",
        referral=referral,
    )
    assert referral in system


def test_the_referral_mirrors_the_customer_s_language():
    from moc.agent.script_engine import ScriptEngine

    engine = ScriptEngine.from_config("scripts/education/fees")
    assert engine.referral("en") != engine.referral("ar")

    system = render_composition(
        message="how much is engineering tuition?",
        register=Register.msa,
        channel="whatsapp",
        lang="en",
        referral=engine.referral("en"),
    )
    assert engine.referral("en") in system
    assert engine.referral("ar") not in system


def test_a_script_with_no_referral_leaves_the_placeholder_unrendered():
    """Not the literal `{referral}` in front of a customer. A script that
    configures none is a script whose author has not decided where those turns
    go, and the prompt must degrade to the instruction it had before rather
    than to a template artifact."""
    system = render_composition(
        message="المصاريف كام؟", register=Register.msa, channel="whatsapp"
    )
    assert "{referral}" not in system


def test_every_shipped_script_configures_a_referral():
    """The same reasoning as the per-node refusals: a script that cannot
    answer something and says nothing about where to go is a dead end for
    every unanswerable turn it has, not just the one a case caught."""
    from moc.agent.script_engine import ScriptEngine

    missing = [
        name
        for name in ("scripts/education/fees", "scripts/realestate/search")
        for lang in ("ar", "en")
        if not ScriptEngine.from_config(name).referral(lang)
    ]
    assert missing == [], f"no referral configured for: {sorted(set(missing))}"


def test_the_referral_rule_says_when_the_sentence_must_not_appear():
    """edu-0014. Told only "end with this sentence", the model read it as a
    sign-off: a correct, grounded accreditation answer that then sent the
    customer to student affairs for no reason, which the judge scored as an
    unsupported claim because on that turn it was one.

    A conditional instruction has to carry both branches, and this one is
    read by a model that will otherwise pick the simpler reading.
    """
    system = render_composition(
        message="الجامعة معتمدة؟",
        register=Register.msa,
        channel="whatsapp",
        referral="اتواصل مع شئون الطلبة.",
    )
    rule = next(line for line in system.splitlines() if "اتواصل مع شئون الطلبة." in line)
    assert "must not appear" in system[system.index(rule):], (
        "the rule states when to include the sentence and never when not to"
    )
