"""The answer-composition prompt — design §3.1, §8.2, §19.

**There was none.** `_compose` sent `system=None` and a message body of
`decision.reason or decision.node`, so the model received the retrieved
passages and the string `"fees"`. It never saw the customer's question, was
never told which register the node had chosen, and was never told how the
channel renders text.

Everything the education suite failed on follows from that. Four Arabic
questions came back as English markdown — the passages were English and
nothing said to mirror the customer — and the markdown then broke the numeric
grounding gate in both directions, discarding two answers that were correct
and fully grounded.

Register and language are separate axes here for the same reason they are in
`moc.agent.replies`: §8.2 makes register the node's policy and F6 makes
language the customer's.
"""

import hashlib
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

from moc.agent.replies import resolve
from moc.agent.state import Register
from moc.arabic.script import reply_language
from moc.config_store import load

_CONFIG = "agent/composition"
_PROMPTS = Path(__file__).parent / "prompts"
_DEFAULT_CHANNEL = "default"

#: What the model is told to write in. Two values, because the product ships
#: two languages — a third would need a lexicon entry, not a string here.
_LANGUAGES = {"ar": "Arabic", "en": "English"}

#: How each register reads to the model. §8.2's policy in words: the node
#: decides this, never the customer's own variety.
_REGISTERS = {
    Register.masri: (
        "Egyptian Arabic (Masri), the way a person speaks. This is a "
        "conversation, not a document."
    ),
    Register.msa: (
        "Modern Standard Arabic (MSA). This is an official statement — a fee, "
        "a regulation, a deadline — and it is quoted, not chatted about."
    ),
    Register.english: "Plain professional English.",
}


def render_composition(
    *,
    message: str,
    register: Register,
    channel: str,
    passages: Sequence[str] = (),
    lang: str | None = None,
) -> str:
    """Fill the template for one turn.

    `passages` is accepted and deliberately not interpolated: they travel as
    cache blocks, which is the stable prefix of the prompt, and the customer's
    question is the volatile part. Taking the argument keeps the caller from
    having to know that.
    """
    del passages
    # The extractor's reading when it gave one — it read the sentence on a
    # model that handles Egyptian Arabic, and the heuristic misses franco that
    # carries no digit substitutions.
    lang = lang or reply_language(message)
    speaks = resolve(register, lang)
    return (
        _template()
        .replace("{message}", message)
        .replace("{language}", _LANGUAGES.get(lang or "ar", _LANGUAGES["ar"]))
        .replace("{register}", _REGISTERS.get(speaks, _REGISTERS[Register.masri]))
        .replace("{formatting}", " ".join(formatting_for(channel).split()))
    )


def formatting_for(channel: str) -> str:
    """The channel's rules, falling back to the strictest.

    A channel nobody has configured gets `default`, which forbids everything.
    The other direction — an unknown channel inheriting the loosest rules —
    is how a new integration ships markdown to a surface that cannot render
    it, and the markdown is not cosmetic: `## 1.` reads to the numeric gate as
    the figure 1, and `**3000**` reads as no figure at all.
    """
    rules = load(_CONFIG)["formatting"]
    return rules.get(channel) or rules[_DEFAULT_CHANNEL]


def prompt_version() -> str:
    """Digest read from the file, not from the render cache.

    `_template` is cached because it is on the turn path. The version must not
    be: a cached digest reports the prompt this process started with, which is
    the §2.3 failure arriving through the mechanism meant to prevent it.
    """
    name = load(_CONFIG)["prompt"]
    digest = hashlib.sha256((_PROMPTS / f"{name}.md").read_bytes()).hexdigest()
    return f"{name}+{digest[:12]}"


@lru_cache(maxsize=1)
def _template() -> str:
    return (_PROMPTS / f"{load(_CONFIG)['prompt']}.md").read_text(encoding="utf-8")


def settings() -> dict[str, Any]:
    return load(_CONFIG)


__all__ = [
    "formatting_for",
    "prompt_version",
    "render_composition",
    "settings",
]
