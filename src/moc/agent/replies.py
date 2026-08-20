"""Choosing the wording of a scripted reply — register and language.

**They are different axes, and one lookup used to resolve both.**

`Register` is per-node policy (§8.2). A fee statement is MSA because fees are
official, and it stays MSA when the customer writes casual Masri — mirroring
register would make the bot sound like it was agreeing rather than stating.

Language mirrors, always. F6 in the harness spec's failure table is replying
in the wrong one, and it is listed as "common with code-switched input".

Conflating them cost both English cases in the real-estate suite. re-0020 asks
about title deeds in English, hands off correctly — and says
`لحظة من فضلك، سيتم تحويلك` because its node pins `register: msa` and the
lookup asked for `msa`. The English wording had been sitting in
`replies.yaml` since the file was written; nothing ever asked for it.
"""

from dataclasses import dataclass
from typing import Any

from moc.agent.state import Action, Register
from moc.arabic.script import reply_language

#: The wording key an English turn takes, whatever register the node declares.
_ENGLISH = "english"


def resolve(register: Register, lang: str | None) -> Register:
    """Make the register speak the language the reply will be written in.

    They are separate decisions and they are not independent: MSA is a variety
    OF Arabic, so "write in English" and "write in MSA" contradict — edu-0015's
    prompt said both, two paragraphs apart, and the model resolved it by
    ignoring the register.

    Language wins, because language mirrors the customer (F6) and register is
    a house style. What survives is the formality, mapped across:

    - English turn, Arabic register -> English. There is no English MSA.
    - Arabic turn, English register -> Masri. A node whose author chose plain
      English chose the conversational end of the scale, and that is Masri.

    The Arabic-to-Arabic case is untouched, which is the point: §8.2 says a
    fee is MSA whatever variety the customer typed, and this must not soften
    that.
    """
    if lang == "en" and register is not Register.english:
        return Register.english
    if lang == "ar" and register is Register.english:
        return Register.masri
    return register


def scripted(entry: dict[str, Any], register: Register, *, lang: str | None) -> str:
    """One reply's wording for this turn.

    English message -> the English string. Anything else -> the node's
    register, falling back to Masri.

    The Masri fallback is not a shrug: a key with no entry for the node's
    register is almost always an apology or an outage message, and those are
    conversation rather than regulation.

    A missing English string falls back rather than raising. An English
    customer reading Arabic is a bad turn; a KeyError is a lost one.
    """
    speaks = resolve(register, lang)
    return entry.get(str(speaks)) or entry[str(Register.masri)]


#: The refusal for a node with no wording of its own.
_DEFAULT = "default"


def refusal(replies: dict[str, Any], node: str | None, voice: Voice) -> str:
    """The refusal this node offers instead, not the one another node offers.

    A refusal that says only "no" reads as evasion, so every one of them names
    what *is* knowable — and what is knowable is a property of the node doing
    the refusing. One shared string served both verticals and it was written
    for one of them: edu-0017 asks which faculty leads to a job, is correctly
    refused, and told "أقدر أقولك السعر الحالي وخطة السداد المتاحة" — a price
    and a payment plan, to a student.

    Lives here rather than in either agent because both of them refuse, and the
    real-estate agent is where the string that leaked was written.

    `default` catches a node with no entry: a KeyError in front of a customer
    is worse than a general sentence. A test asserts every shipped refuse node
    has its own, so the default is a floor and not a destination.
    """
    entries = replies[str(Action.refuse)]
    return voice.say(entries.get(node) or entries[_DEFAULT])


@dataclass(frozen=True)
class Voice:
    """How this turn should sound: the node's register, the customer's language.

    Carried together because every scripted reply needs both and passing only
    the register is exactly the omission that produced F6 — it type-checks,
    it renders, and it answers an English customer in Arabic.
    """

    register: Register
    lang: str | None = None

    @classmethod
    def of(cls, register: Register, message: str) -> Voice:
        # `reply_language`, not `detect_language`: franco is Latin script and
        # Arabic language, and mirroring the script answers a franco customer
        # in English.
        return cls(register=register, lang=reply_language(message))

    def say(self, entry: dict[str, Any]) -> str:
        return scripted(entry, self.register, lang=self.lang)


__all__ = ["Voice", "refusal", "resolve", "scripted"]
