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

from moc.agent.state import Register
from moc.arabic.script import detect_language

#: The wording key an English turn takes, whatever register the node declares.
_ENGLISH = "english"


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
    if lang == "en" and entry.get(_ENGLISH):
        return entry[_ENGLISH]
    return entry.get(str(register)) or entry[str(Register.masri)]


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
        return cls(register=register, lang=detect_language(message))

    def say(self, entry: dict[str, Any]) -> str:
        return scripted(entry, self.register, lang=self.lang)


__all__ = ["Voice", "scripted"]
