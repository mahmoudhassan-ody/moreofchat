"""A scripted reply mirrors the customer's language — F6.

**Register and language are different axes, and one lookup was resolving
both.** `Register` is per-node policy (§8.2): a fee statement is MSA because
fees are official, not because the customer wrote MSA. Language is the
opposite — it mirrors, always, and F6 in the harness spec's own failure table
is replying in the wrong one.

`replies.yaml` has carried an `english` string for every key since it was
written. Nothing ever selected it: the node pins `register: masri`, the
lookup asked for `masri`, and both English cases in the real-estate suite got
an Arabic reply. re-0020 asks about title deeds, hands off correctly, and
answers in Arabic.
"""

import pytest

from moc.agent.replies import scripted
from moc.agent.state import Register

ENTRY = {"masri": "أهلاً", "msa": "مرحباً", "english": "Hello"}


def test_an_english_message_gets_the_english_wording_whatever_the_node_says():
    assert scripted(ENTRY, Register.masri, lang="en") == "Hello"
    assert scripted(ENTRY, Register.msa, lang="en") == "Hello"


def test_an_arabic_message_keeps_the_node_s_register():
    """Language mirrors; register does not. A customer writing casual Masri
    about fees still gets the MSA fee wording — that is §8.2, not a bug."""
    assert scripted(ENTRY, Register.msa, lang="ar") == "مرحباً"
    assert scripted(ENTRY, Register.masri, lang="ar") == "أهلاً"


def test_an_unknown_language_falls_back_to_the_node_s_register():
    assert scripted(ENTRY, Register.msa, lang=None) == "مرحباً"


def test_an_entry_with_no_english_form_falls_back_rather_than_raising():
    """A missing wording must degrade to something sendable. An English
    customer reading Arabic is a bad turn; a KeyError is a lost one."""
    assert scripted({"masri": "أهلاً"}, Register.masri, lang="en") == "أهلاً"


@pytest.mark.parametrize("document", ["agent/replies", "scripts/realestate/search"])
def test_every_scripted_reply_carries_an_english_form(document):
    """The check that keeps this fixed. A key added in Arabic only is a key
    that will answer an English customer in Arabic, and nothing else notices.
    """
    from moc.config_store import load

    config = load(document)
    missing = [
        f"{group}.{key}"
        for group in ("replies", "ask_for_slot")
        for key, entry in (config.get(group) or {}).items()
        if isinstance(entry, dict) and not entry.get("english")
    ]
    assert missing == [], f"no English wording for: {missing}"
