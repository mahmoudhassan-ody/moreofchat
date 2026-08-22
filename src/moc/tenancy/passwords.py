"""Password hashing for console agents — demo plan Task 28.

**Flagged for line-by-line human review**, with `guards.py`, `webhooks.py` and
the Qdrant repository. Everything here fails silently when it is wrong: a
password store built on `sha256` hashes, verifies, and rejects the wrong
password exactly like this one does, and passes every behavioural test anyone
will write against it.

Three properties carry the weight.

**A slow, memory-hard KDF, named rather than implied.** `ALGORITHM` is part of
the encoded output and asserted by a test, because "we hash passwords" is not a
claim about anything. scrypt rather than Argon2id: Argon2id is first on OWASP's
list and is a C extension this project does not otherwise need, scrypt is
second on the same list and is in the standard library. In a file nobody may
merge without reading, one fewer dependency is worth the ranking.

**The work factor is config with a floor in code.** §19 puts it in config
because hardware moves; the floor is here because a config edit that lowers it
downgrades every password written afterwards and produces no error, no log and
no failing test. Config may strengthen the KDF and may not weaken it.

**No `==` in this module, at all.** A timing side channel on a digest
comparison is invisible to every functional test, so the rule is absolute
rather than case-by-case and a test asserts it against the source — the same
treatment `_matches` gets in the Twilio adapter. It costs one `compare_digest`
on the algorithm name, which is not secret and does not need it; that is the
price of a rule with no exceptions to argue about, and it is cheaper than the
whitelist the argument would produce.

What this module deliberately does not do is store or transport anything. It
takes a string and returns a string; the tenant boundary, the session and the
cookie are `agent_auth.py`'s problem, and keeping them apart is what makes
this file short enough to actually review.
"""

import base64
import binascii
import hashlib
import hmac
from typing import Any

#: Named in the encoded output and asserted by a test. A change here is a
#: change to what every stored hash means, so it cannot be a quiet one.
ALGORITHM = "scrypt"
_SEPARATOR = "$"
_SALT_BYTES = 16
_DERIVED_BYTES = 32
#: hashlib.scrypt's default maxmem is 32 MiB and these parameters need twice
#: that, so it is computed from them rather than fixed — raising `n` in config
#: must not fail as an obscure memory error inside the stdlib.
_MEMORY_PER_BLOCK = 128
_MEMORY_SLACK = 2


def parameters(config: dict[str, Any] | None = None) -> dict[str, Any]:
    if config is not None:
        return config
    from moc.config_store import load

    return load("security/agents")["kdf"]


def hash_password(password: str, *, config: dict[str, Any] | None = None) -> str:
    """`scrypt$n$r$p$salt$derived`, base64 for the two binary fields.

    Self-describing on purpose. A bare digest column cannot be re-read after
    the parameters change, so every row would have to be assumed to hold
    whatever the config says today — which is false for exactly the rows
    written before someone strengthened it.
    """
    settings = parameters(config)
    n, r, p = _checked(settings)
    salt = _random(_SALT_BYTES)
    derived = _derive(password, salt=salt, n=n, r=r, p=p)
    return _SEPARATOR.join(
        (ALGORITHM, str(n), str(r), str(p), _encode(salt), _encode(derived))
    )


def verify_password(
    password: str, encoded: str, *, config: dict[str, Any] | None = None
) -> bool:
    """True only if `password` produced `encoded`.

    **Parameters come from the stored value, not from config.** A row written
    under weaker settings still has to verify, or strengthening the KDF logs
    every existing agent out and reads as an outage.

    Anything unparseable is False rather than an exception. A truncated or
    hand-edited row must cost one login, not every request that touches it —
    and an exception escaping here would be an authentication failure
    presenting as a 500, which is the shape of an outage rather than of a
    refusal.
    """
    try:
        name, n, r, p, salt, expected = encoded.split(_SEPARATOR)
        # compare_digest on a value that is not secret, because the rule in
        # this module has no exceptions. A whitelist of "comparisons that are
        # fine" is where the next `==` on a digest hides.
        if not hmac.compare_digest(name.encode("ascii"), ALGORITHM.encode("ascii")):
            return False
        derived = _derive(
            password, salt=_decode(salt), n=int(n), r=int(r), p=int(p)
        )
    except (ValueError, binascii.Error, MemoryError):
        # ValueError covers both the wrong field count and a non-numeric
        # parameter; MemoryError covers a row claiming an `n` this machine
        # cannot honour, which is refusable but not a crash.
        return False
    # The only comparison in this module, and never `==`. See the docstring.
    return hmac.compare_digest(derived, _decode(expected))


def _checked(settings: dict[str, Any]) -> tuple[int, int, int]:
    """The floor, enforced where config cannot reach.

    Raised rather than clamped. Silently substituting the floor would mean a
    config file saying one thing and the system doing another, and the next
    person to read the file would believe it.
    """
    floor = settings.get("floor") or {}
    n, r, p = int(settings["n"]), int(settings["r"]), int(settings["p"])
    for name, value in (("n", n), ("r", r), ("p", p)):
        minimum = floor.get(name)
        if minimum is not None and value < int(minimum):
            raise ValueError(
                f"kdf.{name} is {value}, below the floor of {minimum}. "
                "Config may strengthen the KDF and may not weaken it — a lower "
                "work factor downgrades every password written afterwards and "
                "nothing about the running system looks different."
            )
    return n, r, p


def _derive(password: str, *, salt: bytes, n: int, r: int, p: int) -> bytes:
    return hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt,
        n=n,
        r=r,
        p=p,
        maxmem=_MEMORY_PER_BLOCK * n * r * _MEMORY_SLACK,
        dklen=_DERIVED_BYTES,
    )


def _random(size: int) -> bytes:
    import secrets

    return secrets.token_bytes(size)


def _encode(raw: bytes) -> str:
    return base64.b64encode(raw).decode("ascii")


def _decode(value: str) -> bytes:
    return base64.b64decode(value, validate=True)


__all__ = ["ALGORITHM", "hash_password", "parameters", "verify_password"]
