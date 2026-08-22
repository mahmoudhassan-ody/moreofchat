"""Tenant identity — demo plan Task 30, design §17.

Three tenants see this product and each has to see *themselves*. That is the
whole of this module: their name in the header, their crest, and a fallback
that is theirs rather than ours.

**The fallback is the tenant's initials, never the More Of Chat mark.** A
university seeing our logo where their crest belongs reads as a product that
does not know who they are, which is the opposite of what a pilot is for. Our
mark stays in the corner, on every page, and that is deliberate too: white
label is out of scope, because a working product with our name on it is how the
university's IT director tells the broker about us.

**The media type is sniffed from the bytes.** An extension is a claim made by
whoever is uploading, and the reason to check at all is that the claim might be
false. `logo.png` holding a shell script is the shape that matters: trust the
name and it is stored, and whatever serves it later decides what it is.

**SVG is refused although it is an image.** It is a document format that
executes: a tenant-supplied SVG can carry `<script>`, and a console rendering
one inline has handed an uploader a cross-tenant XSS. Refused rather than
sanitised — sanitising SVG is a losing arms race and nobody's logo needs to be
one.
"""

import uuid
from dataclasses import dataclass
from typing import Any

from sqlalchemy import text

from moc.tenancy.context import tenant_session

#: 512 KiB. This column is read on every console page load and travels with the
#: tenant row; a crest that does not fit is a crest that was never exported for
#: the web.
MAX_LOGO_BYTES = 512 * 1024

#: Magic numbers, in the order they are tested. Every entry is a prefix of the
#: file's first bytes — WebP is the exception and is handled below, because its
#: marker sits at offset 8 behind a RIFF container.
_SIGNATURES: tuple[tuple[bytes, str], ...] = (
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
)
_RIFF = b"RIFF"
_WEBP = b"WEBP"
_WEBP_OFFSET = 8

#: Enough bytes to see any signature above, and to spot an SVG's opening tag
#: past a byte-order mark, a comment or an XML declaration.
_SNIFF = 1024

_FALLBACK_INITIAL = "?"
_INITIALS = 2


class NotAnImage(ValueError):
    """The upload is refused. Never a warning, and never stored anyway."""


@dataclass(frozen=True)
class Brand:
    """What the console header needs, and nothing it does not.

    The bytes are deliberately absent: the header renders a name and either a
    logo URL or initials, and shipping half a megabyte of PNG inside a JSON
    response on every page load would be a strange way to serve an image.
    """

    tenant_id: uuid.UUID
    name: str
    initials: str
    has_logo: bool
    media_type: str | None
    timezone: str
    default_language: str


def initials(name: str) -> str:
    """Two characters that stand for a tenant, in whatever script it writes in.

    Two words give their first letters — "Sinai University" is SU, "جامعة
    سيناء" is جس. One word gives its first two. `.upper()` is a no-op on Arabic
    and correct on Latin, so it is applied to both rather than branched on.

    Empty gives `?` rather than an empty string: an empty box reads as a broken
    console, and a missing name is a data problem worth seeing.
    """
    words = name.split()
    if not words:
        return _FALLBACK_INITIAL
    if len(words) >= _INITIALS:
        return "".join(word[0] for word in words[:_INITIALS]).upper()
    return words[0][:_INITIALS].upper()


def sniff(content: bytes) -> str:
    """The media type, from the content. Raises `NotAnImage` for anything else.

    SVG is named in its own error because "that is not an image" would be a
    lie, and the person uploading a perfectly valid logo deserves to know it
    was refused for what it can do rather than for what it is.
    """
    if not content:
        raise NotAnImage("the upload is empty")
    if len(content) > MAX_LOGO_BYTES:
        raise NotAnImage(
            f"the logo is too large: {len(content)} bytes against a "
            f"{MAX_LOGO_BYTES}-byte cap"
        )

    head = content[:_SNIFF]
    for signature, media_type in _SIGNATURES:
        if head.startswith(signature):
            return media_type
    if head.startswith(_RIFF) and head[_WEBP_OFFSET : _WEBP_OFFSET + len(_WEBP)] == _WEBP:
        return "image/webp"
    if b"<svg" in head.lower():
        raise NotAnImage(
            "SVG is refused. It is a document format that executes — a "
            "tenant-supplied SVG can carry a script, and a console that renders "
            "one inline has handed an uploader a cross-tenant XSS. Export the "
            "crest as PNG."
        )
    raise NotAnImage(
        "the file is not a PNG, JPEG, GIF or WebP. The type is read from the "
        "content rather than the file name, because the name is a claim by "
        "whoever is uploading."
    )


class BrandingStore:
    """Reads and writes one tenant's identity, under that tenant's session.

    Every method takes a `tenant_id` and there is no request in sight: this
    sits below the API, and the tenant reaching it came from a session row
    (Task 28). Callers on the request path pass `principal.tenant_id` and have
    nothing else to pass.
    """

    def __init__(self, *, engine: Any) -> None:
        self._engine = engine

    async def brand(self, *, tenant_id: uuid.UUID) -> Brand | None:
        async with tenant_session(self._engine, tenant_id) as session:
            row = (
                await session.execute(
                    text(
                        "SELECT id, name, default_lang, timezone, logo_media_type, "
                        # The bytes are not selected — only whether there are
                        # any. See `Brand`.
                        "(logo IS NOT NULL) AS has_logo FROM tenants WHERE id = :id"
                    ),
                    {"id": tenant_id},
                )
            ).one_or_none()
        if row is None:
            return None
        return Brand(
            tenant_id=row.id,
            name=row.name,
            initials=initials(row.name),
            has_logo=row.has_logo,
            media_type=row.logo_media_type,
            timezone=row.timezone,
            default_language=row.default_lang,
        )

    async def logo(self, *, tenant_id: uuid.UUID) -> tuple[bytes, str] | None:
        """The bytes and the type they were stored under, or None."""
        async with tenant_session(self._engine, tenant_id) as session:
            row = (
                await session.execute(
                    text(
                        "SELECT logo, logo_media_type FROM tenants "
                        "WHERE id = :id AND logo IS NOT NULL"
                    ),
                    {"id": tenant_id},
                )
            ).one_or_none()
        return (row.logo, row.logo_media_type) if row is not None else None

    async def set_logo(
        self, *, tenant_id: uuid.UUID, content: bytes, filename: str = ""
    ) -> str:
        """Store a logo, typed by its content. Returns the media type.

        `filename` is accepted and never trusted — it exists so a caller can
        pass what the browser sent without having to decide whether to, and so
        that the one place it *could* have been used shows plainly that it was
        not.
        """
        media_type = sniff(content)
        async with tenant_session(self._engine, tenant_id) as session:
            await session.execute(
                text(
                    "UPDATE tenants SET logo = :logo, logo_media_type = :media_type "
                    "WHERE id = :id"
                ),
                {"logo": content, "media_type": media_type, "id": tenant_id},
            )
            await session.commit()
        return media_type

    async def clear_logo(self, *, tenant_id: uuid.UUID) -> None:
        """Back to initials, not to our mark."""
        async with tenant_session(self._engine, tenant_id) as session:
            await session.execute(
                text(
                    "UPDATE tenants SET logo = NULL, logo_media_type = NULL "
                    "WHERE id = :id"
                ),
                {"id": tenant_id},
            )
            await session.commit()


__all__ = ["Brand", "BrandingStore", "MAX_LOGO_BYTES", "NotAnImage", "initials", "sniff"]
