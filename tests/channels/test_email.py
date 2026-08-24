"""Email through SendGrid Inbound Parse — demo plan Task 37.

The last channel, and the one least like the others. Four differences carry
most of these tests.

**Nothing signs the body.** Twilio HMACs the request and Meta HMACs the
payload; SendGrid's Inbound Parse posts an unauthenticated multipart form to
whatever URL is configured. The only credential available is HTTP basic auth
embedded in that URL, so `verify_secret` is a bearer check — the same honest
naming Telegram got, and for the same reason.

**`From:` is not authenticated by SPF.** SPF authenticates the envelope
sender, which nothing in a mail client ever shows. A message can pass SPF on
`evil.example` while claiming `From: registrar@sinai.edu.eg`, and an adapter
that reads the `SPF` field as "this is who they say they are" will hand a
forged identity to the conversation store — where it selects a real student's
thread. `test_spf_pass_on_another_domain_does_not_authenticate_the_from_address`
is the one that matters here.

**Robots answer robots.** An out-of-office autoresponder replies to our reply,
we answer it, it replies again — a loop between two machines that costs a model
call per turn and never stops on its own. This is Meta's echo problem, except
it crosses organisations, so neither end can fix it alone.

**The customer's words arrive wrapped in everything they have said before.**
A reply quotes the whole thread. Left in, every turn re-asks all the earlier
questions and the grounding gate sees figures from our own previous replies as
though the customer had written them.
"""

import json
from uuid import uuid4

import httpx
import pytest

from moc.channels.base import Channel, ChannelAccount, OutsideServiceWindow
from moc.channels.sendgrid_email import (
    NotACustomerTurn,
    SendGridEmail,
    Unauthenticated,
    authentication,
    parse_inbound,
    strip_quoted,
    verify_secret,
)

CREDENTIAL = "sendgrid:parse-password"  # noqa: S105 - a test fixture
API_KEY = "SG.a-real-looking-api-key"  # noqa: S105 - a test fixture
MAILBOX = "admissions@sinai.edu.eg"
STUDENT = "mariam@example.com"
BOUNDARY = "xYzZY"

ACCOUNT = ChannelAccount(
    id=uuid4(),
    tenant_id=uuid4(),
    channel=Channel.email,
    account_ref=MAILBOX,
    secret_ref="sendgrid/sinai/parse",
)

CONTENT_TYPE = f'multipart/form-data; boundary="{BOUNDARY}"'


def form(**fields: bytes) -> bytes:
    """A SendGrid Inbound Parse POST, in the vendor's own wire shape.

    Values are bytes on purpose: SendGrid sends each field in whatever charset
    the original message used and declares it in `charsets`, so a fixture that
    took `str` would quietly assume the one encoding this adapter must not.
    """
    body = b""
    for name, value in fields.items():
        name = name.replace("_", "-") if name.startswith("attachment_") else name
        body += (
            f"--{BOUNDARY}\r\n"
            f'Content-Disposition: form-data; name="{name}"\r\n\r\n'
        ).encode() + value + b"\r\n"
    return body + f"--{BOUNDARY}--\r\n".encode()


def headers(**overrides: str) -> bytes:
    """The original message's RFC 5322 headers, as SendGrid passes them."""
    lines = {
        "From": f"Mariam Adel <{STUDENT}>",
        "To": MAILBOX,
        "Subject": "Fees",
        "Message-ID": "<msg-1@example.com>",
        "Date": "Sat, 23 Aug 2026 10:00:00 +0200",
    }
    lines.update(overrides)
    return "\r\n".join(f"{k}: {v}" for k, v in lines.items() if v is not None).encode()


def email(
    *,
    text: bytes = "كام رسوم الساعة المعتمدة؟".encode(),
    spf: bytes = b"pass",
    dkim: bytes = b"{@example.com : pass}",
    envelope_from: str = STUDENT,
    subject: bytes = b"Fees",
    charsets: bytes | None = None,
    **header_overrides: str,
) -> bytes:
    return form(
        headers=headers(**header_overrides),
        **{"from": f"Mariam Adel <{STUDENT}>".encode()},
        to=MAILBOX.encode(),
        subject=subject,
        text=text,
        envelope=json.dumps({"to": [MAILBOX], "from": envelope_from}).encode(),
        SPF=spf,
        dkim=dkim,
        charsets=charsets or b'{"subject":"UTF-8","text":"UTF-8","from":"UTF-8"}',
    )


def parse(raw: bytes = b"", **kwargs):
    return parse_inbound(
        raw_body=raw or email(),
        content_type=CONTENT_TYPE,
        account=ACCOUNT,
        **kwargs,
    )


# ─────────────────────────── the credential ───────────────────────────


def test_the_verifying_function_contains_no_equality_comparison():
    """Structural, as in every other adapter here.

    Scoped to `verify_secret` alone: this module also parses mail, where
    comparing a header name to a string is ordinary code and `compare_digest`
    on it would be noise that dilutes the signal.
    """
    import ast
    import inspect

    from moc.channels import sendgrid_email

    tree = ast.parse(inspect.getsource(sendgrid_email))
    found = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "verify_secret"
    ]
    assert found, "verify_secret was renamed — this rule now guards nothing"
    compares = [
        inner
        for inner in ast.walk(found[0])
        if isinstance(inner, ast.Compare)
        and any(isinstance(op, ast.Eq | ast.NotEq) for op in inner.ops)
    ]
    assert compares == [], "verify_secret compares with == rather than compare_digest"


def test_the_configured_credential_verifies():
    import base64

    presented = "Basic " + base64.b64encode(CREDENTIAL.encode()).decode()
    assert verify_secret(presented=presented, expected=CREDENTIAL) is True


def test_a_wrong_credential_is_refused():
    import base64

    presented = "Basic " + base64.b64encode(b"sendgrid:guessed").decode()
    assert verify_secret(presented=presented, expected=CREDENTIAL) is False


@pytest.mark.parametrize(
    ("presented", "expected"),
    [(None, CREDENTIAL), ("", CREDENTIAL), ("Basic ", CREDENTIAL), ("Bearer x", CREDENTIAL)],
)
def test_a_missing_credential_fails_closed(presented, expected):
    assert verify_secret(presented=presented, expected=expected) is False


def test_an_unconfigured_mailbox_accepts_nothing():
    """Two empty strings compare equal. Without the guard, a channel account
    whose secret was never set accepts every stranger who omits the header."""
    assert verify_secret(presented="Basic ", expected="") is False
    assert verify_secret(presented=None, expected="") is False


# ─────────────────────── who the message is actually from ───────────────────────


def test_spf_pass_on_another_domain_does_not_authenticate_the_from_address():
    """The forgery this whole check exists for.

    SPF authenticates the envelope sender — the `Return-Path`, which no mail
    client displays. Anyone can pass SPF on a domain they own while writing
    whatever they like in `From:`. Reading the `SPF` field as identity hands
    a forged address to the conversation store, and the conversation store
    keys threads on exactly that address.
    """
    result = authentication(
        from_address="registrar@sinai.edu.eg",
        spf="pass",
        dkim=None,
        envelope_from="bounce@evil.example",
    )
    assert result.aligned is False
    assert "align" in result.reason.lower()


def test_spf_pass_on_the_from_domain_authenticates():
    result = authentication(
        from_address=STUDENT, spf="pass", dkim=None, envelope_from=STUDENT
    )
    assert result.aligned is True


def test_dkim_on_the_from_domain_authenticates():
    result = authentication(
        from_address=STUDENT,
        spf=None,
        dkim="{@example.com : pass}",
        envelope_from="bounce@sendgrid.example",
    )
    assert result.aligned is True


def test_a_forwarded_message_with_broken_spf_still_authenticates_on_dkim():
    """Why DMARC accepts either signal and not both.

    A forwarder rewrites the envelope and SPF fails at the new hop, while the
    DKIM signature travels with the message and still verifies. Requiring SPF
    would refuse every student whose university forwards their mail."""
    result = authentication(
        from_address=STUDENT,
        spf="softfail",
        dkim="{@example.com : pass}",
        envelope_from="forwarder@relay.example",
    )
    assert result.aligned is True


def test_a_failing_dkim_signature_is_not_a_pass():
    result = authentication(
        from_address=STUDENT,
        spf="fail",
        dkim="{@example.com : fail}",
        envelope_from=STUDENT,
    )
    assert result.aligned is False


def test_a_subdomain_is_aligned_with_its_parent():
    """SendGrid signs from a subdomain of the sending domain, so strict
    alignment would refuse the platform's own mail coming back."""
    result = authentication(
        from_address="noreply@sinai.edu.eg",
        spf=None,
        dkim="{@em1234.sinai.edu.eg : pass}",
        envelope_from="bounce@em1234.sinai.edu.eg",
    )
    assert result.aligned is True


def test_a_lookalike_domain_is_not_aligned():
    """`notsinai.edu.eg` ends with `sinai.edu.eg` as a string and is a
    different organisation. Alignment is on label boundaries, not substrings."""
    result = authentication(
        from_address="registrar@sinai.edu.eg",
        spf="pass",
        dkim=None,
        envelope_from="anyone@notsinai.edu.eg",
    )
    assert result.aligned is False


def test_an_unauthenticated_message_is_not_parsed_into_a_turn():
    with pytest.raises(Unauthenticated):
        parse(email(spf=b"fail", dkim=b"{@evil.example : fail}", envelope_from="x@evil.example"))


# ─────────────────────────── robots answering robots ───────────────────────────


@pytest.mark.parametrize(
    "header",
    [
        {"Auto-Submitted": "auto-replied"},
        {"Auto-Submitted": "auto-generated"},
        {"X-Autoreply": "yes"},
        {"Precedence": "bulk"},
        {"List-Id": "<students.sinai.edu.eg>"},
        {"List-Unsubscribe": "<mailto:leave@list.example>"},
        {"Return-Path": "<>"},
    ],
)
def test_a_machine_generated_message_is_not_a_customer_turn(header):
    """Each of these is a mail loop waiting to happen.

    An out-of-office reply answers our reply, which answers it, forever — and
    unlike Meta's echo, the other end belongs to someone else and cannot be
    fixed from here. `Return-Path: <>` is the null sender: a bounce, by
    definition, and replying to it bounces again."""
    with pytest.raises(NotACustomerTurn):
        parse(email(**header))


def test_an_ordinary_message_is_a_customer_turn():
    """The negative control. Every rule above refuses something; without this,
    a rule that refused everything would pass all of them."""
    assert parse().text


# ─────────────────────────── what the customer said ───────────────────────────


def test_the_quoted_thread_is_not_part_of_what_the_customer_said():
    body = (
        "شكرا، وكام المصاريف للطب؟\r\n"
        "\r\n"
        "On Sat, 23 Aug 2026 at 10:00, Sinai <admissions@sinai.edu.eg> wrote:\r\n"
        "> رسوم الساعة المعتمدة 1400 جنيه\r\n"
        "> مع تحياتنا\r\n"
    )
    assert strip_quoted(body) == "شكرا، وكام المصاريف للطب؟"


def test_an_arabic_quote_marker_is_stripped_too():
    body = "تمام\r\n\r\nفي السبت، 23 أغسطس 2026، كتب:\r\n> رسوم الساعة 1400 جنيه\r\n"
    assert strip_quoted(body) == "تمام"


def test_a_figure_from_our_own_earlier_reply_does_not_come_back_as_the_customers_words():
    """The reason this is not cosmetic. The quoted block holds figures this
    system composed; left in `text`, the next turn treats them as something
    the customer stated."""
    body = "طب والماجستير؟\r\n\r\nOn Sat, X wrote:\r\n> الرسوم 1400 جنيه للساعة\r\n"
    assert "1400" not in strip_quoted(body)


def test_a_signature_is_not_part_of_the_question():
    body = "متى موعد التقديم؟\r\n-- \r\nMariam Adel\r\n+20 100 000 0000\r\n"
    assert strip_quoted(body) == "متى موعد التقديم؟"


def test_a_message_that_is_entirely_quoted_keeps_its_original_text():
    """Bottom-posting. Stripping to nothing loses the turn altogether, and a
    noisy message is worth more than an empty one."""
    body = "> كام الرسوم؟\r\n> شكرا\r\n"
    assert strip_quoted(body).strip() != ""


def test_a_subject_only_email_is_still_a_turn():
    """Very common, and the whole message. An empty body must not become an
    empty turn — the customer would get a fallback for a question they asked."""
    message = parse(email(text=b"", subject="متى موعد التقديم؟".encode()))
    assert "التقديم" in (message.text or "")


def test_an_html_only_message_still_carries_words():
    raw = form(
        headers=headers(),
        **{"from": f"Mariam <{STUDENT}>".encode()},
        to=MAILBOX.encode(),
        subject=b"Fees",
        html="<html><body><p>كام الرسوم؟</p></body></html>".encode(),
        envelope=json.dumps({"to": [MAILBOX], "from": STUDENT}).encode(),
        SPF=b"pass",
        dkim=b"{@example.com : pass}",
        charsets=b'{"html":"UTF-8"}',
    )
    assert "كام الرسوم؟" in (parse(raw).text or "")


def test_a_body_is_decoded_with_the_charset_sendgrid_declares():
    """SendGrid sends each field in the original message's charset and names it
    in `charsets`. Assuming UTF-8 turns an Arabic message from an older Windows
    client into mojibake — which reaches retrieval as a query of replacement
    characters and produces a confident answer to nothing."""
    arabic = "كام الرسوم؟"
    message = parse(
        email(
            text=arabic.encode("windows-1256"),
            charsets=b'{"text":"windows-1256","subject":"UTF-8"}',
        )
    )
    assert arabic in (message.text or "")


def test_an_encoded_word_subject_arrives_as_words():
    """`=?UTF-8?B?…?=` is what an Arabic subject looks like on the wire."""
    message = parse(email(text=b"", subject=b"=?UTF-8?B?2YXYsdit2KjYpw==?="))
    assert "مرحبا" in (message.text or "")


def test_an_oversized_body_is_truncated_rather_than_queued_whole():
    """A mail thread has no size bound and this box has 3.3 GB. The cap is on
    what crosses the queue, and the truncation is marked so nobody reads the
    tail as the end of the message."""
    message = parse(email(text=("ا" * 200_000).encode()))
    assert len(message.text or "") < 100_000


# ─────────────────────────── identity and threading ───────────────────────────


def test_the_sender_is_the_address_not_the_display_name():
    """A display name is not an identity: the same student writes from two
    clients with two names, and the conversation store keys on this."""
    message = parse()
    assert message.sender_ref == STUDENT


def test_the_sender_is_lowercased():
    """Addresses are case-insensitive in practice and mail clients vary, so
    `Mariam@Example.com` and `mariam@example.com` must be one conversation."""
    message = parse(
        email(**{"From": "Mariam <Mariam@Example.com>"}, envelope_from="Mariam@Example.com")
    )
    assert message.sender_ref == "mariam@example.com"


def test_the_message_id_is_the_idempotency_key():
    assert parse().provider_message_id == "msg-1@example.com"


def test_a_message_with_no_message_id_still_gets_a_stable_distinct_key():
    """A constant fallback would make every unidentified message a duplicate
    of the first, and the idempotency guard would discard them all."""
    one = parse(email(**{"Message-ID": None}, text=b"first"))
    two = parse(email(**{"Message-ID": None}, text=b"second"))
    again = parse(email(**{"Message-ID": None}, text=b"first"))
    assert one.provider_message_id not in ("", None)
    assert one.provider_message_id != two.provider_message_id
    assert one.provider_message_id == again.provider_message_id


def test_the_thread_root_is_the_first_reference_not_the_last():
    """`References` is the chain from the root forward, and it arrives folded
    across lines. The root is what keeps a long thread one thread."""
    message = parse(
        email(
            **{
                "References": "<root@example.com>\r\n <second@example.com>",
                "In-Reply-To": "<second@example.com>",
            }
        )
    )
    assert message.thread_ref == "root@example.com"


def test_a_message_with_only_in_reply_to_threads_on_it():
    """Outlook sets `In-Reply-To` and often no `References`."""
    message = parse(email(**{"In-Reply-To": "<earlier@example.com>"}))
    assert message.thread_ref == "earlier@example.com"


def test_a_first_email_starts_a_thread_at_itself():
    assert parse().thread_ref == "msg-1@example.com"


def test_the_arrival_time_is_used_and_not_the_senders_own_date_header():
    """The one adapter that ignores the timestamp in the payload.

    Telegram's and Meta's come from the platform; `Date:` comes from the
    sender's own machine. A skewed or forged clock would pin a message to the
    top of the agent's thread permanently."""
    from datetime import UTC, datetime

    message = parse(email(**{"Date": "Tue, 01 Jan 2036 00:00:00 +0000"}))
    assert message.received_at.year == datetime.now(UTC).year


def test_the_original_headers_survive_in_raw_and_the_html_body_does_not():
    """`raw` settles disputes, and for this channel alone it is not the whole
    message: an unbounded mail thread on a 3.3 GB box is a memory problem, so
    the HTML alternative is dropped and the text is capped."""
    message = parse()
    assert "headers" in message.raw
    assert "html" not in message.raw


def test_an_attachment_is_recorded_by_name_and_its_bytes_are_not_queued():
    """Inbound Parse hands over the bytes and keeps nothing, so an attachment
    dropped here is gone. Recording the name is the honest half: the agent can
    see that a certificate was sent and ask for it again."""
    raw = form(
        headers=headers(),
        **{"from": f"Mariam <{STUDENT}>".encode()},
        to=MAILBOX.encode(),
        subject=b"Certificate",
        text=b"attached",
        envelope=json.dumps({"to": [MAILBOX], "from": STUDENT}).encode(),
        SPF=b"pass",
        dkim=b"{@example.com : pass}",
        charsets=b'{"text":"UTF-8"}',
    )
    raw = raw[: -len(f"--{BOUNDARY}--\r\n")] + (
        f"--{BOUNDARY}\r\n"
        'Content-Disposition: form-data; name="attachment1"; filename="shahada.pdf"\r\n'
        "Content-Type: application/pdf\r\n\r\n"
    ).encode() + b"%PDF-1.4 fake bytes" + f"\r\n--{BOUNDARY}--\r\n".encode()

    message = parse(raw)
    assert [ref.url for ref in message.media] == ["shahada.pdf"]
    assert b"%PDF" not in json.dumps(message.raw, ensure_ascii=False).encode()


def test_a_malformed_post_is_a_value_error_not_a_crash():
    with pytest.raises(ValueError):
        parse_inbound(raw_body=b"not multipart at all", content_type="text/plain", account=ACCOUNT)


# ─────────────────────────── the reply ───────────────────────────


def sender(capture) -> SendGridEmail:
    return SendGridEmail(
        api_key=API_KEY,
        sender=MAILBOX,
        sender_name="Sinai University",
        transport=httpx.MockTransport(capture),
    )


def accepted(request: httpx.Request) -> httpx.Response:
    return httpx.Response(202, headers={"X-Message-Id": "sg-1"})


async def test_the_reply_threads_on_the_customers_message():
    sent = []

    def capture(request):
        sent.append(json.loads(request.content))
        return accepted(request)

    bot = sender(capture)
    await bot.send(
        to=STUDENT, text="الرسوم 1400 جنيه", thread_ref="root@example.com", subject="Fees"
    )
    await bot.aclose()

    sent_headers = sent[0]["personalizations"][0]["headers"]
    assert sent_headers["In-Reply-To"] == "<root@example.com>"
    assert "root@example.com" in sent_headers["References"]


async def test_the_reply_says_it_is_automatic_so_the_other_end_stops():
    """The outbound half of the loop rule. An autoresponder that honours
    RFC 3834 will not answer this, which is the only way the loop is stopped
    from a mailbox we do not control."""
    sent = []

    def capture(request):
        sent.append(json.loads(request.content))
        return accepted(request)

    bot = sender(capture)
    await bot.send(to=STUDENT, text="أهلا", subject="Fees")
    await bot.aclose()

    assert sent[0]["personalizations"][0]["headers"]["Auto-Submitted"] == "auto-generated"


async def test_the_reply_subject_does_not_stack_re_prefixes():
    sent = []

    def capture(request):
        sent.append(json.loads(request.content))
        return accepted(request)

    bot = sender(capture)
    await bot.send(to=STUDENT, text="أهلا", subject="Re: Fees")
    await bot.aclose()

    assert sent[0]["subject"] == "Re: Fees"


async def test_the_reply_comes_from_the_tenants_own_address():
    """One platform sender for every tenant would put Sinai's replies under
    the broker's name — and, worse, under a domain neither of them can
    authenticate."""
    sent = []

    def capture(request):
        sent.append(json.loads(request.content))
        return accepted(request)

    bot = sender(capture)
    await bot.send(to=STUDENT, text="أهلا", subject="Fees")
    await bot.aclose()

    assert sent[0]["from"]["email"] == MAILBOX


async def test_email_has_no_service_window():
    """WhatsApp's 24-hour rule is Meta's, not the platform's. Inheriting it
    here would refuse every reply to a mail older than a day."""
    bot = sender(accepted)
    receipt = await bot.send(to=STUDENT, text="أهلا", subject="Fees", last_inbound_at=None)
    await bot.aclose()
    assert receipt.provider_message_id == "sg-1"


async def test_the_window_refusal_is_not_reachable_from_here():
    bot = sender(accepted)
    try:
        await bot.send(to=STUDENT, text="أهلا", subject="Fees")
    except OutsideServiceWindow:  # pragma: no cover - the assertion is that this is unreachable
        pytest.fail("email has no service window and must not refuse on one")
    await bot.aclose()


async def test_the_api_key_is_absent_from_a_failed_send():
    """Whatever escapes ends up as `repr(exc)` in a dead-letter row."""
    from moc.channels.sendgrid_email import EmailRefused

    def refuse(request):
        return httpx.Response(401, json={"errors": [{"message": "permission denied"}]})

    bot = sender(refuse)
    with pytest.raises(EmailRefused) as raised:
        await bot.send(to=STUDENT, text="أهلا", subject="Fees")
    await bot.aclose()

    assert API_KEY not in str(raised.value)
    assert API_KEY not in repr(raised.value)
    assert "permission denied" in str(raised.value)
