# Email: what each tenant must put in DNS

> Owner: whoever runs the tenant's DNS, not this repository. Nothing here is
> code, and nothing here can be fixed on demo day — the shortest change below
> is minutes and the longest is a university IT ticket.

Email is the only channel where the platform can be completely correct and the
customer still never sees the reply. WhatsApp either delivers or returns an
error; a message that fails authentication is *accepted* by the receiving
server and filed in spam, silently, with no signal back to us. Design doc §6.2
calls this out as a common silent failure in Egypt-hosted setups.

Two separate things have to be true, and they are configured in different
places:

| | Purpose | Record | Who notices when it is wrong |
|---|---|---|---|
| **Receiving** | Mail addressed to the tenant reaches Inbound Parse | `MX` | Everyone, immediately — mail bounces |
| **Sending** | The tenant's replies are trusted | `CNAME` ×3, `TXT` (DMARC) | Nobody — replies go to spam |

---

## 1. Receiving: never point the main domain's MX at SendGrid

Inbound Parse works by taking over MX for a hostname. Pointing
`sinai.edu.eg` at SendGrid would route **all** of the university's mail —
payroll, staff, students — into this platform. Use a dedicated subdomain:

```
ask.sinai.edu.eg.    MX    10 mx.sendgrid.net.
```

The tenant's mailbox is then `admissions@ask.sinai.edu.eg`, and that address is
what goes in `channel_accounts.address` and in the Inbound Parse URL
(`/webhooks/email/{account_ref}`).

Use the **same** subdomain for sending, so that a customer replying to our
reply comes back through Parse rather than into a mailbox nobody reads.

The Inbound Parse URL carries HTTP basic auth. That credential is the only
thing authenticating deliveries — SendGrid signs nothing — so it is per tenant
and belongs in the secret store, never in a ticket.

---

## 2. Sending: three CNAMEs and one TXT

In SendGrid, **Settings → Sender Authentication → Authenticate Your Domain**,
for `ask.sinai.edu.eg`, with automated security on. SendGrid then asks for
three CNAMEs:

```
em1234.ask.sinai.edu.eg.        CNAME   u1234567.wl.sendgrid.net.
s1._domainkey.ask.sinai.edu.eg. CNAME   s1.domainkey.u1234567.wl.sendgrid.net.
s2._domainkey.ask.sinai.edu.eg. CNAME   s2.domainkey.u1234567.wl.sendgrid.net.
```

The exact hostnames are generated per SendGrid account — copy them from the
console, do not transcribe the ones above.

- The `em####` record is the **return path**. It is what makes SPF *align* with
  the From domain, which is the part that matters and the part that is usually
  missing.
- The two `_domainkey` records are **DKIM**. Two of them so a key can be
  rotated without a gap.

Then DMARC, which is a `TXT` record and is not created by SendGrid:

```
_dmarc.ask.sinai.edu.eg.  TXT  "v=DMARC1; p=none; rua=mailto:dmarc@sinai.edu.eg; fo=1"
```

Start at `p=none`. It changes nothing about delivery and turns on the reports
that say whether alignment actually works. Move to `p=quarantine`, then
`p=reject`, once a week of reports shows every legitimate source passing.
Publishing `p=reject` first rejects the tenant's own mail.

### Do not add `include:sendgrid.net` to the main SPF record

This is the step everyone takes and it is usually both unnecessary and
harmful.

- With domain authentication set up, the envelope sender is the `em####`
  subdomain, which SendGrid already publishes SPF for. Alignment comes from
  there; the apex record is not consulted.
- **Two SPF records on one name is a permanent error**, not a merge. Sinai
  almost certainly already publishes one for Microsoft 365 or Google
  Workspace. A second `v=spf1` TXT makes SPF fail for *all* of their mail,
  including the mail this platform never touches.
- SPF allows **10 DNS lookups** total. An apex record that already includes a
  mail provider is often close to that limit, and an extra `include:` pushes
  it over — which fails silently, exactly like everything else here.

If a merge really is needed, edit the existing record; never add a second one.

---

## 3. The two tenants, and how long each takes

| | Sinai University | The broker |
|---|---|---|
| Domain | `sinai.edu.eg` (`.edu.eg` — registry-managed) | a commercial domain at a registrar |
| Who edits DNS | university IT, by ticket | the broker, in a hosting panel |
| Realistic lead time | **1–2 weeks** | **an hour** |
| Likely existing SPF | yes — Microsoft 365 or Google | maybe, from a website mail form |
| Risk | ticket bounces between IT and the registrar | pasting the record with the domain appended twice |

Two failure modes are specific to this list:

- A panel that appends the domain automatically turns
  `s1._domainkey.ask.sinai.edu.eg` into
  `s1._domainkey.ask.sinai.edu.eg.ask.sinai.edu.eg`. Enter the host part only
  when the panel shows a domain suffix beside the field.
- `.edu.eg` records sometimes have to go through the academic network's own
  registry rather than the university's own DNS. Ask which, in the first
  ticket, not the third.

**Start both before Task 41.** Everything else in the demo can be built and
rebuilt in an afternoon; this cannot, and a reply in the spam folder is not
something anyone can debug in the room.

---

## 4. Verifying, before anyone claims it works

```bash
dig +short MX ask.sinai.edu.eg
dig +short CNAME s1._domainkey.ask.sinai.edu.eg
dig +short TXT _dmarc.ask.sinai.edu.eg
```

Then send one real message to a Gmail address and open **Show original**. It
must read `SPF: PASS`, `DKIM: PASS` and `DMARC: PASS`, with the domain beside
each one matching the From domain. `SPF: PASS` on a different domain is the
exact condition this platform's inbound adapter refuses, and it is what a
half-finished setup looks like.

Lower the TTL on any record before changing it, not after — a record cached at
86400 is a day of not knowing whether the fix worked.

New sending domains have no reputation. The first day of real volume should be
tens of messages, not thousands.

---

## 5. What this buys on the inbound side

The same records are what lets *this* platform trust mail arriving from the
tenant's own people. `moc.channels.sendgrid_email.authentication` refuses any
message whose From domain is backed by neither SPF nor DKIM, because
`sender_ref` is the key a conversation is looked up by — an unauthenticated
From is a request to be handed somebody else's thread.

A tenant whose correspondents fail that check has a DNS problem at the
*sender's* end. The check is deliberately the stricter of the two possible
mistakes: a refusal we can explain beats an acceptance we cannot.
