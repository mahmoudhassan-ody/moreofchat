# The real-phone path

> What this repository can prove on its own, what it cannot, and the exact
> procedure for the part that needs a phone.

Until Task 39 nothing in this system had ever been started. Every app was
assembled inside a test, from fakes, over `ASGITransport`. That is not a
missing convenience — an object graph only ever built by tests is one whose
production shape nobody has checked, and driving it for the first time found
six faults, none of which any test could have failed on.

---

## 1. What runs, and what it can reach

Three processes from one image, plus Caddy.

| Process | Command | Can reach |
|---|---|---|
| `api` | `uvicorn --factory moc.api.main:webhook_app` | `moc_lookup` (SELECT on a five-column view), Valkey, the signing secrets |
| `worker-inbound` | `python -m moc.workers.run inbound` | `moc_app`, the model providers, Qdrant, Meilisearch |
| `worker-outbound` | `python -m moc.workers.run outbound` | `moc_app`, each tenant's vendor credentials |
| `caddy` | image | ports 80 and 443 |

**The internet-facing process holds no application database credentials.** Not
by policy — it has no login that would let it read a conversation, a message, a
fee or a unit price. Everything slow and everything tenant-scoped is behind the
queue. `tests/api/test_main.py` asserts it structurally, because the failure
mode of a convenience import here is that everything keeps working.

Caddy is the only service publishing outside loopback. CLAUDE.md says compose
binds to 127.0.0.1 only; a webhook is a URL a vendor's servers must reach, so
the rule is narrowed rather than broken, and `tests/test_infra.py` asserts that
about `compose.yaml`.

---

## 2. Before anything

```bash
uv run python scripts/preflight.py
```

Every check names the one thing that is wrong and how to fix it. Three of them
exist because the first real run hit exactly that fault.

Then, to prove the wiring without a vendor:

```bash
uv run python scripts/drive_the_path.py
```

This starts the actual programs and sends a genuinely signed Twilio webhook
over TCP. Twilio's host is pointed at a stub on loopback through
`MOC_CONFIG_DIR`; everything else — the signature, the queue, the consumer
group, the model call, the adapter, the form encoding — is real. It cleans up
the tenant it seeds.

---

## 3. The leg that needs a phone

Nothing above touches a vendor. This is the part a human does.

### 3.1 A public hostname with real TLS

Twilio and Meta both refuse a webhook URL that is not HTTPS with a valid
certificate. A self-signed certificate is not a shortcut — it is a webhook that
never fires, with no error anywhere on this side.

```
A    moc.example.com    -> the VPS address
```

Set `MOC_DOMAIN` and `MOC_TLS_EMAIL`, bring Caddy up, and confirm the
certificate before going near a vendor console:

```bash
curl -sS -o /dev/null -w '%{http_code} %{ssl_verify_result}\n' https://$MOC_DOMAIN/webhooks/twilio/whatsapp
```

`403` is the right answer here. It means the request arrived, TLS verified, and
the signature check refused an unsigned POST — which is three separate things
working. A `000`, a timeout or a certificate error is DNS or Caddy, not us.

### 3.2 Connect the number

The row and the secrets are two separate steps and both are required.

```sql
INSERT INTO tenants (id, slug, name, vertical, default_lang)
VALUES (gen_random_uuid(), 'sinai', 'Sinai University', 'education', 'ar');

INSERT INTO channel_accounts (id, tenant_id, channel, address, secret_ref)
VALUES (gen_random_uuid(),
        (SELECT id FROM tenants WHERE slug = 'sinai'),
        'whatsapp', '+2015XXXXXXXX', 'twilio/sinai/wa');
```

`default_lang` is spelled out deliberately: the column is `NOT NULL` with only
a Python-side ORM default, so any insert that is not a `Tenant(...)` fails on
it.

Then the secrets. The inbound-verification secret is what `secret_ref` names;
outbound credentials hang off the same reference with a suffix, so every
variable name is derivable from the row:

```
MOC_SECRET_TWILIO__SINAI__WA        the Twilio auth token   (verifies inbound)
MOC_SECRET_TWILIO__SINAI__WA__SID   the Twilio account SID  (sends)
```

For Telegram the pair is worth stating twice, because they are *different
secrets* and swapping them produces a bot that authenticates nothing and sends
nothing, in that order:

```
MOC_SECRET_TELEGRAM__SINAI__BOT         the webhook secret token (verifies inbound)
MOC_SECRET_TELEGRAM__SINAI__BOT__TOKEN  the bot token            (sends)
```

Re-run the preflight. It reads `channel_accounts` and checks that this host has
every secret each connected account needs — an account missing its *outbound*
credential accepts messages and answers none of them: the question arrives, the
turn runs, the model is paid for, and the reply dead-letters.

### 3.3 Register the webhook with the vendor

**Telegram** — the cheapest to prove, and the only one that will tell you what
it thinks of your URL:

```bash
curl -sS "https://api.telegram.org/bot$BOT_TOKEN/setWebhook" \
  -d "url=https://$MOC_DOMAIN/webhooks/telegram/sinai_bot" \
  -d "secret_token=$WEBHOOK_SECRET"

curl -sS "https://api.telegram.org/bot$BOT_TOKEN/getWebhookInfo"
```

`getWebhookInfo` reports `last_error_message` and `last_error_date`. That field
is the single most useful diagnostic in this whole document: it is the vendor
telling you, from outside, why delivery failed.

**Twilio** — set the WhatsApp sender's inbound webhook to
`https://$MOC_DOMAIN/webhooks/twilio/whatsapp`, method POST.

**Meta** — the GET handshake happens once, at subscription. It needs
`MOC_SECRET_META__APP__VERIFY_TOKEN` set to the same string entered in the
app dashboard, and `MOC_SECRET_META__APP__SECRET` for the POST signature. Both
are platform-wide rather than per tenant: one Meta app serves every page.

### 3.4 Send a message

From a real phone, to the connected number. Then, on the host:

```bash
docker compose logs -f worker-inbound worker-outbound
```

What each failure looks like:

| Symptom | Where it is |
|---|---|
| Nothing in any log | The vendor never reached Caddy. DNS, TLS, or the URL in their console. |
| `403` in the api log | The signature failed. The auth token in the environment is not the one the vendor signs with. |
| `200` and nothing after | The message is on the queue and no worker is consuming. Check the consumer group. |
| Entry in `moc:inbound:dead` | The turn failed. The reason is on the dead-letter entry, not in the logs. |
| Reply composed, nothing delivered | The outbound credential, or a tenant with no `channel_accounts` row for that channel. |

---

## 4. What the first real run found

None of these had a behavioural signature short of "no reply arrives", and none
could have been caught by the test suite as it was written.

1. **Nothing could be started.** There was no ASGI app factory and no worker
   entrypoint. `compose.yaml` held four backing stores and no application.

2. **The dev database was eight migrations behind** — at `0009` while the code
   was at `0017`. Every test migrates a fresh `moc_test`, so the database the
   system would actually run against had not been touched since Task 23.

3. **`.env` defined `MOC_LOOKUP_PASSWORD` twice**, with different values. The
   last assignment wins in both `set -a; . .env` and pydantic-settings, and the
   role's password had been set from the other one. The webhook process could
   not authenticate at all.

4. **A worker blocking on an idle stream died within five seconds.**
   `block_ms` is 5000 and redis-py 8 defaults `socket_timeout` to exactly 5.
   Nothing caught it because nothing had ever called `run_once(block=True)` —
   every test polls with `block=False` and returns immediately, so the one call
   shape a deployed worker uses was the one shape never exercised.

5. **The retriever was bound to one tenant for the life of the process.** A
   second tenant's question would have been answered from the first tenant's
   corpus — a cross-tenant read arriving as a fluent, correctly-cited reply.
   RLS cannot catch it: the retriever holds the tenant id it filters on, and it
   held the wrong one.

6. **The outbound worker held one sender per channel for the life of the
   process**, so every tenant's replies would have gone out from one number.
   The Twilio adapter's own docstring says the sender is never platform-wide;
   there was no wiring through which that could be honoured.

Five and six are fixed and pinned by tests that fail when the fix is removed.
Two, three and four are fixed and checked by the preflight.

---

## 5. What is still not proven

- **No message has been sent from a real phone.** Everything above is loopback
  and a stub. §3 is the procedure, not a record.
- **The console is not deployed.** It builds, it passes its own smoke gate in a
  real browser, and no process serves it. It needs the same treatment this
  document gives the webhook: a composition root, a service, and a Caddy route.
- **Real estate has no worker path.** Inventory turns are a different agent
  with a different result type, and `worker-inbound` refuses a tenant whose
  vertical it cannot serve rather than answering them with the education
  script. A broker's WhatsApp number cannot be connected until that exists.
- **The image has never run against the real stack.** It builds and its
  entrypoints import; `docker compose up` has not been driven end to end.
