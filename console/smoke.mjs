/* Does the built console actually mount, and does the toggle flip direction?
 *
 * **The structural gates in `tests/console/` cannot answer this.** They read
 * source as text: they catch a hard-coded string and a physical margin, and
 * they are blind to a bad import path, a wrong react-i18next API, a component
 * that throws on mount, or a `margin-inline-start` that does not do what its
 * author believed. This runs the real bundle in a real layout engine.
 *
 * Served from `dist/`, not from the dev server. The dev server transforms on
 * the fly and would let a build-only failure through, and the bundle is what a
 * demo actually serves.
 *
 *     npm run build && npm run smoke
 *
 * Chromium needs system libraries this VPS does not have and cannot install
 * without root, so `npm run smoke` runs it inside the Playwright image. The
 * container gets `--network none`: nothing here should need the internet, and
 * the one thing that reaches for it — the Google Fonts stylesheet — is
 * reported rather than tolerated silently. See the note at the bottom.
 */
import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import { extname, join } from "node:path";

import { chromium } from "playwright";

// Relative to this file, because this runs both on the host and inside the
// container where the console is mounted at /work.
const ROOT = join(import.meta.dirname, "dist");
const PORT = 4173;
const TYPES = {
  ".html": "text/html",
  ".js": "text/javascript",
  ".css": "text/css",
  ".json": "application/json",
};

const failures = [];
function check(name, actual, expected) {
  const ok = JSON.stringify(actual) === JSON.stringify(expected);
  console.log(`${ok ? "  ok  " : "FAIL  "}${name}: ${JSON.stringify(actual)}`);
  if (!ok) failures.push(`${name}: expected ${JSON.stringify(expected)}`);
}

const server = createServer(async (request, response) => {
  const path = request.url === "/" ? "/index.html" : request.url.split("?")[0];
  try {
    const body = await readFile(join(ROOT, path));
    response.writeHead(200, {
      "content-type": TYPES[extname(path)] ?? "application/octet-stream",
    });
    response.end(body);
  } catch {
    response.writeHead(404).end("not found");
  }
});
await new Promise((resolve) => server.listen(PORT, "127.0.0.1", resolve));

const browser = await chromium.launch();
const page = await browser.newPage();
const problems = [];
page.on("pageerror", (error) => problems.push(`pageerror: ${error.message}`));
// Every failed request counts now. The font CDN used to be exempted here
// because `--network none` made it unreachable by design; the fonts are in the
// bundle, so a request leaving this page at all is the defect.
page.on("requestfailed", (request) =>
  problems.push(`requestfailed: ${request.url()} ${request.failure()?.errorText}`));
page.on("request", (request) => {
  if (!request.url().startsWith("http://127.0.0.1:")) {
    problems.push(`offsite request: ${request.url()}`);
  }
});

/* The tenant API, stubbed at the network layer. The console is a static
 * bundle here with no backend behind it, and the question this answers is not
 * "does FastAPI work" — `tests/api/test_tenant_identity.py` answers that — but
 * whether the header renders a tenant it is given, and their initials when
 * there is no crest. Routed rather than mocked in the source, so the code
 * under test is exactly the code that ships. */
const CREST = Buffer.from(
  "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4890000000a4944415478da6360000002000154a24f6f0000000049454e44ae426082",
  "hex",
);
let brand = {
  name: "Cairo Homes",
  initials: "CH",
  hasLogo: false,
  timezone: "Africa/Cairo",
  defaultReplyLanguage: "ar",
};
await page.route("**/tenant", (route) =>
  route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(brand) }),
);
await page.route("**/tenant/logo", (route) =>
  brand.hasLogo
    ? route.fulfill({ status: 200, contentType: "image/png", body: CREST })
    : route.fulfill({ status: 404, body: "no logo" }),
);

await page.goto(`http://127.0.0.1:${PORT}/`, { waitUntil: "load" });
await page.waitForSelector("#root .app", { timeout: 15000 });

check("mounts", await page.locator("#root .app").count(), 1);
check("starts ltr", await page.getAttribute("html", "dir"), "ltr");
check("starts en", await page.getAttribute("html", "lang"), "en");
check("english nav", (await page.locator(".nav a").allInnerTexts())[0], "Inbox");

// Through the control a person clicks, not by calling i18next directly: the
// question is whether the toggle flips the document, not whether i18next can.
await page.locator('.lang button:has-text("AR")').click();
await page.waitForFunction(() => document.documentElement.dir === "rtl", null, {
  timeout: 5000,
});

check("flips to rtl", await page.getAttribute("html", "dir"), "rtl");
check("flips to ar", await page.getAttribute("html", "lang"), "ar");
check("arabic nav", (await page.locator(".nav a").allInnerTexts())[0], "صندوق الوارد");
check("arabic logo", await page.locator(".logo").innerText(), "مور أوف تشات");
check("toggle state", await page.locator('.lang button[aria-pressed="true"]').innerText(), "AR");

/* The assertion the CSS scanner cannot make, and it took a sabotage to write
 * it correctly: the first version compared which of the two sat further left,
 * and in RTL the nav is left of the logo whether the margin is logical or
 * physical, so `margin-left: auto` passed it.
 *
 * What `margin-inline-start: auto` actually buys is the GAP. The auto margin
 * absorbs the free space on the nav's start side, so logo and nav end up at
 * opposite ends of the bar in both directions. The physical version puts that
 * space on the nav's left in RTL, which pulls the nav up against the logo and
 * strands the language toggle — a bar that looks deliberately arranged and is
 * wrong. Measured as distance rather than order, because order does not move
 * and distance does. */
const GAP = 100;
const gap = () =>
  page.evaluate(() => {
    const nav = document.querySelector(".nav").getBoundingClientRect();
    const logo = document.querySelector(".logo").getBoundingClientRect();
    return document.documentElement.dir === "rtl" ? logo.left - nav.right : nav.left - logo.right;
  });

const rtlGap = await gap();
check(`rtl separates logo and nav by >${GAP}px`, rtlGap > GAP, true);

await page.locator('.lang button:has-text("EN")').click();
await page.waitForFunction(() => document.documentElement.dir === "ltr", null, {
  timeout: 5000,
});
const ltrGap = await gap();
check(`ltr separates logo and nav by >${GAP}px`, ltrGap > GAP, true);
/* **Which font the browser actually painted with**, not which one the CSS
 * asked for. `getComputedStyle` returns the declared stack and says nothing
 * about whether the first family in it loaded — a page that fell all the way
 * back to a system Arabic face reports exactly the same string. CDP's
 * `getPlatformFontsForNode` reports what the renderer resolved to.
 *
 * This runs with `--network none`, so a pass is proof the file came from the
 * bundle: there is no CDN reachable to have served it. Checked on Arabic text
 * specifically, because the Arabic subset is a separate file and a build that
 * shipped only the latin ones would still look perfect in English.
 */
await page.locator('.lang button:has-text("AR")').click();
await page.waitForFunction(() => document.documentElement.dir === "rtl", null, { timeout: 5000 });
await page.evaluate(() => document.fonts.ready);

const cdp = await page.context().newCDPSession(page);
await cdp.send("DOM.enable");
await cdp.send("CSS.enable");
async function paintedFont(selector) {
  // The document is re-fetched every call. CDP node ids are invalidated by a
  // navigation, and a single `root` captured up front worked right up until
  // this file navigated twice — then failed as "could not find node", which
  // reads like a missing element rather than a stale handle.
  const { root } = await cdp.send("DOM.getDocument");
  const { nodeId } = await cdp.send("DOM.querySelector", { nodeId: root.nodeId, selector });
  const { fonts } = await cdp.send("CSS.getPlatformFontsForNode", { nodeId });
  return fonts.map((font) => font.familyName);
}

/* Every element that paints text, not two hand-picked selectors. CDP reports
 * the resolved FACE — "IBM Plex Sans Arabic SemiBold", not the family — so the
 * assertion is on the prefix.
 *
 * Written as a sweep because the failure it found was not in a selector anyone
 * would have thought to check: the version number's sibling paragraph carried
 * `.mono`, IBM Plex Mono has no Arabic glyphs, and the Arabic ran in FreeSerif.
 * It rendered, it looked plausible to someone reading English, and nothing
 * logged. Task 32 puts figures beside Arabic labels on every row of the
 * inbox, which is the same trap with forty chances to spring. */
async function strayFonts() {
  const selectors = await page.evaluate(() =>
    [...document.querySelectorAll("body *")]
      .filter((element) =>
        [...element.childNodes].some(
          (node) => node.nodeType === 3 && node.textContent.trim(),
        ),
      )
      .map((element, index) => {
        element.dataset.smoke = String(index);
        return `[data-smoke="${index}"]`;
      }),
  );
  const stray = [];
  for (const selector of selectors) {
    for (const family of await paintedFont(selector)) {
      if (!family.startsWith("IBM Plex")) {
        stray.push(`${selector} -> ${family}`);
      }
    }
  }
  return stray;
}

check("arabic paints in the self-hosted families", await strayFonts(), []);

await page.locator('.lang button:has-text("EN")').click();
await page.waitForFunction(() => document.documentElement.dir === "ltr", null, { timeout: 5000 });
await page.evaluate(() => document.fonts.ready);
check("latin paints in the self-hosted families", await strayFonts(), []);

/* Identity. The fallback is the TENANT's initials — ours in that slot reads
 * as a product that does not know who they are. */
await page.waitForSelector(".tenant b", { timeout: 5000 });
check("header names the tenant", await page.locator(".tenant b").innerText(), "Cairo Homes");
check("no crest means their initials", await page.locator(".initials").innerText(), "CH");
check("and no image element at all", await page.locator(".crest").count(), 0);

brand = { ...brand, hasLogo: true, name: "Sinai University", initials: "SU" };
await page.reload({ waitUntil: "load" });
await page.waitForSelector(".crest", { timeout: 5000 });
check("a crest renders as an image", await page.locator(".crest").count(), 1);
check("and replaces the initials", await page.locator(".initials").count(), 0);
check(
  "the crest actually loaded",
  await page.locator(".crest").evaluate((img) => img.naturalWidth > 0),
  true,
);
check("header names the tenant", await page.locator(".tenant b").innerText(), "Sinai University");
check("powered-by is on the page", (await page.locator(".powered").innerText()).length > 0, true);

/* ── the knowledge screen ──────────────────────────────────────────────
 *
 * The order is the property: preview, THEN confirm. A screen that ingests
 * first and reports afterwards has already spent the money and already put a
 * broken corpus behind the bot, and no amount of reporting undoes either. */
let ingested = 0;
await page.route("**/knowledge/preview", (route) =>
  route.fulfill({
    status: 200,
    contentType: "application/json",
    body: JSON.stringify({
      chunkCount: 12,
      sample: [{ ordinal: 0, content: "رسوم الساعة المعتمدة لكلية الهندسة 1400 جنيه." }],
      warnings: [
        { name: "no_sentence_boundaries", ordinal: 0, reason: "one chunk, no terminators" },
      ],
      contentHash: "abc",
      unchanged: false,
    }),
  }),
);
await page.route("**/knowledge/documents", (route) => {
  if (route.request().method() === "POST") {
    ingested += 1;
    return route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ docId: "fees", chunkCount: 12, unchanged: false, failures: [] }),
    });
  }
  return route.fulfill({ status: 200, contentType: "application/json", body: "[]" });
});

await page.goto(`http://127.0.0.1:${PORT}/#knowledge`, { waitUntil: "load" });
await page.waitForSelector(".knowledge", { timeout: 5000 });

check("confirm does not exist before a preview", await page.locator(".preview").count(), 0);
check("and nothing has been ingested", ingested, 0);

await page.locator(".upload .field").first().fill("fees-2026");
await page.locator(".upload .body").fill("رسوم الساعة المعتمدة 1400 جنيه.");
await page.locator(".upload .act").click();
await page.waitForSelector(".preview", { timeout: 5000 });

check("the preview shows the chunk count", await page.locator(".count .mono").innerText(), "12");
check("and the chunk text itself", (await page.locator(".chunk").innerText()).includes("1400"), true);
/* Translated from the warning's NAME, not echoed from the wire. The reason
 * string in the response is written in English by whoever added the check;
 * the person reading this screen is an admissions officer in Egypt.
 *
 * The marker is "no terminators", which appears only in the stubbed wire
 * reason. An earlier version of this check looked for "one chunk" and could
 * not tell the two apart — the catalogue entry contains that phrase too. */
const warning = await page.locator(".warning").innerText();
check("the warning is translated, not echoed", warning.includes("no terminators"), false);
check(
  "and it is the catalogue's wording",
  warning.includes("No sentence ends were found"),
  true,
);
check("still nothing ingested", ingested, 0);

await page.locator(".preview .act").click();
await page.waitForSelector(".result", { timeout: 5000 });
check("confirm ingests exactly once", ingested, 1);

/* ── the inbox, and the pane the whole screen is for ──────────────────── */
const HANDOFF = "11111111-1111-1111-1111-111111111111";
await page.route("**/inbox", (route) =>
  route.fulfill({
    status: 200,
    contentType: "application/json",
    body: JSON.stringify([
      {
        id: HANDOFF,
        conversation_id: "22222222-2222-2222-2222-222222222222",
        reason: "three clarifications",
        status: "open",
        channel: "whatsapp",
        sender_ref: "+201012345678",
        opened_at: "2026-08-22T09:00:00+00:00",
        claimed_by: null,
        team: "villas",
        lead_qualified: true,
      },
    ]),
  }),
);
await page.route("**/inbox/*/thread", (route) =>
  route.fulfill({
    status: 200,
    contentType: "application/json",
    body: JSON.stringify([
      {
        channel: "whatsapp",
        author: "customer",
        body: "كام رسوم الساعة؟",
        created_at: "2026-08-22T09:00:00+00:00",
        provenance: null,
      },
      {
        channel: "whatsapp",
        author: "bot",
        body: "رسوم الساعة المعتمدة 1400 جنيه.",
        created_at: "2026-08-22T09:00:05+00:00",
        provenance: {
          figures: [
            {
              value: 1400,
              raw: "1400",
              grounded: true,
              source: "chunk",
              chunkId: "sinai_fee_hour_ar",
              title: "رسوم الساعة",
              asOf: "2026-01-01",
              excerpt: "رسوم الساعة المعتمدة لكلية الهندسة 1400 جنيه.",
            },
          ],
          gates: {
            numeric_grounding: true,
            figure_audit: true,
            figure_audit_degraded: false,
          },
        },
      },
      {
        /* §3.2's grounding mode, in the same pane. A price traces to a unit
           row and its snapshot date; an instalment to the calculator and the
           inputs it ran with. There is no figure audit on this one — no model
           composed the numbers — and a pane holding its own list of gates drew
           that absence as a failed check. */
        channel: "whatsapp",
        author: "bot",
        body: "شقة في مدينتي بـ 5,500,000 جنيه، المقدم 1,100,000.",
        created_at: "2026-08-22T09:00:09+00:00",
        provenance: {
          figures: [
            {
              value: 5500000,
              raw: "5,500,000",
              grounded: true,
              source: "inventory",
              chunkId: "MD-1",
              title: "Madinaty",
              asOf: "2026-08-01",
              excerpt: "price = 5,500,000",
            },
            {
              value: 1100000,
              raw: "1,100,000",
              grounded: true,
              source: "calculator",
              chunkId: "MD-1",
              title: null,
              asOf: "2026-08-01",
              excerpt:
                "down_payment = 1,100,000 — payment_plan_calculator(down_payment_pct=20, price=5,500,000, years=8)",
            },
          ],
          gates: { numeric_grounding: true },
        },
      },
    ]),
  }),
);

await page.goto(`http://127.0.0.1:${PORT}/#inbox`, { waitUntil: "load" });
await page.waitForSelector(".conv", { timeout: 5000 });

check("the conversation needing a human is marked", await page.locator(".pill.needs").count(), 1);

/* §11.2's routing, on the row. A team chosen and not shown is a lead routed to
   nobody: the column is written and whoever picks the conversation up is
   whoever happened to be looking. */
check("the routed sales team is on the row", await page.locator(".conv-team").count(), 1);
const routed = await page.locator(".conv-team").allInnerTexts();
check("it names the team", routed.join(" ").includes("villas"), true);
check("and it is labelled from the catalogue", routed.join(" ").includes("Routed to"), true);

await page.locator(".conv").click();
await page.waitForSelector(".bubble", { timeout: 5000 });
check("the thread renders every turn", await page.locator(".bubble").count(), 3);
check("no source pane until a reply is picked", await page.locator(".sources").count(), 0);

await page.locator(".row.out .bubble").first().click();
await page.waitForSelector(".sources", { timeout: 5000 });

check("the figure is shown", await page.locator(".figure .mono").innerText(), "1400");
check(
  "with the sentence it came from",
  (await page.locator(".excerpt").innerText()).includes("رسوم الساعة المعتمدة"),
  true,
);
check(
  "and the chunk that supplied it",
  (await page.locator(".attribution").innerText()).includes("رسوم الساعة"),
  true,
);
check("and the gates that passed", await page.locator(".gate.on").count(), 2);

/* The other grounding mode, in the same pane — demo plan Task 41b. The source
   pane is the demo's centrepiece and it was blank for two of the three
   tenants: a price traces to a unit row and its snapshot date, an instalment
   to the calculator and the inputs it ran with. */
await page.locator(".row.out .bubble").nth(1).click();
await page.waitForSelector(".sources", { timeout: 5000 });

const traced = await page.locator(".source").allInnerTexts();
check("a broker's price traces to the row it came from", traced[0].includes("price ="), true);
check("named by the compound", traced[0].includes("Madinaty"), true);
check("with the date the row was snapshotted", traced[0].includes("2026-08-01"), true);
check(
  "an instalment traces to the calculator and its inputs",
  traced[1].includes("payment_plan_calculator") && traced[1].includes("price=5,500,000"),
  true,
);
check(
  "labelled from the catalogue rather than by a tool identifier",
  traced[1].includes("Payment calculator"),
  true,
);
/* One gate, not two. There is no figure audit on an inventory reply — no model
   composed those numbers — and a pane holding its own list drew that absence
   as a failed check: a red mark for something that never ran, on the screen
   whose whole job is to say what was verified. */
check("only the gate this vertical runs is drawn", await page.locator(".gate").count(), 1);

await page.locator(".row.out .bubble").first().click();
await page.waitForSelector(".sources", { timeout: 5000 });

/* The trap from Task 30, on the screen that has forty chances at it: the
 * figure is monospace and the Arabic label is not. Asked of the renderer,
 * which is the only thing that knows what it actually painted with. */
// Prefixes, because CDP reports the resolved FACE — "IBM Plex Mono Medium" at
// weight 600, not the family. The same correction the font sweep needed.
const family = async (selector) =>
  (await paintedFont(selector)).map((face) => face.split(/ (?:Thin|Extra|Light|Regular|Medium|Semi|Bold|Black)/)[0]);

check("the figure paints in the mono family", await family(".figure .mono"), [
  "IBM Plex Mono",
]);
check("the Arabic excerpt does not", await family(".excerpt"), ["IBM Plex Sans Arabic"]);

/* ── settings: what the screen may draw ───────────────────────────────── */
await page.route("**/settings", (route) => {
  if (route.request().method() === "PUT") {
    const asked = JSON.parse(route.request().postData() ?? "{}").changes ?? {};
    if ((asked.min_score ?? 1) < 0) {
      return route.fulfill({
        status: 422,
        contentType: "application/json",
        body: JSON.stringify({ detail: "min_score is -0.5, below the platform floor of 0.0" }),
      });
    }
    return route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ values: { min_score: 0.5, synonyms: {} } }),
    });
  }
  return route.fulfill({
    status: 200,
    contentType: "application/json",
    body: JSON.stringify({
      bounds: {
        min_score: { kind: "number", min: 0, max: 1, description: "How similar a passage must be." },
        synonyms: { kind: "map", min: null, max: null, description: "Your words." },
      },
      values: { min_score: 0, synonyms: { التجمع: ["التجمع الخامس"] } },
    }),
  });
});

await page.goto(`http://127.0.0.1:${PORT}/#settings`, { waitUntil: "load" });
await page.waitForSelector(".settings", { timeout: 5000 });

check("one control per declared setting", await page.locator("#min_score").count(), 1);
check("bounded by the server's floor", await page.getAttribute("#min_score", "min"), "0");
check("nothing is drawn disabled", await page.locator("[disabled]").count(), 0);
check(
  "the tenant's own word is listed",
  (await page.locator(".documents").innerText()).includes("التجمع الخامس"),
  true,
);

/* A refusal is the server's sentence, not a control snapping back. */
await page.locator("#min_score").fill("-0.5");
await page.locator("#min_score").blur();
await page.waitForSelector(".settings .warning", { timeout: 5000 });
check(
  "a refusal says why",
  (await page.locator(".settings .warning").innerText()).includes("below the platform floor"),
  true,
);

await page.goto(`http://127.0.0.1:${PORT}/`, { waitUntil: "load" });

check("no page errors", problems, []);

await browser.close();
server.close();

if (failures.length) {
  console.error(`\n${failures.length} failed:\n${failures.join("\n")}`);
  process.exit(1);
}
console.log("\nall checks passed");
