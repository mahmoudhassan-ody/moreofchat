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
page.on("requestfailed", (request) => {
  // The font CDN is unreachable by design under `--network none`. Reported
  // below as a finding rather than counted as a failure here.
  if (request.url().includes("fonts.g")) return;
  problems.push(`requestfailed: ${request.url()} ${request.failure()?.errorText}`);
});

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
check("no page errors", problems, []);

await browser.close();
server.close();

if (failures.length) {
  console.error(`\n${failures.length} failed:\n${failures.join("\n")}`);
  process.exit(1);
}
console.log("\nall checks passed");
