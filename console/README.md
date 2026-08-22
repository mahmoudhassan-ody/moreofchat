# The console

React + Vite + TypeScript. Demo plan Task 29 built the shell; the screens come
after it.

## Built, type-checked and rendered — 2026-08-22

```bash
export PATH="$HOME/.local/opt/node/bin:$PATH"
cd console
npm install
npm run typecheck   # tsc -p tsconfig.json --noEmit
npm run build       # tsc -b && vite build
npm run smoke       # the built bundle, in a real browser
```

**Node is installed user-locally**, at `~/.local/opt/node` (v22.22.0), because
this account has no passwordless sudo. Nothing was installed system-wide and
nothing outside `$HOME` was touched. `export PATH` as above, or add it to your
profile.

**The browser runs in Docker.** Chromium needs `libatk-1.0.so.0` and a dozen
other system libraries that need root to install, so `npm run smoke` runs the
Playwright image against the local `dist/` with `--network none`. The container
gets no network deliberately: nothing in the console should need one, and the
single thing that reaches for the internet is named below.

### What `npm run smoke` checks, and why it exists

The structural gates in `tests/console/` read source as text. They catch a
hard-coded string and a physical margin; they are blind to a bad import path, a
wrong `react-i18next` API, a component that throws on mount, and — the one that
matters — a logical property that does not do what its author believed.

So the smoke check runs the built bundle in a real layout engine and asserts
the shell mounts, the toggle flips `dir` and `lang` on the root element, both
catalogues render, and **the logo and the nav end up at opposite ends of the
bar in both directions**.

That last assertion was wrong when first written, and a sabotage caught it: it
compared which element sat further left, and in RTL the nav is left of the logo
whether the margin is logical or physical, so `margin-left: auto` passed. What
`margin-inline-start: auto` actually buys is the *gap*, so the check measures
distance rather than order. Re-sabotaged: `margin-left: auto` now fails the RTL
case and passes the LTR one, which is exactly the asymmetry that makes physical
properties dangerous — the layout is correct in the language the author was
reading.

### Type is self-hosted

IBM Plex Sans Arabic (400/500/600/700) and IBM Plex Mono (400/500) live in
`src/theme/fonts/` as woff2 — 16 files, 332 KiB, arabic + latin + latin-ext
only. Cyrillic and Vietnamese are dropped; `unicode-range` is kept verbatim
from the vendor, so an English screen never downloads the Arabic files.

There is no CDN link. The console's identity is carried by the logo and the
typography — colour does no branding work at all — which makes the font a
dependency of the brand rather than a refinement of it, and a first impression
should not depend on a third party being reachable from wherever the laptop is
plugged in.

The smoke check asserts, via CDP, that **every element painting text resolves
to an IBM Plex face** — the resolved face, not the declared CSS stack, which
reports the same string whether the font loaded or not. It runs with
`--network none`, so a pass is proof the files came from the bundle. Sabotaged
by removing the `@import`: everything fell to `FreeSerif` and
`WenQuanYi Zen Hei`, and both assertions failed.

To refresh the files:

```bash
curl -fsS -A "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120.0" \
  "https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Arabic:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap"
```

then download each `arabic`/`latin`/`latin-ext` woff2 into `src/theme/fonts/`
named `<family-slug>-<weight>-<subset>.woff2` and regenerate
`src/theme/faces.css`. The names derive from family, weight and subset rather
than from the vendor's hashes, so a refresh is a content diff and not a rename.

**`.mono` is for figures, never for prose.** IBM Plex Mono carries no Arabic
glyphs, so Arabic text placed in it falls back to a system serif — it renders,
it looks plausible to anyone reading English, and nothing logs. That is exactly
the bug the font sweep caught in this shell, and Task 32 puts figures beside
Arabic labels on every row of the inbox.

## What the gates enforce

They live in `tests/console/test_shell.py` and run with everything else — one
command, one green, no separate frontend gate that a backend change never
triggers.

- **Every visible string comes from the catalogue.** A hard-coded string is a
  string that never translates, and it is found by a customer.
- **Both catalogues hold the same keys, and every key a component asks for
  exists.** i18next renders a missing key as the key itself, so `nav.inbox`
  appears in the navigation bar and nothing reports a problem.
- **No physical direction properties.** `margin-inline-start`, never
  `margin-left`. In an RTL layout the physical version is the bug that looks
  like a design choice.
- **One accent.** Colours are grouped by hue, so a second accent fails rather
  than being added to a list of permitted values.

All four have been proven by sabotage: a hard-coded string, a hard-coded
`placeholder`, `margin-left: auto`, and a second accent added "just for the
success state" each failed the gate that owns them.

## Language

The console's language is per agent (`agents.console_language`, migration
0012). It is not the language the bot replies in, which is decided per turn by
mirroring the customer — and no module under `moc/agent/` can see the console
setting, which a test asserts.
