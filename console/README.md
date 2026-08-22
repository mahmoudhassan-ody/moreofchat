# The console

React + Vite + TypeScript. Demo plan Task 29 built the shell; the screens come
after it.

## Not built or type-checked yet

**There is no `node` or `npm` on this VPS**, so nothing here has been compiled,
type-checked, or rendered in a browser. What is verified is what `pytest`
verifies: the structural gates in `tests/console/`, which read the source as
text. A type error, a bad import path or a wrong `react-i18next` API would pass
every test in this repository today.

To close that gap:

```bash
curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash -
sudo apt-get install -y nodejs
cd console && npm install && npm run typecheck && npm run build
```

Until that has been run, treat this directory as reviewed source rather than as
a working console.

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
