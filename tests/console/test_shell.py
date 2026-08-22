"""The console shell — demo plan Task 29.

**Everything after this task inherits it**, which is the only reason these
tests are worth their weight. A missing translation, a physical margin or a
second accent colour is a five-minute fix in one component and an unfixable
sprawl across forty, and none of the three announces itself: the console looks
right in the language and direction the person who wrote it was using.

So the gates are structural and they run in pytest with everything else. One
command, one green, no separate frontend gate that a backend change never runs.

**`test_no_component_uses_left_or_right_css` is the one that matters.**
`margin-left` in an RTL layout is the bug that looks like a design choice —
nothing crashes, nothing logs, and a reviewer reading the diff in English sees
a margin exactly where they expect it. It is caught here or it is caught by an
Arabic-speaking customer.
"""

import json
import re
from pathlib import Path

CONSOLE = Path(__file__).parents[2] / "console"
SOURCE = CONSOLE / "src"
CATALOGUES = SOURCE / "i18n"
LANGUAGES = ("en", "ar")


def components() -> list[Path]:
    return sorted(SOURCE.rglob("*.tsx"))


def stylesheets() -> list[Path]:
    return sorted(SOURCE.rglob("*.css"))


def catalogue(language: str) -> dict:
    return json.loads((CATALOGUES / f"{language}.json").read_text(encoding="utf-8"))


def test_the_console_shell_exists():
    """Named first so the rest of this file fails as assertions rather than as
    an empty scan — a glob over a missing directory matches nothing and every
    "no violations" test passes vacuously."""
    assert components(), "no .tsx components found"
    assert stylesheets(), "no stylesheets found"


# ─────────────────────────── strings ───────────────────────────

#: JSX text between tags, minus anything that is an expression.
#: `(`, `)`, `;` and `=` are excluded so a TypeScript generic does not read as
#: JSX text: `useState<string>("en");\n  return (\n    <div` otherwise matches
#: from the generic's closing bracket to the next tag. The cost is that a
#: hard-coded string containing a bracket or a semicolon is missed, which is a
#: narrower hole than a scanner nobody trusts because it cries wolf.
_JSX_TEXT = re.compile(r">([^<>{}();=]+)<")
#: Attributes a person reads. `className` and `id` are not on this list because
#: nobody reads them; `placeholder` and `aria-label` are because somebody does,
#: including somebody using a screen reader in Arabic.
_VISIBLE_ATTRIBUTE = re.compile(
    r"\b(placeholder|title|alt|aria-label|aria-description)\s*=\s*[\"']([^\"']+)[\"']"
)
_LETTER = re.compile(r"[A-Za-z؀-ۿ]")
_TRANSLATION_KEY = re.compile(r"\bt\(\s*[\"']([a-zA-Z0-9_.]+)[\"']")
#: `t(`nav.${section}`)` — a key assembled at runtime. The scanner cannot
#: resolve it, so it checks the literal prefix is a real namespace instead of
#: shrugging: a dynamic key over a namespace that does not exist renders
#: `nav.inbox` into the navigation bar just as loudly as a static one.
_DYNAMIC_KEY = re.compile(r"\bt\(\s*`([a-zA-Z0-9_.]*)\$\{")


def test_every_visible_string_comes_from_the_catalogue():
    """AST/scan over components. A literal that ships is a string that never
    translates, and it is found by a customer, not a test."""
    assert components(), "nothing scanned — this test cannot pass vacuously"
    violations = []
    for path in components():
        source = path.read_text(encoding="utf-8")
        for text in _JSX_TEXT.findall(source):
            if _LETTER.search(text):
                violations.append(f"{path.name}: text {text.strip()!r}")
        for attribute, value in _VISIBLE_ATTRIBUTE.findall(source):
            if _LETTER.search(value):
                violations.append(f"{path.name}: {attribute}={value!r}")
    assert violations == [], (
        "hard-coded visible strings — use t('key') and add it to both catalogues:\n"
        + "\n".join(violations)
    )


def test_both_catalogues_have_the_same_keys():
    """A missing Arabic key renders English inside an Arabic sentence."""
    keys = {language: set(_flatten(catalogue(language))) for language in LANGUAGES}
    assert keys["en"] == keys["ar"], (
        f"only in en: {sorted(keys['en'] - keys['ar'])}; "
        f"only in ar: {sorted(keys['ar'] - keys['en'])}"
    )


def test_every_key_a_component_asks_for_exists():
    """The other direction, and the one that reaches a customer fastest:
    i18next renders a missing key as the key itself, so `nav.inbox` appears in
    the navigation bar and nothing anywhere reports a problem."""
    available = set(_flatten(catalogue("en")))
    asked = {
        key
        for path in components()
        for key in _TRANSLATION_KEY.findall(path.read_text(encoding="utf-8"))
    }
    assert asked, "no t() calls found — the scanner is looking at the wrong thing"
    assert asked <= available, f"missing from the catalogues: {sorted(asked - available)}"

    namespaces = {
        prefix.rstrip(".")
        for path in components()
        for prefix in _DYNAMIC_KEY.findall(path.read_text(encoding="utf-8"))
        if prefix
    }
    known = {key.rsplit(".", 1)[0] for key in available if "." in key}
    assert namespaces <= known, (
        f"dynamic keys over namespaces that do not exist: {sorted(namespaces - known)}"
    )


def _before_line_comment(line: str) -> str:
    return line.split("//")[0]


def _flatten(node, prefix: str = "") -> list[str]:
    if not isinstance(node, dict):
        return [prefix]
    return [
        key
        for name, value in node.items()
        for key in _flatten(value, f"{prefix}.{name}" if prefix else name)
    ]


# ─────────────────────────── direction ───────────────────────────

#: Physical properties, by name. Word-boundary anchored because "right" is a
#: substring of "bright" and "copyright", and a scanner that cries wolf gets
#: deleted by the third person who hits it.
_PHYSICAL_CSS = re.compile(
    r"\b("
    r"(?:margin|padding|border|scroll-margin|scroll-padding|inset)-(?:left|right)"
    r"|border-(?:top|bottom)-(?:left|right)-radius"
    r"|(?:left|right)\s*:"
    r"|text-align\s*:\s*(?:left|right)"
    r"|float\s*:\s*(?:left|right)"
    r")",
    re.IGNORECASE,
)
#: The same properties as React style-object keys.
_PHYSICAL_JSX = re.compile(
    r"\b("
    r"(?:margin|padding|border|scrollMargin|scrollPadding|inset)(?:Left|Right)"
    r"|borderTop(?:Left|Right)Radius|borderBottom(?:Left|Right)Radius"
    r"|textAlign\s*:\s*[\"'](?:left|right)[\"']"
    r")"
)


def test_no_component_uses_left_or_right_css():
    """margin-left in an RTL layout is the bug that looks like a design
    choice. Logical properties only.

    `margin-inline-start`, `inset-inline-start`, `text-align: start`,
    `border-start-start-radius`. They read as awkwardly the first week and
    they are correct in both directions forever, which is the trade.

    `direction: ltr` is deliberately NOT caught: a run of Latin digits inside
    Arabic text is genuinely left-to-right, and forcing it is correct rather
    than a shortcut.
    """
    assert stylesheets() and components(), "nothing scanned"
    violations = []
    for path in stylesheets() + components():
        source = _without_comments(path.read_text(encoding="utf-8"))
        pattern = _PHYSICAL_CSS if path.suffix == ".css" else _PHYSICAL_JSX
        for line_number, line in enumerate(source.splitlines(), start=1):
            found = pattern.search(_before_line_comment(line))
            if found:
                violations.append(f"{path.name}:{line_number}: {found.group(1).strip()}")
    assert violations == [], (
        "physical direction properties — use the logical equivalent "
        "(margin-inline-start, inset-inline-start, text-align: start):\n"
        + "\n".join(violations)
    )


def _without_comments(source: str) -> str:
    """Blank out `/* ... */` regions, keeping the line count.

    Necessary rather than tidy: `shell.css` explains at length why
    `margin-left` is banned, and a scanner that reads its own rationale as a
    violation is a scanner that gets an exception carved into it — and the
    exception is where the real one hides.
    """
    out, depth, index = [], 0, 0
    while index < len(source):
        if source.startswith("/*", index):
            depth += 1
            index += 2
        elif source.startswith("*/", index) and depth:
            depth -= 1
            index += 2
        else:
            out.append(source[index] if not depth or source[index] == "\n" else " ")
            index += 1
    return "".join(out)


def test_the_root_element_carries_a_direction():
    """`dir` at the root, set from the language, and not a per-component
    decision. A component that sets its own direction is a component that is
    wrong inside the other layout."""
    source = (SOURCE / "i18n" / "index.ts").read_text(encoding="utf-8")
    assert "documentElement" in source
    assert "dir" in source


# ─────────────────────────── the accent ───────────────────────────

_HEX = re.compile(r"#([0-9a-fA-F]{6})\b")
_RGBA = re.compile(r"rgba?\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)")
#: A colour whose channels are within this of each other is a grey — including
#: the cool, slightly blue greys this palette actually uses. Channel spread
#: rather than HLS saturation, because saturation is computed differently above
#: and below 50% lightness and #ECECF0 comes out more "saturated" than #191920
#: while being visibly the same neutral. The separation is not close: the
#: greys here spread at most 11, and the brand spreads 178.
_GREY_SPREAD = 16
#: Hues within this many degrees are the same accent — a tint and a shade of
#: one brand colour are one colour, which is the point of having tints.
_HUE_TOLERANCE = 12.0


def test_the_theme_exposes_exactly_one_accent():
    """One accent, and a second one has to fail a test rather than pass review.

    Identity is carried by the logo and the typography; colour is reserved for
    "this is live, actionable or verified". A palette that grows a second
    accent has lost that meaning before anyone notices, because each addition
    is individually reasonable.

    Hue-based rather than a list of permitted hex values: a list is satisfied
    by adding to the list, which is the edit this is meant to make visible.
    """
    import colorsys

    tokens = (SOURCE / "theme" / "tokens.css").read_text(encoding="utf-8")
    hues = []
    for red, green, blue in _colours(tokens):
        if max(red, green, blue) - min(red, green, blue) <= _GREY_SPREAD:
            continue
        hue, _light, _saturation = colorsys.rgb_to_hls(red / 255, green / 255, blue / 255)
        hues.append(hue * 360)

    assert hues, "no colour in the theme at all"
    families: list[float] = []
    for hue in hues:
        if not any(abs(hue - seen) <= _HUE_TOLERANCE for seen in families):
            families.append(hue)
    assert len(families) == 1, (
        f"{len(families)} accent families in the theme: {[round(h) for h in families]}. "
        "Everything that is not the brand accent is greyscale."
    )


def test_the_failure_colour_is_declared_separately_from_the_theme():
    """The one exception, and it is kept out of the palette on purpose.

    A genuine failure needs a colour nothing else uses, and putting it beside
    the accent is how it becomes available for decoration — at which point the
    console has two accents and no way to say "this actually broke".
    """
    tokens = (SOURCE / "theme" / "tokens.css").read_text(encoding="utf-8")
    failure = (SOURCE / "theme" / "failure.css").read_text(encoding="utf-8")

    assert "--stop" not in tokens
    assert "--stop" in failure


def _colours(text: str) -> list[tuple[int, int, int]]:
    found = [
        tuple(int(value[index : index + 2], 16) for index in (0, 2, 4))
        for value in _HEX.findall(text)
    ]
    return found + [tuple(int(part) for part in group) for group in _RGBA.findall(text)]


# ─────────────────────────── whose console this is ───────────────────────────


def test_powered_by_appears_on_every_page():
    """Rendered by the shell, not by each screen.

    A per-screen `<PoweredBy />` is a per-screen chance to forget it, and the
    screen that forgets is the one somebody screenshots. Asserted structurally
    because there is one page today and four after Task 32 — by which time
    "every page" is no longer something a reader can check by eye.
    """
    shell = (SOURCE / "App.tsx").read_text(encoding="utf-8")
    assert "<PoweredBy />" in shell

    elsewhere = [
        path.name
        for path in components()
        if path.name not in {"App.tsx", "PoweredBy.tsx"}
        and "<PoweredBy" in path.read_text(encoding="utf-8")
    ]
    assert elsewhere == [], (
        f"{elsewhere} render it themselves — it belongs to the shell, once, "
        "so a screen cannot opt out of it"
    )


def test_nothing_falls_back_to_the_more_of_chat_mark():
    """A tenant with no crest sees THEIR initials.

    Ours in that slot reads as a product that does not know who they are,
    which is the opposite of what a pilot is for — and it is the kind of
    default somebody adds later because the empty state looked bare.
    """
    brand = (SOURCE / "components" / "TenantBrand.tsx").read_text(encoding="utf-8")

    assert "initials" in brand
    assert "app.name" not in brand, "the shell's own name is not a tenant fallback"


def test_the_console_never_asks_for_a_tenant_by_id():
    """Task 28's rule, at the other end of the wire.

    A `/tenants/${id}` in the frontend is a request the backend would have to
    authorize, and the backend deliberately has no route that could.
    """
    import re

    offenders = [
        path.name
        for path in [*components(), *sorted(SOURCE.rglob("*.ts"))]
        if re.search(r"[\"'`]/tenants?/\$\{", path.read_text(encoding="utf-8"))
    ]
    assert offenders == []
