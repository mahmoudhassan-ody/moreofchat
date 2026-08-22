/* Console i18n and direction — demo plan Task 29.
 *
 * **The console's language is not the bot's language, and they are set in
 * different places.** An admissions officer works in English while the bot
 * answers students in Masri; a broker's agent works in Arabic while the bot
 * answers an expatriate buyer in English. Conflating them repeats the
 * register/language collapse that composition already had to be fixed for.
 *
 * So: this file governs what the *console* is rendered in, and it is stored
 * per agent (`agents.console_language`) rather than per tenant. What the bot
 * replies in is decided per turn by mirroring the customer, and is never read
 * from here — a test asserts that no module under `moc/agent/` mentions
 * `console_language` at all.
 *
 * **`dir` is set on the root element, once.** Not per component: a component
 * that sets its own direction is a component that is wrong inside the other
 * layout, and the bug only appears when somebody nests it. Setting it before
 * React renders also avoids a flash of LTR, which is the entire layout jumping
 * on load.
 */

import i18next from "i18next";
import { initReactI18next } from "react-i18next";

import ar from "./ar.json";
import en from "./en.json";

export const LANGUAGES = ["en", "ar"] as const;
export type Language = (typeof LANGUAGES)[number];

const RTL: readonly Language[] = ["ar"];
const STORAGE_KEY = "moc.console.language";

export function directionOf(language: Language): "rtl" | "ltr" {
  return RTL.includes(language) ? "rtl" : "ltr";
}

/** Write the language and its direction onto the document root. */
export function applyDirection(language: Language): void {
  const root = document.documentElement;
  root.lang = language;
  root.dir = directionOf(language);
}

/* The agent's own preference, not the tenant's. Read from local storage here
 * so the console renders correctly before the session request returns; the
 * authoritative copy is the `agents` row, and this is a cache of it. */
function preferred(): Language {
  const stored = window.localStorage.getItem(STORAGE_KEY);
  return LANGUAGES.includes(stored as Language) ? (stored as Language) : "en";
}

export function rememberLanguage(language: Language): void {
  window.localStorage.setItem(STORAGE_KEY, language);
  applyDirection(language);
}

export async function initialiseI18n(): Promise<typeof i18next> {
  const language = preferred();
  applyDirection(language);
  await i18next.use(initReactI18next).init({
    lng: language,
    fallbackLng: "en",
    resources: { en: { translation: en }, ar: { translation: ar } },
    interpolation: { escapeValue: false },
    /* A missing key renders the key itself — `nav.inbox` in the navigation
     * bar — and nothing reports it. `test_every_key_a_component_asks_for_exists`
     * is the thing that reports it, at build time rather than to a customer. */
    saveMissing: false,
  });
  return i18next;
}

export default i18next;
