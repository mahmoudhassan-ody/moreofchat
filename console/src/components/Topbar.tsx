import { useTranslation } from "react-i18next";

import { LanguageToggle } from "./LanguageToggle";

const SECTIONS = ["inbox", "knowledge", "scripts", "analytics", "settings"] as const;

/**
 * The one bar every screen sits under.
 *
 * The navigation is pushed to the end edge with `margin-inline-start: auto` —
 * the right in English, the left in Arabic. `margin-left: auto` would pin it
 * to the right in both, which is the middle of an Arabic layout, and it would
 * look completely deliberate to anyone reviewing the diff in English.
 */
export function Topbar({ active }: { active: string }) {
  const { t } = useTranslation();

  return (
    <header className="topbar">
      <div className="logo">{t("app.name")}</div>
      <nav className="nav">
        {SECTIONS.map((section) => (
          <a
            key={section}
            href={`#${section}`}
            className={section === active ? "on" : undefined}
          >
            {t(`nav.${section}`)}
          </a>
        ))}
      </nav>
      <LanguageToggle />
    </header>
  );
}
