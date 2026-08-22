import { useTranslation } from "react-i18next";

import { LANGUAGES, rememberLanguage, type Language } from "../i18n";

/**
 * The console's own language, switched per agent.
 *
 * Deliberately not a "site language": it changes what this person reads and
 * nothing about what the bot says to customers. The note under it says so in
 * both catalogues, because the first support question a console like this
 * gets is "I switched to Arabic, why are the students still getting English?"
 */
export function LanguageToggle() {
  const { t, i18n } = useTranslation();

  async function choose(language: Language) {
    await i18n.changeLanguage(language);
    rememberLanguage(language);
  }

  return (
    <div className="lang" role="group" aria-label={t("language.label")}>
      {LANGUAGES.map((language) => (
        <button
          key={language}
          type="button"
          aria-pressed={i18n.language === language}
          onClick={() => void choose(language)}
        >
          {t(`language.${language}`)}
        </button>
      ))}
    </div>
  );
}
