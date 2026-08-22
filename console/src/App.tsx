import { useTranslation } from "react-i18next";

import { Topbar } from "./components/Topbar";

/**
 * The shell, and nothing else yet.
 *
 * Task 29 is the frame every later screen hangs inside: catalogues, direction
 * and one accent. Building a screen before the frame is how a console ends up
 * with forty components holding hard-coded English and physical margins, at
 * which point neither is fixable in bulk.
 */
export function App() {
  const { t } = useTranslation();

  return (
    <div className="app">
      <Topbar active="inbox" />
      <main className="content">
        <p>{t("language.note")}</p>
        {/* `.mono` is for figures, never for prose: IBM Plex Mono carries no
            Arabic glyphs, so Arabic text placed in it falls back to whatever
            system serif exists — which renders, looks wrong, and reports
            nothing. The smoke check asserts every painted family is IBM Plex,
            which is what caught it here. */}
        <p>
          {t("app.poweredBy")} <span className="mono">0.1.0</span>
        </p>
      </main>
    </div>
  );
}
