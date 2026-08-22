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
        <p className="mono">{t("app.poweredBy")}</p>
      </main>
    </div>
  );
}
