import { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";

import { fetchBrand, type Brand } from "./api/tenant";
import { PoweredBy } from "./components/PoweredBy";
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
  const [brand, setBrand] = useState<Brand | null>(null);

  /* Fetched once, in the shell, and passed down. Every screen shows the same
   * header, so a per-screen fetch would be one request per navigation for a
   * value that changes about never — and four chances for two screens to
   * disagree about whose console this is. */
  useEffect(() => {
    void fetchBrand().then(setBrand);
  }, []);

  return (
    <div className="app">
      <Topbar active="inbox" brand={brand} />
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
      {/* In the shell, outside the routed content, so a screen added later
          cannot forget it. */}
      <PoweredBy />
    </div>
  );
}
