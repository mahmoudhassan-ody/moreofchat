import { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";

import { readAnalytics, type Analytics as Report } from "../api/analytics";

/**
 * How many the bot answered, how many needed a person, and what it cost.
 *
 * **Containment is shown and never gated.** A target on it pays the bot to
 * answer where the honest behaviour is to hand off, which is the exact
 * pressure §19.3 exists to remove. It is a number to look at, not one to move.
 *
 * **An unpriced call makes the total "at least".** A model with no rate
 * contributes nothing to the sum, and a sum that silently omits it is a
 * smaller number wearing the word "total". The screen says which models it
 * could not price.
 *
 * Figures go in `.mono` and labels do not — IBM Plex Mono has no Arabic
 * glyphs, and this screen is nothing but figures beside Arabic labels.
 */
export function Analytics() {
  const { t } = useTranslation();
  const [report, setReport] = useState<Report | null>(null);

  useEffect(() => {
    void readAnalytics().then(setReport);
  }, []);

  if (!report) {
    return <p>{t("shell.loading")}</p>;
  }

  return (
    <section className="analytics">
      <h2>{t("analytics.title")}</h2>

      <div className="tiles">
        <div className="tile">
          <span className="tile-label">{t("analytics.conversations")}</span>
          <span className="tile-value mono">{report.conversations}</span>
        </div>
        <div className="tile">
          <span className="tile-label">{t("analytics.handedOff")}</span>
          <span className="tile-value mono">{report.handedOff}</span>
        </div>
        <div className="tile">
          <span className="tile-label">{t("analytics.containment")}</span>
          <span className="tile-value mono">
            {report.containmentRate === null
              ? "—"
              : `${Math.round(report.containmentRate * 100)}%`}
          </span>
        </div>
        <div className="tile">
          <span className="tile-label">
            {report.costIsComplete ? t("analytics.cost") : t("analytics.costAtLeast")}
          </span>
          <span className="tile-value mono">${report.costUsd}</span>
        </div>
        <div className="tile">
          <span className="tile-label">{t("analytics.perConversation")}</span>
          <span className="tile-value mono">
            {report.costPerConversation === null ? "—" : `$${report.costPerConversation}`}
          </span>
        </div>
      </div>

      {!report.costIsComplete && (
        <p className="warning">
          {t("analytics.unpriced")} {report.unpricedModels.join("، ")}
        </p>
      )}

      <h3>{t("analytics.byModel")}</h3>
      <ul className="documents">
        {report.byModel.map((row) => (
          <li key={row.model}>
            <b>{row.model}</b> <span className="mono">{row.calls}</span>{" "}
            <span className="mono">{row.costUsd === null ? "—" : `$${row.costUsd}`}</span>
          </li>
        ))}
      </ul>

      <h3>{t("analytics.whyHandedOff")}</h3>
      <ul className="documents">
        {report.handoffReasons.map((row) => (
          <li key={row.reason}>
            <b>{row.reason}</b> <span className="mono">{row.times}</span>
          </li>
        ))}
      </ul>
    </section>
  );
}
