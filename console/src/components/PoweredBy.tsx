import { useTranslation } from "react-i18next";

/**
 * Our mark, in the corner, on every page.
 *
 * Rendered by the shell rather than by each screen, so a page added later
 * cannot forget it. White label — their colours, their domain — is out of
 * scope and stays out: a working product with our name on it is how the
 * university's IT director tells the broker about us.
 */
export function PoweredBy() {
  const { t } = useTranslation();

  return <footer className="powered">{t("app.poweredBy")}</footer>;
}
