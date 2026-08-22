import { useTranslation } from "react-i18next";

import { LOGO_URL, type Brand } from "../api/tenant";

/**
 * Whose console this is.
 *
 * **The fallback is the tenant's initials, never the More Of Chat mark.** A
 * university seeing our logo where their crest belongs reads as a product that
 * does not know who they are, which is the opposite of what a pilot is for.
 *
 * While the brand is loading it renders nothing identifying rather than a
 * placeholder name: an empty space for 200ms is honest, and a wrong name for
 * 200ms is the exact failure this component exists to prevent.
 */
export function TenantBrand({ brand }: { brand: Brand | null }) {
  const { t } = useTranslation();

  if (!brand) {
    return <div className="tenant" aria-busy="true" />;
  }
  return (
    <div className="tenant">
      {brand.hasLogo ? (
        <img className="crest" src={LOGO_URL} alt={t("brand.logoAlt")} />
      ) : (
        <span className="initials" aria-hidden="true">
          {brand.initials}
        </span>
      )}
      <b>{brand.name}</b>
    </div>
  );
}
