/* The tenant's identity, fetched from the console's own origin.
 *
 * Same-origin and relative on purpose: the session is an httpOnly cookie, and
 * a cross-origin call would need CORS plus credentials plus a decision about
 * which origins may ask — three things to get wrong in exchange for nothing.
 *
 * There is no tenant id in the URL, and there must never be one. `/tenant`
 * means "the tenant whose cookie you presented"; a `/tenants/{id}` would make
 * authorization a decision on every request instead of a property of the
 * session.
 */

export type Brand = {
  name: string;
  initials: string;
  hasLogo: boolean;
  timezone: string;
  /** The tenant's default REPLY language, which is not the console's. */
  defaultReplyLanguage: string;
};

export const LOGO_URL = "/tenant/logo";

/** The brand, or null when it cannot be read.
 *
 * Null rather than a thrown error, and never a guess: a header that invents a
 * name is a header that shows the wrong tenant, which is the one failure this
 * whole screen exists to avoid. The caller renders nothing identifying until
 * this resolves.
 */
export async function fetchBrand(): Promise<Brand | null> {
  try {
    const response = await fetch("/tenant", { credentials: "same-origin" });
    return response.ok ? ((await response.json()) as Brand) : null;
  } catch {
    return null;
  }
}
