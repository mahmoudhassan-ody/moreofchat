import { useTranslation } from "react-i18next";

import type { Provenance } from "../api/inbox";

/**
 * Where each figure in a reply came from.
 *
 * **This is the pane that makes a dean believe the product.** A claim of
 * groundedness is a brochure; their own sentence, from their own document,
 * with the date it was current, is evidence. The data was already computed by
 * the grounding gate on every composed turn and discarded at this boundary.
 *
 * Ungrounded figures are shown, not hidden. A pane where every figure has a
 * source would be making §19.3's claim as a rendering default rather than as a
 * measurement — and the one time it matters is the one time it would lie.
 *
 * **One pane for both grounding modes.** A document answer's figure traces to
 * a chunk; an inventory answer's traces to a unit row and its `as_of`, and an
 * instalment to the calculator's inputs. Same keys, same renderer — a second
 * shape would be a second thing this file can fail to render, and the promise
 * made to a university and to a broker is one promise.
 *
 * **Gates are rendered from what the payload carries, never from a list here.**
 * The two verticals run different checks: there is no figure audit on an
 * inventory reply, because no model composed those numbers and there is
 * nothing for a second model to check. A hardcoded pair drew that missing gate
 * as a failed one — a red mark for a check that never ran, on the screen whose
 * whole job is to say what was and was not verified.
 *
 * **The figure goes in `.mono`; the label does not.** IBM Plex Mono carries no
 * Arabic glyphs, so a translated string placed there falls back to a system
 * serif — it renders, it looks right in English, and nothing logs. A test
 * enforces the split, because this screen puts a figure beside an Arabic label
 * on every row.
 */
export function SourcePane({ provenance }: { provenance: Provenance | null }) {
  const { t } = useTranslation();

  if (!provenance) {
    return null;
  }
  return (
    <aside className="sources">
      <div className="pane-head">
        <span className="pane-title">{t("inbox.sources")}</span>
      </div>

      <div className="gates">
        {Object.entries(provenance.gates)
          .filter(([name]) => name !== "figure_audit_degraded")
          .map(([name, passed]) => (
            <p className={passed ? "gate on" : "gate"} key={name}>
              {t(`inbox.gate.${name}`)}
            </p>
          ))}
        {provenance.gates.figure_audit_degraded && (
          <p className="gate degraded">{t("inbox.gate.degraded")}</p>
        )}
      </div>

      {provenance.figures.length === 0 && (
        <p className="empty">{t("inbox.noFigures")}</p>
      )}

      {provenance.figures.map((figure, index) => (
        <div className={figure.grounded ? "source" : "source orphan"} key={index}>
          <p className="figure">
            <span className="mono">{figure.raw}</span>
          </p>
          {figure.grounded ? (
            <>
              <p className="excerpt">{figure.excerpt}</p>
              <p className="attribution">
                {figure.title ?? t(`inbox.from.${figure.source ?? "chunk"}`)}
                {figure.asOf && (
                  <>
                    {" "}
                    <span className="mono">{figure.asOf}</span>
                  </>
                )}
              </p>
            </>
          ) : (
            <p className="attribution">{t("inbox.orphan")}</p>
          )}
        </div>
      ))}
    </aside>
  );
}
