import { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";

import { readSettings, writeSettings, type Settings as Config } from "../api/settings";

/**
 * Everything the tenant may change, and nothing they may not.
 *
 * **Every control is drawn from the server's declared bounds.** There is no
 * list of settings in this file — a setting the engine refuses simply has no
 * entry to render, which is the honest form of "not settable". A greyed-out
 * slider would imply the setting exists and you are not allowed it; the truth
 * is that a floor is a property of the system, not a permission level.
 *
 * A refusal is shown verbatim from the server. The tenant who tried to make
 * retrieval more permissive did something reasonable and deserves the reason,
 * not a control that silently snapped back.
 */
export function Settings() {
  const { t } = useTranslation();
  const [config, setConfig] = useState<Config | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [synonymKey, setSynonymKey] = useState("");
  const [synonymValues, setSynonymValues] = useState("");

  useEffect(() => {
    void readSettings().then(setConfig);
  }, []);

  if (!config) {
    return <p>{t("shell.loading")}</p>;
  }

  async function apply(changes: Record<string, unknown>) {
    const result = await writeSettings(changes);
    if ("error" in result) {
      setError(result.error);
      return;
    }
    setError(null);
    setConfig(await readSettings());
  }

  const synonyms = (config.values.synonyms ?? {}) as Record<string, string[]>;

  return (
    <section className="settings">
      <h2>{t("settings.title")}</h2>
      {error && <p className="warning">{error}</p>}

      {Object.entries(config.bounds)
        .filter(([, bound]) => bound.kind === "number")
        .map(([name, bound]) => (
          <div className="setting" key={name}>
            <label htmlFor={name}>{t(`settings.name.${name}`)}</label>
            <p className="hint">{bound.description}</p>
            <input
              id={name}
              className="field"
              type="number"
              step="0.05"
              /* The bounds are the server's, not this file's. A control that
                 could express a refused value would make the refusal a
                 surprise rather than a boundary. */
              min={bound.min ?? undefined}
              max={bound.max ?? undefined}
              defaultValue={String(config.values[name])}
              onBlur={(event) => void apply({ [name]: Number(event.target.value) })}
            />
          </div>
        ))}

      <div className="setting">
        <label htmlFor="synonym-key">{t("settings.name.synonyms")}</label>
        <p className="hint">{t("settings.synonymsHint")}</p>

        <ul className="documents">
          {Object.entries(synonyms).map(([word, alternatives]) => (
            <li key={word}>
              <b>{word}</b> → {alternatives.join("، ")}
              <button
                className="act ghost"
                type="button"
                onClick={() => {
                  const { [word]: _dropped, ...rest } = synonyms;
                  void apply({ synonyms: rest });
                }}
              >
                {t("settings.remove")}
              </button>
            </li>
          ))}
        </ul>

        <input
          id="synonym-key"
          className="field"
          value={synonymKey}
          onChange={(event) => setSynonymKey(event.target.value)}
          placeholder={t("settings.theirWord")}
        />
        <input
          className="field"
          value={synonymValues}
          onChange={(event) => setSynonymValues(event.target.value)}
          placeholder={t("settings.ourWords")}
        />
        <button
          className="act"
          type="button"
          onClick={() => {
            if (!synonymKey.trim()) return;
            void apply({
              synonyms: {
                ...synonyms,
                [synonymKey.trim()]: synonymValues
                  .split(/[,،]/)
                  .map((word) => word.trim())
                  .filter(Boolean),
              },
            });
            setSynonymKey("");
            setSynonymValues("");
          }}
        >
          {t("settings.addSynonym")}
        </button>
      </div>
    </section>
  );
}
