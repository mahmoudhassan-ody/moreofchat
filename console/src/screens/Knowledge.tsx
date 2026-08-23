import { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";

import {
  confirmDocument,
  listDocuments,
  previewDocument,
  removeDocument,
  type Preview,
  type StoredDocument,
} from "../api/knowledge";

/**
 * Upload, look at what the chunker did, then confirm.
 *
 * **The preview is not a courtesy.** Chunking is where a corpus quietly
 * breaks: nothing errors, the document ingests, and months later a figure
 * surfaces with no sentence around it. The tenant is the only person who can
 * look at their own text and see that it came out wrong, so the screen shows
 * the count *and* the chunks before anything is written or bought.
 *
 * The confirm button does not exist until a preview has been taken. That is
 * the order the whole screen is for, and leaving both live would make it a
 * suggestion.
 */
export function Knowledge() {
  const { t } = useTranslation();
  const [docId, setDocId] = useState("");
  const [title, setTitle] = useState("");
  const [body, setBody] = useState("");
  const [preview, setPreview] = useState<Preview | null>(null);
  const [ingested, setIngested] = useState<string | null>(null);
  const [documents, setDocuments] = useState<StoredDocument[]>([]);

  useEffect(() => {
    void listDocuments().then(setDocuments);
  }, []);

  const upload = { docId, title, text: body };

  async function onPreview() {
    setIngested(null);
    setPreview(await previewDocument(upload));
  }

  async function onConfirm() {
    const result = await confirmDocument(upload);
    setIngested(
      result.failures.length
        ? result.failures[0].reason
        : result.unchanged
          ? t("knowledge.unchanged")
          : t("knowledge.ingested"),
    );
    setPreview(null);
    setDocuments(await listDocuments());
  }

  async function onRemove(id: string) {
    await removeDocument(id);
    setDocuments(await listDocuments());
  }

  return (
    <section className="knowledge">
      <h2>{t("knowledge.title")}</h2>

      <div className="upload">
        <input
          className="field"
          value={docId}
          onChange={(event) => setDocId(event.target.value)}
          placeholder={t("knowledge.docId")}
        />
        <input
          className="field"
          value={title}
          onChange={(event) => setTitle(event.target.value)}
          placeholder={t("knowledge.docTitle")}
        />
        <textarea
          className="field body"
          value={body}
          onChange={(event) => setBody(event.target.value)}
          placeholder={t("knowledge.body")}
        />
        <button className="act" type="button" onClick={() => void onPreview()}>
          {t("knowledge.preview")}
        </button>
      </div>

      {preview && (
        <div className="preview">
          <p className="count">
            <span className="mono">{preview.chunkCount}</span> {t("knowledge.chunks")}
          </p>

          {/* Named warnings, translated here. The reason string on the wire is
              written in English by whoever added the check; the person reading
              it is an admissions officer in Egypt. */}
          {preview.warnings.map((warning) => (
            <p className="warning" key={`${warning.name}-${warning.ordinal}`}>
              {t(`knowledge.warning.${warning.name}`)}
            </p>
          ))}

          {preview.sample.map((chunk) => (
            <pre className="chunk" key={chunk.ordinal}>
              {chunk.content}
            </pre>
          ))}

          {/* Only after a preview. */}
          <button className="act" type="button" onClick={() => void onConfirm()}>
            {preview.unchanged ? t("knowledge.unchanged") : t("knowledge.confirm")}
          </button>
        </div>
      )}

      {ingested && <p className="result">{ingested}</p>}

      <ul className="documents">
        {documents.map((document) => (
          <li key={document.docId}>
            <b>{document.title || document.docId}</b>{" "}
            <span className="mono">{document.chunkCount}</span>{" "}
            <button
              className="act ghost"
              type="button"
              onClick={() => void onRemove(document.docId)}
            >
              {t("knowledge.remove")}
            </button>
          </li>
        ))}
      </ul>
    </section>
  );
}
