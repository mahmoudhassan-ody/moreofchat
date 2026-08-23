/* The knowledge screen's calls. Same-origin, no tenant in any URL.
 *
 * Two calls where one would do, and the split is the point: `preview` writes
 * nothing and `confirm` spends money. A single call with a `dryRun` flag would
 * put that difference inside a boolean the client sets, and the flag would be
 * wrong exactly once.
 */

export type Chunk = { ordinal: number; content: string };
export type Finding = { name: string; ordinal: number; reason: string };

export type Preview = {
  chunkCount: number;
  sample: Chunk[];
  warnings: Finding[];
  contentHash: string;
  unchanged: boolean;
};

export type Ingested = {
  docId: string;
  chunkCount: number;
  unchanged: boolean;
  failures: { docId: string; reason: string }[];
};

export type StoredDocument = {
  docId: string;
  title: string | null;
  vertical: string;
  chunkCount: number;
  createdAt: string;
};

export type Upload = { docId: string; title: string; text: string };

async function post<T>(path: string, body: unknown): Promise<T> {
  const response = await fetch(path, {
    method: "POST",
    credentials: "same-origin",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    throw new Error(`${path} returned ${response.status}`);
  }
  return (await response.json()) as T;
}

export const previewDocument = (upload: Upload) =>
  post<Preview>("/knowledge/preview", upload);

export const confirmDocument = (upload: Upload) =>
  post<Ingested>("/knowledge/documents", upload);

export async function listDocuments(): Promise<StoredDocument[]> {
  const response = await fetch("/knowledge/documents", { credentials: "same-origin" });
  return response.ok ? ((await response.json()) as StoredDocument[]) : [];
}

export async function removeDocument(docId: string): Promise<boolean> {
  const response = await fetch(`/knowledge/documents/${encodeURIComponent(docId)}`, {
    method: "DELETE",
    credentials: "same-origin",
  });
  return response.ok;
}
