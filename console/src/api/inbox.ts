/* The inbox's calls. Same-origin, and no tenant anywhere in a URL.
 *
 * `provenance` is the reason this screen exists: every composed reply carries
 * where each of its figures came from, computed by the grounding gate and — up
 * to Task 32 — thrown away at this boundary.
 */

export type FigureSource = {
  value: number;
  raw: string;
  grounded: boolean;
  source: "chunk" | "script" | null;
  chunkId: string | null;
  title: string | null;
  asOf: string | null;
  excerpt: string;
};

export type Provenance = {
  figures: FigureSource[];
  gates: {
    numeric_grounding: boolean;
    figure_audit: boolean;
    figure_audit_degraded: boolean;
  };
};

export type ThreadMessage = {
  channel: string;
  author: "customer" | "bot" | "agent";
  body: string | null;
  created_at: string;
  provenance: Provenance | null;
};

export type Conversation = {
  id: string;
  conversation_id: string;
  reason: string;
  status: string;
  channel: string;
  sender_ref: string;
  opened_at: string;
  claimed_by: string | null;
};

async function get<T>(path: string, fallback: T): Promise<T> {
  const response = await fetch(path, { credentials: "same-origin" });
  return response.ok ? ((await response.json()) as T) : fallback;
}

export const listConversations = () => get<Conversation[]>("/inbox", []);

export const readThread = (handoffId: string) =>
  get<ThreadMessage[]>(`/inbox/${encodeURIComponent(handoffId)}/thread`, []);

async function act(path: string, body?: unknown): Promise<boolean> {
  const response = await fetch(path, {
    method: "POST",
    credentials: "same-origin",
    headers: { "content-type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  return response.ok;
}

export const claimConversation = (handoffId: string) =>
  act(`/inbox/${encodeURIComponent(handoffId)}/claim`);

export const sendReply = (handoffId: string, text: string) =>
  act(`/inbox/${encodeURIComponent(handoffId)}/reply`, { text });

export const returnToBot = (handoffId: string) =>
  act(`/inbox/${encodeURIComponent(handoffId)}/return`);
