/* What a buyer asks, as one call.
 *
 * `costIsComplete` travels with the total on purpose: a screen must not render
 * a number as final that the ledger could not fully price.
 */

export type ModelCost = {
  model: string;
  calls: number;
  inputTokens: number;
  outputTokens: number;
  /** Null when nothing in the price table covers this model. Never "0.00". */
  costUsd: string | null;
};

export type Analytics = {
  conversations: number;
  handedOff: number;
  messages: number;
  /** Null for a tenant with no traffic — not 1.0. */
  containmentRate: number | null;
  costUsd: string;
  costPerConversation: string | null;
  unpricedCalls: number;
  unpricedModels: string[];
  costIsComplete: boolean;
  byModel: ModelCost[];
  handoffReasons: { reason: string; times: number }[];
};

export async function readAnalytics(): Promise<Analytics | null> {
  const response = await fetch("/analytics", { credentials: "same-origin" });
  return response.ok ? ((await response.json()) as Analytics) : null;
}
