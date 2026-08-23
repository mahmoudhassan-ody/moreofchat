/* Settings the tenant may change, and the bounds they may change them within.
 *
 * **The bounds come from the server and the console renders one control per
 * declared setting.** A list held here would draw a control for something the
 * backend has withdrawn — which is the disabled-slider failure from the other
 * side: a control that exists on screen and not in the system.
 */

export type Bound = {
  kind: "number" | "map";
  min: number | null;
  max: number | null;
  description: string;
};

export type Settings = {
  bounds: Record<string, Bound>;
  values: Record<string, unknown>;
};

export async function readSettings(): Promise<Settings | null> {
  const response = await fetch("/settings", { credentials: "same-origin" });
  return response.ok ? ((await response.json()) as Settings) : null;
}

/** The new values, or the server's reason for refusing. */
export async function writeSettings(
  changes: Record<string, unknown>,
): Promise<{ values: Record<string, unknown> } | { error: string }> {
  const response = await fetch("/settings", {
    method: "PUT",
    credentials: "same-origin",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ changes }),
  });
  const body = await response.json();
  return response.ok ? body : { error: String(body.detail ?? response.status) };
}
