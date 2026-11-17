// The typed API client. Every call goes to the relative path /api on the page's own
// origin; nginx (or the Vite dev server) forwards it to the service and adds the API key.
import createClient from "openapi-fetch";

import type { components, paths } from "./schema";

export type Schemas = components["schemas"];
export type Alert = Schemas["AlertOut"];
export type AlertPage = Schemas["AlertPage"];
export type DecisionRecord = Schemas["DecisionRecordOut"];
export type Explanation = Schemas["ExplanationOut"];
export type CardTransaction = Schemas["CardTransactionOut"];
export type ReasonCode = Schemas["StoredReasonCode"];
export type Disposition = Schemas["DispositionOut"];
export type ThresholdLab = Schemas["ThresholdLab"];
export type ModelInfo = Schemas["ModelInfo"];
export type MonitoringSummary = Schemas["MonitoringSummary"];
export type Decision = Alert["decision"];

export const api = createClient<paths>({
  baseUrl: new URL("/api", window.location.origin).toString(),
  // Resolve fetch at call time, so tests can stub it.
  fetch: (request) => globalThis.fetch(request),
});

/** An API response that was not 2xx. */
export class ApiError extends Error {
  readonly status: number;

  constructor(status: number, detail: string) {
    super(detail);
    this.name = "ApiError";
    this.status = status;
  }
}

/** Return the data of an openapi-fetch result, or throw an ApiError carrying its detail. */
export async function unwrap<T>(
  call: Promise<{ data?: T; error?: unknown; response: Response }>,
): Promise<T> {
  const { data, error, response } = await call;
  if (response.ok && data !== undefined) {
    return data;
  }
  const detail =
    typeof error === "object" && error !== null && "detail" in error
      ? String((error as { detail: unknown }).detail)
      : response.statusText || "request failed";
  throw new ApiError(response.status, detail);
}
