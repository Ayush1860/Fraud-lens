// Server state through TanStack Query: one hook per resource the screens read.
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { api, unwrap, type Decision } from "./client";
import type { ExperimentsFile } from "../lib/experiments";

/** How often the alert queue polls, in milliseconds. */
export const QUEUE_POLL_MS = 5_000;

export type AlertFilter = { decision?: Exclude<Decision, "approve">; status: "open" | "closed" };

export function useAlerts(filter: AlertFilter) {
  return useQuery({
    queryKey: ["alerts", filter],
    queryFn: () =>
      unwrap(
        api.GET("/v1/alerts", {
          params: { query: { status: filter.status, decision: filter.decision, limit: 100 } },
        }),
      ),
    refetchInterval: QUEUE_POLL_MS,
  });
}

export function useDecision(transactionId: string) {
  return useQuery({
    queryKey: ["decision", transactionId],
    queryFn: () =>
      unwrap(
        api.GET("/v1/transactions/{transaction_id}", {
          params: { path: { transaction_id: transactionId } },
        }),
      ),
  });
}

export function useExplanation(transactionId: string) {
  return useQuery({
    queryKey: ["explanation", transactionId],
    queryFn: () =>
      unwrap(
        api.GET("/v1/transactions/{transaction_id}/explanation", {
          params: { path: { transaction_id: transactionId } },
        }),
      ),
  });
}

export function useCardHistory(transactionId: string) {
  return useQuery({
    queryKey: ["card-history", transactionId],
    queryFn: () =>
      unwrap(
        api.GET("/v1/transactions/{transaction_id}/card-history", {
          params: { path: { transaction_id: transactionId }, query: { limit: 20 } },
        }),
      ),
  });
}

export function useThresholdLab() {
  return useQuery({
    queryKey: ["threshold-lab"],
    queryFn: () => unwrap(api.GET("/v1/model/threshold-curve")),
    staleTime: Infinity,
  });
}

export function useModel() {
  return useQuery({ queryKey: ["model"], queryFn: () => unwrap(api.GET("/v1/model")) });
}

export function useMonitoring() {
  return useQuery({
    queryKey: ["monitoring"],
    queryFn: () => unwrap(api.GET("/v1/monitoring/summary")),
    refetchInterval: 15_000,
  });
}

/** The imbalance study, a static file generated from docs/results/experiments.json. */
export function useExperiments() {
  return useQuery({
    queryKey: ["experiments"],
    queryFn: async (): Promise<ExperimentsFile> => {
      const response = await globalThis.fetch(
        new URL("/data/experiments.json", window.location.origin),
      );
      if (!response.ok) {
        throw new Error(`experiments.json: ${response.status}`);
      }
      return (await response.json()) as ExperimentsFile;
    },
    staleTime: Infinity,
  });
}

export type DispositionInput = {
  alertId: number;
  label: "fraud" | "genuine";
  analyst: string;
  note: string;
};

export function useDisposition() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({ alertId, label, analyst, note }: DispositionInput) =>
      unwrap(
        api.POST("/v1/alerts/{alert_id}/disposition", {
          params: { path: { alert_id: alertId } },
          body: { label, analyst, note: note.trim() || null },
        }),
      ),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ["alerts"] });
      void queryClient.invalidateQueries({ queryKey: ["monitoring"] });
    },
  });
}
