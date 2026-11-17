import { QueryClient } from "@tanstack/react-query";

/** The app's query cache: one retry, and no refetch merely because the tab regained focus. */
export function makeQueryClient(): QueryClient {
  return new QueryClient({
    defaultOptions: { queries: { retry: 1, refetchOnWindowFocus: false } },
  });
}
