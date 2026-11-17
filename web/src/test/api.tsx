// A fetch stub that answers the console's requests from a route table, and a renderer
// that mounts the whole app at a given URL.
import { QueryClient } from "@tanstack/react-query";
import { render } from "@testing-library/react";
import { MemoryRouter } from "react-router";
import { vi } from "vitest";

import { App } from "../App";

export type Reply = { status?: number; body: unknown };
export type Handler = (request: Request) => Reply | Promise<Reply>;
export type Calls = { method: string; path: string; body: unknown }[];

/**
 * Stub ``fetch``. Routes are keyed "METHOD /path" with the path after /api and without
 * the query string, for example "GET /v1/alerts". Unknown routes answer 404.
 */
export function mockApi(routes: Record<string, Handler | Reply>): Calls {
  const calls: Calls = [];
  vi.stubGlobal(
    "fetch",
    vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const request = input instanceof Request ? input : new Request(input, init);
      const url = new URL(request.url);
      const path = url.pathname.replace(/^\/api/, "");
      const text = request.method === "GET" ? "" : await request.clone().text();
      calls.push({ method: request.method, path, body: text ? JSON.parse(text) : undefined });
      const route = routes[`${request.method} ${path}`];
      const reply = typeof route === "function" ? await route(request) : route;
      if (!reply) return json(404, { detail: `no route for ${request.method} ${path}` });
      return json(reply.status ?? 200, reply.body);
    }),
  );
  return calls;
}

function json(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

/** Mount the app at ``url`` with a fresh query cache that does not retry. */
export function renderAt(url: string) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  return render(
    <MemoryRouter initialEntries={[url]}>
      <App client={client} />
    </MemoryRouter>,
  );
}
