import { fireEvent, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";

import { mockApi, renderAt } from "../test/api";
import {
  alert,
  decision,
  driftReport,
  experiments,
  explanation,
  history,
  lab,
  model,
  monitoring,
} from "../test/fixtures";

const failing = { status: 503, body: { detail: "audit store unavailable" } };

describe("shell", () => {
  it("always shows the simulated-data banner", async () => {
    mockApi({ "GET /v1/alerts": { body: { items: [], next_cursor: null } } });
    renderAt("/");

    expect(screen.getByRole("note")).toHaveTextContent("Simulated data");
    expect(await screen.findByText(/No open alerts/)).toBeInTheDocument();
  });
});

describe("alert queue", () => {
  it("lists open alerts with a worded top reason and a labelled decision", async () => {
    const calls = mockApi({ "GET /v1/alerts": { body: { items: [alert], next_cursor: null } } });
    renderAt("/");

    const row = (await screen.findByText("$528.25")).closest("tr");
    expect(row).not.toBeNull();
    const cells = within(row as HTMLElement);
    expect(cells.getByText("Block")).toBeInTheDocument();
    expect(cells.getByText(alert.top_reason as string)).toBeInTheDocument();
    expect(cells.getByRole("link")).toHaveAttribute("href", "/cases/tx-7?alert=7");
    expect(calls[0]?.path).toBe("/v1/alerts");
  });

  it("filters by decision", async () => {
    const seen: string[] = [];
    mockApi({
      "GET /v1/alerts": (request) => {
        seen.push(new URL(request.url).search);
        return { body: { items: [], next_cursor: null } };
      },
    });
    renderAt("/");
    await screen.findByText(/No open alerts/);

    await userEvent.selectOptions(screen.getByLabelText("Decision"), "block");

    await waitFor(() => expect(seen.at(-1)).toContain("decision=block"));
    expect(seen[0]).toContain("status=open");
  });

  it("shows a loading state, then an error with a retry", async () => {
    mockApi({ "GET /v1/alerts": failing });
    renderAt("/");

    expect(screen.getByRole("status")).toHaveTextContent("Loading alerts");
    expect(await screen.findByRole("alert")).toHaveTextContent("audit store unavailable");
    expect(screen.getByRole("button", { name: "Try again" })).toBeInTheDocument();
  });
});

const caseRoutes = {
  "GET /v1/transactions/tx-7": { body: decision },
  "GET /v1/transactions/tx-7/explanation": { body: explanation },
  "GET /v1/transactions/tx-7/card-history": { body: history },
};

describe("case view", () => {
  it("puts the reasons in words first, with the waterfall and the card's history", async () => {
    mockApi(caseRoutes);
    renderAt("/cases/tx-7?alert=7");

    expect(
      await screen.findByRole("heading", { name: /\$528\.25 · shopping_net/ }),
    ).toBeInTheDocument();
    const reasons = await screen.findAllByRole("listitem");
    expect(reasons[0]).toHaveTextContent("Amount is 8.2 times");
    expect(reasons[0]).toHaveTextContent("55% of the push");
    expect(screen.getByText(/Pulling the other way/).closest("p")).toHaveTextContent(
      "12 km from home",
    );
    const current = (await screen.findByText("· this one")).closest("tr");
    expect(current).toHaveAttribute("aria-current", "true");
  });

  it("posts a disposition and confirms it", async () => {
    const calls = mockApi({
      ...caseRoutes,
      "POST /v1/alerts/7/disposition": {
        status: 201,
        body: {
          id: 1,
          alert_id: 7,
          label: "fraud",
          analyst: "maria",
          note: "called the cardholder",
          created_at: "2020-08-14T04:00:00Z",
        },
      },
    });
    renderAt("/cases/tx-7?alert=7");
    const analyst = await screen.findByLabelText("Analyst");

    await userEvent.clear(analyst);
    await userEvent.type(analyst, "maria");
    await userEvent.type(screen.getByLabelText("Note"), "called the cardholder");
    await userEvent.click(screen.getByRole("button", { name: "Confirm fraud" }));

    expect(await screen.findByText(/Recorded: confirmed fraud by maria/)).toBeInTheDocument();
    const post = calls.find((c) => c.method === "POST");
    expect(post).toEqual({
      method: "POST",
      path: "/v1/alerts/7/disposition",
      body: { label: "fraud", analyst: "maria", note: "called the cardholder" },
    });
  });

  it("shows the API's reason when a disposition fails", async () => {
    mockApi({ ...caseRoutes, "POST /v1/alerts/7/disposition": failing });
    renderAt("/cases/tx-7?alert=7");

    await userEvent.click(await screen.findByRole("button", { name: "Mark genuine" }));

    expect(await screen.findByRole("alert")).toHaveTextContent("audit store unavailable");
  });

  it("asks for the queue when opened without an alert, and handles an empty history", async () => {
    mockApi({ ...caseRoutes, "GET /v1/transactions/tx-7/card-history": { body: [] } });
    renderAt("/cases/tx-7");

    expect(await screen.findByText(/Open this case from the alert queue/)).toBeInTheDocument();
    expect(await screen.findByText(/No earlier transactions/)).toBeInTheDocument();
  });

  it("shows an error for an unknown transaction", async () => {
    mockApi({});
    renderAt("/cases/nope");

    expect(await screen.findByRole("alert")).toHaveTextContent("no route");
  });
});

describe("threshold lab", () => {
  it("starts at the released thresholds and reprices when a slider moves", async () => {
    mockApi({ "GET /v1/model/threshold-curve": { body: lab } });
    renderAt("/thresholds");

    const stats = await screen.findByRole("table", { name: "Operating point" });
    const costRow = within(stats)
      .getByText(/Expected cost/)
      .closest("tr") as HTMLElement;
    expect(within(costRow).getAllByText("$25.50")).toHaveLength(2);
    expect(screen.getByText("These are the released thresholds.")).toBeInTheDocument();

    fireEvent.change(screen.getByLabelText(/Review threshold/), { target: { value: "0" } });

    await waitFor(() => expect(within(costRow).getByText("$395.00")).toBeInTheDocument());
    expect(screen.getByText(/Over the review capacity/)).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: /Back to the released/ }));
    expect(screen.getByText("These are the released thresholds.")).toBeInTheDocument();
  });

  it("shows an error when the curve cannot load", async () => {
    mockApi({ "GET /v1/model/threshold-curve": failing });
    renderAt("/thresholds");

    expect(await screen.findByRole("alert")).toBeInTheDocument();
  });
});

describe("model and monitoring", () => {
  it("compares the live decision mix with validation and waits for drift", async () => {
    mockApi({
      "GET /v1/model": { body: model },
      "GET /v1/monitoring/summary": { body: monitoring },
    });
    renderAt("/model");

    const mix = await screen.findByRole("table", { name: "Decision mix" });
    const block = within(mix).getByText("block").closest("tr") as HTMLElement;
    expect(within(block).getByText("2.00%")).toBeInTheDocument();
    expect(within(block).getByText("0.60%")).toBeInTheDocument();
    expect(screen.getByText("75.0% (3 fraud, 1 genuine)")).toBeInTheDocument();
    expect(screen.getByText("8 ms / 14 ms / 21 ms")).toBeInTheDocument();
    expect(screen.getByText(/first drift report is computed/)).toBeInTheDocument();
    expect(screen.getByText("0.9900")).toBeInTheDocument();
    expect(screen.getByText("0.9998")).toBeInTheDocument();
  });

  it("shows a drift alarm, and never alarms on calendar features", async () => {
    mockApi({
      "GET /v1/model": { body: model },
      "GET /v1/monitoring/summary": { body: { ...monitoring, drift: driftReport } },
    });
    renderAt("/model");

    const alarm = await screen.findByText("Drift alarm.");
    expect(alarm.closest("div")).toHaveTextContent("amount in shopping_net (PSI 0.34)");
    const segments = screen.getByRole("table", { name: "Amount by category" });
    const travel = within(segments).getByText("travel").closest("tr") as HTMLElement;
    expect(within(travel).getByText("Too few rows")).toBeInTheDocument();
    const features = screen.getByRole("table", { name: "Features against training" });
    const calendar = within(features).getByText("day_of_week").closest("tr") as HTMLElement;
    expect(within(calendar).getByText(/shorter than a week/)).toBeInTheDocument();
  });

  it("has an empty state before any traffic", async () => {
    mockApi({
      "GET /v1/model": { body: model },
      "GET /v1/monitoring/summary": { body: { ...monitoring, window: 0 } },
    });
    renderAt("/model");

    expect(await screen.findByText(/No decisions yet/)).toBeInTheDocument();
  });

  it("shows an error when the model cannot load", async () => {
    mockApi({ "GET /v1/model": failing });
    renderAt("/model");

    expect(await screen.findByRole("alert")).toBeInTheDocument();
  });
});

describe("experiments", () => {
  it("tabulates each arm with its interval", async () => {
    mockApi({ "GET /data/experiments.json": { body: experiments } });
    renderAt("/experiments");

    const row = (await screen.findByRole("rowheader", { name: /A1/ })).closest("tr") as HTMLElement;
    expect(within(row).getByText("0.9870")).toBeInTheDocument();
    expect(within(row).getByText("[0.9820, 0.9920]")).toBeInTheDocument();
    expect(within(row).getAllByText("n/a").length).toBeGreaterThan(0);
  });

  it("has empty and error states", async () => {
    mockApi({ "GET /data/experiments.json": { body: { ...experiments, arms: [] } } });
    renderAt("/experiments");
    expect(await screen.findByText(/No experiment results yet/)).toBeInTheDocument();
  });

  it("reports a missing results file", async () => {
    mockApi({});
    renderAt("/experiments");
    expect(await screen.findByRole("alert")).toHaveTextContent("experiments.json: 404");
  });
});
